"""Audit all public THCA XML supplements without promoting dates to availability."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET

import requests


API = 'https://api.gdc.cancer.gov'
FORMATS = ['BCR XML', 'BCR SSF XML', 'BCR OMF XML']


def pin_xml(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    filters = dict(op='and', content=[dict(op='in', content=dict(field=field, value=values)) for field, values in (
        ('cases.project.project_id', ['TCGA-THCA']), ('data_type', ['Clinical Supplement', 'Biospecimen Supplement']),
        ('data_format', FORMATS), ('access', ['open']))])
    params = dict(filters=json.dumps(filters), fields='file_id,file_name,file_size,md5sum,data_type,data_format,access,cases.submitter_id',
                  size=500, sort='file_id:asc')
    hits, offset, total = [], 0, None
    while total is None or offset < total:
        response = requests.get(API+'/files', params={**params, 'from': offset}, timeout=60)
        response.raise_for_status()
        data = response.json()['data']
        total = data['pagination']['total']
        if not data['hits']:
            raise ValueError('XML pagination ended early')
        hits.extend(data['hits'])
        offset += len(data['hits'])
    if not hits or len({row['file_id'] for row in hits}) != len(hits):
        raise ValueError('Empty or duplicate XML source snapshot')
    report = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), project='TCGA-THCA', endpoint=API+'/files', query=params, files=hits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def verify_record(record, content):
    if (record.get('access') != 'open' or record.get('data_format') not in FORMATS
            or not re.fullmatch(r'[0-9a-f-]{36}', record['file_id']) or not record['file_name'].endswith('.xml')):
        raise ValueError('Unsupported XML source metadata')
    if len(content) != record['file_size'] or hashlib.md5(content).hexdigest() != record['md5sum']:
        raise ValueError('XML publisher checksum or size differs')


def consume_archive(content, expected, root):
    """Read only pinned regular files. Never extract a publisher archive pathname."""
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        for member in archive:
            if member.name == 'MANIFEST.txt' and member.isfile():
                continue
            parts = member.name.split('/')
            if len(parts) != 2 or not member.isfile() or parts[0] not in expected:
                raise ValueError('Unexpected or unsafe XML archive member')
            record = expected[parts[0]]
            if parts[0] in seen or parts[1] != record['file_name'] or member.size != record['file_size']:
                raise ValueError('XML archive identity or size differs')
            body = archive.extractfile(member).read(record['file_size']+1)
            verify_record(record, body)
            path = Path(root)/(record['file_id']+'.xml')
            temporary = path.with_suffix('.part')
            temporary.write_bytes(body)
            temporary.replace(path)
            seen.add(parts[0])
    if seen != set(expected):
        raise ValueError('XML archive omitted pinned files')


def acquire_xml(manifest, root, batch_size=50):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    required = sum(row['file_size'] for row in manifest['files'])
    if shutil.disk_usage(root).free < required+200*1024**2:
        raise ValueError('Insufficient disk for public XML supplements and reserve')
    missing = []
    for row in manifest['files']:
        if not re.fullmatch(r'[0-9a-f-]{36}', row['file_id']):
            raise ValueError('Unsafe XML file UUID')
        path = root/(row['file_id']+'.xml')
        if path.exists():
            verify_record(row, path.read_bytes())
        else:
            missing.append(row)
    failures = []
    def individual(record):
        last = None
        for attempt in range(2):
            try:
                response = requests.get(API+'/data/'+record['file_id'], timeout=(20,60))
                response.raise_for_status()
                verify_record(record, response.content)
                path = root/(record['file_id']+'.xml')
                temporary = path.with_suffix('.part')
                temporary.write_bytes(response.content)
                temporary.replace(path)
                return None
            except (requests.RequestException, ValueError) as error:
                last = str(error)
        return dict(file_id=record['file_id'], file_name=record['file_name'], error=last)
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset:offset+batch_size]
        try:
            response = requests.post(API+'/data', json=dict(ids=[row['file_id'] for row in batch]), timeout=(30,120))
            response.raise_for_status()
            consume_archive(response.content, {row['file_id']: row for row in batch}, root)
        except (requests.RequestException, tarfile.TarError, ValueError):
            # A bad publisher batch must not prevent checking unrelated public files.
            with ThreadPoolExecutor(max_workers=4) as pool:
                failures.extend(row for row in pool.map(individual, batch) if row is not None)
        print(f'Checked {min(offset+batch_size,len(missing))}/{len(missing)} remaining XML files; unavailable={len(failures)}', flush=True)
    return failures


def inspect_xml(record, body):
    verify_record(record, body)
    if b'<!DOCTYPE' in body.upper() or b'<!ENTITY' in body.upper():
        raise ValueError('XML declarations with entities are unsupported')
    tree = ET.fromstring(body)
    local = lambda tag: tag.split('}')[-1]
    cases = {str(node.text).strip() for node in tree.iter() if local(node.tag) == 'bcr_patient_barcode' and node.text}
    expected = {row['submitter_id'] for row in record.get('cases', [])}
    if not cases or cases != expected:
        raise ValueError('XML patient barcodes disagree with publisher case metadata')
    leaves = []
    def visit(node, path):
        name = local(node.tag)
        path = path+'/'+name
        if not list(node) and re.search(r'days_to|day_of|month_of|year_of|date|time|availab|result|patholog|genotyp|mutation|callab|coverage|quality|depth|sequenc|capture|report|assay', name):
            leaves.append((path, (node.text or '').strip()))
        for child in node:
            visit(child, path)
    visit(tree, '')
    return cases, leaves


def inventory_xml(manifest, root):
    inventory = {}
    files = []
    all_cases = set()
    for record in manifest['files']:
        path = Path(root)/(record['file_id']+'.xml')
        if not path.is_file():
            continue
        body = path.read_bytes()
        cases, leaves = inspect_xml(record, body)
        all_cases |= cases
        group = record['data_type']+' / '+record['data_format']
        for field, value in leaves:
            entry = inventory.setdefault((group, field), dict(values=Counter(), files=set(), cases=set(), numeric=[]))
            entry['values'][value] += 1
            if value and not value.startswith('['):
                entry['files'].add(record['file_id'])
                entry['cases'].update(cases)
                if 'days_to' in field.split('/')[-1]:
                    try:
                        number = float(value)
                        if math.isfinite(number):
                            entry['numeric'].append(number)
                    except ValueError:
                        pass
        files.append(dict(file_id=record['file_id'], file_name=record['file_name'], sha256=hashlib.sha256(body).hexdigest()))
    fields = []
    for (group, field), entry in sorted(inventory.items()):
        row = dict(source_group=group, field_path=field, occurrences=sum(entry['values'].values()),
                   populated_files=len(entry['files']), populated_cases=len(entry['cases']))
        if 'days_to' in field.split('/')[-1]:
            row.update(finite_numeric_occurrences=len(entry['numeric']), minimum=min(entry['numeric'], default=None), maximum=max(entry['numeric'], default=None))
        if any(word in field.split('/')[-1] for word in ('genotyp', 'mutation', 'callab', 'coverage', 'assay')):
            row['reported_values'] = dict(sorted(entry['values'].items()))
        fields.append(row)
    return dict(files=files, file_count=len(files), expected_files=len(manifest['files']),
                total_source_bytes=sum(record['file_size'] for record in manifest['files'] if (Path(root)/(record['file_id']+'.xml')).is_file()),
                acquisition_complete=len(files)==len(manifest['files']), case_count=len(all_cases), fields=fields,
                formats=dict(Counter(row['data_format'] for row in manifest['files'] if (Path(root)/(row['file_id']+'.xml')).is_file())),
                cross_table_chronology_verified=False, callable_panel_verified=False,
                interpretation='All pinned public XML supplement leaves matching timing/pathology/genotyping/assay terms were inventoried with full element paths. Specimen review, form completion and upload dates are not clinical/genomic result availability. No unavailable mutation was changed to a negative call.')

"""Pin the publisher's extraction code and distinguish it from exact WiDS provenance."""
import hashlib
import json
from pathlib import Path

import requests

PIN = '95b69b99620d322b088262a41cd8098a1aed2f3e'
FILES = ('README.md', 'eicu/sql/data.sql', 'eicu/sql/lab.sql', 'eicu/sql/lab-d1.sql',
         'eicu/load-data.ipynb', 'merge-data.ipynb', 'hdr/variable-definitions.yaml')


def audit():
    response = requests.get(f'https://api.github.com/repos/MIT-LCP/gossis/git/trees/{PIN}',
                            params={'recursive': 1}, timeout=60)
    response.raise_for_status()
    tree = response.json()
    if tree.get('truncated'):
        raise ValueError('Publisher source tree was truncated')
    expected = {row['path']: row['sha'] for row in tree['tree'] if row['type'] == 'blob'}
    rows, texts = [], {}
    for name in FILES:
        url = f'https://raw.githubusercontent.com/MIT-LCP/gossis/{PIN}/{name}'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = response.content
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        if blob != expected[name]:
            raise ValueError('Source bytes differ from pinned publisher Git blob')
        path = Path('.cache/gossis-pinned')/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        text = content.decode('utf-8')
        if name.endswith('.ipynb'):
            notebook = json.loads(text)
            text = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'])
        texts[name] = text
        rows.append(dict(path=name, url=url, bytes=len(content), git_blob=blob,
                         sha256=hashlib.sha256(content).hexdigest()))
    def lines(name, fragment):
        found = [i+1 for i, line in enumerate(texts[name].splitlines()) if fragment in line]
        if not found:
            raise ValueError(f'Expected extraction-code anchor absent: {name}/{fragment}')
        return found
    report = dict(publisher_repository='https://github.com/MIT-LCP/gossis', pinned_revision=PIN, files=rows,
        observations=[
            dict(file='eicu/sql/data.sql', lines=lines('eicu/sql/data.sql', 'lab_d1.creatinine_max'),
                 observation='The eICU extraction projects aggregated day-one creatinine directly into the named output field.'),
            dict(file='eicu/sql/lab.sql', lines=lines('eicu/sql/lab.sql', "lab.labname = 'creatinine'"),
                 observation='The eICU raw-lab filter uses a 0.1 to 28.28 creatinine range, not the micromol-scale example in the dictionary.'),
            dict(file='eicu/sql/data.sql', lines=lines('eicu/sql/data.sql', 'predictedhospitalmortality'),
                 observation='The APACHE hospital prediction is cast from the eICU APACHE result table in this extraction.'),
            dict(file='hdr/variable-definitions.yaml', lines=lines('hdr/variable-definitions.yaml', 'd1_creatinine_max:'),
                 observation='The header source retains the micromol/L creatinine definition also found in the downloaded WiDS dictionary.')],
        exact_wids_extraction_revision_verified=False, creatinine_unit_resolved=False,
        apache_score_availability_verified=False,
        interpretation='Publisher extraction code supplies a plausible source lineage and exposes an internal unit-definition conflict. It does not prove that the inspected revision generated the frozen WiDS release, resolve clinical availability, or establish independent cohort identity.')
    target = Path('results/research_checks/gossis_source.json')
    target.write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    report = audit()
    print(json.dumps(dict(files=len(report['files']), pinned_revision=PIN,
                         exact_wids_extraction_revision_verified=False, creatinine_unit_resolved=False)))

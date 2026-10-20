import json
import zipfile

import pytest

from scripts.package_reproducibility import TOP, build, verify
from scripts.reproduce_studies import ROOT, reproduce


def test_source_package_excludes_local_data_and_detects_changed_bytes(tmp_path):
    root = tmp_path/'project'
    root.mkdir()
    for name in TOP:
        (root/name).write_text('source')
    for folder in ('src', 'scripts', 'tests', 'docs', 'paper', 'data/private', 'results/private'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    (root/'src/model.py').write_text('pass')
    (root/'data/private/patients.csv').write_text('private')
    (root/'results/private/key.json').write_text('private')
    archive = tmp_path/'source.zip'
    assert build(root, archive)['passed']
    with zipfile.ZipFile(archive) as package:
        content = {name:package.read(name) for name in package.namelist()}
        assert not any(name.startswith(('data/', 'results/')) for name in content)
    content['src/model.py'] = b'changed'
    with zipfile.ZipFile(archive, 'w') as package:
        for name, value in content.items():
            package.writestr(name, value)
    with pytest.raises(ValueError, match='hash/size mismatch'):
        verify(archive)


def test_package_rejects_unlisted_members(tmp_path):
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as package:
        package.writestr('PACKAGE_MANIFEST.json', json.dumps(dict(files=[])))
        package.writestr('data/private.csv', 'private')
    with pytest.raises(ValueError, match='inventory differs'):
        verify(path)


def test_recipe_dry_run_and_original_output_protection(tmp_path):
    plan = reproduce(tmp_path/"reproduction")
    assert not plan["executed"] and plan["fits"] == 36
    assert len({row["output"] for row in plan["plan"]}) == 36
    assert not (tmp_path/"reproduction").exists()
    with pytest.raises(ValueError, match="separate output root"):
        reproduce(ROOT/"results")

"""Build a self-contained public source ZIP with assets and license notices."""
import hashlib
import json
from pathlib import Path
import zipfile
from combatsim.env import Config
from combatsim.factory import load_settings

ROOT = Path(__file__).resolve().parents[1]

def main():
    for path in (ROOT/'configs').glob('scenario*.json'):
        Config(**load_settings(path))
    fixed = ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'SOURCE.md',
             'pyproject.toml', 'uv.lock', '.python-version', '.gitignore']
    files = {ROOT/name for name in fixed}
    for folder in ('combatsim', 'configs', 'experiments', 'examples', 'scripts', 'docs'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    payload.update({'artifacts/.gitkeep': b'', 'models/.gitkeep': b''})
    manifest = {'files': {name: hashlib.sha256(data).hexdigest() for name,data in payload.items()}}
    payload['distribution-manifest.json'] = json.dumps(manifest,indent=2).encode()
    output = ROOT/'artifacts/distribution/CombatSim.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in payload.items():
            archive.writestr('CombatSim/'+name,data)
    print(output)

if __name__ == '__main__':
    main()

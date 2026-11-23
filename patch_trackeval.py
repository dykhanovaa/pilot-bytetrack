"""NumPy alias compatibility patch; works with vendored files or an upstream clone."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path(__file__).resolve().parent
vendor = root / 'vendor' / 'TrackEval'
manifest = vendor / 'UPSTREAM.json'
patch_file = root / 'trackeval_numpy.patch'
files = [
    'trackeval/datasets/mot_challenge_2d_box.py',
    'trackeval/metrics/hota.py',
    'trackeval/metrics/identity.py',
    'trackeval/metrics/clear.py',
]

def sha(data):
    return hashlib.sha256(data).hexdigest()

if manifest.exists():
    metadata = json.loads(manifest.read_text())
    if sha(patch_file.read_bytes()) != metadata['numpy_patch_sha256']:
        raise RuntimeError('trackeval_numpy.patch differs from the pinned patch')
    for relative, expected in metadata['compatibility_files'].items():
        path = vendor / relative
        original = path.read_bytes()
        if sha(original) == expected['patched_sha256']:
            continue
        if sha(original) != expected['upstream_sha256']:
            raise RuntimeError(f'Unexpected TrackEval changes: {relative}')
        updated = re.sub(r'np\.(float|int|bool)\b', lambda m: m.group(1), original.decode()).encode()
        if sha(updated) != expected['patched_sha256']:
            raise RuntimeError(f'Compatibility patch mismatch: {relative}')
        path.write_bytes(updated)
else:
    # Support the documented setup from a fresh, independently cloned upstream.
    if not (vendor / '.git').is_dir():
        raise RuntimeError('TrackEval requires UPSTREAM.json or its own upstream Git metadata')
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '12c8791b303e0a0b50f753af204249e622d0281a':
        raise RuntimeError('Unexpected upstream TrackEval commit')
    metadata = {'repository': 'https://github.com/JonathonLuiten/TrackEval',
                'commit': commit, 'compatibility_files': {}}
    for relative in files:
        path = vendor / relative
        upstream = subprocess.check_output(['git', '-C', str(vendor), 'show', f'HEAD:{relative}'])
        updated = re.sub(r'np\.(float|int|bool)\b', lambda m: m.group(1), upstream.decode()).encode()
        if sha(path.read_bytes()) not in {sha(upstream), sha(updated)}:
            raise RuntimeError(f'Unexpected TrackEval changes: {relative}')
        path.write_bytes(updated)
        metadata['compatibility_files'][relative] = {
            'upstream_sha256': sha(upstream), 'patched_sha256': sha(updated)}
    patch = subprocess.check_output(['git', '-C', str(vendor), 'diff', '--', *files])
    if patch_file.exists() and patch != patch_file.read_bytes():
        raise RuntimeError('Generated patch differs from the supplied patch')
    patch_file.write_bytes(patch)
    metadata['numpy_patch_sha256'] = sha(patch)
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
print('Patch SHA256', sha(patch_file.read_bytes()))

"""Standard-library integrity and syntax check; does not deserialize models."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for name, expected in manifest['files'].items():
    path = root / name
    assert path.resolve().is_relative_to(root), name
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, name
    if path.suffix == '.py':
        ast.parse(raw, filename=name)
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual == set(manifest['files']) | {'EXPORT_MANIFEST.json'}, 'Unexpected/missing files'
print(f"Verified {len(manifest['files'])} files; source syntax passed. Models not executed.")

"""Load pinned official mechanics without importing unrelated Kaggle environments."""
import ast
import hashlib
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / 'expansion_research/schedule_training_r3/sources/official_engine.py'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'


def load_engine():
    raw = ENGINE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ENGINE_SHA
    tree = ast.parse(raw)
    # Preserve official constants/functions unchanged, omit framework bootstrap
    # and JSON/renderer file loading at module scope.
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.Import))
                 or (isinstance(node, ast.ImportFrom) and node.module == 'os')
                 or (isinstance(node, ast.Assign) and all(
                     isinstance(t, ast.Name) and t.id.isupper() for t in node.targets))]
    module = types.ModuleType('pinned_kaggriculture')
    module.__file__ = str(ENGINE_PATH)
    exec(compile(tree, str(ENGINE_PATH), 'exec'), module.__dict__)
    return module


E = load_engine()

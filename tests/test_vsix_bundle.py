"""The VS Code extension shipped inside the Python package matches its source."""

import json
from pathlib import Path
import zipfile

from arrayview._vscode_extension import _VSCODE_EXT_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_vsix_is_built_from_current_extension_source():
    with zipfile.ZipFile(ROOT / "src/arrayview/arrayview-opener.vsix") as zf:
        package = json.loads(zf.read("extension/package.json"))
        assert package["version"] == _VSCODE_EXT_VERSION
        for name in ("extension.js", "lifecycle_helpers.js", "package.json"):
            assert zf.read(f"extension/{name}") == (
                ROOT / "vscode-extension" / name
            ).read_bytes(), f"bundled {name} is stale; rebuild the .vsix"

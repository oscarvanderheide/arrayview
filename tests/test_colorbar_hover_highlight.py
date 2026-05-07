from pathlib import Path


VIEWER_HTML = Path(__file__).parent.parent / "src" / "arrayview" / "_viewer.html"


def test_colorbar_hover_dimming_branch_present():
    src = VIEWER_HTML.read_text()
    assert "const hoverBinIdx = (hoverBinCount > 0 && this._hoverFrac >= 0 && this._hoverFrac <= 1)" in src
    assert "if (pxBinIdx !== hoverBinIdx) drawAlpha *= 0.28;" in src

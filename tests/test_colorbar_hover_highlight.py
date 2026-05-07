from pathlib import Path


VIEWER_HTML = Path(__file__).parent.parent / "src" / "arrayview" / "_viewer.html"


def test_colorbar_hover_bar_bin_mapping_branch_present():
    src = VIEWER_HTML.read_text()
    assert "_hoverBarValueAt(frac)" in src
    assert "_hoverBinIndexForPoint(frac, y, cssH)" in src
    assert "const isBarStrip = y >= Math.max(0, cssH - CB_COLLAPSED_H);" in src

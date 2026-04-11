"""Unit tests for the View Component System (Phase 1+).

Each test calls into the browser via page.evaluate() and asserts the return
value. No DOM assertions in this file — that's in test_view_component_integration.py.
"""
import pytest

pytestmark = pytest.mark.browser


def test_display_state_factory_defaults(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("() => makeDisplayState()")
    assert result["vmin"] is None
    assert result["vmax"] is None
    assert result["quantileIdx"] == -1
    assert result["cmapIdx"] == 0
    assert result["logScale"] is False
    assert result["complexMode"] == 0
    assert result["projectionMode"] == 0
    assert result["renderMode"] == "scalar"
    assert result["alphaThreshold"] == 0
    assert result["overlaySids"] == []
    assert result["overlayAlpha"] == 0.45


def test_display_state_overrides(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("() => makeDisplayState({vmin: 5, logScale: true, cmapIdx: 3})")
    assert result["vmin"] == 5
    assert result["logScale"] is True
    assert result["cmapIdx"] == 3
    # other fields still defaulted
    assert result["vmax"] is None
    assert result["quantileIdx"] == -1


def test_free_slice_slicer_request(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("""() => {
        const slicer = new FreeSliceSlicer();
        const ds = makeDisplayState({sliceIndices: [5, 0, 0], currentSliceDim: 0});
        const session = {sid: 'test-sid', ndim: 3};
        return slicer.getRequest(session, ds);
    }""")
    assert result["sid"] == "test-sid"
    assert result["params"]["slice"] == [5, 0, 0]
    assert result["params"]["sliceDim"] == 0


def test_orthogonal_slicer_axial(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("""() => {
        const slicer = new OrthogonalSlicer('axial');
        const ds = makeDisplayState({sliceIndices: [10, 20, 30]});
        return slicer.getRequest({sid: 'test', ndim: 3}, ds);
    }""")
    assert result["params"]["axis"] == "axial"
    assert result["params"]["index"] == 10   # axial slices along first spatial axis


def test_image_layer_is_duck_typed(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("""() => {
        const layer = new ImageLayer();
        return {
            name: layer.name,
            hasDraw: typeof layer.draw === 'function',
            hasDestroy: typeof layer.destroy === 'function',
        };
    }""")
    assert result["name"] == "image"
    assert result["hasDraw"] is True
    assert result["hasDestroy"] is True

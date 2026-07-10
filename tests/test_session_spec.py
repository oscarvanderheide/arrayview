import json

import pytest

from arrayview._session_spec import (
    SessionCapability,
    SessionSpec,
    SourceKind,
    session_spec_from_cli_file,
    session_spec_from_python_array,
)


def test_cli_file_request_is_normalized() -> None:
    spec = session_spec_from_cli_file(
        "image.npz",
        name="T1",
        rgb=True,
        compare_sources=["followup.npy"],
        overlays=["mask.npy"],
        dims=":,:,x,y",
        select=["*t1*"],
        key="volume",
        watch=True,
        vectorfield="flow.npy",
        vectorfield_components_dim=3,
    )

    assert spec.source_kind is SourceKind.FILE
    assert spec.compare_sources == ("followup.npy",)
    assert spec.overlays == ("mask.npy",)
    assert spec.dims == (2, 3)
    assert spec.select == ("*t1*",)
    assert spec.required_capabilities == (
        SessionCapability.FILESYSTEM,
        SessionCapability.WATCH,
        SessionCapability.MULTI_SOURCE,
        SessionCapability.OVERLAYS,
        SessionCapability.VECTORFIELD,
    )


def test_python_array_request_uses_registration_ids() -> None:
    spec = session_spec_from_python_array(
        "array:0",
        compare_source_ids=["array:1"],
        overlay_source_ids=["overlay:0"],
        dims=[0, 2],
    )

    assert spec.source == "array:0"
    assert spec.source_kind is SourceKind.ARRAY
    assert spec.dims == (0, 2)
    assert spec.required_capabilities == (
        SessionCapability.IN_MEMORY,
        SessionCapability.MULTI_SOURCE,
        SessionCapability.OVERLAYS,
    )


def test_serialization_is_json_safe_and_stable() -> None:
    spec = session_spec_from_cli_file("image.npy", dims="2,3")

    encoded = spec.to_json()
    decoded = json.loads(encoded)

    assert decoded["source_kind"] == "file"
    assert decoded["dims"] == [2, 3]
    assert decoded["required_capabilities"] == ["filesystem"]


def test_named_dims_preserve_x_y_meaning() -> None:
    assert session_spec_from_cli_file("image.npy", dims=":,y,x").dims == (2, 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": ""}, "source"),
        ({"source": "x", "dims": (1, 1)}, "different axes"),
        ({"source": "x", "dims": (-1, 2)}, "non-negative"),
        ({"source": "x", "vectorfield_components_dim": 0}, "requires vectorfield"),
        ({"source": "x", "source_kind": SourceKind.ARRAY, "watch": True}, "file sources"),
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict, message: str) -> None:
    kwargs.setdefault("source_kind", SourceKind.FILE)
    with pytest.raises(ValueError, match=message):
        SessionSpec(**kwargs)


@pytest.mark.parametrize("dims", ["x,x,:,:", "x,:,:", "1,a", "1,2,3"])
def test_invalid_cli_dims_are_rejected(dims: str) -> None:
    with pytest.raises(ValueError, match="dims"):
        session_spec_from_cli_file("image.npy", dims=dims)

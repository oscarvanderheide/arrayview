from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arrayview import _launcher
from arrayview._tutorial import (
    cleanup_tutorial_bundle,
    create_tutorial_bundle,
    make_tutorial_arrays,
)


def test_tutorial_arrays_are_deterministic_and_composable():
    base_a, compare_a, overlay_a = make_tutorial_arrays()
    base_b, compare_b, overlay_b = make_tutorial_arrays()

    assert base_a.shape == (72, 72, 24, 12)
    assert compare_a.shape == base_a.shape
    assert overlay_a.shape == base_a.shape
    assert base_a.dtype == compare_a.dtype == np.float32
    assert overlay_a.dtype == np.uint8
    np.testing.assert_array_equal(base_a, base_b)
    np.testing.assert_array_equal(compare_a, compare_b)
    np.testing.assert_array_equal(overlay_a, overlay_b)
    assert not np.array_equal(base_a, compare_a)
    assert set(np.unique(overlay_a)) == {0, 1, 2, 3}


def test_tutorial_bundle_has_named_files_and_idempotent_cleanup():
    bundle = create_tutorial_bundle()

    try:
        assert Path(bundle.base_file).name == "tutorial-volume.npy"
        assert Path(bundle.compare_file).name == "comparison-volume.npy"
        assert Path(bundle.overlay_file).name == "regions-overlay.npy"
        assert all(
            Path(path).is_file()
            for path in (bundle.base_file, bundle.compare_file, bundle.overlay_file)
        )
    finally:
        cleanup_tutorial_bundle(bundle.directory)
        cleanup_tutorial_bundle(bundle.directory)

    assert not Path(bundle.directory).exists()


def test_no_file_cli_arguments_use_scalar_tutorial_bundle():
    args = SimpleNamespace(files=[], overlay=None, rgb=True)
    tutorial = _launcher._configure_tutorial_args(args)

    try:
        assert args.files == [tutorial.base_file, tutorial.compare_file]
        assert args.overlay == [f"Regions={tutorial.overlay_file}"]
        assert args._demo_name == "tutorial"
        assert args._demo_cleanup is False
        assert args._tutorial_cleanup_dir == tutorial.directory
        assert args.rgb is False
    finally:
        cleanup_tutorial_bundle(tutorial.directory)

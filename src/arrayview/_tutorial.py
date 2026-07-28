"""Deterministic generated data used by the interactive tutorial."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import tempfile


@dataclass(frozen=True)
class TutorialBundle:
    """Temporary file-backed inputs for the normal CLI registration path."""

    directory: str
    base_file: str
    compare_file: str
    overlay_file: str


def make_tutorial_arrays():
    """Return a scalar 4D volume, a comparison volume, and an integer mask."""
    import numpy as np

    height, width, depth, frames = 72, 72, 24, 12
    yy, xx, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        np.linspace(-1.0, 1.0, depth, dtype=np.float32),
        indexing="ij",
    )

    base = np.empty((height, width, depth, frames), dtype=np.float32)
    compare = np.empty_like(base)
    overlay = np.zeros(base.shape, dtype=np.uint8)

    for frame in range(frames):
        phase = np.float32(2.0 * np.pi * frame / frames)
        cx = np.float32(0.28 * np.sin(phase))
        cy = np.float32(0.22 * np.cos(phase))
        cz = np.float32(0.18 * np.sin(phase * 2.0))

        core = np.exp(
            -(
                ((xx - cx) / 0.42) ** 2
                + ((yy - cy) / 0.34) ** 2
                + ((zz - cz) / 0.55) ** 2
            )
        )
        satellite = np.exp(
            -(
                ((xx + 0.48) / 0.20) ** 2
                + ((yy - 0.32) / 0.18) ** 2
                + ((zz + 0.15) / 0.30) ** 2
            )
        )
        ripple = 0.12 * np.sin(8.0 * xx - phase) * np.cos(7.0 * yy + phase)
        base[..., frame] = (1.35 * core + 0.72 * satellite + ripple).astype(
            np.float32
        )

        shifted_core = np.exp(
            -(
                ((xx - cx - 0.07) / 0.44) ** 2
                + ((yy - cy + 0.04) / 0.36) ** 2
                + ((zz - cz) / 0.55) ** 2
            )
        )
        compare[..., frame] = (
            1.22 * shifted_core + 0.78 * satellite + ripple * 0.65 + 0.04
        ).astype(np.float32)

        overlay[..., frame][core > 0.58] = 1
        overlay[..., frame][satellite > 0.50] = 2
        ring = (core > 0.34) & (core < 0.43)
        overlay[..., frame][ring] = 3

    return base, compare, overlay


def create_tutorial_bundle() -> TutorialBundle:
    """Write tutorial inputs to a private temporary directory."""
    import numpy as np

    directory = tempfile.mkdtemp(prefix="arrayview-tutorial-")
    bundle = TutorialBundle(
        directory=directory,
        base_file=os.path.join(directory, "tutorial-volume.npy"),
        compare_file=os.path.join(directory, "comparison-volume.npy"),
        overlay_file=os.path.join(directory, "regions-overlay.npy"),
    )
    try:
        base, compare, overlay = make_tutorial_arrays()
        np.save(bundle.base_file, base)
        np.save(bundle.compare_file, compare)
        np.save(bundle.overlay_file, overlay)
    except Exception:
        cleanup_tutorial_bundle(directory)
        raise
    return bundle


def cleanup_tutorial_bundle(directory: str | None) -> None:
    """Remove a tutorial bundle after its sessions own the loaded data."""
    if directory:
        shutil.rmtree(directory, ignore_errors=True)

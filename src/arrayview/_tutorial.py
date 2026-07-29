"""Deterministic generated data used by the interactive tutorial."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import tempfile


@dataclass(frozen=True)
class TutorialBundle:
    """Temporary file-backed inputs for the normal CLI registration path.

    The first three are the tour's main array, its comparison partner, and a
    label mask, and they share one session set. The last three back sections
    that cannot live on that session: a vector field disables statistical
    projections for whatever session it is attached to, and a ragged
    collection is a single session built from differently shaped files. Both
    are registered separately and the tour navigates to them.
    """

    directory: str
    base_file: str
    compare_file: str
    overlay_file: str
    # The extras get their own directory because `directory` is removed as
    # soon as the main array is registered — in the spawned-daemon path that
    # happens inside the child, before the parent gets to register anything
    # else. Same lifetime rule, just released separately.
    extras_directory: str
    flow_file: str
    flow_field_file: str
    stack_pattern: str


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


def make_flow_arrays():
    """Return a volume and a matching displacement field.

    The field is a swirl around the z axis, which reads clearly as arrows at
    any arrow length — the point of the section is that the arrows are data,
    not decoration, so they have to obviously follow something.
    """
    import numpy as np

    height, width, depth = 64, 64, 16
    yy, xx, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        np.linspace(-1.0, 1.0, depth, dtype=np.float32),
        indexing="ij",
    )
    radius = np.sqrt(xx**2 + yy**2)
    volume = (
        np.exp(-((xx / 0.52) ** 2 + (yy / 0.46) ** 2 + (zz / 0.72) ** 2))
        + 0.18 * np.sin(6.0 * radius)
    ).astype(np.float32)

    falloff = np.exp(-((radius / 0.85) ** 2)).astype(np.float32)
    field = np.stack(
        [(-yy * falloff), (xx * falloff), (0.22 * zz * falloff)], axis=-1
    ).astype(np.float32)
    return volume, field


def make_stack_arrays():
    """Return several volumes that share a rank and dtype but not a shape.

    Differing shapes are the whole point: a dense stack would refuse these,
    and the collection keeps each item at its own size.
    """
    import numpy as np

    shapes = ((52, 44, 14), (44, 60, 10), (60, 52, 18), (48, 48, 12))
    volumes = []
    for index, (height, width, depth) in enumerate(shapes):
        yy, xx, zz = np.meshgrid(
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            np.linspace(-1.0, 1.0, depth, dtype=np.float32),
            indexing="ij",
        )
        offset = np.float32(0.16 * index - 0.24)
        volumes.append(
            (
                np.exp(-(((xx - offset) / 0.55) ** 2 + (yy / 0.48) ** 2 + (zz / 0.8) ** 2))
                + 0.12 * np.cos(5.0 * xx + 3.0 * yy)
            ).astype(np.float32)
        )
    return volumes


def create_tutorial_bundle() -> TutorialBundle:
    """Write tutorial inputs to a private temporary directory."""
    import numpy as np

    directory = tempfile.mkdtemp(prefix="arrayview-tutorial-")
    extras = tempfile.mkdtemp(prefix="arrayview-tutorial-extra-")
    cases_dir = os.path.join(extras, "cases")
    bundle = TutorialBundle(
        directory=directory,
        base_file=os.path.join(directory, "tutorial-volume.npy"),
        compare_file=os.path.join(directory, "comparison-volume.npy"),
        overlay_file=os.path.join(directory, "regions-overlay.npy"),
        extras_directory=extras,
        flow_file=os.path.join(extras, "flow-volume.npy"),
        flow_field_file=os.path.join(extras, "flow-field.npy"),
        stack_pattern=os.path.join(cases_dir, "*", "scan.npy"),
    )
    try:
        base, compare, overlay = make_tutorial_arrays()
        np.save(bundle.base_file, base)
        np.save(bundle.compare_file, compare)
        np.save(bundle.overlay_file, overlay)

        flow, field = make_flow_arrays()
        np.save(bundle.flow_file, flow)
        np.save(bundle.flow_field_file, field)

        for index, volume in enumerate(make_stack_arrays(), start=1):
            case_dir = os.path.join(cases_dir, f"case{index:02d}")
            os.makedirs(case_dir, exist_ok=True)
            np.save(os.path.join(case_dir, "scan.npy"), volume)
    except Exception:
        cleanup_tutorial_bundle(directory)
        cleanup_tutorial_bundle(extras)
        raise
    return bundle


def cleanup_tutorial_bundle(directory: str | None) -> None:
    """Remove a tutorial bundle after its sessions own the loaded data."""
    if directory:
        shutil.rmtree(directory, ignore_errors=True)

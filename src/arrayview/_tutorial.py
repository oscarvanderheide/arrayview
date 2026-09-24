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
    """Return a scalar 4D volume, a comparison volume, and an integer mask.

    The volume is a made-up head scan rather than a single blob, so that
    stepping through any dimension shows something change: a skull and
    folded tissue, two dark cavities, a tilted ring that splits in two, a
    bright tube that spirals through the slices, a small beating "heart"
    and a patch of fine stripes that rewards zooming and the Fourier view.
    The last dimension is time: the heart beats and the spiral turns.
    """
    import numpy as np

    height, width, depth, frames = 96, 96, 36, 12
    yy, xx, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        np.linspace(-1.0, 1.0, depth, dtype=np.float32),
        indexing="ij",
    )

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2

    def soft(inside, width=0.03):
        # 1 inside, 0 outside, with a smooth edge instead of a staircase.
        return (1.0 / (1.0 + np.exp(np.minimum((inside - 1.0) / width, 60.0)))).astype(
            np.float32
        )

    head = soft(ellipsoid(0.0, 0.0, 0.0, 0.86, 0.94, 0.92))
    brain = soft(ellipsoid(0.0, 0.0, 0.0, 0.76, 0.84, 0.82))
    skull = np.clip(head - brain, 0.0, 1.0)
    folds = 0.18 * np.sin(11.0 * xx + 3.0 * zz) * np.cos(9.0 * yy - 2.0 * zz)
    cavities = np.maximum(
        soft(ellipsoid(-0.2, -0.08, 0.05, 0.13, 0.32, 0.38)),
        soft(ellipsoid(0.2, -0.08, 0.05, 0.13, 0.32, 0.38)),
    )

    # A ring tilted out of the screen: scrolling through it shows a circle
    # that breaks into two spots and then closes again.
    tilt = np.float32(0.6)
    rx = xx - 0.1
    ry = (yy + 0.45) * np.cos(tilt) - zz * np.sin(tilt)
    rz = (yy + 0.45) * np.sin(tilt) + zz * np.cos(tilt)
    ring = soft(((np.sqrt(rx**2 + ry**2) - 0.26) ** 2 + rz**2) / 0.06**2)

    # Fine stripes that get finer to the right: a resolution chart.
    patch = (
        (xx > 0.25) & (xx < 0.62) & (yy > 0.38) & (yy < 0.62) & (np.abs(zz) < 0.5)
    ).astype(np.float32)
    stripes = patch * 0.5 * (1.0 + np.sin((30.0 + 90.0 * (xx - 0.25)) * xx))

    static = (
        1.0 * skull
        + brain * (0.45 + folds)
        - 0.35 * cavities
        + 0.8 * ring
        + 0.6 * stripes
    )

    rng = np.random.default_rng(0)
    base = np.empty((height, width, depth, frames), dtype=np.float32)
    compare = np.empty_like(base)
    overlay = np.zeros(base.shape, dtype=np.uint8)

    for frame in range(frames):
        phase = np.float32(2.0 * np.pi * frame / frames)

        # A bright tube that spirals through the slices and turns with time.
        angle = np.pi * 2.0 * zz + phase
        spiral = soft(
            ((xx - 0.5 * np.cos(angle)) ** 2 + (yy - 0.5 * np.sin(angle)) ** 2)
            / 0.07**2
        )

        # A small heart that beats.
        beat = np.float32(0.16 + 0.05 * np.sin(phase))
        heart = soft(ellipsoid(-0.05, 0.35, -0.2, beat, beat, beat * 1.4))

        frame_base = static + 1.1 * spiral + 1.3 * heart
        base[..., frame] = (
            frame_base + 0.03 * rng.standard_normal(frame_base.shape)
        ).astype(np.float32)

        # The comparison scan: a little later in the beat, and a new spot
        # has appeared, so the difference view has something to find.
        late_beat = np.float32(0.16 + 0.05 * np.sin(phase + 0.9))
        late_heart = soft(ellipsoid(-0.05, 0.35, -0.2, late_beat, late_beat, late_beat * 1.4))
        spot = soft(ellipsoid(-0.45, 0.1, 0.3, 0.09, 0.09, 0.14))
        frame_compare = static + 1.1 * spiral + 1.3 * late_heart + 0.7 * spot
        compare[..., frame] = (
            frame_compare + 0.03 * rng.standard_normal(frame_compare.shape)
        ).astype(np.float32)

        overlay[..., frame][cavities > 0.5] = 1
        overlay[..., frame][heart > 0.5] = 2
        overlay[..., frame][ring > 0.5] = 3

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

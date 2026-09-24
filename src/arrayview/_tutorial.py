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


# The main tutorial array is a small pixel-art campsite. It is 4D so every
# part of the tour has something to show: height and width are the picture,
# the third dimension is a looping animation (rain, wind in the trees, the
# fire, a camper walking about) and the fourth is the time of day, from dawn
# through noon to the middle of the night. Values are brightness, so any
# colormap works and the fire lights up its surroundings after dark.

_CAMP_HEIGHT, _CAMP_WIDTH, _CAMP_FRAMES, _CAMP_HOURS = 64, 96, 24, 8
_CAMP_GROUND = 44
# Ambient light for each time of day: dawn, morning, noon, afternoon, dusk,
# evening, night, late night.
_CAMP_LIGHT = (0.55, 0.85, 1.0, 0.9, 0.55, 0.3, 0.2, 0.22)
_CAMP_FIRE = (58, 52)

_CAMPER_SHADES = {"h": 1.0, "s": 0.78, "e": 0.02, "c": 0.62, "p": 0.18, "b": 0.08}
_CAMPER_WALK = (
    (
        "..hhh..",
        ".hhhhh.",
        "..sss..",
        "..sse..",
        "..sss..",
        ".ccccc.",
        "c.ccc.c",
        "..ccc..",
        "..p.p..",
        ".p...p.",
        ".b...b.",
    ),
    (
        "..hhh..",
        ".hhhhh.",
        "..sss..",
        "..sse..",
        "..sss..",
        ".ccccc.",
        ".ccccc.",
        "..ccc..",
        "..p.p..",
        "..p.p..",
        "..b.b..",
    ),
)
_CAMPER_SIT = (
    "..hhh..",
    ".hhhhh.",
    "..sse..",
    "..sss..",
    ".cccc..",
    ".ccccss",
    ".cccc..",
    ".ppppp.",
    ".pp..pb",
    ".bb....",
)


def _camp_scene(frame: int, hour: int, snow: bool):
    """Draw one picture of the campsite and its region labels."""
    import numpy as np

    H, W, F = _CAMP_HEIGHT, _CAMP_WIDTH, _CAMP_FRAMES
    ground = _CAMP_GROUND
    light = _CAMP_LIGHT[hour]
    night = light < 0.4
    rng = np.random.default_rng(7)  # the fixed layout: stars, drops, grass
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    loop = 2.0 * np.pi * frame / F
    sway = float(np.sin(2.0 * loop) + 0.4 * np.sin(3.0 * loop + 1.0))

    albedo = np.zeros((H, W), dtype=np.float32)
    glow = np.zeros((H, W), dtype=np.float32)  # light that ignores the hour
    labels = np.zeros((H, W), dtype=np.uint8)

    def put(y, x, value, target=albedo):
        y, x = int(round(y)), int(round(x))
        if 0 <= y < H and 0 <= x < W:
            target[y, x] = value

    def sprite(rows, left, bottom, flip=False):
        top = bottom - len(rows) + 1
        for r, row in enumerate(rows):
            for c, ch in enumerate(row[::-1] if flip else row):
                if ch != ".":
                    put(top + r, left + c, _CAMPER_SHADES[ch])
                    if 0 <= top + r < H and 0 <= left + c < W:
                        labels[top + r, left + c] = 2

    # Sky, brighter toward the horizon.
    sky = 0.62 + 0.25 * (yy / ground)
    albedo[:] = np.where(yy < ground, sky, 0.0)

    # Stars, twinkling.
    star_y = rng.integers(0, 30, 45)
    star_x = rng.integers(0, W, 45)
    stars = []
    if night:
        for i, (sy, sx) in enumerate(zip(star_y, star_x)):
            stars.append((sy, sx, 0.9 if (i + frame // 3) % 4 else 0.45))

    # The sun crosses the sky by day; a crescent moon by night.
    if not night:
        t = min(hour, 4) / 4.0
        cx, cy = 8 + 80 * t, 28 - 22 * np.sin(np.pi * t)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 16
        glow[disc] = 1.3
        albedo[disc] = 0.0
    else:
        cx, cy = 18 + (hour - 5) * 28, 11
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 10) & (
            (xx - cx - 1.6) ** 2 + (yy - cy + 1) ** 2 > 7
        )
        glow[disc] = 0.95

    # Clouds drifting past; they loop with the animation.
    for i, (x0, y0, speed) in enumerate(((10, 8, 1), (55, 15, 2), (80, 5, 1))):
        cx = (x0 + frame * speed * W / F) % W
        for dx, dy, rx, ry in ((0, 0, 7, 2.6), (-5, 1, 4, 2), (5, 1, 5, 2), (1, -2, 4, 2)):
            dxw = (xx - cx - dx + W / 2) % W - W / 2
            blob = (dxw / rx) ** 2 + ((yy - y0 - dy) / ry) ** 2 <= 1
            albedo[blob] = 0.95
            glow[blob] = 0.0
            if night:
                albedo[blob] = 0.55
    for sy, sx, v in stars:
        if albedo[sy, sx] < 0.9:
            glow[sy, sx] = v

    # Mountains, with snow on the peaks.
    ridge = 31 + 5 * np.sin(xx[0] * 0.08 + 0.5) + 4 * np.sin(xx[0] * 0.21 + 2.0)
    for x in range(W):
        top = int(ridge[x])
        albedo[top:ground, x] = 0.42
        glow[top:ground, x] = 0.0
        if top < 30:
            albedo[top : top + 2, x] = 0.95

    # Ground, grass with a little texture; snow when it snows.
    texture = rng.normal(0.0, 0.02, (H, W)).astype(np.float32)
    base = 0.86 if snow else 0.4
    albedo[ground:] = base + texture[ground:]

    # Grass tufts lean with the wind.
    for gx, gy in zip(rng.integers(0, W, 40), rng.integers(ground + 2, H, 40)):
        lean = int(round(sway * 0.8))
        shade = 0.7 if snow else 0.58
        put(gy - 1, gx, shade)
        put(gy - 2, gx + lean, shade)
        put(gy - 1, gx + 1, shade)

    # Pine trees sway, more at the top.
    for tx, base_y, height in ((6, 47, 20), (14, 45, 15), (86, 46, 19), (93, 48, 14), (76, 45, 12)):
        for r in range(height):
            y = base_y - 3 - height + r
            half = 1 + int(r * 0.42) - (1 if r % 4 == 0 and r > 3 else 0)
            shift = int(round(sway * 1.6 * (1.0 - r / height)))
            for dx in range(-half, half + 1):
                v = 0.24 if dx > 0 else 0.32
                if snow and (r % 4 == 0 or dx == -half):
                    v = 0.9
                put(y, tx + dx + shift, v)
        for r in range(3):
            put(base_y - 2 + r, tx, 0.18)
            put(base_y - 2 + r, tx + 1, 0.18)

    # The tent, with a lantern inside after dark.
    apex_x, apex_y, bottom = 22, 34, 49
    for y in range(apex_y, bottom + 1):
        half = int((y - apex_y) * 0.85)
        for dx in range(-half, half + 1):
            v = 0.55 if dx < 0 else 0.68
            if snow and y - apex_y < 3:
                v = 0.92
            put(y, apex_x + dx, v)
            if 0 <= y < H and 0 <= apex_x + dx < W:
                labels[y, apex_x + dx] = 3
        door = int((y - 43) * 0.6) if y >= 43 else -1
        for dx in range(-door, door + 1):
            put(y, apex_x + dx, 0.1)
            if night:
                put(y, apex_x + dx, 0.5, glow)

    # The camper walks about by day and sits by the fire at night.
    if night:
        sprite(_CAMPER_SIT, 46, 55)
    else:
        going = frame < F // 2
        t = (frame if going else F - 1 - frame) / (F // 2 - 1)
        x = int(round(32 + 36 * t))
        sprite(_CAMPER_WALK[(frame // 2) % 2], x, 58, flip=not going)

    # The fire: stones, logs, flickering flames and rising sparks.
    fx, fy = _CAMP_FIRE
    for dx in (-5, -4, 4, 5):
        put(fy + 1, fx + dx, 0.55)
    for dx in range(-3, 4):
        put(fy + 1, fx + dx, 0.22)
        put(fy, fx + dx if abs(dx) < 3 else fx, 0.28)
    flicker = np.random.default_rng(100 + frame)
    heights = np.array([2, 4, 6, 8, 6, 4, 2]) + flicker.integers(-1, 3, 7)
    for i, h in enumerate(heights):
        x = fx - 3 + i
        for r in range(h):
            y = fy - 1 - r
            put(y, x, 1.25 if r < h - 2 and 1 <= i <= 5 else 0.9, glow)
            put(y, x, 0.0)
            if 0 <= y < H:
                labels[y, x] = 1
    for k in range(4):
        age = (frame * 2 + k * 5) % 16
        put(fy - 6 - age, fx + int(round(np.sin(k + age * 0.5) * 2 + sway)), 0.9, glow)

    # Smoke drifts off with the wind.
    for k in range(6):
        age = ((frame + 4 * k) % F) / F
        sx = fx + 16 * age + sway
        sy = fy - 9 - 30 * age
        puff = (xx - sx) ** 2 + (yy - sy) ** 2 <= (1 + 2.2 * age) ** 2
        albedo[puff] = albedo[puff] * (0.5 + 0.5 * age) + 0.6 * (0.5 - 0.5 * age)

    # Firelight: the hour's light, plus the fire nearby.
    fire_d2 = (xx - fx) ** 2 + (yy - fy + 3) ** 2
    firelight = (1.0 + 0.12 * np.sin(frame * 2.3)) * np.exp(-fire_d2 / 16.0**2)
    lit = albedo * (light + (1.0 - light) * 0.9 * firelight) + glow

    # Rain streaks, or snowflakes, slanted by the wind.
    drops = zip(rng.integers(0, W, 55), rng.integers(0, H, 55), rng.integers(2, 4, 55))
    for i, (x0, y0, speed) in enumerate(drops):
        if snow:
            y = (y0 + frame * H / F) % H
            x = (x0 + 1.5 * np.sin(loop + i) + sway) % W
            if i % 2 == 0:
                put(y, x, 0.35 + 0.6 * light, lit)
            continue
        y = (y0 + frame * speed * H / F) % H
        for r in range(3):
            px = int(round((x0 + (y - r) * 0.35) % W)) % W
            py = int(round(y - r))
            if 0 <= py < H:
                lit[py, px] = 0.6 * lit[py, px] + 0.4 * (0.3 + 0.5 * light)

    # Fireflies after dark.
    if night:
        for k in range(5):
            a = loop * (1 + k % 2) + k * 1.3
            if (frame + k) % 3:
                put(47 + 3 * np.sin(a) + k, 70 + 7 * k % 20 + 4 * np.cos(a), 0.8, lit)

    return lit.astype(np.float32), labels


def make_tutorial_arrays():
    """Return the campsite, the same campsite in the snow, and its regions.

    The regions mark the fire (1), the camper (2) and the tent (3).
    """
    import numpy as np

    shape = (_CAMP_HEIGHT, _CAMP_WIDTH, _CAMP_FRAMES, _CAMP_HOURS)
    base = np.empty(shape, dtype=np.float32)
    compare = np.empty_like(base)
    overlay = np.zeros(shape, dtype=np.uint8)
    for hour in range(_CAMP_HOURS):
        for frame in range(_CAMP_FRAMES):
            base[..., frame, hour], overlay[..., frame, hour] = _camp_scene(
                frame, hour, snow=False
            )
            compare[..., frame, hour], _ = _camp_scene(frame, hour, snow=True)
    # The viewer draws the first dimension across the screen, so the
    # picture is stored as (x, y, frame, hour).
    return tuple(
        np.ascontiguousarray(a.transpose(1, 0, 2, 3)) for a in (base, compare, overlay)
    )


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

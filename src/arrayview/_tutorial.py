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
# part of the tour has something to show: width and height are the picture,
# the third dimension is a looping animation (rain, wind in the trees, the
# fire, a camper walking about) and the fourth is the time of day, a full
# day starting in the morning. Values are brightness, so any colormap works
# and the fire lights up its surroundings after dark.
#
# The layout is written in "design units" on a 96 x 64 grid and drawn at
# `_CAMP_SCALE` pixels per unit, so shapes get finer rather than blockier.

_CAMP_SCALE = 2
_CAMP_FRAMES = 24
_CAMP_UNITS_W, _CAMP_UNITS_H = 96, 64
_CAMP_GROUND = 44
_CAMP_FIRE = (58, 52)
# The time-of-day dimension, in hours on the clock. Steps are closer
# together around sunset and sunrise, where the light changes fastest.
_CAMP_CLOCK = (8, 10, 12, 14, 16, 17, 18, 19, 20, 22, 0, 2, 4, 5, 6, 7)

_CAMPER_SHADES = {
    "h": 1.0, "s": 0.78, "e": 0.02, "c": 0.62, "k": 0.36,
    "p": 0.2, "b": 0.08, "l": 0.3,
}
_CAMPER_WALK = (
    (
        "....hhhh....",
        "...hhhhhh...",
        "..hhhhhhhhh.",
        "....ssss....",
        "....sssse...",
        "....ssss....",
        ".....ss.....",
        "...cccccc...",
        "..kcccccc...",
        "..kccccccc..",
        "..kcccc.cs..",
        "..kcccc.....",
        "...cccc.....",
        "...pppp.....",
        "...pp.pp....",
        "..pp...pp...",
        "..pp....pp..",
        ".pp.....pp..",
        ".bbb....bbb.",
    ),
    (
        "....hhhh....",
        "...hhhhhh...",
        "..hhhhhhhhh.",
        "....ssss....",
        "....sssse...",
        "....ssss....",
        ".....ss.....",
        "...cccccc...",
        "..kcccccc...",
        "..kcccccc...",
        "..kccccc....",
        "..kccccs....",
        "...cccc.....",
        "...pppp.....",
        "...pppp.....",
        "....pp......",
        "....pp......",
        "....pp......",
        "...bbbb.....",
    ),
)
_CAMPER_SIT = (
    "....hhhh.....",
    "...hhhhhh....",
    "..hhhhhhhhh..",
    "....ssss.....",
    "....sssse....",
    "....ssss.....",
    ".....ss......",
    "...cccccc....",
    "..kccccccccss",
    "..kcccccc....",
    "..kcccccc....",
    "...cccccc....",
    "...pppppppp..",
    "...pppppppp..",
    ".........pp..",
    ".llllllllpp..",
    "llllllllbbb..",
    ".llllllll....",
)


def _smoothstep(lo, hi, x):
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def _camp_light(clock: float):
    """How bright the hour is, how dark it is, and whether it is night."""
    import numpy as np

    # How high the sun is: 1 at noon, 0 at six, -1 at midnight.
    sun_up = float(np.sin(2.0 * np.pi * (clock - 6.0) / 24.0))
    light = 0.2 + 0.8 * _smoothstep(-0.45, 0.5, sun_up)
    dark = 1.0 - _smoothstep(0.2, 0.6, light)  # 1 once it is fully dark
    return sun_up, light, dark


def _camp_layers(frame: int, snow: bool, night: bool):
    """Everything in one animation frame that does not depend on the hour.

    The hour only changes the light, the sky's sun, moon and stars, the
    shade of the clouds and the lantern in the tent, so those are left as
    masks for `_camp_scene` to fill in.
    """
    import numpy as np

    S = _CAMP_SCALE
    H, W = _CAMP_UNITS_H * S, _CAMP_UNITS_W * S
    F = _CAMP_FRAMES
    ground = _CAMP_GROUND
    rng = np.random.default_rng(7)  # the fixed layout: grass, drops
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    yy, xx = (rows + 0.5) / S, (cols + 0.5) / S  # design units
    loop = 2.0 * np.pi * frame / F
    sway = float(np.sin(2.0 * loop) + 0.4 * np.sin(3.0 * loop + 1.0))

    albedo = np.zeros((H, W), dtype=np.float32)
    glow = np.zeros((H, W), dtype=np.float32)  # light that ignores the hour
    labels = np.zeros((H, W), dtype=np.uint8)

    def put(y, x, value, target=albedo):
        """Set one pixel, in pixel coordinates."""
        y, x = int(round(y)), int(round(x))
        if 0 <= y < H and 0 <= x < W:
            target[y, x] = value

    def sprite(rows_, left, bottom, flip=False):
        """Draw the camper, one character per pixel, feet at `bottom`."""
        width = max(len(r) for r in rows_)
        top = bottom - len(rows_) + 1
        for r, row in enumerate(rows_):
            row = row.ljust(width, ".")
            for c, ch in enumerate(row[::-1] if flip else row):
                y, x = top + r, left + c
                if ch != "." and 0 <= y < H and 0 <= x < W:
                    albedo[y, x] = _CAMPER_SHADES[ch]
                    if ch != "l":
                        labels[y, x] = 2

    # Sky, brighter toward the horizon.
    albedo[:] = np.where(yy < ground, 0.62 + 0.25 * (yy / ground), 0.0)

    # Clouds drift past and loop with the animation.
    clouds = np.zeros((H, W), dtype=bool)
    for x0, y0, speed in ((10, 8, 1), (55, 15, 2), (80, 5, 1)):
        cx = (x0 + frame * speed * _CAMP_UNITS_W / F) % _CAMP_UNITS_W
        dxw = (xx - cx + _CAMP_UNITS_W / 2) % _CAMP_UNITS_W - _CAMP_UNITS_W / 2
        for dx, dy, rx, ry in ((0, 0, 7, 2.6), (-5, 1, 4, 2), (5, 1, 5, 2), (1, -2, 4, 2)):
            clouds |= ((dxw - dx) / rx) ** 2 + ((yy - y0 - dy) / ry) ** 2 <= 1

    # Mountains, with snow on the peaks and a nearer, darker range.
    ridge = 31 + 5 * np.sin(xx * 0.08 + 0.5) + 4 * np.sin(xx * 0.21 + 2.0)
    near = 38 + 2.5 * np.sin(xx * 0.13 + 4.0) + 1.5 * np.sin(xx * 0.37)
    rock = (yy >= ridge) & (yy < ground)
    albedo[rock] = 0.46
    albedo[rock & (yy < ridge + 1.2) & (ridge < 30)] = 0.95
    albedo[(yy >= near) & (yy < ground)] = 0.36
    # Where the sun, moon and stars can show: sky not behind anything.
    open_sky = (yy < ridge) & ~clouds

    # Ground, grass with a little texture; snow when it snows.
    texture = rng.normal(0.0, 0.02, (H, W)).astype(np.float32)
    below = yy >= ground
    albedo[below] = (0.86 if snow else 0.4) + texture[below]

    # Grass tufts lean with the wind.
    blade = 0.7 if snow else 0.58
    lean = sway * 1.2
    for gx, gy in zip(rng.integers(0, W, 90), rng.integers((ground + 2) * S, H, 90)):
        for r in range(4):
            put(gy - r, gx + lean * r / 3, blade)
            put(gy - r * 0.7, gx + 2 + lean * r / 4, blade)

    # Pine trees sway, more at the top; tiers of branches get wider down.
    for tx, base_y, height in (
        (6, 47, 20), (14, 45, 15), (86, 46, 19), (93, 48, 14), (76, 45, 12)
    ):
        top = base_y - 3 - height
        r = yy - top
        shift = sway * 1.6 * (1.0 - r / height)
        half = 0.8 + 0.42 * r - 1.3 * ((r % 4.0) / 4.0)
        dx = xx - tx - shift
        tree = (r >= 0) & (r < height) & (np.abs(dx) <= half)
        albedo[tree] = np.where(dx[tree] > 0, 0.24, 0.32)
        open_sky &= ~tree
        if snow:
            albedo[tree & (dx < -half + 0.9)] = 0.9
            albedo[tree & ((r % 4.0) < 0.6)] = 0.9
        trunk = (np.abs(xx - tx) <= 0.9) & (yy >= base_y - 3) & (yy < base_y)
        albedo[trunk] = 0.18

    # The tent. Its doorway is where the lantern shows after dark.
    apex_x, apex_y, bottom = 22, 34, 49
    across = xx - apex_x
    tent = (yy >= apex_y) & (yy <= bottom) & (np.abs(across) <= (yy - apex_y) * 0.85)
    albedo[tent] = np.where(across[tent] < 0, 0.55, 0.68)
    if snow:
        albedo[tent & (yy - apex_y < 2.5)] = 0.92
    labels[tent] = 3
    door = tent & (yy >= 42) & (np.abs(across) <= (yy - 42) * 0.6)
    albedo[door] = 0.1

    # The camper walks about by day and sits by the fire at night.
    if night:
        sprite(_CAMPER_SIT, 44 * S, 55 * S)
    else:
        going = frame < F // 2
        step = (frame if going else F - 1 - frame) / (F // 2 - 1)
        sprite(_CAMPER_WALK[(frame // 2) % 2], int(round((30 + 38 * step) * S)),
               58 * S, flip=not going)

    # The fire: a ring of stones, crossed logs, flickering flames, sparks.
    fx, fy = _CAMP_FIRE
    stones = (((xx - fx) / 7.0) ** 2 + ((yy - fy - 1.0) / 1.6) ** 2 <= 1) & (
        ((xx - fx) / 5.2) ** 2 + ((yy - fy - 1.0) / 1.0) ** 2 > 1
    )
    albedo[stones] = 0.55
    logs = (np.abs(yy - fy - 0.2 - 0.25 * (xx - fx)) < 0.7) | (
        np.abs(yy - fy - 0.2 + 0.25 * (xx - fx)) < 0.7
    )
    albedo[logs & (np.abs(xx - fx) < 4.5)] = 0.26
    flicker = np.random.default_rng(100 + frame)
    profile = 9.0 * np.exp(-((np.arange(14) - 6.5) / 4.0) ** 2)
    heights = profile + flicker.normal(0.0, 1.0, 14) + 1.5 * np.sin(loop * 3 + np.arange(14))
    base_row = (fy - 0.5) * S
    for i, h in enumerate(np.maximum(heights, 0.5) * S):
        x = (fx - 3.5) * S + i
        for r in range(int(h)):
            core = r < h * 0.6 and 3 <= i <= 10
            put(base_row - r, x, 1.25 if core else 0.9, glow)
            put(base_row - r, x, 0.0)
            if 0 <= int(base_row - r) < H:
                labels[int(base_row - r), int(x)] = 1
    for k in range(6):
        age = (frame * 2 + k * 5) % 16
        put((fy - 7 - age) * S, (fx + np.sin(k + age * 0.5) * 2 + sway) * S, 0.95, glow)

    # Smoke drifts off with the wind.
    for k in range(6):
        age = ((frame + 4 * k) % F) / F
        sx = fx + 16 * age + sway
        sy = fy - 10 - 30 * age
        puff = (xx - sx) ** 2 + (yy - sy) ** 2 <= (1 + 2.2 * age) ** 2
        albedo[puff] = albedo[puff] * (0.5 + 0.5 * age) + 0.6 * (0.5 - 0.5 * age)
        clouds &= ~puff
        open_sky &= ~puff

    fire_d2 = (xx - fx) ** 2 + (yy - fy + 3) ** 2
    firelight = (1.0 + 0.12 * np.sin(frame * 2.3)) * np.exp(-fire_d2 / 16.0**2)

    # Rain streaks, or snowflakes, slanted by the wind. Both loop.
    count = 160
    drop_y, drop_x = [], []
    drops = zip(rng.integers(0, W, count), rng.integers(0, H, count), rng.integers(2, 4, count))
    for i, (x0, y0, speed) in enumerate(drops):
        if snow:
            if i % 2 == 0:
                drop_y.append(int(round((y0 + frame * H / F) % H)) % H)
                drop_x.append(int(round((x0 + 3.0 * np.sin(loop + i) + sway * S) % W)) % W)
            continue
        y = (y0 + frame * speed * H / F) % H
        for r in range(5):
            py = int(round(y - r))
            if 0 <= py < H:
                drop_y.append(py)
                drop_x.append(int(round((x0 + (y - r) * 0.35) % W)) % W)

    return {
        "albedo": albedo, "glow": glow, "labels": labels, "clouds": clouds,
        "open_sky": open_sky, "door": door, "firelight": firelight,
        "drops": (np.array(drop_y, dtype=np.intp), np.array(drop_x, dtype=np.intp)),
        "xx": xx, "yy": yy, "loop": loop,
    }


def _camp_scene(frame: int, clock: float, snow: bool, layers=None):
    """Draw one picture of the campsite and its region labels.

    `clock` is the hour of the day, 0 to 24.
    """
    import numpy as np

    S = _CAMP_SCALE
    sun_up, light, dark = _camp_light(clock)
    if layers is None:
        layers = _camp_layers(frame, snow, sun_up < 0.0)
    albedo = layers["albedo"].copy()
    glow = layers["glow"].copy()
    xx, yy, loop = layers["xx"], layers["yy"], layers["loop"]
    open_sky = layers["open_sky"]
    H, W = albedo.shape

    # The sun crosses the sky by day and the moon by night, rising and
    # setting behind the mountains.
    t = ((clock - 6.0) % 24.0) / 12.0
    if t <= 1.0:
        cx, cy = 8 + 80 * t, 44 - 38 * max(sun_up, -0.1)
        disc = open_sky & ((xx - cx) ** 2 + (yy - cy) ** 2 <= 4.2**2)
        glow[disc] = 1.3
    else:
        cx, cy = 8 + 80 * (t - 1.0), 44 - 38 * max(-sun_up, -0.1)
        disc = open_sky & ((xx - cx) ** 2 + (yy - cy) ** 2 <= 3.4**2) & (
            (xx - cx - 1.6) ** 2 + (yy - cy + 1.0) ** 2 > 2.8**2
        )
        glow[disc] = 0.3 + 0.65 * dark
    albedo[disc] = 0.0

    # Stars come out as it gets dark, and twinkle.
    if dark > 0:
        rng = np.random.default_rng(11)
        star_y = rng.integers(0, 30 * S, 90)
        star_x = rng.integers(0, W, 90)
        twinkle = np.where((np.arange(90) + frame // 3) % 4 > 0, 0.95, 0.45)
        shown = open_sky[star_y, star_x] & ~disc[star_y, star_x]
        glow[star_y[shown], star_x[shown]] = dark * twinkle[shown]

    # Clouds are grey against the day and dark against the night sky.
    albedo[layers["clouds"]] = 0.55 + 0.4 * (1.0 - dark)
    glow[layers["door"]] = 0.5 * dark

    # The light: the hour's, plus the fire nearby, strongest after dark.
    lit = albedo * (light + (1.0 - light) * 0.9 * layers["firelight"]) + glow

    drop_y, drop_x = layers["drops"]
    if snow:
        lit[drop_y, drop_x] = 0.35 + 0.6 * light
    else:
        lit[drop_y, drop_x] = 0.6 * lit[drop_y, drop_x] + 0.4 * (0.3 + 0.5 * light)

    # Fireflies after dark.
    if dark > 0.5:
        for k in range(7):
            a = loop * (1 + k % 2) + k * 1.3
            if (frame + k) % 3:
                y = int(round((46 + 3 * np.sin(a) + k) * S))
                x = int(round((68 + (7 * k) % 22 + 4 * np.cos(a)) * S))
                if 0 <= y < H and 0 <= x < W:
                    lit[y, x] = 0.8

    return lit.astype(np.float32), layers["labels"]


def make_tutorial_arrays():
    """Return the campsite, the same campsite in the snow, and its regions.

    The regions mark the fire (1), the camper (2) and the tent (3).
    """
    import numpy as np

    S = _CAMP_SCALE
    shape = (_CAMP_UNITS_H * S, _CAMP_UNITS_W * S, _CAMP_FRAMES, len(_CAMP_CLOCK))
    base = np.empty(shape, dtype=np.float32)
    compare = np.empty_like(base)
    overlay = np.zeros(shape, dtype=np.uint8)
    for frame in range(_CAMP_FRAMES):
        cache = {}
        for step, clock in enumerate(_CAMP_CLOCK):
            night = _camp_light(clock)[0] < 0.0
            for snow, target in ((False, base), (True, compare)):
                key = (snow, night)
                if key not in cache:
                    cache[key] = _camp_layers(frame, snow, night)
                image, labels = _camp_scene(frame, clock, snow, cache[key])
                target[..., frame, step] = image
                if not snow:
                    overlay[..., frame, step] = labels
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

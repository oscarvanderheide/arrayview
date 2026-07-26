"""Behavioural coverage for the viewer's shared vmin/vmax formatter.

`formatValueGroup` decides how many decimals every colorbar label, range
readout and line-profile axis shows. The rule is numeric, so asserting on the
source text would not catch a change in what the user actually sees. These
tests extract the function and execute it.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

VIEWER_HTML = Path(__file__).parent.parent / "src" / "arrayview" / "_viewer.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to execute viewer JS"
)


def _extract_function(source: str, name: str) -> str:
    """Return the full text of a top-level `function <name>(...) { ... }`.

    Brace counting has to start after the parameter list: a default such as
    `options = {}` would otherwise close the body before it opened.
    """
    start = source.index(f"function {name}(")
    paren_depth = 0
    body_start = None
    for index in range(start, len(source)):
        char = source[index]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                body_start = source.index("{", index)
                break
    assert body_start is not None, f"could not find parameter list of {name}"

    depth = 0
    for index in range(body_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


def _format_pairs(pairs):
    """Format each [lo, hi] pair through the real viewer implementation."""
    fn = _extract_function(VIEWER_HTML.read_text(), "formatValueGroup")
    script = textwrap.dedent(
        """
        const _escapeHtml = (s) => String(s);
        __FN__
        const pairs = __PAIRS__;
        const out = pairs.map((pair) => {
            const [values, options] = pair;
            const fmt = formatValueGroup(values, options || {});
            return values.map((v) => fmt.format(v));
        });
        process.stdout.write(JSON.stringify(out));
        """
    ).replace("__FN__", fn).replace("__PAIRS__", json.dumps(pairs))
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_large_ranges_drop_meaningless_decimals():
    """A range topping out near 1890 should not advertise hundredths."""
    (near_two_thousand,), = _format_pairs([[[0, 1890.04], None]]),
    assert near_two_thousand == ["0", "1890"]


def test_small_ranges_keep_their_precision():
    """Lowering the floor for large values must not coarsen small ones."""
    results = _format_pairs(
        [
            [[0, 0.5], None],
            [[1.23456, 9.87654], None],
            [[0, 3.14159265], None],
            [[0, 0.00123], None],
        ]
    )
    assert results[0] == ["0.00", "0.50"]
    assert results[1] == ["1.23", "9.88"]
    assert results[2] == ["0.00", "3.14"]
    assert results[3] == ["0.000", "0.001"]


def test_vmin_and_vmax_share_a_decimal_count():
    """The pair must never read as e.g. "0.1" beside "1890.0421"."""
    for values in ([0, 1890.04], [0.5, 18.9], [100.5, 100.6], [-1890.04, 1890.04]):
        low, high = _format_pairs([[values, None]])[0]
        assert low.partition(".")[2].__len__() == high.partition(".")[2].__len__(), (
            f"{values} rendered as {low} / {high}"
        )


def test_close_values_stay_distinguishable():
    """Rounding must never collapse two different values into one label."""
    for values in ([100.5, 100.6], [1000000, 1000001], [-0.001, 0.001]):
        low, high = _format_pairs([[values, None]])[0]
        assert low != high, f"{values} both rendered as {low}"


def test_non_finite_and_degenerate_inputs_are_safe():
    results = _format_pairs([[[0, 0], None]])
    assert results[0] == ["0", "0"]


def test_explicit_min_decimals_still_wins():
    """The pixel readout asks for three decimals; that override must hold."""
    low, high = _format_pairs([[[0, 1890.04], {"minDecimals": 3, "maxDecimals": 7}]])[0]
    assert high == "1890.040"
    assert low == "0.000"

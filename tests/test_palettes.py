"""Lock the colourblind-safe plotting contract.

Categorical series use Okabe-Ito; continuous scales use cividis. These are easy
to undo by accident -- one `color="red"` in a new plot and the guarantee is
gone -- so the rules are asserted rather than only documented.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

from utils import palettes  # noqa: E402

OKABE_ITO_HEXES = set(palettes.OKABE_ITO.values())

# Every .py that draws. palettes.py is the one place literals may appear.
PLOTTING_SOURCES = [p for p in (REPO / "src").glob("*.py")] + [
    REPO / "lib" / "python" / "utils" / "plotting.py"
]


def test_okabe_ito_has_the_canonical_eight():
    assert len(palettes.OKABE_ITO) == 8
    assert palettes.OKABE_ITO["bluish_green"] == "#009E73"
    assert palettes.OKABE_ITO["vermillion"] == "#D55E00"


def test_cycle_covers_all_eight_without_repeats():
    assert len(palettes.OKABE_ITO_CYCLE) == 8
    assert set(palettes.OKABE_ITO_CYCLE) == OKABE_ITO_HEXES


def test_cycle_colors_wraps_past_eight():
    assert palettes.cycle_colors(3) == palettes.OKABE_ITO_CYCLE[:3]
    assert palettes.cycle_colors(10)[8:] == palettes.OKABE_ITO_CYCLE[:2]


def test_sequential_scale_is_cividis():
    assert palettes.SEQUENTIAL_CMAP == "cividis"


@pytest.mark.parametrize(
    "mapping",
    [
        "DATASET_COLORS_COMBINED",
        "DATASET_COLORS_QC",
        "BIAS_COLORS",
        "STATUS_COLORS",
        "THRESHOLD_COLORS",
    ],
)
def test_categorical_maps_use_only_okabe_ito(mapping):
    assert set(getattr(palettes, mapping).values()) <= OKABE_ITO_HEXES


def test_fragment_size_colors_use_only_okabe_ito():
    assert set(palettes.FRAGMENT_SIZE_COLORS) <= OKABE_ITO_HEXES


def test_each_dataset_gets_a_distinct_colour():
    for mapping in (palettes.DATASET_COLORS_COMBINED, palettes.DATASET_COLORS_QC):
        assert len(set(mapping.values())) == len(mapping)


def test_status_colours_are_mutually_distinct():
    """pass/warn/fail must not collapse -- that is the whole point."""
    assert len(set(palettes.STATUS_COLORS.values())) == 3


def test_unmapped_default_is_outside_the_palette():
    """A missing mapping should look unmapped, not like a real series."""
    assert palettes.DATASET_COLOR_DEFAULT not in OKABE_ITO_HEXES


# ── the rules that are easy to break later ────────────────────────────────────


@pytest.mark.parametrize("source", PLOTTING_SOURCES, ids=lambda p: p.name)
def test_no_banned_colormap_anywhere(source):
    text = source.read_text()
    for banned in ("viridis", "jet", "rainbow", "magma", "inferno", "plasma"):
        assert not re.search(rf'cmap\s*=\s*["\']{banned}["\']', text), (
            f"{source.name} uses {banned}; use palettes.SEQUENTIAL_CMAP (cividis)"
        )


@pytest.mark.parametrize("source", PLOTTING_SOURCES, ids=lambda p: p.name)
def test_no_hardcoded_hex_outside_palettes(source):
    """Colour belongs in palettes.py so the guarantee holds in one place."""
    hexes = re.findall(r"#[0-9A-Fa-f]{6}\b", source.read_text())
    assert not hexes, f"{source.name} hardcodes {hexes}; move them to utils.palettes"

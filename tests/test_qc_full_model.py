"""src/qc_full_model.py reads chrombpnet's own outputs; pin how it parses them."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import qc_full_model  # noqa: E402


@pytest.mark.parametrize("status", ["corrected", "uncorrected"])
def test_bias_response_follows_chrombpnets_format(tmp_path, status):
    # marginal_footprinting.py: status + "_" + round(mean, 3) + "_" + "/".join(responses)
    f = tmp_path / "chrombpnet_nobias_max_bias_response.txt"
    f.write_text(f"{status}_0.002_0.001/0.002/0.003/0.004/0.005")
    assert qc_full_model.parse_bias_response(f) == {
        "tn5_1": 0.001,
        "tn5_2": 0.002,
        "tn5_3": 0.003,
        "tn5_4": 0.004,
        "tn5_5": 0.005,
    }


def test_bias_response_in_an_unexpected_format_is_empty(tmp_path):
    f = tmp_path / "chrombpnet_nobias_max_bias_response.txt"
    f.write_text("garbage")
    assert qc_full_model.parse_bias_response(f) == {}

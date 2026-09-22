"""The low-memory one-hot encoder must be byte-identical to chrombpnet's.

The value of `utils.onehot` rests entirely on producing exactly what the
upstream encoder produces, so the reference implementation is reproduced here
verbatim (it lives in the container, which the test suite cannot import) and
the two are compared on the cases that actually differ between naive
implementations: lowercase, N, other IUPAC codes, and the all-one-base
sequence that makes `np.unique` see fewer than four distinct values.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

from utils import onehot  # noqa: E402


def upstream_dna_to_one_hot(seqs):
    """chrombpnet/training/utils/one_hot.py, verbatim. Written by Alex Tseng."""
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)
    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


@pytest.mark.parametrize(
    "seqs",
    [
        pytest.param(["ACGT"], id="minimal"),
        pytest.param(["acgt"], id="lowercase"),
        pytest.param(["AcGt"], id="mixed-case"),
        pytest.param(["NNNN"], id="all-N"),
        pytest.param(["ACGN"], id="trailing-N"),
        pytest.param(["RYKM"], id="other-iupac"),
        pytest.param(["AAAA"], id="single-base-only"),
        pytest.param(["AAAA", "CCCC", "GGGG", "TTTT"], id="one-base-per-seq"),
        pytest.param(["ACGT", "TGCA", "NNNN", "acgt"], id="mixed-batch"),
    ],
)
def test_matches_upstream(seqs):
    assert np.array_equal(onehot.dna_to_one_hot(seqs), upstream_dna_to_one_hot(seqs))


def test_matches_upstream_on_random_batch():
    random.seed(0)
    seqs = ["".join(random.choice("ACGTNacgtn") for _ in range(101)) for _ in range(50)]
    assert np.array_equal(onehot.dna_to_one_hot(seqs), upstream_dna_to_one_hot(seqs))


def test_spans_more_than_one_chunk():
    """The chunked loop must not drop or reorder regions at a boundary."""
    random.seed(1)
    n = onehot.CHUNK + 7
    seqs = ["".join(random.choice("ACGT") for _ in range(8)) for _ in range(n)]
    got = onehot.dna_to_one_hot(seqs)
    assert got.shape == (n, 8, 4)
    assert np.array_equal(got, upstream_dna_to_one_hot(seqs))


def test_output_is_int8():
    assert onehot.dna_to_one_hot(["ACGT"]).dtype == np.int8


def test_non_acgt_encodes_to_zeros():
    got = onehot.dna_to_one_hot(["ACGTN"])
    assert got[0, 4].tolist() == [0, 0, 0, 0]
    assert got[0, :4].sum() == 4  # the real bases are still one-hot


def test_ragged_input_rejected():
    with pytest.raises(AssertionError):
        onehot.dna_to_one_hot(["ACGT", "AC"])


def test_empty_input():
    assert onehot.dna_to_one_hot([]).shape == (0, 0, 4)


def test_install_without_chrombpnet_is_a_noop():
    """The test env has no chrombpnet; install() must report False, not raise."""
    pytest.importorskip  # noqa: B018 - documents intent; no chrombpnet here
    try:
        import chrombpnet  # noqa: F401
    except ImportError:
        assert onehot.install() is False

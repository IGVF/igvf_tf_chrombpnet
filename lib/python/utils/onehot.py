"""A low-memory replacement for chrombpnet's one-hot encoder.

Upstream (`chrombpnet/training/utils/one_hot.py`, by Alex Tseng) encodes every
region in the training set in a single pass:

    seq_concat = "".join(seqs).upper() + "ACGT"
    base_vals  = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape(...)

For the 03.0 bias sweep that call arrives with ~130k-290k regions of 2114 bp at
once, and the intermediates dwarf the int8 result:

    ".join(seqs)" + ".upper()"     2 bytes/base   (two whole Python strings)
    base_vals, int8               1 byte/base
    np.isin(...) and its "~"      2 bytes/base   (two bool arrays)
    np.unique sort                1 byte/base
    return_inverse, INT64         8 bytes/base   <-- dominates
    one-hot output, int8          4 bytes/base

Measured at 20,000 x 2114 bp: 1.142 GiB peak for 0.16 GiB of useful output, a
7.2x overhead. This module gets the same bytes out in 0.262 GiB (1.6x) by

  * indexing a 256-row lookup table with the raw byte, so no index array is
    materialised at all -- `np.unique` is the only reason an int64 array ever
    existed, and it is not needed to map a byte to a row;
  * folding lowercase into the table, which removes the `.upper()` copy;
  * encoding in chunks straight into the output array, so the largest
    temporary is one chunk rather than the whole training set.

Output is byte-identical to upstream, including the "anything not ACGT encodes
to all zeros" rule. `install()` verifies that on a probe before swapping the
function in, so a change in upstream semantics falls back rather than silently
encoding differently.

Two ways in, one implementation. On the cluster the container is the stock
.sif, so `src/chrombpnet_train.py` calls `install()` at runtime. On molab,
`workflows/molab/setup_molab.sh` copies this file into the unpacked sandbox
as `chrombpnet/training/utils/igvf_onehot.py` and rebinds `one_hot.dna_to_one_hot`
there, so every chrombpnet entry point uses it (interpretation and bigwig
helpers too, not just training) and `install()` finds it already in place.
That copy is why this module imports nothing from `utils`: it must run as a
standalone file inside chrombpnet's package.
"""

from __future__ import annotations  # py3.8 in the chrombpnet container

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Rows 0-3 are the ACGT identity; every other byte stays all-zero, which is
# upstream's encoding for N and any other non-ACGT character.
_LUT = np.zeros((256, 4), dtype=np.int8)
for _byte, _row in zip(b"ACGT", np.identity(4, dtype=np.int8)):
    _LUT[_byte] = _row
    _LUT[_byte + 32] = _row  # lowercase acgt

# Big enough that the per-chunk Python join is not the bottleneck, small enough
# that the transient stays a few MB: 4096 x 2114 x 4 = 33 MB.
CHUNK = 4096


def dna_to_one_hot(seqs) -> np.ndarray:
    """N x L x 4 int8 one-hot encoding of equal-length DNA strings.

    Drop-in replacement for `chrombpnet.training.utils.one_hot.dna_to_one_hot`.
    """
    seqs = list(seqs)
    if not seqs:
        return np.empty((0, 0, 4), dtype=np.int8)

    seq_len = len(seqs[0])
    # Upstream asserts this; keep the same contract rather than producing a
    # confusingly-shaped array from a ragged input.
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    out = np.empty((len(seqs), seq_len, 4), dtype=np.int8)
    for start in range(0, len(seqs), CHUNK):
        block = seqs[start : start + CHUNK]
        # "replace" maps any non-ASCII byte to "?", which the table encodes as
        # all zeros -- the same fate upstream gives it.
        buf = np.frombuffer("".join(block).encode("ascii", "replace"), dtype=np.uint8)
        np.take(_LUT, buf, axis=0, out=out[start : start + len(block)].reshape(-1, 4))
    return out


# Carried by the function itself, so a copy baked into the container is
# recognised by `install()` as already in place.
dna_to_one_hot._igvf_low_memory = True


def _matches_upstream(upstream) -> bool:
    """Does our encoder still agree with the one in the container?"""
    probe = [
        "ACGTACGT",
        "acgtACGT",
        "NNNNACGT",  # N -> all zeros
        "ACGTNRYK",  # other IUPAC codes -> all zeros
        "TTTTTTTT",
    ]
    try:
        return np.array_equal(upstream(probe), dna_to_one_hot(probe))
    except Exception:  # upstream signature changed; don't risk the swap
        logger.exception("one-hot probe raised; leaving chrombpnet's encoder in place")
        return False


def install() -> bool:
    """Swap our encoder into chrombpnet, if it reproduces upstream exactly.

    Returns True when the patch was applied. Safe to call more than once and
    safe to call when chrombpnet is not importable (returns False).
    """
    try:
        from chrombpnet.training.utils import one_hot
    except ImportError:
        return False

    if getattr(one_hot.dna_to_one_hot, "_igvf_low_memory", False):
        return True

    if not _matches_upstream(one_hot.dna_to_one_hot):
        logger.warning(
            "chrombpnet's one-hot encoder does not match ours on the probe; "
            "leaving it in place (training will use more memory)"
        )
        return False

    one_hot.dna_to_one_hot = dna_to_one_hot
    # data_utils imported the name directly (`from ... import one_hot` then
    # `one_hot.dna_to_one_hot`), but rebind any direct import too.
    try:
        from chrombpnet.training.utils import data_utils

        if getattr(data_utils, "dna_to_one_hot", None) is not None:
            data_utils.dna_to_one_hot = dna_to_one_hot
    except ImportError:
        pass

    logger.info("installed low-memory one-hot encoder (~4x less peak RAM in data loading)")
    return True

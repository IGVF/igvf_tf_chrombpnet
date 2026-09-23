"""Peak/region file schemas shared by the pipeline scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ChromBPNet's 10-column narrowPeak layout. The trailing "summit" is an offset
# from ``start``, not an absolute coordinate. The six numbered columns are
# unused by this pipeline but must be present for chrombpnet to parse the file.
NARROWPEAK_SCHEMA = [
    "chr",
    "start",
    "end",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "summit",
]


def read_narrowpeak(path) -> pd.DataFrame:
    """Read a 10-column narrowPeak into a DataFrame with NARROWPEAK_SCHEMA names."""
    return pd.read_csv(path, sep="\t", header=None, names=NARROWPEAK_SCHEMA)


def subsample_regions(bed, out, n: int = 30000, seed: int = 1234) -> Path:
    """Write chrombpnet's interpretation subsample of a BED: n rows, seeded.

    Exactly chrombpnet's own rule (pipelines.py, for both the bias QC and the
    full-model pipeline): if the file has MORE than n rows, pandas
    `sample(n, random_state=seed)`, else every row -- same row order, same
    header-less TSV. Kept identical so a split-out QC step interprets the same
    regions `chrombpnet pipeline` would have.
    """
    df = pd.read_csv(bed, sep="\t", header=None)
    sub = df.sample(n, random_state=seed) if df.shape[0] > n else df
    out = Path(out)
    sub.to_csv(out, sep="\t", header=False, index=False)
    return out

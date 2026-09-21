"""Peak/region file schemas shared by the pipeline scripts."""

from __future__ import annotations

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

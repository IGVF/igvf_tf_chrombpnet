"""Resolve a reference file given as an accession, a URL, or a local path.

The bash side of this is `lib/bash/references.sh`, which declares where the
cluster's reference copies live. This is the Python counterpart for inputs that
may instead be named by accession: `ENCFF356LFX` resolves to its ENCODE download
URL, which pandas reads directly.

Accepting an accession means a step can state *which* blacklist it used rather
than a path that says nothing about provenance. The cost is a network call from
wherever the step runs -- on a SLURM compute node that may be firewalled, so a
local path stays supported and is what `lib/bash/references.sh` passes by default.
"""

from __future__ import annotations

import re

# ENCODE file accessions are ENCFF + 6 alphanumerics, e.g. ENCFF356LFX.
ENCODE_ACCESSION_RE = re.compile(r"^ENCFF[0-9A-Z]{6}$")

ENCODE_FILE_URL = "https://www.encodeproject.org/files/{accession}/@@download/{accession}.bed.gz"


def is_encode_accession(spec: str) -> bool:
    """True if ``spec`` looks like an ENCODE file accession rather than a path."""
    return bool(ENCODE_ACCESSION_RE.match(str(spec).strip()))


def encode_file_url(accession: str) -> str:
    """The ENCODE download URL for a BED file accession."""
    accession = str(accession).strip()
    if not is_encode_accession(accession):
        raise ValueError(f"not an ENCODE file accession: {accession!r}")
    return ENCODE_FILE_URL.format(accession=accession)


def resolve_bed_source(spec: str) -> str:
    """Turn an accession into a URL; leave URLs and local paths untouched."""
    return encode_file_url(spec) if is_encode_accession(spec) else str(spec)


def describe_source(spec: str) -> str:
    """Human-readable provenance for a log line."""
    spec = str(spec).strip()
    if is_encode_accession(spec):
        return f"ENCODE {spec} ({encode_file_url(spec)})"
    if spec.startswith(("http://", "https://", "ftp://")):
        return f"URL {spec}"
    return f"file {spec}"

#!/usr/bin/env python3
"""Has this step (and array index) already run successfully on this data?

Used by run_step.sh to skip work that is already done. The evidence is the
run-metadata record every step writes (schema v2, see CLAUDE.md): a step
counts as done for an array index when a record for it exists with
``run_status == "ok"`` AND every output that record declared is on disk now
with the size and md5 it recorded. The second half is what makes a record
trustworthy after a restore from the bucket or a partial cleanup -- an ok
record whose outputs are gone, truncated or changed is not "done".

A SIGKILLed run writes no record at all (CLAUDE.md), so it is never mistaken
for a finished one. What this cannot see is a change of *configuration*: an ok
record with intact outputs from different parameters still counts, which is
why run_step.sh has --force.

Stdlib only: it runs on the box's bare python3, before any environment.

Usage:
    step_done.py <metadata_dir> <step> <array_index>
Exit 0 and print the record's path if done; exit 1 otherwise (reason on stderr).
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("step_done")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _outputs_intact(record: dict) -> str | None:
    """None if every declared output matches the record, else why not."""
    for p in record.get("parameters") or []:
        if p.get("parameter_type") != "output":
            continue
        path = Path(p["filepath"])
        if not p.get("file_exists"):
            return f"{path} was declared but not produced"
        if not path.is_file():
            return f"{path} is missing"
        if p.get("file_size") is not None and path.stat().st_size != p["file_size"]:
            return f"{path} has changed size"
        if p.get("md5sum") and _md5(path) != p["md5sum"]:
            return f"{path} has changed content"
    return None


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="    %(message)s", stream=sys.stderr)
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    metadata_dir, step, index = Path(argv[0]), argv[1].removesuffix(".sh"), argv[2]

    candidates = []
    for f in sorted(metadata_dir.glob(f"*_{step}_*.json"), reverse=True):
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("step") != step:
            continue
        if str((rec.get("slurm") or {}).get("array_task_id")) != index:
            continue
        candidates.append((f, rec))

    if not candidates:
        logger.info("no record of %s index %s under %s", step, index, metadata_dir)
        return 1
    # Newest first (file names start with the timestamp): the latest run is
    # the one that describes what is on disk.
    f, rec = candidates[0]
    if rec.get("run_status") != "ok":
        logger.info(
            "latest run of %s index %s is %s: %s", step, index, rec.get("run_status"), f.name
        )
        return 1
    why = _outputs_intact(rec)
    if why:
        logger.info("%s index %s ran (%s), but %s", step, index, f.name, why)
        return 1
    print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Run a chrombpnet training command in this process, after checking its bigwig.

Why a launcher at all
---------------------
chrombpnet 2.x does natively what this script used to patch into 1.x: `-bw`
trains from a prepared bigwig without converting reads on the GPU node,
`pipeline --skip-interpretation` stops after the marginal footprints and
writes the train-mode report, and its one-hot encoder is the low-memory lookup
table. So nothing of chrombpnet is replaced any more; the argv after `--` is
handed to chrombpnet.CHROMBPNET.main() unchanged. Two jobs are left that the
`chrombpnet` console script cannot do:

1. Peak RSS. The step's metadata trap runs in a SIBLING process
   (emit_metadata.py), which only sees its own few MB. Running chrombpnet
   in-process and calling metadata.report_peak_rss() in a `finally` records
   the process that allocated the training arrays, failed runs included. See
   docs/resource-measurements.md.

2. The prepared-bigwig check. chrombpnet uses a -bw file as given and cannot
   know what it was made from. 00.0.prepare_signal.sh writes
   prepared_bigwig.json beside the bigwig, recording the resolved signal path,
   the signal's md5 and the assay. Given --signal and --assay, this script
   checks that sidecar against the signal the step is configured with and
   exits 1 on any difference, before chrombpnet creates its output directory.
   There is no fallback conversion any more: a stale bigwig -- the config now
   points at other reads, the reads changed, or the assay changed -- would
   otherwise train on the wrong signal without a word. The fix is to remove
   the stale bigwig and sidecar and re-run 00.0.prepare_signal.sh, which
   skips while both exist.

   The check md5s the whole signal file once per job: seconds to a minute for
   a multi-GB fragments file, against hours of GPU training.

   The sidecar is looked up beside the -bw path AS GIVEN, not resolved: when
   the configured signal is itself a bigwig, 00.0 registers it as a symlink,
   and the sidecar sits beside the symlink, not beside its target.

Usage (see workflows/SLURM/03.0 and 04.0):
  python chrombpnet_train.py --signal <configured signal> --assay ATAC \\
      -- bias train -bw <prepared bigwig> -d ATAC ... --device gpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata  # noqa: E402

logger = log.get_logger(__name__)

SIDECAR = "prepared_bigwig.json"
#: chrombpnet 2.x's spellings of the bigwig input to its training commands.
BIGWIG_FLAGS = ("-ibw", "-bw", "--bigwig")
#: ... and of the assay.
ASSAY_FLAGS = ("-d", "--data-type")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chrombpnet_train.py",
        usage="%(prog)s [--signal PATH --assay {ATAC,DNASE}] [-v | -q] -- <chrombpnet args>",
        description="Run a chrombpnet command in this process. With --signal/--assay, "
        "first check the -bw bigwig against the prepared_bigwig.json 00.0 wrote beside it.",
    )
    p.add_argument(
        "--signal",
        help="the configured signal file (reads or bigwig) the -bw bigwig must be prepared from",
    )
    p.add_argument(
        "--assay",
        choices=["ATAC", "DNASE"],
        help="the assay the -bw bigwig must be prepared for",
    )
    log.add_logging_args(p)
    return p


def split_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split our options from the chrombpnet argv after the first ``--``."""
    parser = _parser()
    if "--" not in argv:
        parser.parse_args(argv)  # --help exits 0 here, a bad option exits 2
        parser.error("no chrombpnet command: give it after `--`")
    cut = argv.index("--")
    opts = parser.parse_args(argv[:cut])
    theirs = argv[cut + 1 :]
    if (opts.signal is None) != (opts.assay is None):
        parser.error("--signal and --assay go together")
    if not theirs:
        parser.error("no chrombpnet command after `--`")
    return opts, theirs


def option_value(argv: list[str], flags: tuple[str, ...]) -> str | None:
    """The value argparse would take for ``flags`` from ``argv``: the last one given."""
    value = None
    for i, token in enumerate(argv):
        if token in flags:
            if i + 1 < len(argv):
                value = argv[i + 1]
        else:
            name, eq, rest = token.partition("=")
            if eq and name in flags:
                value = rest
    return value


def sidecar_mismatch(bigwig: Path, signal: str, assay: str) -> str | None:
    """Why ``bigwig`` is not the one prepared from ``signal`` for ``assay``; None if it is."""
    meta = bigwig.parent / SIDECAR  # beside the path as given: see the module docstring
    if not Path(signal).is_file():
        return f"signal file {signal} does not exist"
    if not bigwig.is_file():
        return f"prepared bigwig {bigwig} does not exist"
    if not meta.is_file():
        return f"{meta} does not exist, so nothing records what {bigwig} was made from"
    try:
        rec = json.loads(meta.read_text())
    except (OSError, ValueError) as exc:
        return f"cannot read {meta}: {exc}"
    if not isinstance(rec, dict):
        return f"{meta} is not a JSON object"

    want = str(Path(signal).resolve())
    if rec.get("signal_path") != want:
        return f"{bigwig} was prepared from {rec.get('signal_path')}, not {want}"
    if rec.get("assay") != assay:
        return f"{bigwig} was prepared for assay {rec.get('assay')}, not {assay}"
    recorded = rec.get("signal_md5")
    if not recorded:
        return f"{meta} records no signal_md5 to check {want} against"
    actual = metadata.md5sum(signal)
    if actual != recorded:
        return (
            f"{want} changed after {bigwig} was prepared "
            f"(md5 now {actual}, sidecar says {recorded})"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    opts, chrombpnet_argv = split_argv(sys.argv[1:] if argv is None else argv)
    log.setup_from_args(opts)

    bigwig = option_value(chrombpnet_argv, BIGWIG_FLAGS)
    if opts.signal is not None:
        if bigwig is None:
            logger.error("--signal checks the -bw bigwig, and the chrombpnet command has none")
            return 1
        data_type = option_value(chrombpnet_argv, ASSAY_FLAGS)
        if data_type is not None and data_type != opts.assay:
            logger.error("--assay is %s but chrombpnet is given -d %s", opts.assay, data_type)
            return 1
        problem = sidecar_mismatch(Path(bigwig), opts.signal, opts.assay)
        if problem:
            # 00.0 skips while the bigwig and its sidecar both exist, so the
            # stale pair has to go before re-running it does anything.
            logger.error(
                "%s. Remove %s and %s, then re-run 00.0.prepare_signal.sh: chrombpnet "
                "trains from the prepared bigwig only, there is no fallback conversion.",
                problem,
                bigwig,
                Path(bigwig).parent / SIDECAR,
            )
            return 1
        logger.info("prepared bigwig %s matches %s (%s)", bigwig, opts.signal, opts.assay)
    elif bigwig is not None:
        logger.warning("no --signal/--assay given: %s is used unchecked", bigwig)

    # Imported only now: chrombpnet pulls in Keras 3 and JAX, and a failed
    # check should cost nothing.
    import chrombpnet.CHROMBPNET as chrombpnet_cli

    sys.argv = ["chrombpnet", *chrombpnet_argv]
    try:
        chrombpnet_cli.main()
    finally:
        # This process allocated the training arrays, so its peak RSS is the
        # step's real footprint; the metadata trap runs in a sibling that
        # cannot see it, so the number is handed over through a file.
        metadata.report_peak_rss()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

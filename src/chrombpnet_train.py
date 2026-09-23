#!/usr/bin/env python3
"""Run a chrombpnet training command, reusing a bigwig prepared on CPU.

Why
---
`chrombpnet train`, `chrombpnet pipeline` and `chrombpnet bias train` all begin
by converting reads to a bigwig:

    awk (Tn5 shift) | sort -k1,1 | bedtools genomecov -bg -5 | sort | bedGraphToBigWig

plus an enzyme-shift auto-detection pass that samples reads and compares them to
a reference motif. None of that touches the GPU, and it is unconditional -- there
is no existence check to short-circuit. A 5-fold x 4-bias-factor sweep therefore
repeats the identical conversion 20 times, each inside its own GPU allocation.

This wrapper does the conversion once on a CPU node (`cli.py prepare-bigwig`)
and has each GPU job reuse the result.

How
---
`chrombpnet.pipelines` calls `reads_to_bigwig.main(args)` through a module
attribute looked up at call time, so replacing that attribute before invoking
chrombpnet's own `main()` is enough -- we do not reimplement any argument
parsing, we hand chrombpnet the exact argv it expects.

The replacement *copies the prepared bigwig into place* rather than doing
nothing. It has to: chrombpnet creates `auxiliary/` itself with
`exist_ok=False`, so the file cannot be staged beforehand.

Why not just pass the bigwig on the command line? chrombpnet 1.0.1 DOES have a
`--bigwig` flag, but only on the subcommands that *read* one -- `qc`,
`bias qc` and `pred_bw` (chrombpnet/parsers.py:146, 195, 226). The training
subcommands, `pipeline` and `bias train`, take reads and nothing else: their
required group is `-ibam | -ifrag | -itag`, and `--bigwig` there is rejected as
an unrecognized argument. The ChromBPNet tutorial shows `chrombpnet pipeline
--bigwig ...`, which documents GitHub main; the newest PyPI release is 1.0.1,
so that form does not work against anything installable today. If a future
chrombpnet adds `--bigwig` to `pipeline`, delete this module and pass the flag.

Safety
------
Reuse only happens when the prepared bigwig's sidecar records the same signal
file, md5, assay and chrombpnet version. Anything else and it falls through to
chrombpnet's own conversion, which is always correct, only slower. That matters
because this depends on chrombpnet internals: if an upgrade moves
`reads_to_bigwig`, the import below fails loudly instead of silently training on
a stale bigwig.

Usage (see workflows/SLURM/03.0 and 04.0):
  python chrombpnet_train.py --prepared-bigwig <dir> [--require-prepared] \\
      -- bias train -ifrag ... -o ...
"""

from __future__ import annotations  # py3.8 in the chrombpnet container: PEP 585/604 annotations

import json
import os
import shutil
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata, onehot  # noqa: E402

logger = log.get_logger(__name__)

SIDECAR = "prepared_bigwig.json"
BIGWIG = "data_unstranded.bw"


def _split_argv(argv):
    """Split our options from the chrombpnet argv after ``--``."""
    if "--" not in argv:
        raise SystemExit("usage: chrombpnet_train.py --prepared-bigwig <dir> -- <chrombpnet args>")
    cut = argv.index("--")
    ours, theirs = argv[:cut], argv[cut + 1 :]
    prepared = None
    require = False
    stop = False
    for i, token in enumerate(ours):
        if token == "--prepared-bigwig":
            prepared = ours[i + 1]
        elif token == "--require-prepared":
            require = True
        elif token == "--stop-before-interpretation":
            stop = True
    return prepared, require, stop, theirs


def _sidecar_matches(prepared_dir: Path, signal_path: str | None, assay: str | None) -> bool:
    """Is this prepared bigwig actually for the signal we are about to train on?"""
    meta = prepared_dir / SIDECAR
    if not (prepared_dir / BIGWIG).is_file() or not meta.is_file():
        return False
    try:
        rec = json.loads(meta.read_text())
    except (OSError, ValueError):
        return False

    if signal_path and rec.get("signal_path") != str(Path(signal_path).resolve()):
        logger.warning("prepared bigwig is for a different signal file; reconverting")
        return False
    if assay and rec.get("assay") != assay:
        logger.warning("prepared bigwig was made for assay %s; reconverting", rec.get("assay"))
        return False
    if signal_path and rec.get("signal_md5"):
        actual = metadata.md5sum(signal_path)
        if actual != rec["signal_md5"]:
            logger.warning("signal file changed since the bigwig was prepared; reconverting")
            return False
    return True


class _StopBeforeInterpretation(Exception):
    """Raised in place of chrombpnet's interpretation step (--stop-before-interpretation)."""


INTERPRET_MODULE = "chrombpnet.evaluation.interpret.interpret"


def _install_interpretation_stop():
    """Make chrombpnet's interpretation step raise instead of running.

    The real module must NOT be imported to do it: interpret.py calls
    tf.compat.v1.disable_eager_execution() at import time (for SHAP's deep
    explainer), which switches the whole process out of eager mode -- the
    first attempt at this imported it to swap `main` and training died in
    find_chrombpnet_hyperparams on "numpy() is only available when eager
    execution is enabled". So a stand-in module is registered under its name
    instead. pipelines.py imports it inside the function (`import ... as
    interpret`), which resolves through sys.modules and gets the stand-in;
    the only other importer, CHROMBPNET.py's contribs_bw branch, does not run
    here, and both package __init__.py files are empty.
    """
    import types

    if INTERPRET_MODULE in sys.modules and not getattr(
        sys.modules[INTERPRET_MODULE], "_igvf_stop", False
    ):
        raise RuntimeError(
            f"{INTERPRET_MODULE} was already imported; eager execution is already off"
        )

    def _stop(_args):
        raise _StopBeforeInterpretation

    stub = types.ModuleType(INTERPRET_MODULE)
    stub.main = _stop
    stub._igvf_stop = True
    sys.modules[INTERPRET_MODULE] = stub
    # `import a.b.c as x` binds x by getattr on the parent package (py3.8), so
    # the parent packages must exist and carry the stand-in as an attribute.
    # They are empty __init__.py files: importing them imports nothing else.
    import importlib

    parent_name, _, leaf = INTERPRET_MODULE.rpartition(".")
    setattr(importlib.import_module(parent_name), leaf, stub)
    return stub


def _write_train_report(output_dir, data_type, file_prefix) -> None:
    """chrombpnet's own `train`-mode HTML report, which needs no interpretation.

    The `pipeline`-mode report needs the motif report too, so 04.5 writes it
    once TF-MoDISco has run.
    """
    import argparse

    import chrombpnet.helpers.generate_reports.make_html as make_html

    make_html.main(
        argparse.Namespace(
            input_dir=output_dir,
            command="train",
            data_type=data_type,
            file_prefix=file_prefix,
            html_prefix="./",
        )
    )


def main() -> int:
    prepared, require_prepared, stop_before_interpretation, chrombpnet_argv = _split_argv(
        sys.argv[1:]
    )
    log.setup()

    def _arg(flag):
        return chrombpnet_argv[chrombpnet_argv.index(flag) + 1] if flag in chrombpnet_argv else None

    signal_path = _arg("-ifrag") or _arg("-ibam") or _arg("-itag")
    assay = _arg("-d")

    reuse = False
    if require_prepared and not prepared:
        logger.error("--require-prepared given without --prepared-bigwig")
        return 1

    if prepared:
        prepared_dir = Path(prepared)
        reuse = _sidecar_matches(prepared_dir, signal_path, assay)
        if not reuse:
            if require_prepared:
                # The configured signal is a bigwig. chrombpnet's parser required
                # an -ifrag placeholder, and without the prepared bigwig it would
                # now try to read that bigwig as a fragment file -- silent garbage.
                logger.error(
                    "the configured signal is a bigwig, so a valid prepared bigwig in %s "
                    "is required and none matched. Re-run 00.0.prepare_signal.sh.",
                    prepared_dir,
                )
                return 1
            logger.warning(
                "not reusing %s; chrombpnet will do its own conversion", prepared_dir / BIGWIG
            )

    if reuse:
        import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig

        source = Path(prepared) / BIGWIG

        def _install_prepared(args):
            """Stand in for reads_to_bigwig.main: drop the prepared bigwig in place.

            Copies rather than no-ops because chrombpnet creates auxiliary/ itself
            with exist_ok=False, so nothing can be staged there in advance.
            """
            dest = Path(f"{args.output_prefix}_unstranded.bw")
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, dest)  # same filesystem: free
            except OSError:
                shutil.copy2(source, dest)
            logger.info("reused prepared bigwig %s -> %s (skipped conversion)", source, dest)

        reads_to_bigwig.main = _install_prepared

    # chrombpnet's one-hot encoder routes through np.unique(return_inverse=True),
    # whose int64 inverse costs 8 bytes per base for the whole training set at
    # once. Ours is byte-identical and ~4x lighter; install() verifies that
    # against the container's own function before swapping. See
    # lib/python/utils/onehot.py and docs/resource-measurements.md.
    onehot.install()

    # `chrombpnet pipeline` goes on, after training, predictions and marginal
    # footprinting, to DeepLIFT on a 30K peak subsample and TF-MoDISco -- inside
    # this GPU job, where TF-MoDISco (CPU-only, the long pole) holds the GPU
    # idle for hours. With --stop-before-interpretation, chrombpnet's own
    # interpretation entry point is replaced by a stop signal, so everything
    # up to it runs as chrombpnet's unmodified code and nothing after it runs;
    # 04.4 (DeepLIFT) and 04.5 (TF-MoDISco) do the rest from the outputs left.
    if stop_before_interpretation:
        if not chrombpnet_argv or chrombpnet_argv[0] != "pipeline":
            logger.error("--stop-before-interpretation only applies to `chrombpnet pipeline`")
            return 1
        _install_interpretation_stop()

    import chrombpnet.CHROMBPNET as chrombpnet_cli

    sys.argv = ["chrombpnet", *chrombpnet_argv]
    try:
        chrombpnet_cli.main()
    except _StopBeforeInterpretation:
        logger.info("stopped `chrombpnet pipeline` before interpretation (04.4/04.5 run it)")
        _write_train_report(_arg("-o"), _arg("-d"), _arg("-fp"))
    finally:
        # This process is the one that allocated the training arrays, so its
        # peak RSS is the step's real footprint. The step's metadata trap runs
        # in a SIBLING process (emit_metadata.py) that cannot see it, so hand
        # the number over through a file. See docs/resource-measurements.md.
        metadata.report_peak_rss()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

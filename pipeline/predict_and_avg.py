"""
10.predict_and_avg.py
Generate genome-wide accessibility prediction bigwigs by averaging ChromBPNet
model outputs across any number of folds.

Adapted from the Greenleaf HDMA pipeline (08-predict_and_avg.py), with
support for a variable number of folds (not hard-coded to 5).

Outputs (given --output-prefix PREFIX and --output-key KEY):
  PREFIX_chrombpnet_KEY.bw                    – predicted profile bigwig
  PREFIX_chrombpnet_KEY_preds_w_logcounts.bed – per-peak predicted log counts
"""

import argparse

import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
import chrombpnet.training.utils.losses as losses
import numpy as np
import pandas as pd
import pyfaidx
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Average ChromBPNet predictions across folds and write bigwig."
    )
    p.add_argument(
        "-cm",
        "--chrombpnet-model",
        type=str,
        action="append",
        required=True,
        help="Path to one chrombpnet model .h5 (repeat for each fold)",
    )
    p.add_argument(
        "-r",
        "--regions",
        type=str,
        required=True,
        help="10-column narrowPeak BED file of prediction loci",
    )
    p.add_argument(
        "-g", "--genome", type=str, required=True, help="Reference genome FASTA"
    )
    p.add_argument(
        "-c",
        "--chrom-sizes",
        type=str,
        required=True,
        help="Chromosome sizes (2-column TSV)",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output path prefix (no extension)",
    )
    p.add_argument(
        "--output-key",
        type=str,
        required=True,
        help="Label inserted into output filenames (e.g. 'nobias' or 'uncorrected')",
    )
    p.add_argument("-b", "--batch-size", type=int, default=64)
    p.add_argument(
        "-ob",
        "--output-bed",
        type=bool,
        default=False,
        help="Write BED file with per-peak predicted log counts",
    )
    p.add_argument(
        "-d",
        "--debug-chr",
        nargs="+",
        type=str,
        default=None,
        help="Restrict to specific chromosomes for debugging",
    )
    return p.parse_args()


def softmax(x):
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    e = np.exp(x_centered)
    return e / np.sum(e, axis=1, keepdims=True)


def load_model_wrapper(path):
    custom_objects = {"multinomial_nll": losses.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(path, compile=False)
    print(f"Loaded model: {path}")
    model.summary()
    return model


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    models, inputlens, outputlens = [], [], []
    for path in args.chrombpnet_model:
        m = load_model_wrapper(path)
        models.append(m)
        inputlens.append(int(m.input_shape[1]))
        outputlens.append(int(m.output_shape[0][1]))

    assert len(set(inputlens)) == 1, "All models must share the same input length"
    assert len(set(outputlens)) == 1, "All models must share the same output length"
    inputlen = inputlens[0]
    outputlen = outputlens[0]
    n_models = len(models)
    print(f"Loaded {n_models} model(s). inputlen={inputlen}, outputlen={outputlen}")

    # ------------------------------------------------------------------
    # Load regions and sequences
    # ------------------------------------------------------------------
    regions_df = pd.read_csv(args.regions, sep="\t", names=NARROWPEAK_SCHEMA)
    if args.debug_chr:
        regions_df = regions_df[regions_df["chr"].isin(args.debug_chr)]

    with pyfaidx.Fasta(args.genome) as g:
        seqs, regions_used = bigwig_helper.get_seq(regions_df, g, inputlen)

    gs = bigwig_helper.read_chrom_sizes(args.chrom_sizes)
    regions = bigwig_helper.get_regions(args.regions, outputlen, regions_used)

    # Write the set of regions actually used
    regions_df[regions_used].to_csv(
        f"{args.output_prefix}_chrombpnet_{args.output_key}_preds.bed",
        sep="\t",
        header=False,
        index=False,
    )

    # ------------------------------------------------------------------
    # Run inference, accumulate across models
    # ------------------------------------------------------------------
    sum_logits = None
    sum_logcounts = None

    for model in models:
        pred_logits, pred_logcts = model.predict(
            [seqs], batch_size=args.batch_size, verbose=True
        )
        pred_logits = np.squeeze(pred_logits)

        if sum_logits is None:
            sum_logits = pred_logits.copy()
            sum_logcounts = pred_logcts.copy()
        else:
            sum_logits += pred_logits
            sum_logcounts += pred_logcts

    avg_logits = sum_logits / n_models
    avg_logcounts = sum_logcounts / n_models
    avg_prob = softmax(avg_logits)
    avg_counts = np.expand_dims(np.exp(avg_logcounts)[:, 0], axis=1)
    avg_profile = avg_counts * avg_prob

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    bw_out = f"{args.output_prefix}_chrombpnet_{args.output_key}.bw"
    bigwig_helper.write_bigwig(
        avg_profile, regions, gs, bw_out, debug_chr=args.debug_chr
    )
    print(f"Wrote bigwig: {bw_out}")

    if args.output_bed:
        bed_out = (
            f"{args.output_prefix}_chrombpnet_{args.output_key}_preds_w_logcounts.bed"
        )
        df_used = regions_df[regions_used].copy()
        df_used.insert(10, "logcounts", avg_logcounts)
        df_used.to_csv(bed_out, sep="\t", header=False, index=False)
        print(f"Wrote log-counts BED: {bed_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
motif_compendium.py
Build a non-redundant motif compendium from per-dataset MoDISco H5 files
using MotifCompendium (Kundaje lab).

Called by 09.0.cross_dataset_compendium.sh, which pools every dataset and
whose modisco_compiled.h5 is what 10.0 calls hits with, and by the superseded
deprecated/motif_compendium.sh (per dataset).

The clustering algorithm is passed explicitly (--algorithm, default
cpm_leiden), because MotifCompendium v1.0.19 changed cluster()'s default from
"cpm_leiden" to ["cpm_leiden", "k_centroids"]. The k-centroids pass then
reassigns every motif to its nearest cluster average without looking at
similarity_threshold, and at its default n_iterations=-1 repeats until no
motif moves, with no iteration cap. Taking the new default would silently
change what --threshold means; cpm_leiden alone is what v1.0.16 ran.

Input:  2-column TSV (-c): dataset_name <TAB> /path/to/modisco_counts_results.h5
Output (inside --out-dir):
  modisco_compiled.h5          - cluster-averaged motifs in MoDISco format, for Fi-NeMo (10.0)
  modisco_compendium.meme      - same cluster-averaged motifs in MEME format
  modisco_compendium.mc        - pickled MotifCompendium object (for inspection)
  modisco_compendium_meta.tsv  - per-motif metadata with TF annotations + cluster IDs

Usage:
  python src/motif_compendium.py \\
      -c  results/compendium/modisco_compiled/modisco_config.tsv \\
      -o  results/compendium/modisco_compiled \\
      --ref-db /path/to/MotifCompendium-1.0.19-Database-Human.meme.txt \\
      --threshold 0.95 \\
      --algorithm cpm_leiden \\
      --cpus 16

Prerequisites:
  This repo's motif-compendium pixi environment (MotifCompendium v1.0.19,
  CPU only): `pixi install -e motif-compendium`. The steps enter it with
  activate_env "${motif_compendium_env}".
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis

from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a non-redundant motif compendium using MotifCompendium."
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        help="2-column TSV: dataset_name <TAB> modisco_h5_path",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Output directory (modisco_compiled.h5 written here for FiNeMo)",
    )
    p.add_argument(
        "--ref-db",
        required=True,
        help="Reference motif database in MEME format for annotation (the steps pass ${ref_db_meme}, MotifCompendium-1.0.19-Database-Human.meme.txt)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for Leiden clustering (default: 0.95)",
    )
    p.add_argument(
        "--algorithm",
        nargs="+",
        default=["cpm_leiden"],
        metavar="NAME",
        help=(
            "MotifCompendium clustering algorithm(s), run in order, each starting "
            "from the previous one's clusters (default: cpm_leiden, as in "
            "MotifCompendium v1.0.16). v1.0.19's own default, cpm_leiden "
            "k_centroids, adds a k-centroids pass that ignores --threshold."
        ),
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="CPUs for pairwise similarity computation (default: 8)",
    )
    p.add_argument(
        "--min-annotation-score",
        type=float,
        default=0.7,
        help="Min similarity to reference database for annotation match (default: 0.7)",
    )
    log.add_logging_args(p)
    return p.parse_args()


def load_config(config_path):
    modisco_dict = {}
    with open(config_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning(f"  [WARN] skipping malformed line: {line}")
                continue
            dataset, h5_path = parts
            if not os.path.exists(h5_path):
                logger.warning(f"  [WARN] {dataset}: H5 not found, skipping: {h5_path}")
                continue
            modisco_dict[dataset] = h5_path
    return modisco_dict


def main():
    args = parse_args()
    log.setup_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    h5_out = os.path.join(args.out_dir, "modisco_compiled.h5")
    meme_out = os.path.join(args.out_dir, "modisco_compendium.meme")
    mc_out = os.path.join(args.out_dir, "modisco_compendium.mc")
    meta_out = os.path.join(args.out_dir, "modisco_compendium_meta.tsv")

    # 1. Load MoDISco H5s
    modisco_dict = load_config(args.config)
    logger.info(f"@ loaded {len(modisco_dict)} MoDISco H5 paths: {list(modisco_dict.keys())}")
    if not modisco_dict:
        sys.exit("ERROR: no valid MoDISco H5 files found in config.")

    # 2. Build MotifCompendium (reads pos_patterns + neg_patterns from each H5)
    MotifCompendium.set_compute_options(max_cpus=args.cpus, progress_bar=True)
    logger.info("@ building MotifCompendium from MoDISco H5s...")
    mc = MotifCompendium.build_from_modisco(modisco_dict)
    logger.info(f"@ {len(mc.metadata)} motifs loaded across all datasets")

    # 3. Annotate individual motifs against reference database before clustering
    logger.info(f"@ annotating against reference database ({args.ref_db})...")
    utils_analysis.assign_label_from_pfms(
        mc,
        pfm_file=args.ref_db,
        save_col_prefix="annotation",
        min_score=args.min_annotation_score,
        save_images=False,
    )
    logger.info("@ annotation done")

    # 4. Cluster at similarity threshold (Leiden community detection)
    cluster_col = "cluster_id"
    logger.info(
        f"@ clustering with {' -> '.join(args.algorithm)} at similarity threshold "
        f"{args.threshold}..."
    )
    mc.cluster(
        algorithm=args.algorithm,
        similarity_threshold=args.threshold,
        save_name=cluster_col,
    )
    n_clusters = mc.metadata[cluster_col].nunique()
    logger.info(f"@ {len(mc.metadata)} motifs → {n_clusters} clusters (threshold={args.threshold})")

    # 5. Export clustered modisco H5 for Fi-NeMo (10.0)
    #    One averaged pattern per cluster, written to modisco_compiled.h5, the
    #    name 10.0.run_finemo_unified.sh reads.
    logger.info(f"@ exporting clustered modisco H5 → {h5_out}")
    utils_analysis.export_compendium_clustered_modisco(mc, cluster_col, h5_out)
    logger.info(f"@ modisco_compiled.h5 written ({n_clusters} patterns)")

    # 6. Export MEME format — cluster averages, matching the H5 content
    logger.info(f"@ exporting MEME → {meme_out}")
    mc_avg = mc.cluster_averages(clustering=cluster_col, aggregations=[], weight_col=None)
    utils_analysis.export_compendium_meme(mc_avg, meme_out, name_col="source_cluster")

    # 7. Save MotifCompendium object (for interactive inspection / re-analysis)
    logger.info(f"@ saving MotifCompendium object → {mc_out}")
    with open(mc_out, "wb") as fh:
        pickle.dump(mc, fh)

    # 8. Save full metadata TSV (motif names, source dataset, TF annotations,
    #    cluster IDs)
    mc.metadata.to_csv(meta_out, sep="\t", index=False)
    logger.info(f"@ metadata TSV → {meta_out} ({len(mc.metadata)} rows)")

    logger.info(
        f"\n@ done: {n_clusters} distinct motifs from "
        f"{len(mc.metadata)} input patterns across "
        f"{len(modisco_dict)} datasets"
    )
    logger.info(f"  FiNeMo input : {h5_out}")
    logger.info(f"  Annotations  : {meta_out}")


if __name__ == "__main__":
    main()

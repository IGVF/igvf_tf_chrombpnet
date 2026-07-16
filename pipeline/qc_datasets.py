#!/usr/bin/env python
# qc_datasets.py
# Purpose: Compute QC statistics across all ChromBPNet training datasets.
# Metrics: total fragments, cells, FRIP, peaks, fragment size distribution,
#          per-chromosome counts, and ChromBPNet training readiness.
# Outputs: summary TSV, per-metric TSVs, and plots in results/plots/dataset_qc/

# %% Parameters

import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

core_path = "/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab"
output_path = f"{core_path}/results/plots/dataset_qc"
save_plots = True

DATASETS = {
    "igvf3_cardiomyocyte": "Cardiomyocyte\n(igvf3)",
    "igvf6_definitive_endoderm": "Def. Endoderm\n(igvf6)",
    "igvf11_h7_hesc": "hESC H7\n(igvf11)",
    "igvf17_endothelial": "Endothelial\n(igvf17)",
}

DATASET_COLORS = {
    "igvf3_cardiomyocyte": "#E07B54",
    "igvf6_definitive_endoderm": "#5E8C61",
    "igvf11_h7_hesc": "#5B84C4",
    "igvf17_endothelial": "#B666D2",
}

MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# Approximate minimum total fragments for ChromBPNet bias model to converge
CHROMBPNET_MIN_FRAGS = 20_000_000

CHUNK_SIZE = 1_000_000


# %% Matplotlib style

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True

os.makedirs(output_path, exist_ok=True)


# %% Helper functions


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{output_path}/{stem}.{ext}")
    print(f"Saved: {output_path}/{stem}.pdf/.png")


def collect_fragment_stats(fragments_gz):
    """Parse fragment file in chunks; return total, per-chrom, n_cells, size bins."""
    valid_chroms = set(MAIN_CHROMS)
    chrom_counts = defaultdict(int)
    barcode_set = set()
    size_bins = {
        "NFR (<200 bp)": 0,
        "Mono (200-400 bp)": 0,
        "Di (400-600 bp)": 0,
        "Multi (>600 bp)": 0,
    }
    total = 0

    for chunk in pd.read_csv(
        fragments_gz,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "barcode", "count"],
        comment="#",
        chunksize=CHUNK_SIZE,
        compression="gzip",
    ):
        chunk = chunk[chunk["chrom"].isin(valid_chroms)]
        for chrom, n in chunk["chrom"].value_counts().items():
            chrom_counts[chrom] += n
        barcode_set.update(chunk["barcode"].unique())
        sizes = chunk["end"] - chunk["start"]
        size_bins["NFR (<200 bp)"] += int((sizes < 200).sum())
        size_bins["Mono (200-400 bp)"] += int(((sizes >= 200) & (sizes < 400)).sum())
        size_bins["Di (400-600 bp)"] += int(((sizes >= 400) & (sizes < 600)).sum())
        size_bins["Multi (>600 bp)"] += int((sizes >= 600).sum())
        total += len(chunk)

    return total, dict(chrom_counts), len(barcode_set), size_bins


def collect_peak_stats(peaks_file):
    """Read narrowPeak: total, per-chrom counts, median width, total bp."""
    peaks = pd.read_csv(
        peaks_file,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signal",
            "pval",
            "qval",
            "summit",
        ],
    )
    peaks = peaks[peaks["chrom"].isin(set(MAIN_CHROMS))]
    per_chrom = peaks["chrom"].value_counts().to_dict()
    widths = peaks["end"] - peaks["start"]
    return len(peaks), per_chrom, float(widths.median()), int(widths.sum())


def compute_frip(fragments_gz, peaks_file):
    """Fragments overlapping any peak via bedtools intersect."""
    cmd = (
        f"bedtools intersect "
        f"-a <(zcat {fragments_gz} | grep -v '^#') "
        f"-b {peaks_file} -u | wc -l"
    )
    result = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())


# %% Compute stats for each dataset

stats = {}

for ds_id, ds_label in DATASETS.items():
    print(f"\n[{ds_id}]")
    ds_dir = f"{core_path}/{ds_id}"
    fragments_gz = f"{ds_dir}/data/fragments/{ds_id}_atac_fragments_main_chrs.tsv.gz"
    peaks_file = (
        f"{ds_dir}/results/preprocessing/{ds_id}_all_peaks_no_blacklist.narrowPeak"
    )

    assert os.path.exists(fragments_gz), f"Missing: {fragments_gz}"
    assert os.path.exists(peaks_file), f"Missing: {peaks_file}"

    print("  fragments...")
    total_frags, frags_per_chrom, n_cells, size_bins = collect_fragment_stats(
        fragments_gz
    )

    print("  peaks...")
    total_peaks, peaks_per_chrom, median_peak_width, total_peak_bp = collect_peak_stats(
        peaks_file
    )

    print("  FRIP (bedtools)...")
    in_peaks = compute_frip(fragments_gz, peaks_file)
    frip = in_peaks / total_frags if total_frags > 0 else 0.0

    stats[ds_id] = {
        "label": ds_label,
        "total_fragments": total_frags,
        "n_cells": n_cells,
        "frags_per_cell": total_frags / n_cells if n_cells > 0 else 0,
        "total_peaks": total_peaks,
        "median_peak_width": median_peak_width,
        "total_peak_bp": total_peak_bp,
        "frip": frip,
        "frip_pct": frip * 100,
        "frip_count": in_peaks,
        "frags_per_chrom": frags_per_chrom,
        "peaks_per_chrom": peaks_per_chrom,
        "size_bins": size_bins,
        "chrombpnet_ready": total_frags >= CHROMBPNET_MIN_FRAGS,
    }
    print(
        f"  {total_frags:,} frags | {n_cells:,} cells | "
        f"{total_peaks:,} peaks | FRIP={frip:.3f} | "
        f"{'OK' if total_frags >= CHROMBPNET_MIN_FRAGS else 'INSUFFICIENT'}"
    )


# %% Build summary table

summary = pd.DataFrame(
    [
        {
            "dataset": ds_id,
            "label": v["label"].replace("\n", " "),
            "total_fragments": v["total_fragments"],
            "n_cells": v["n_cells"],
            "frags_per_cell": round(v["frags_per_cell"]),
            "total_peaks": v["total_peaks"],
            "median_peak_width": v["median_peak_width"],
            "total_peak_bp": v["total_peak_bp"],
            "frip": round(v["frip"], 4),
            "frip_pct": round(v["frip_pct"], 2),
            "chrombpnet_ready": v["chrombpnet_ready"],
        }
        for ds_id, v in stats.items()
    ]
).set_index("dataset")

summary.to_csv(f"{output_path}/summary_stats_data.tsv", sep="\t")
print("\nSummary:")
print(summary.to_string())


# %% Plot 1: Summary metrics (5 panels)

ds_order = list(DATASETS.keys())
labels = [DATASETS[d] for d in ds_order]
colors = [DATASET_COLORS[d] for d in ds_order]

METRICS = [
    ("total_fragments", "Total fragments", 1e6, "M"),
    ("n_cells", "Unique barcodes (cells)", 1e3, "K"),
    ("frags_per_cell", "Mean frags per cell", 1e3, "K"),
    ("total_peaks", "Total peaks", 1e3, "K"),
    ("frip_pct", "FRIP", 1, "%"),
]

fig, axes = plt.subplots(1, len(METRICS), figsize=(3 * len(METRICS), 5))

for ax, (col, title, scale, unit) in zip(axes, METRICS):
    vals = [summary.loc[d, col] / scale for d in ds_order]
    bars = ax.bar(
        range(len(ds_order)), vals, color=colors, width=0.6, edgecolor="white"
    )
    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(unit)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}{unit}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if col == "total_fragments":
        threshold_scaled = CHROMBPNET_MIN_FRAGS / scale
        ax.axhline(threshold_scaled, color="red", linestyle="--", linewidth=1)
        ax.text(
            len(ds_order) - 0.5,
            threshold_scaled,
            f"min {threshold_scaled:.0f}M",
            color="red",
            fontsize=8,
            va="bottom",
            ha="right",
        )

plt.tight_layout()
if save_plots:
    save_fig(fig, "summary_metrics")


# %% Build per-chromosome DataFrames

chrom_sizes = pd.read_csv(
    f"{core_path}/genome/hg38.chrom.sizes",
    sep="\t",
    header=None,
    names=["chrom", "size"],
).set_index("chrom")["size"]

peaks_chr_df = (
    pd.DataFrame({ds_id: v["peaks_per_chrom"] for ds_id, v in stats.items()})
    .reindex(MAIN_CHROMS)
    .fillna(0)
    .astype(int)
)

frags_chr_df = (
    pd.DataFrame({ds_id: v["frags_per_chrom"] for ds_id, v in stats.items()})
    .reindex(MAIN_CHROMS)
    .fillna(0)
    .astype(int)
)

chrom_mbp = chrom_sizes.reindex(MAIN_CHROMS).values / 1e6
frags_chr_norm = frags_chr_df.div(chrom_mbp, axis=0)
peaks_chr_norm = peaks_chr_df.div(chrom_mbp, axis=0)

peaks_chr_df.to_csv(f"{output_path}/peaks_per_chromosome_data.tsv", sep="\t")
frags_chr_df.to_csv(f"{output_path}/fragments_per_chromosome_raw_data.tsv", sep="\t")
frags_chr_norm.to_csv(f"{output_path}/fragments_per_chromosome_norm_data.tsv", sep="\t")
peaks_chr_norm.to_csv(f"{output_path}/peaks_per_chromosome_norm_data.tsv", sep="\t")


# %% Plot 2: Peaks per chromosome (grouped bar)

x = np.arange(len(MAIN_CHROMS))
bar_width = 0.18

fig, ax = plt.subplots(figsize=(14, 4))
for i, ds_id in enumerate(ds_order):
    ax.bar(
        x + i * bar_width,
        peaks_chr_df[ds_id],
        width=bar_width,
        label=DATASETS[ds_id].replace("\n", " "),
        color=DATASET_COLORS[ds_id],
        edgecolor="white",
    )
ax.set_xticks(x + bar_width * (len(ds_order) - 1) / 2)
ax.set_xticklabels(MAIN_CHROMS, rotation=90, fontsize=8)
ax.set_ylabel("Number of peaks")
ax.set_title("Peaks per chromosome")
ax.legend(fontsize=9)
plt.tight_layout()
if save_plots:
    save_fig(fig, "peaks_per_chromosome")


# %% Plot 3: Fragment density per chromosome (normalised to fragments per Mbp)

fig, ax = plt.subplots(figsize=(14, 4))
for i, ds_id in enumerate(ds_order):
    ax.bar(
        x + i * bar_width,
        frags_chr_norm[ds_id],
        width=bar_width,
        label=DATASETS[ds_id].replace("\n", " "),
        color=DATASET_COLORS[ds_id],
        edgecolor="white",
    )
ax.set_xticks(x + bar_width * (len(ds_order) - 1) / 2)
ax.set_xticklabels(MAIN_CHROMS, rotation=90, fontsize=8)
ax.set_ylabel("Fragments per Mbp")
ax.set_title("Fragment density per chromosome")
ax.legend(fontsize=9)
plt.tight_layout()
if save_plots:
    save_fig(fig, "fragments_per_chromosome_norm")


# %% Plot 4: Fragment size distribution (stacked % bars)

size_bins_order = [
    "NFR (<200 bp)",
    "Mono (200-400 bp)",
    "Di (400-600 bp)",
    "Multi (>600 bp)",
]
size_colors = ["#3B7AB5", "#7CB9E8", "#BDBDBD", "#636363"]

size_df = pd.DataFrame(
    {ds_id: v["size_bins"] for ds_id, v in stats.items()},
    index=size_bins_order,
).T
size_df_pct = size_df.div(size_df.sum(axis=1), axis=0) * 100
size_df_pct.index = [DATASETS[d].replace("\n", " ") for d in size_df_pct.index]
size_df_pct.to_csv(f"{output_path}/fragment_size_distribution_data.tsv", sep="\t")

fig, ax = plt.subplots(figsize=(4, 5))
bottoms = np.zeros(len(ds_order))
for size_bin, color in zip(size_bins_order, size_colors):
    vals = np.array([size_df_pct.iloc[i][size_bin] for i in range(len(ds_order))])
    bars = ax.bar(
        range(len(ds_order)),
        vals,
        bottom=bottoms,
        color=color,
        label=size_bin,
        edgecolor="white",
        width=0.6,
    )
    for i, (bar, val) in enumerate(zip(bars, vals)):
        if val > 5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottoms[i] + val / 2,
                f"{val:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    bottoms += vals

ax.set_xticks(range(len(ds_order)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Fragments (%)")
ax.set_title("Fragment size distribution")
ax.legend(loc="upper right", fontsize=8, frameon=False)
plt.tight_layout()
if save_plots:
    save_fig(fig, "fragment_size_distribution")


# %% CONCLUSIONS.txt

hline = "=" * 60

ready = [d for d in ds_order if stats[d]["chrombpnet_ready"]]
not_ready = [d for d in ds_order if not stats[d]["chrombpnet_ready"]]

lines = [
    hline,
    "DATASET QC CONCLUSIONS",
    hline,
    "",
    "SUMMARY",
    "-" * 40,
]
for ds_id in ds_order:
    v = stats[ds_id]
    status = "READY" if v["chrombpnet_ready"] else "INSUFFICIENT DEPTH"
    lines.append(
        f"  {ds_id}:"
        f" {v['total_fragments']:,} frags"
        f" | {v['n_cells']:,} cells"
        f" | {round(v['frags_per_cell']):,} frags/cell"
        f" | {v['total_peaks']:,} peaks"
        f" | FRIP={v['frip_pct']:.1f}%"
        f" | {status}"
    )

lines += [
    "",
    "TRAINING FAILURES",
    "-" * 40,
    "  igvf3_cardiomyocyte (~5.3M frags, 925 cells) and",
    "  igvf17_endothelial (~5.0M frags, 380 cells) fail ChromBPNet bias",
    "  training: assert(counts_threshold > 0) because mean non-peak window",
    "  coverage is ~0 -- nearly all 1000bp windows have 0 reads.",
    f"  Recommended minimum: ~{CHROMBPNET_MIN_FRAGS / 1e6:.0f}M fragments.",
    "",
    "IGVF11 WARNING",
    "-" * 40,
    "  igvf11_h7_hesc (90M frags) completed bias training but received",
    "  'warn' status: nonpeaks_pearsonr is negative (-0.40 to +0.02).",
    "  Likely cause: broad hESC chromatin accessibility creates noisy",
    "  non-peak signal, making bias harder to learn.",
    "",
    "RECOMMENDED FIXES",
    "-" * 40,
    "  igvf17_endothelial: merge with igvf0/wtc11_endo_endothelial_1 (98MB)",
    "    + wtc11_endo_endothelial_2 (60MB) -> ~20M fragments.",
    "  igvf3_cardiomyocyte: merge with igvf3/h9_cardio_cardiomyocte_d15 (285MB)",
    "    and/or h9_cardio_cardiomyocte_d8 (332MB) -> 30-64M fragments.",
    "",
    "NEXT STEPS",
    "-" * 40,
    "  1. Decide on merge strategy (iPSC line mixing vs same-line only).",
    "  2. Update 00.copy_and_prepare_data.sh for merged fragments and peaks.",
    "  3. Re-run 01-02 preprocessing on merged datasets.",
    "  4. Re-run 03 bias sweep.",
]

text = "\n".join(lines) + "\n"
with open(f"{output_path}/CONCLUSIONS.txt", "w") as fh:
    fh.write(text)
print("\n" + text)

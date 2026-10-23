#!/usr/bin/env python3
"""Quality control for tandem-repeat calls, before they are screened for novelty.

``trfcaller`` returns every repeat the finder reports, including weak ones: a
repeat that is mostly noise, one that covers a sliver of a long insertion, or one
in an insertion too poorly supported to trust. Calling any of those "novel" says
nothing about the genome, so this step removes them first. It judges the *call*,
not whether the locus is already known; that is the catalogue screen's job.

What it needs
-------------
The table ``trfcaller`` returns: chrom, ins_coord, SVID, depth, insert_size,
sample, allele, rep_start, rep_end, motif, purity, motif_length, rep_length and
rep_units (listed in ``REQUIRED_COLUMNS``, which nothing enforces). ``depth`` is
the ``"<variant>,<reference>"`` read-count string. The numeric columns must
already be numeric.

What it adds
------------
``repeat_coverage = rep_length / insert_size``: the fraction of the insertion the
repeat call spans (0.0 when ``insert_size`` is not positive).

What it removes
---------------
Filters run in this order, each on what the previous one left, so a row must pass
all of them. The first two only run when their option is non-empty.

    --exclude-motif-sizes   drop calls whose motif_length is in the list
    --keep-motif-sizes      keep only calls whose motif_length is in the list
    --min-repeat-coverage   repeat_coverage >= this (default 0.8)
    --min-purity            purity >= this, on pytrf's 0-1 scale (default 0.7)
    --max-insert-size       insert_size <= this, in bp (default 10000)
    --min-depth             BOTH numbers in ``depth`` >= this (default 30)


Returns
-------
``filter_trf`` returns a new DataFrame of the passing rows, with
``repeat_coverage`` added and the original row labels kept. Those labels have
gaps wherever rows were removed, so a caller that needs positional alignment
(``screen_frame`` does) must ``reset_index(drop=True)`` first, as ``cli`` does.

"""

from __future__ import annotations

import argparse
import csv
import statistics as stats
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "chrom", "ins_coord", "SVID", "depth", "insert_size", "sample", "allele",
    "rep_start", "rep_end", "motif", "purity", "motif_length",
    "rep_length", "rep_units",
]

NUMERIC_FIELDS = {
    "depth": lambda x: sum([(int(v)) for v in x.split(",")]) if "," in x else int(x),
    "insert_size": int,
    "rep_start": int,
    "rep_end": int,
    "purity": float,
    "motif_length": int,
    "rep_length": int,
    "rep_units": float,
}


# --------------------------------------------------------------------------- #
# File i/o
# --------------------------------------------------------------------------- #



def write_filtered_tsv(rows, fieldnames, path):
    out_fields = list(fieldnames) + ["repeat_coverage"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in out_fields}
            if isinstance(row_out["repeat_coverage"], float):
                row_out["repeat_coverage"] = f"{row_out['repeat_coverage']:.4f}"
            writer.writerow(row_out)


def write_stats(funnel, motif_summary, rows_in, rows_out, path):
    uniq_ins_in = len({insertion_key(r) for r in rows_in})
    uniq_ins_out = len({insertion_key(r) for r in rows_out})

    with open(path, "w", newline="") as fh:
        fh.write("## Filtering funnel\n")
        writer = csv.DictWriter(
            fh, fieldnames=["step", "rows_in", "rows_removed", "rows_remaining"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in funnel:
            writer.writerow(row)

        fh.write("\n## Unique insertion counts (chrom+ins_coord+SVID+sample)\n")
        fh.write(f"unique_insertions_input\t{uniq_ins_in}\n")
        fh.write(f"unique_insertions_output\t{uniq_ins_out}\n")

        fh.write("\n## Summary by motif_length (final filtered set)\n")
        if motif_summary:
            writer = csv.DictWriter(
                fh, fieldnames=list(motif_summary[0].keys()), delimiter="\t"
            )
            writer.writeheader()
            for row in motif_summary:
                writer.writerow(row)
        else:
            fh.write("(no rows remaining after filtering)\n")


# --------------------------------------------------------------------------- #
# File ops
# --------------------------------------------------------------------------- #

def insertion_key(r):
    return (r["chrom"], r["ins_coord"], r["SVID"], r["sample"])


def summarize_by_motif_length(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["motif_length"], []).append(r)

    summary = []
    for motif_len in sorted(groups):
        grp = groups[motif_len]
        purities = [r["purity"] for r in grp]
        coverages = [r["repeat_coverage"] for r in grp]
        ins_sizes = [r["insert_size"] for r in grp]
        uniq_ins = {insertion_key(r) for r in grp}
        uniq_samples = {r["sample"] for r in grp}
        summary.append({
            "motif_length": motif_len,
            "n_trf_hits": len(grp),
            "n_unique_insertions": len(uniq_ins),
            "n_unique_samples": len(uniq_samples),
            "mean_purity": round(stats.mean(purities), 4),
            "median_purity": round(stats.median(purities), 4),
            "mean_repeat_coverage": round(stats.mean(coverages), 4),
            "median_repeat_coverage": round(stats.median(coverages), 4),
            "mean_insert_size": round(stats.mean(ins_sizes), 1),
            "median_insert_size": round(stats.median(ins_sizes), 1),
        })
    return summary


def add_repeat_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Add repeat_coverage column to the DataFrame vectorially."""
    ins_len = df["insert_size"]
    
    # Vectorized conditional: if insert_size > 0, compute division; otherwise assign 0.0
    df["repeat_coverage"] = np.where(
        ins_len > 0, 
        df["rep_length"] / ins_len, 
        0.0
    )
    
    return df


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def parse_int_list(s):
    if not s:
        return set()
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


def run_filters(df: pd.DataFrame, args):
    """Apply each filter in sequence on a pandas DataFrame, logging a funnel of counts."""
    exclude_sizes = parse_int_list(args.exclude_motif_sizes)
    keep_sizes = parse_int_list(args.keep_motif_sizes)

    funnel = []
    current = df.copy() 

    def log_step(name, before, after):
        removed = before - after
        funnel.append({
            "step": name, 
            "rows_in": before,
            "rows_removed": removed, 
            "rows_remaining": after,
        })

    # Log initial input state
    log_step("input", len(current), len(current))

    def apply_step(name, mask):
        nonlocal current
        before = len(current)
        current = current[mask]
        after = len(current)
        log_step(name, before, after)

    if exclude_sizes:
        apply_step(
            f"exclude_motif_sizes({sorted(exclude_sizes)})",
            ~current["motif_length"].isin(exclude_sizes),
        )

    if keep_sizes:
        apply_step(
            f"keep_motif_sizes_only({sorted(keep_sizes)})",
            current["motif_length"].isin(keep_sizes),
        )

    apply_step(
        f"min_repeat_coverage(>={args.min_repeat_coverage})",
        current["repeat_coverage"] >= args.min_repeat_coverage,
    )

    apply_step(
        f"min_purity(>={args.min_purity})",
        current["purity"] >= args.min_purity,
    )

    apply_step(
        f"max_insert_size(<={args.max_insert_size})",
        current["insert_size"] <= args.max_insert_size,
    )

    depth_split = current["depth"].astype(str).str.split(",")
    
    apply_step(
        f"min_depth(>={args.min_depth})",
        (depth_split.str[0].str.strip().astype(int) >= args.min_depth) #& 
        #(depth_split.str[1].str.strip().astype(int) >= args.min_depth), disabled for now, so reference depths are not filtering variants
    )

    return current, funnel



def filter_trf(in_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:

    print(f"[filter] Input TR rows:    {len(in_df)}")
    
    in_df = add_repeat_coverage(in_df)

    filtered, funnel = run_filters(in_df, args)

    #write_filtered_tsv(filtered, fieldnames, args.output)

    # stats_path = args.stats_output or (args.output + ".stats.tsv")
    # motif_summary = summarize_by_motif_length(filtered)
    # write_stats(funnel, motif_summary, rows, filtered, stats_path)

    print(f"[filter] Output TR rows:   {len(filtered)}")
    #print(f"Filtered TSV:  {args.output}")
    #print(f"Stats TSV:     {stats_path}")

    return pd.DataFrame(filtered)
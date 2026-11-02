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
# File ops
# --------------------------------------------------------------------------- #

def insertion_key(r):
    return (r["chrom"], r["ins_coord"], r["SVID"], r["sample"])


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
        if not tok:
            continue
        # "3-6" -> 3, 4, 5, 6 (inclusive on both ends)
        lo, sep, hi = tok.partition("-")
        if sep:
            if int(hi) < int(lo):
                raise ValueError(f"range {tok!r} ends before it starts")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(tok))
    return out


def run_filters(df: pd.DataFrame, args):
    """Apply each filter in sequence on a pandas DataFrame."""
    exclude_sizes = parse_int_list(args.exclude_motif_sizes)
    keep_sizes = parse_int_list(args.keep_motif_sizes)

    current = df.copy() 

    def apply_step(name, mask):
        nonlocal current
        current = current[mask]

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
        f"min_depth(>={args.min_depth})",
        (current["depth"] >= args.min_depth)
    )

    return current



def filter_trf(in_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:

    print(f"[filter] Input TR rows:    {len(in_df)}")
    
    in_df = add_repeat_coverage(in_df)
    filtered = run_filters(in_df, args)

    print(f"[filter] Output TR rows:   {len(filtered)}")

    return pd.DataFrame(filtered)
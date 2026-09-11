"""Load the input table, normalize dtype quirks, build feature blocks, and
assign each row to a motif-length x flank-presence stratum.

Three columns change shape between pipeline stages, and this module is where
that gets normalized once rather than three different callers guessing:

``depth``
    ``"DV,DR"`` (read support for variant/reference) straight off
    ``sv_trfcaller.py``, or a single summed int once ``filter_ins_trf.py`` has
    run. Both are accepted.

``insert_size``
    Sometimes the literal ``"[138]"`` (a stringified one-element list, from a
    VCF ``SVLEN`` that came back as a list); reused from
    ``novelty.insertions.parse_sizes``, which already handles this.

``rep_units``
    An int from stage 01, a float after ``filter_ins_trf.py``. Just needs a
    consistent numeric cast.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from intruder.pipeline.novelty.insertions import parse_sizes
from intruder.trcore.motifs import STR_MAX_MOTIF, canonical_motif, gc_fraction, shannon_entropy

from .columns import COHORT_FEATURES, FEATURE_BLOCKS, FeatureBlock

REQUIRED_INPUT_COLUMNS = (
    "purity", "motif_length", "rep_length", "insert_size", "depth", "motif",
)


def normalize_depth_total(values) -> pd.Series:
    """``depth`` -> total read support, whether it is ``"DV,DR"`` or a plain int."""
    raw = pd.Series(values).astype("string")
    has_comma = raw.str.contains(",", na=False).fillna(False)
    total = pd.to_numeric(raw.where(~has_comma), errors="coerce").astype("Float64")
    if has_comma.any():
        parts = raw[has_comma].str.split(",", expand=True).apply(
            lambda col: pd.to_numeric(col, errors="coerce"))
        total.loc[has_comma] = parts.sum(axis=1).astype("Float64")
    return total


def normalize_rep_units(values) -> pd.Series:
    """``rep_units`` -> float, whichever stage wrote it."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("Float64")


def compute_vaf(values) -> pd.Series:
    """``DV / (DV + DR)`` from the raw ``"DV,DR"`` depth string; ``NaN`` once collapsed.

    Mirrors :func:`normalize_depth_total`'s comma-detection -- same input, the
    other half of the split survives here. No imputation: a row whose
    ``depth`` is already a bare total gets ``NaN``, not a guessed ratio.
    """
    raw = pd.Series(values).astype("string")
    has_comma = raw.str.contains(",", na=False).fillna(False)
    vaf = pd.Series(pd.NA, index=raw.index, dtype="Float64")
    if has_comma.any():
        parts = raw[has_comma].str.split(",", n=1, expand=True).apply(
            lambda col: pd.to_numeric(col, errors="coerce"))
        dv, dr = parts[0], parts[1]
        vaf.loc[has_comma] = (dv / (dv + dr)).astype("Float64")
    return vaf


def motif_strata(motif_length, bounds: Sequence[int] = (STR_MAX_MOTIF,)) -> pd.Series:
    """Label each row's motif length against ``bounds`` (upper edges, sorted).

    ``bounds=(6,)`` (the default, :data:`trcore.motifs.STR_MAX_MOTIF`) produces
    two strata: ``"len<=6"`` and ``"len>6"``. This is a convention the model
    stratifies on, not a claim about where STRs end and VNTRs begin --
    ``calibration.py`` has the diagnostic for where feature distributions
    actually shift.
    """
    bounds = sorted(int(b) for b in bounds)
    edges = [-np.inf, *bounds, np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == -np.inf:
            labels.append(f"len<={hi:g}")
        elif hi == np.inf:
            labels.append(f"len>{lo:g}")
        else:
            labels.append(f"{lo:g}<len<={hi:g}")
    values = pd.to_numeric(motif_length, errors="coerce")
    binned = pd.cut(values, bins=edges, labels=labels, right=True)
    return binned.astype(object).where(values.notna(), "unknown_len").astype(str)


def audit_correlations(features: pd.DataFrame, columns: Sequence[str], *,
                        threshold: float = 0.95, mode: str = "warn"
                        ) -> list[tuple[str, str, float]]:
    """Pairwise Spearman audit of the survivor features. Runs by default during `fit`.

    ``mode="error"`` escalates a pair above ``threshold`` to a hard failure;
    the default ``"warn"`` prints and continues. Correlated features split
    SHAP credit arbitrarily per row, so this is a diagnostic, not a filter --
    nothing here drops a column automatically.
    """
    cols = [c for c in columns if c in features.columns]
    corr = features[cols].astype("float64").corr(method="spearman").abs()
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > threshold:
                pairs.append((a, b, float(r)))
    if pairs:
        detail = "; ".join(f"{a}~{b} (rho={r:.3f})" for a, b, r in pairs)
        message = f"[isolation-forest] correlated feature pair(s) above {threshold}: {detail}"
        if mode == "error":
            raise ValueError(message)
        print(message, file=sys.stderr)
    return pairs


def assert_no_missing(features: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise loudly, naming every offending column, if any of ``columns`` has a NaN.

    There is no imputation anywhere in this package: a missing value here means
    the input was bad (a blank cell, an unparseable dtype quirk) or a block was
    asked for without what it needs, and either way silently filling it in
    would corrupt the model rather than fix the row.
    """
    sub = features[list(columns)]
    bad = sub.columns[sub.isna().any()].tolist()
    if bad:
        raise ValueError(f"NaN in feature column(s) {bad}; fix the input or drop "
                         f"the offending block rather than impute")


@dataclass(frozen=True)
class FeatureTable:
    """The input table plus its computed feature columns and stratum labels."""

    frame: pd.DataFrame          # original input, unmodified
    features: pd.DataFrame       # feature columns requested, index-aligned to frame
    stratum: pd.Series           # motif-length x flank-presence label per row
    has_flank: pd.Series | None  # bool; None when flank presence was never determined
    has_vaf: pd.Series | None    # bool; None when vaf was dropped or never attempted
    blocks: tuple[FeatureBlock, ...]


def _cohort_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("SVID", "sample")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"the 'cohort' block needs column(s) {missing}")

    svid = frame["SVID"]
    canon = frame["motif"].astype(str).map(canonical_motif)
    rep_length = pd.to_numeric(frame["rep_length"], errors="coerce")

    n_carriers = frame.groupby(svid)["sample"].transform("nunique").astype("Float64")
    mean = rep_length.groupby(svid).transform("mean")
    std = rep_length.groupby(svid).transform("std")
    rep_length_cv = (std / mean).where(mean != 0).astype("Float64")
    n_distinct = canon.groupby(svid).transform("nunique").astype("Float64")

    out = pd.DataFrame(index=frame.index)
    out["n_carriers"] = n_carriers
    out["rep_length_cv"] = rep_length_cv
    out["n_distinct_canonical_motifs"] = n_distinct
    return out


def build_features(frame: pd.DataFrame, blocks: Sequence[FeatureBlock], *,
                   strata_bounds: Sequence[int] = (STR_MAX_MOTIF,),
                   flanks: pd.DataFrame | None = None,
                   drop_vaf: bool = False) -> FeatureTable:
    """Build every column ``blocks`` asks for, plus the stratum label.

    ``flanks`` is the output of :func:`flanks.compute_flank_features`, keyed
    to ``frame``'s index -- computed separately because it needs the VCF, not
    just the table.

    ``drop_vaf`` (default ``False``) skips computing ``vaf`` even when the
    input's ``depth`` column still has read-support to split -- vaf is
    computed by default whenever it's available, never imputed when it
    isn't. Unlike the flank partition, the motif-length stratum only gets a
    ``|vaf``/``|novaf`` suffix when ``has_vaf`` actually varies within the
    table; a table that is uniformly one depth format (every table in this
    repo, today) keeps its plain stratum labels rather than being fragmented
    for a distinction with no rows on the other side of it.
    """
    names = {b.name for b in blocks}
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"missing required input column(s): {missing}")

    insert_size = parse_sizes(frame["insert_size"]).astype("Float64")
    depth_total = normalize_depth_total(frame["depth"])

    out = pd.DataFrame(index=frame.index)
    coverage = None
    if "repeat_coverage" in frame.columns:
        coverage = pd.to_numeric(frame["repeat_coverage"], errors="coerce")
    else:
        rep_length = pd.to_numeric(frame["rep_length"], errors="coerce")
        coverage = rep_length / insert_size

    has_vaf = None
    if "intrinsic" in names:
        out["purity"] = pd.to_numeric(frame["purity"], errors="coerce")
        out["log_motif_length"] = np.log(pd.to_numeric(frame["motif_length"], errors="coerce"))
        out["log_rep_length"] = np.log(pd.to_numeric(frame["rep_length"], errors="coerce"))
        out["repeat_coverage"] = coverage
        out["log_depth_total"] = np.log(depth_total)
        out["motif_gc"] = frame["motif"].astype(str).map(gc_fraction)
        out["motif_entropy"] = frame["motif"].astype(str).map(shannon_entropy)
        if not drop_vaf:
            vaf = compute_vaf(frame["depth"])
            out["vaf"] = vaf
            has_vaf = vaf.notna()

    # Flank presence drives the structural-missingness partition regardless of
    # whether the `flank` block itself was requested -- `model.py` needs it to
    # route rows, and `has_flank` on the returned table is how it does that.
    has_flank = coverage < 1.0

    if "flank" in names:
        if flanks is None:
            raise ValueError("the 'flank' block needs --vcf to re-read ALT sequences")
        flank_cols = list(FEATURE_BLOCKS["flank"].columns)
        out[flank_cols] = flanks.reindex(frame.index)[flank_cols]
        # Structural missingness, not a sentinel: a row with no room for a
        # flank gets NaN here and is routed to the intrinsic-only partition by
        # `has_flank`, so this NaN never reaches a model.
        out.loc[~has_flank, flank_cols] = np.nan

    if "cohort" in names:
        cohort = _cohort_features(frame)
        for column in COHORT_FEATURES:
            out[column] = cohort[column]

    stratum = motif_strata(frame["motif_length"], strata_bounds)
    if "flank" in names:
        stratum = stratum + np.where(has_flank, "|flanked", "|unflanked")
    if has_vaf is not None and has_vaf.nunique() > 1:
        stratum = stratum + np.where(has_vaf, "|vaf", "|novaf")

    return FeatureTable(frame=frame, features=out,
                        stratum=pd.Series(stratum, index=frame.index, name="if_stratum"),
                        has_flank=has_flank, has_vaf=has_vaf, blocks=tuple(blocks))

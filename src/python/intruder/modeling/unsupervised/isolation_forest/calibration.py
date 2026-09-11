"""STRchive as a sanity set, and diagnostics for the stratification choices.

STRchive is ~82 curated, disease-biased loci that by construction excludes the
population-specific novel loci this model is hunting for -- it cannot be
positive training data. What it *can* do is check that candidates overlapping
a known-real TR are not scored anomalous: a low recall here means the model is
flagging real repeats as weird, which is a problem regardless of what it does
on genuinely novel ones.

Two checks live here, both diagnostic rather than filters:

* :func:`calibrate` -- the STRchive recall gate (>=95% of overlapping
  candidates survive a chosen operating threshold) plus a KS test comparing
  the overlap and background score distributions.
* :func:`stratum_boundary_shift` / :func:`motif_length_dominance` -- whether
  the motif-length stratification boundary lines up with an actual shift in
  feature distributions, and whether ``log_motif_length`` still dominates the
  SHAP attributions once stratified (if it does, the stratification is not
  doing its job).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from scipy.stats import ks_2samp

from intruder.pipeline.strchive.catalog import Catalog
from intruder.trcore.coords import to_internal

from .columns import shap_column


def _summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {"n": len(s), "mean": float(s.mean()), "median": float(s.median()),
           "std": float(s.std()) if len(s) > 1 else 0.0}


def _overlap_locus_ids(frame: pd.DataFrame, catalog: Catalog, *, chrom_col: str,
                       pos_col: str, coord_base: int, window: int) -> pd.Series:
    """The STRchive locus id each row overlaps (nearest within ``window``), or ``NA``."""
    points = to_internal(pd.to_numeric(frame[pos_col], errors="raise"), coord_base)
    ids: list[str | None] = []
    for chrom, pos in zip(frame[chrom_col], points):
        hits = catalog.nearby(str(chrom), int(pos), int(pos) + 1, window=window)
        ids.append(hits[0].id if hits else None)
    return pd.Series(ids, index=frame.index, dtype=object)


@dataclass(frozen=True)
class CalibrationReport:
    """Everything :func:`calibrate` found, as one row's worth of numbers."""

    n_candidates: int
    n_strchive_loci: int
    n_overlapping: int
    dropped_loci: tuple[str, ...]     # STRchive loci with zero overlapping candidates
    threshold: float
    min_recall: float
    recall: float
    passed: bool
    ks_statistic: float
    ks_pvalue: float
    overlap_summary: dict
    background_summary: dict

    def as_row(self) -> dict:
        """Flattened for a one-row TSV, the way ``novelty --metrics`` does it."""
        row = {
            "n_candidates": self.n_candidates,
            "n_strchive_loci": self.n_strchive_loci,
            "n_overlapping": self.n_overlapping,
            "n_dropped_loci": len(self.dropped_loci),
            "dropped_loci": ",".join(self.dropped_loci),
            "threshold": self.threshold,
            "min_recall": self.min_recall,
            "recall": self.recall,
            "passed": self.passed,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
        }
        for prefix, summary in (("overlap", self.overlap_summary),
                               ("background", self.background_summary)):
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        return row


def calibrate(frame: pd.DataFrame, catalog: Catalog, *, score_col: str = "if_score",
             chrom_col: str = "chrom", pos_col: str = "ins_coord", coord_base: int = 1,
             window: int = 0, min_recall: float = 0.95,
             threshold: float | None = None) -> CalibrationReport:
    """Recall of STRchive-overlapping candidates, plus a KS diagnostic against the rest.

    ``threshold`` is the operating point on ``score_col`` at/above which a call
    counts as "kept" (``score_samples`` runs low = anomalous, so higher is
    more normal). When not given, it is chosen as the loosest threshold that
    clears ``min_recall`` on the overlapping candidates themselves -- i.e. the
    ``(1 - min_recall)`` quantile of their scores.
    """
    overlap_locus = _overlap_locus_ids(frame, catalog, chrom_col=chrom_col,
                                       pos_col=pos_col, coord_base=coord_base,
                                       window=window)
    is_overlap = overlap_locus.notna()
    overlap_scores = pd.to_numeric(frame.loc[is_overlap, score_col], errors="coerce")
    background_scores = pd.to_numeric(frame.loc[~is_overlap, score_col], errors="coerce")

    covered = set(overlap_locus.dropna().unique())
    dropped = tuple(sorted(locus.id for locus in catalog if locus.id not in covered))

    if threshold is None:
        threshold = (float(overlap_scores.quantile(1 - min_recall))
                    if len(overlap_scores) else float("nan"))

    recall = float((overlap_scores >= threshold).mean()) if len(overlap_scores) else float("nan")

    if len(overlap_scores.dropna()) >= 2 and len(background_scores.dropna()) >= 2:
        ks_stat, ks_p = ks_2samp(overlap_scores.dropna(), background_scores.dropna())
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    return CalibrationReport(
        n_candidates=len(frame), n_strchive_loci=len(catalog),
        n_overlapping=int(is_overlap.sum()), dropped_loci=dropped,
        threshold=threshold, min_recall=min_recall, recall=recall,
        passed=bool(recall >= min_recall) if pd.notna(recall) else False,
        ks_statistic=float(ks_stat), ks_pvalue=float(ks_p),
        overlap_summary=_summary(overlap_scores), background_summary=_summary(background_scores),
    )


def stratum_boundary_shift(features: pd.DataFrame, motif_length: pd.Series,
                           bounds: Sequence[int],
                           columns: Sequence[str] | None = None) -> pd.DataFrame:
    """KS statistic per feature, on each side of each candidate boundary in ``bounds``.

    The 6bp STR/VNTR line is a convention, not a biological claim -- this is
    the check for whether it (or any other candidate boundary) actually lines
    up with a shift in feature distributions, one row per (bound, feature).
    """
    columns = list(columns) if columns is not None else list(features.columns)
    motif_length = pd.to_numeric(motif_length, errors="coerce")
    rows = []
    for bound in sorted(int(b) for b in bounds):
        below = features.loc[motif_length <= bound]
        above = features.loc[motif_length > bound]
        for column in columns:
            a = pd.to_numeric(below[column], errors="coerce").dropna()
            b = pd.to_numeric(above[column], errors="coerce").dropna()
            if len(a) >= 2 and len(b) >= 2:
                stat, p = ks_2samp(a, b)
            else:
                stat, p = float("nan"), float("nan")
            rows.append({"bound": bound, "feature": column, "ks_statistic": float(stat),
                        "ks_pvalue": float(p), "n_le_bound": len(a), "n_gt_bound": len(b)})
    return pd.DataFrame(rows)


def motif_length_dominance(shap_frame: pd.DataFrame, *,
                           feature: str = "log_motif_length",
                           threshold: float = 0.5) -> dict:
    """Share of total mean |SHAP| attribution held by ``feature``.

    If this stays high once the model is stratified by motif length, the
    stratification is not doing its job -- see the design note in
    ``features.motif_strata``.
    """
    shap_cols = [c for c in shap_frame.columns if c.startswith("shap_anom_")]
    abs_mean = shap_frame[shap_cols].abs().mean()
    total = float(abs_mean.sum())
    column = shap_column(feature)
    share = float(abs_mean.get(column, 0.0) / total) if total else float("nan")
    return {"feature": feature, "share": share, "dominant": bool(share > threshold)}

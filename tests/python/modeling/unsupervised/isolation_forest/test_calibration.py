"""Recall-gate arithmetic against a small synthetic STRchive catalog."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("scipy")

from intruder.modeling.unsupervised.isolation_forest.calibration import (
    calibrate,
    motif_length_dominance,
    stratum_boundary_shift,
)
from intruder.pipeline.strchive.catalog import Catalog, DiseaseLocus


def _locus(locus_id, start, end):
    return DiseaseLocus(
        id=locus_id, gene="GENE", chrom="chr1", start=start, end=end,
        disease="disease", disease_id="D1", inheritance=(), evidence=(),
        location_in_gene="", gene_strand="+", ref_copies=None, motif_len=3,
        novel=None, benign_min=None, benign_max=None, intermediate_min=None,
        intermediate_max=None, pathogenic_min=None, pathogenic_max=None,
        motifs={}, canonical={},
    )


@pytest.fixture
def catalog():
    # 0-based half-open, so 1-based POS 1001..1010 overlaps locus "A".
    return Catalog([_locus("A", 1000, 1010), _locus("B", 5000, 5010)],
                   build="hg38", version="test")


def _frame(rows):
    return pd.DataFrame(rows, columns=["chrom", "ins_coord", "if_score"])


def test_recall_is_one_when_every_overlapping_candidate_clears_the_threshold(catalog):
    frame = _frame([
        ("chr1", 1005, -0.40),
        ("chr1", 1006, -0.42),
        ("chr1", 9000, -0.90),   # background, non-overlapping
    ])
    report = calibrate(frame, catalog, threshold=-0.5, min_recall=0.95)
    assert report.n_overlapping == 2
    assert report.recall == pytest.approx(1.0)
    assert report.passed is True


def test_recall_drops_when_an_overlapping_candidate_scores_low(catalog):
    frame = _frame([
        ("chr1", 1005, -0.40),
        ("chr1", 1006, -0.80),   # scores below the threshold
    ])
    report = calibrate(frame, catalog, threshold=-0.5, min_recall=0.95)
    assert report.n_overlapping == 2
    assert report.recall == pytest.approx(0.5)
    assert report.passed is False


def test_dropped_loci_lists_strchive_loci_with_no_overlapping_candidate(catalog):
    frame = _frame([("chr1", 1005, -0.40)])   # only overlaps locus "A"
    report = calibrate(frame, catalog, threshold=-0.5, min_recall=0.95)
    assert report.dropped_loci == ("B",)


def test_auto_threshold_is_the_1_minus_min_recall_quantile_of_overlap_scores(catalog):
    frame = _frame([
        ("chr1", 1001, -0.10),
        ("chr1", 1002, -0.20),
        ("chr1", 1003, -0.30),
        ("chr1", 1004, -0.40),
    ])
    report = calibrate(frame, catalog, min_recall=0.5)
    assert report.threshold == pytest.approx(
        frame["if_score"].quantile(0.5))


def test_no_overlapping_candidates_reports_nan_recall_and_fails(catalog):
    frame = _frame([("chr1", 9000, -0.10)])
    report = calibrate(frame, catalog, threshold=-0.5, min_recall=0.95)
    assert report.n_overlapping == 0
    assert report.passed is False


def test_as_row_is_flat_and_includes_dropped_loci_count(catalog):
    frame = _frame([("chr1", 1005, -0.40)])
    report = calibrate(frame, catalog, threshold=-0.5, min_recall=0.95)
    row = report.as_row()
    assert row["n_dropped_loci"] == 1
    assert row["dropped_loci"] == "B"
    assert "overlap_mean" in row
    assert "background_mean" in row


# --------------------------------------------------------------------------- #
# stratification diagnostics
# --------------------------------------------------------------------------- #

def test_stratum_boundary_shift_detects_an_actual_distribution_shift():
    motif_length = pd.Series([1] * 50 + [10] * 50)
    features = pd.DataFrame({
        "x": [0.0] * 50 + [10.0] * 50,   # a real shift at the boundary
    })
    result = stratum_boundary_shift(features, motif_length, bounds=(6,))
    row = result.loc[result["feature"] == "x"].iloc[0]
    assert row["ks_statistic"] == pytest.approx(1.0)


def test_motif_length_dominance_flags_a_dominant_feature():
    shap_frame = pd.DataFrame({
        "shap_anom_log_motif_length": [10.0, -10.0, 9.0],
        "shap_anom_purity": [0.1, -0.1, 0.05],
    })
    result = motif_length_dominance(shap_frame, threshold=0.5)
    assert result["dominant"] is True
    assert result["share"] > 0.9


def test_motif_length_dominance_is_false_when_shared_evenly():
    shap_frame = pd.DataFrame({
        "shap_anom_log_motif_length": [1.0, -1.0],
        "shap_anom_purity": [1.0, -1.0],
        "shap_anom_motif_gc": [1.0, -1.0],
    })
    result = motif_length_dominance(shap_frame, threshold=0.5)
    assert result["dominant"] is False
    assert result["share"] == pytest.approx(1 / 3)

"""Dtype normalization, stratum assignment, and the flank structural-missingness partition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intruder.modeling.unsupervised.isolation_forest.columns import FLANK_FEATURES, resolve_blocks
from intruder.modeling.unsupervised.isolation_forest.features import (
    assert_no_missing,
    build_features,
    compute_vaf,
    motif_strata,
    normalize_depth_total,
    normalize_rep_units,
)
from intruder.modeling.unsupervised.isolation_forest.flanks import compute_flank_features

# --------------------------------------------------------------------------- #
# dtype quirks
# --------------------------------------------------------------------------- #

def test_normalize_depth_total_sums_the_comma_form():
    assert normalize_depth_total(pd.Series(["10,12", "5,5"])).tolist() == [22, 10]


def test_normalize_depth_total_accepts_a_plain_int():
    assert normalize_depth_total(pd.Series(["33", "52"])).tolist() == [33, 52]


def test_normalize_depth_total_handles_a_mix_of_both_forms():
    assert normalize_depth_total(pd.Series(["10,12", "33"])).tolist() == [22, 33]


def test_compute_vaf_splits_the_comma_form():
    assert compute_vaf(pd.Series(["10,10", "3,1"])).tolist() == [0.5, 0.75]


def test_compute_vaf_is_nan_once_depth_is_collapsed():
    result = compute_vaf(pd.Series(["33", "52"]))
    assert result.isna().all()


def test_compute_vaf_handles_a_mix_of_both_forms():
    result = compute_vaf(pd.Series(["10,10", "33"]))
    assert result.iloc[0] == 0.5
    assert pd.isna(result.iloc[1])


def test_normalize_rep_units_accepts_int_or_float_strings():
    result = normalize_rep_units(pd.Series(["4", "4.0", 5]))
    assert result.tolist() == [4.0, 4.0, 5.0]


def test_insert_size_bracket_form_is_handled_via_parse_sizes(make_candidates):
    frame = make_candidates(n=10)
    frame["insert_size"] = frame["insert_size"].astype(object)
    frame.loc[0, "insert_size"] = f"[{int(frame.loc[0, 'insert_size'])}]"
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks)
    assert np.isfinite(table.features.loc[0, "repeat_coverage"])


# --------------------------------------------------------------------------- #
# stratum assignment
# --------------------------------------------------------------------------- #

def test_motif_strata_boundary_is_inclusive_on_the_lower_side():
    lengths = pd.Series([5, 6, 7])
    strata = motif_strata(lengths, bounds=(6,))
    assert strata.tolist() == ["len<=6", "len<=6", "len>6"]


def test_motif_strata_supports_multiple_bounds():
    lengths = pd.Series([1, 1, 6, 7, 50])
    strata = motif_strata(lengths, bounds=(1, 6))
    assert strata.tolist() == ["len<=1", "len<=1", "1<len<=6", "len>6", "len>6"]


# --------------------------------------------------------------------------- #
# NaN fails loudly, no imputation
# --------------------------------------------------------------------------- #

def test_assert_no_missing_names_every_offending_column():
    features = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0], "c": [np.nan, np.nan]})
    with pytest.raises(ValueError, match=r"a.*c|c.*a"):
        assert_no_missing(features, ["a", "b", "c"])


def test_assert_no_missing_passes_when_clean():
    features = pd.DataFrame({"a": [1.0, 2.0]})
    assert_no_missing(features, ["a"])  # no raise


def test_a_blank_purity_cell_is_caught_by_the_missing_check(make_candidates):
    frame = make_candidates(n=10)
    frame["purity"] = frame["purity"].astype(object)
    frame.loc[0, "purity"] = ""
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks)
    with pytest.raises(ValueError, match="purity"):
        assert_no_missing(table.features, ["purity"])


# --------------------------------------------------------------------------- #
# structural missingness: unflanked rows, no sentinel
# --------------------------------------------------------------------------- #

def test_unflanked_rows_route_to_the_intrinsic_only_partition(make_candidates, write_vcf):
    frame = make_candidates(n=20, flanked=False)
    sequences = {
        row["SVID"]: row["motif"] * int(row["rep_units"])
        for _, row in frame.iterrows()
    }
    vcf = write_vcf(frame, sequences)
    blocks = resolve_blocks(["intrinsic", "flank"])
    flanks = compute_flank_features(frame, vcf)
    table = build_features(frame, blocks, flanks=flanks)

    # repeat_coverage == 1.0 for every unflanked row here
    assert (table.features["repeat_coverage"] >= 1.0).all()
    assert (~table.has_flank).all()
    assert table.stratum.str.endswith("|unflanked").all()


def test_no_sentinel_value_appears_in_the_flank_columns_for_unflanked_rows(
        make_candidates, write_vcf):
    frame = make_candidates(n=20, flanked=False)
    sequences = {
        row["SVID"]: row["motif"] * int(row["rep_units"])
        for _, row in frame.iterrows()
    }
    vcf = write_vcf(frame, sequences)
    blocks = resolve_blocks(["intrinsic", "flank"])
    flanks = compute_flank_features(frame, vcf)
    table = build_features(frame, blocks, flanks=flanks)

    # NaN (structural missingness), never a numeric sentinel like -1.0.
    for column in FLANK_FEATURES:
        assert table.features[column].isna().all()


def test_flanked_rows_get_real_flank_values(make_candidates, write_vcf):
    frame = make_candidates(n=20, flanked=True)
    sequences = {}
    for _, row in frame.iterrows():
        five = "T" * row["rep_start"]
        three = "T" * (row["insert_size"] - row["rep_end"])
        repeat = row["motif"] * int(row["rep_units"])
        sequences[row["SVID"]] = five + repeat + three
    vcf = write_vcf(frame, sequences)
    blocks = resolve_blocks(["intrinsic", "flank"])
    flanks = compute_flank_features(frame, vcf)
    table = build_features(frame, blocks, flanks=flanks)

    assert table.has_flank.all()
    assert table.stratum.str.endswith("|flanked").all()
    for column in FLANK_FEATURES:
        assert table.features[column].notna().all()


# --------------------------------------------------------------------------- #
# vaf: structural missingness, included by default, never imputed
# --------------------------------------------------------------------------- #

def test_vaf_is_included_by_default_when_depth_has_the_split(make_candidates):
    frame = make_candidates(n=20)  # depth is "DV,DR" by construction
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks)
    assert "vaf" in table.features.columns
    assert table.features["vaf"].notna().all()
    assert table.has_vaf.all()


def test_drop_vaf_excludes_it_even_when_available(make_candidates):
    frame = make_candidates(n=20)
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks, drop_vaf=True)
    assert "vaf" not in table.features.columns
    assert table.has_vaf is None


def test_vaf_is_nan_not_imputed_once_depth_is_collapsed(make_candidates):
    frame = make_candidates(n=20)
    frame["depth"] = frame["depth"].str.split(",").apply(lambda p: str(sum(int(x) for x in p)))
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks)
    assert table.features["vaf"].isna().all()
    assert (~table.has_vaf).all()


def test_uniform_vaf_availability_does_not_fragment_the_stratum(make_candidates):
    # every row here has the comma split -- has_vaf is uniformly True, so no
    # "|vaf" suffix should appear (see build_features's drop_vaf docstring).
    frame = make_candidates(n=20)
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks, strata_bounds=(6,))
    assert not table.stratum.str.contains(r"\|vaf|\|novaf").any()


def test_mixed_vaf_availability_fragments_the_stratum(make_candidates):
    frame = make_candidates(n=20)
    frame.loc[:9, "depth"] = frame.loc[:9, "depth"].str.split(",").apply(
        lambda p: str(sum(int(x) for x in p)))
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks, strata_bounds=(6,))
    assert table.stratum.str.endswith("|novaf").any()
    assert table.stratum.str.endswith("|vaf").any()


# --------------------------------------------------------------------------- #
# cohort block
# --------------------------------------------------------------------------- #

def test_cohort_n_carriers_counts_distinct_samples_per_svid():
    frame = pd.DataFrame({
        "SVID": ["A", "A", "A", "B"],
        "sample": ["s1", "s2", "s1", "s1"],
        "motif": ["AC", "AC", "AC", "AG"],
        "rep_length": [10, 12, 10, 8],
        "motif_length": [2, 2, 2, 2],
        "purity": [0.9, 0.9, 0.9, 0.9],
        "insert_size": [10, 12, 10, 8],
        "depth": [20, 20, 20, 20],
    })
    blocks = resolve_blocks(["cohort"])
    table = build_features(frame, blocks)
    assert table.features["n_carriers"].tolist() == [2.0, 2.0, 2.0, 1.0]


def test_cohort_n_distinct_canonical_motifs_counts_unique_motifs_per_svid():
    frame = pd.DataFrame({
        "SVID": ["A", "A", "A"],
        "sample": ["s1", "s2", "s3"],
        # CA is a rotation of AC -> same canonical motif; AG is different.
        "motif": ["AC", "CA", "AG"],
        "rep_length": [10, 10, 10],
        "motif_length": [2, 2, 2],
        "purity": [0.9, 0.9, 0.9],
        "insert_size": [10, 10, 10],
        "depth": [20, 20, 20],
    })
    blocks = resolve_blocks(["cohort"])
    table = build_features(frame, blocks)
    assert table.features["n_distinct_canonical_motifs"].tolist() == [2.0, 2.0, 2.0]


def test_cohort_rep_length_cv_is_zero_when_all_carriers_agree():
    frame = pd.DataFrame({
        "SVID": ["A", "A"],
        "sample": ["s1", "s2"],
        "motif": ["AC", "AC"],
        "rep_length": [10, 10],
        "motif_length": [2, 2],
        "purity": [0.9, 0.9],
        "insert_size": [10, 10],
        "depth": [20, 20],
    })
    blocks = resolve_blocks(["cohort"])
    table = build_features(frame, blocks)
    assert table.features["rep_length_cv"].tolist() == [0.0, 0.0]


def test_cohort_block_needs_svid_and_sample_columns():
    frame = pd.DataFrame({"motif": ["AC"], "rep_length": [10], "motif_length": [2],
                          "purity": [0.9], "insert_size": [10], "depth": [20]})
    blocks = resolve_blocks(["cohort"])
    with pytest.raises(KeyError):
        build_features(frame, blocks)

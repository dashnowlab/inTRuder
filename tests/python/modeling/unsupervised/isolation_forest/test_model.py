"""Determinism, the partition-row floor fallback, and the constant-column guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from intruder.modeling.unsupervised.isolation_forest import model as model_mod
from intruder.modeling.unsupervised.isolation_forest.columns import resolve_blocks
from intruder.modeling.unsupervised.isolation_forest.features import build_features

COLUMNS = ["a", "b", "c"]


def _features(n=400, seed=0, constant=None):
    rng = np.random.RandomState(seed)
    data = {c: rng.rand(n) for c in COLUMNS}
    if constant is not None:
        data[constant] = np.full(n, 0.5)
    return pd.DataFrame(data)


def test_fit_is_deterministic_with_the_same_random_state():
    features = _features(n=400)
    stratum = pd.Series(["s"] * len(features))
    m1 = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                       strata_bounds=(6,), n_jobs=1, audit=None)
    m2 = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                       strata_bounds=(6,), n_jobs=1, audit=None)
    s1 = model_mod.score(m1, features, stratum)
    s2 = model_mod.score(m2, features, stratum)
    assert np.allclose(s1["if_score"], s2["if_score"])


def test_n_jobs_does_not_change_scores():
    features = _features(n=400)
    stratum = pd.Series(["s"] * len(features))
    m1 = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                       strata_bounds=(6,), n_jobs=1, audit=None)
    m2 = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                       strata_bounds=(6,), n_jobs=-1, audit=None)
    s1 = model_mod.score(m1, features, stratum)
    s2 = model_mod.score(m2, features, stratum)
    assert np.allclose(s1["if_score"], s2["if_score"])


def test_a_thin_partition_falls_back_to_the_pooled_fit_and_warns(capsys):
    features = _features(n=400)
    # Two strata: one well above the 256-row floor, one far below it.
    stratum = pd.Series(["big"] * 300 + ["small"] * 100)
    fitted = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                           strata_bounds=(6,), n_jobs=1, audit=None)
    captured = capsys.readouterr()
    assert "small" in captured.err
    assert "pooled" in captured.err.lower()
    assert fitted.partitions["small"].fallback is True
    assert fitted.partitions["big"].fallback is False
    # The thin partition is scored under the same fitted estimator as the pool.
    assert fitted.partitions["small"].model is fitted.partitions["small"].model


def test_a_constant_feature_is_dropped_and_named_in_the_warning(capsys):
    features = _features(n=400, constant="b")
    stratum = pd.Series(["s"] * len(features))
    fitted = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                           strata_bounds=(6,), n_jobs=1, audit=None)
    captured = capsys.readouterr()
    assert "'b'" in captured.err
    assert "b" not in fitted.partitions["s"].columns
    assert "a" in fitted.partitions["s"].columns
    assert "c" in fitted.partitions["s"].columns
    assert fitted.partitions["s"].dropped_constant == ("b",)


def test_fit_table_includes_vaf_when_uniformly_available(make_candidates):
    frame = make_candidates(n=400)  # depth is "DV,DR" throughout
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks, strata_bounds=(6,))
    fitted = model_mod.fit_table(table, blocks, strata_bounds=(6,), n_jobs=1, audit=None)
    for partition in fitted.partitions.values():
        assert "vaf" in partition.columns


def test_fit_table_splits_on_a_genuinely_mixed_vaf_partition(make_candidates):
    frame = make_candidates(n=400)
    collapsed = frame.index[:200]
    frame.loc[collapsed, "depth"] = frame.loc[collapsed, "depth"].str.split(",").apply(
        lambda p: str(sum(int(x) for x in p)))
    blocks = resolve_blocks(["intrinsic"])
    table = build_features(frame, blocks, strata_bounds=(6,))
    fitted = model_mod.fit_table(table, blocks, strata_bounds=(6,), n_jobs=1, audit=None)

    scores = model_mod.score(fitted, table.features, table.stratum)
    assert scores["if_score"].notna().all()
    for stratum, partition in fitted.partitions.items():
        assert ("vaf" in partition.columns) == stratum.endswith("|vaf")


def test_save_and_load_round_trip(tmp_path):
    features = _features(n=400)
    stratum = pd.Series(["s"] * len(features))
    fitted = model_mod.fit(features, stratum, COLUMNS, blocks=("intrinsic",),
                           strata_bounds=(6,), n_jobs=1, audit=None)
    path = tmp_path / "model.joblib"
    model_mod.save(fitted, path)
    loaded = model_mod.load(path)

    s1 = model_mod.score(fitted, features, stratum)
    s2 = model_mod.score(loaded, features, stratum)
    assert np.allclose(s1["if_score"], s2["if_score"])
    assert loaded.blocks == fitted.blocks
    assert loaded.partitions["s"].columns == fitted.partitions["s"].columns

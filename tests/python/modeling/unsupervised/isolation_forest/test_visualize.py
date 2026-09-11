"""explain_by_stratum/plot_waterfall wire the right (partition, X, shap_frame) together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from sklearn.ensemble import IsolationForest

from intruder.modeling.unsupervised.isolation_forest.explain import explain
from intruder.modeling.unsupervised.isolation_forest.model import (
    MODEL_PARAMS,
    IsolationForestModel,
    PartitionFit,
)
from intruder.modeling.unsupervised.isolation_forest.paths import explain_paths
from intruder.modeling.unsupervised.isolation_forest.visualize import (
    explain_by_stratum,
    plot_isolation_tree,
    plot_waterfall,
    print_motif_length_dominance,
)

COLUMNS = ["a", "b", "c", "d"]


@pytest.fixture
def model_and_table():
    rng = np.random.RandomState(0)
    features = pd.DataFrame(rng.rand(300, 4), columns=COLUMNS)
    stratum = pd.Series(["only"] * 300, index=features.index)

    fitted = IsolationForest(n_jobs=1, **MODEL_PARAMS)
    fitted.fit(features.to_numpy())
    partition = PartitionFit("only", tuple(COLUMNS), (), fitted, False, 300)
    model = IsolationForestModel(partitions={"only": partition}, blocks=("intrinsic",),
                        strata_bounds=(6,), n_jobs=1)
    return model, features, stratum


def test_explain_by_stratum_matches_a_direct_explain_call(model_and_table):
    model, features, stratum = model_and_table
    result = explain_by_stratum(model, features, stratum, n_jobs=1, chunk_size=None)

    assert set(result) == {"only"}
    s = result["only"]
    assert s.partition is model.partitions["only"]
    assert list(s.X.index) == list(features.index)

    expected = explain(s.partition.model, s.X)
    assert np.allclose(s.shap_frame.to_numpy(), expected.to_numpy(), atol=1e-6)


def test_plot_waterfall_resolves_the_right_row(model_and_table):
    pytest.importorskip("matplotlib")
    model, features, stratum = model_and_table
    shap_by_stratum = explain_by_stratum(model, features, stratum, n_jobs=1, chunk_size=None)

    frame = pd.DataFrame({"if_stratum": stratum})
    row_id = features.index[5]

    result = plot_waterfall(shap_by_stratum, frame, row_id)

    assert result.row_id == row_id
    assert result.label == "only"
    s = shap_by_stratum["only"]
    assert result.row_pos == s.X.index.get_loc(row_id)
    assert np.allclose(result.shap_frame.to_numpy()[result.row_pos],
                       s.shap_frame.to_numpy()[result.row_pos])


def test_print_motif_length_dominance_skips_strata_without_the_feature(capsys, model_and_table):
    model, features, stratum = model_and_table
    shap_by_stratum = explain_by_stratum(model, features, stratum, n_jobs=1, chunk_size=None)

    print_motif_length_dominance(shap_by_stratum, feature="log_motif_length")
    assert capsys.readouterr().out == ""

    print_motif_length_dominance(shap_by_stratum, feature="a")
    out = capsys.readouterr().out
    assert "only" in out
    assert "share=" in out


def test_plot_isolation_tree_positions_every_node_it_draws(model_and_table):
    pytest.importorskip("matplotlib")
    model, features, _ = model_and_table
    fitted = model.partitions["only"].model
    traces = explain_paths(fitted, features.iloc[0], COLUMNS)
    trace = max(traces, key=lambda t: len(t.steps))

    ax = plot_isolation_tree(fitted, trace, COLUMNS)
    assert len(ax.texts) >= len(trace.steps) + 1

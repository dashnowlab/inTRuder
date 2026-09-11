"""Mean `h_tree` from an isolation-path trace reproduces `score_samples`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from intruder.modeling.unsupervised.isolation_forest.model import MODEL_PARAMS
from intruder.modeling.unsupervised.isolation_forest.paths import explain_paths, mean_path_length

COLUMNS = ["a", "b", "c", "d"]


@pytest.fixture
def fitted():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(300, 4), columns=COLUMNS)
    model = IsolationForest(n_jobs=1, **MODEL_PARAMS)
    model.fit(X.to_numpy())
    return model, X


def test_mean_path_length_reproduces_score_samples_on_a_toy_set(fitted):
    model, X = fitted
    c = float(_average_path_length(np.array([model.max_samples_]))[0])
    actual = model.score_samples(X.to_numpy())

    for i in range(10):
        traces = explain_paths(model, X.iloc[i], COLUMNS)
        assert len(traces) == model.n_estimators
        h = mean_path_length(traces)
        reconstructed = -(2.0 ** (-h / c))
        assert reconstructed == pytest.approx(actual[i], abs=1e-6)


def test_every_trace_ends_at_a_leaf_with_at_least_one_sample(fitted):
    model, X = fitted
    traces = explain_paths(model, X.iloc[0], COLUMNS)
    for trace in traces:
        assert trace.leaf_samples >= 1
        assert trace.h_tree >= len(trace.steps)


def test_split_steps_name_real_columns(fitted):
    model, X = fitted
    traces = explain_paths(model, X.iloc[0], COLUMNS)
    for trace in traces:
        for step in trace.steps:
            assert step.feature in COLUMNS
            assert step.direction in ("left", "right")

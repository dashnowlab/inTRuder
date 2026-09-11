"""The SHAP additivity identity is the correctness guard for the sign inversion.

``score_samples(X) == -(2 ** (-(raw_shap.sum(1) + base) / c))``, where
``raw_shap`` is what shap returns *before* this package negates it, and
``c = _average_path_length([max_samples_])``. Everything else here (the
negated sign, the output shape) is checked against that same identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")

from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from intruder.modeling.unsupervised.isolation_forest.explain import base_value, explain
from intruder.modeling.unsupervised.isolation_forest.model import MODEL_PARAMS

COLUMNS = ["a", "b", "c", "d"]


@pytest.fixture
def fitted():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(300, 4), columns=COLUMNS)
    model = IsolationForest(n_jobs=1, **MODEL_PARAMS)
    model.fit(X.to_numpy())
    return model, X


def test_shap_anom_is_negated_relative_to_the_raw_isotree_output(fitted):
    model, X = fitted
    negated = explain(model, X)
    raw = shap.TreeExplainer(model).shap_values(X.to_numpy(), check_additivity=True)
    assert np.allclose(negated.to_numpy(), -raw, atol=1e-6)


def test_additivity_identity_reproduces_score_samples(fitted):
    model, X = fitted
    negated = explain(model, X)
    base = base_value(model)
    c = float(_average_path_length(np.array([model.max_samples_]))[0])

    # Undo the negation to get back shap's own h(x) decomposition.
    raw_sum = -negated.to_numpy().sum(axis=1)
    h = raw_sum + base
    reconstructed_score = -(2.0 ** (-h / c))

    actual_score = model.score_samples(X.to_numpy())
    assert np.allclose(reconstructed_score, actual_score, atol=1e-7)


def test_shap_output_shape_is_two_dimensional(fitted):
    model, X = fitted
    negated = explain(model, X)
    assert negated.to_numpy().ndim == 2
    assert negated.shape == (len(X), len(COLUMNS))


def test_shap_columns_are_named_shap_anom_prefixed(fitted):
    model, X = fitted
    negated = explain(model, X)
    assert list(negated.columns) == [f"shap_anom_{c}" for c in COLUMNS]


def test_shap_dtype_is_float32(fitted):
    model, X = fitted
    negated = explain(model, X)
    assert negated.to_numpy().dtype == np.float32


def test_chunked_shap_matches_unchunked(fitted):
    model, X = fitted
    unchunked = explain(model, X, n_jobs=1)
    chunked = explain(model, X, n_jobs=2, chunk_size=50)
    assert np.allclose(unchunked.to_numpy(), chunked.to_numpy(), atol=1e-6)

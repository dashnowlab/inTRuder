"""SHAP attribution for an `IsolationForest`, via `TreeExplainer`.

shap dispatches `IsolationForest` to a dedicated `IsoTree` explainer (shap PR
#784, shipped 0.32) whose values explain the *expected isolation path length*
`h(x)`, not `score_samples` or `decision_function`: ``shap.sum(1) + base_value
== h(x)``. Path length runs the *opposite* way from anomaly intuition --
shorter path = isolated faster = more anomalous -- so a raw positive SHAP
value means "pushed this call toward looking NORMAL", which reads backwards to
a biologist. Every value here is therefore negated before being named
``shap_anom_<feature>``: positive = drove this call to look anomalous.

No background data is passed to `TreeExplainer`, so `feature_perturbation`
falls back to its default, `tree_path_dependent` -- the mode shap's own
additivity test covers, and the right question here ("which measured property
isolated this call", not "how does this call compare to some reference
distribution"). `interventional` is untested upstream for `IsolationForest`
and would evaluate off-manifold points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest

from .columns import shap_column


def _shap_chunk(model: IsolationForest, X: np.ndarray, *,
                check_additivity: bool) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X, check_additivity=check_additivity)


def explain(model: IsolationForest, X: pd.DataFrame, *, n_jobs: int = 1,
           chunk_size: int | None = None, check_additivity: bool = True) -> pd.DataFrame:
    """Negated SHAP attribution matrix, one ``shap_anom_<feature>`` column per feature.

    shap's TreeSHAP C extension is single-threaded, so there is no internal
    threading to compete with -- but row-chunking with `joblib` copies the
    model and data to every worker, and that overhead can exceed the gain on
    a small chunk count. ``n_jobs=1`` (the default) never chunks; passing
    ``n_jobs`` together with ``chunk_size`` splits ``X`` into row chunks and
    reassembles them, which is exact because TreeSHAP is per-row independent.
    """
    columns = list(X.columns)
    arr = X.to_numpy(dtype="float64")
    if n_jobs == 1 or not chunk_size or len(arr) <= chunk_size:
        raw = _shap_chunk(model, arr, check_additivity=check_additivity)
    else:
        chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_shap_chunk)(model, chunk, check_additivity=check_additivity)
            for chunk in chunks)
        raw = np.concatenate(results, axis=0)
    negated = (-np.asarray(raw)).astype("float32")
    return pd.DataFrame(negated, index=X.index,
                        columns=[shap_column(c) for c in columns])


def base_value(model: IsolationForest) -> float:
    """``TreeExplainer(model).expected_value`` -- the additivity identity's constant term."""
    value = shap.TreeExplainer(model).expected_value
    return float(np.asarray(value).reshape(-1)[0])

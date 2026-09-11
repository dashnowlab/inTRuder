"""Per-candidate isolation-path drill-down: which splits isolated this row, and how fast.

Complements ``explain.py``'s SHAP matrix -- SHAP attributes a *score* to each
feature, this shows the actual sequence of splits one row took down one tree,
for the notebook's per-locus waterfall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length


@dataclass(frozen=True)
class SplitStep:
    """One split a row passed through on its way to a leaf."""

    feature: str
    threshold: float
    value: float
    direction: str   # "left" (value <= threshold) or "right" (value > threshold)


@dataclass(frozen=True)
class PathTrace:
    """One tree's traversal of one row, and that tree's contribution to `h(x)`."""

    tree_index: int
    steps: tuple[SplitStep, ...]
    leaf_samples: int
    h_tree: float


def _trace_one(tree, feature_index: np.ndarray, x: np.ndarray,
               columns: Sequence[str]) -> tuple[tuple[SplitStep, ...], int]:
    t = tree.tree_
    node = 0
    steps: list[SplitStep] = []
    while t.feature[node] >= 0:                      # -2 (TREE_UNDEFINED) marks a leaf
        local_feature = t.feature[node]
        original_feature = feature_index[local_feature]
        threshold = float(t.threshold[node])
        value = float(x[original_feature])
        if value <= threshold:
            steps.append(SplitStep(columns[original_feature], threshold, value, "left"))
            node = t.children_left[node]
        else:
            steps.append(SplitStep(columns[original_feature], threshold, value, "right"))
            node = t.children_right[node]
    return tuple(steps), int(t.n_node_samples[node])


def explain_paths(model: IsolationForest, x: pd.Series,
                  columns: Sequence[str]) -> tuple[PathTrace, ...]:
    """Every tree's traversal of one row ``x`` (indexed/aligned to ``columns``).

    ``h_tree = len(steps) + c(leaf_samples)``: the usual isolation-forest path
    length, with ``_average_path_length``'s correction for the unbuilt subtree
    below an early-stopped leaf (a leaf with more than one sample left in it
    when the tree's depth cap was hit). Averaging ``h_tree`` over every tree
    and plugging the mean into ``-2 ** (-mean/c(max_samples_))`` reproduces
    ``score_samples`` for this row -- the identity :func:`mean_path_length`'s
    caller checks in ``test_paths.py``.
    """
    columns = list(columns)
    arr = x.reindex(columns).to_numpy(dtype="float64")
    traces = []
    for i, (estimator, feature_index) in enumerate(
            zip(model.estimators_, model.estimators_features_)):
        steps, leaf_samples = _trace_one(estimator, feature_index, arr, columns)
        correction = (float(_average_path_length(np.array([leaf_samples]))[0])
                     if leaf_samples > 1 else 0.0)
        traces.append(PathTrace(i, steps, leaf_samples, len(steps) + correction))
    return tuple(traces)


def mean_path_length(traces: Sequence[PathTrace]) -> float:
    """``E[h(x)]`` over every tree's trace -- what ``score_samples`` is derived from."""
    return float(np.mean([t.h_tree for t in traces]))

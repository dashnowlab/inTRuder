"""Notebook plotting helpers: per-stratum SHAP orchestration, beeswarms, waterfalls,
and the isolating-tree diagram.

Kept separate from ``explain.py`` (the SHAP matrix for one stratum) and
``paths.py`` (one row's split trace) so those stay import-light for non-notebook
callers (the CLI, ``annotate``) that never need matplotlib or shap's plotting
module. Every function here is a thin wrapper the calibration notebook used to
inline -- no new analysis logic, just one place to define it so it isn't
duplicated across notebooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Hashable

import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from .explain import explain
from .model import IsolationForestModel, PartitionFit
from .paths import PathTrace


@dataclass
class StratumExplanation:
    """One stratum's fitted partition, the rows it was explained on, and their SHAP matrix."""

    partition: PartitionFit
    X: pd.DataFrame
    shap_frame: pd.DataFrame


def explain_by_stratum(model: IsolationForestModel, features: pd.DataFrame, stratum: pd.Series, *,
                       n_jobs: int | None = None, chunk_size: int | None = 2_000
                       ) -> dict[str, StratumExplanation]:
    """SHAP-explain every row, one :class:`~sklearn.ensemble.IsolationForest` call per stratum.

    TreeSHAP's C extension is single-threaded, so explaining a full population
    (hundreds of thousands of rows) is the slow step in the calibration
    notebook. ``n_jobs``/``chunk_size`` forward to :func:`explain`'s
    row-chunked joblib parallelism, which is exact -- TreeSHAP is per-row
    independent, so splitting and reassembling changes nothing about the
    result. ``n_jobs=None`` (the default) uses every core.
    """
    resolved_n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 1)

    out: dict[str, StratumExplanation] = {}
    for label in pd.unique(stratum):
        idx = stratum.index[stratum == label]
        partition = model.partitions.get(label) or next(
            p for p in model.partitions.values() if p.fallback)
        X = features.loc[idx, list(partition.columns)]
        shap_frame = explain(partition.model, X, n_jobs=resolved_n_jobs, chunk_size=chunk_size)
        out[label] = StratumExplanation(partition, X, shap_frame)
    return out


def plot_beeswarms(shap_by_stratum: dict[str, StratumExplanation]) -> None:
    """One SHAP beeswarm per stratum.

    Positive = drove that call toward looking anomalous (already negated
    relative to shap's raw ``IsoTree`` output -- see ``explain.py``).
    """
    import matplotlib.pyplot as plt

    for label, s in shap_by_stratum.items():
        explanation = shap.Explanation(
            values=s.shap_frame.to_numpy(),
            data=s.X.to_numpy(),
            feature_names=list(s.partition.columns),
        )
        plt.figure()
        plt.title(f"stratum: {label}")
        shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        plt.show()


@dataclass
class RowExplanation:
    """One row's stratum, features, and SHAP attribution.

    What the waterfall plot resolves and what its follow-on drill-down
    (:func:`~intruder.modeling.unsupervised.isolation_forest.paths.explain_paths`,
    :func:`plot_isolation_tree`) needs to keep going on the same row.
    """

    row_id: Hashable
    label: str
    partition: PartitionFit
    X: pd.DataFrame
    shap_frame: pd.DataFrame
    row_pos: int


def plot_waterfall(shap_by_stratum: dict[str, StratumExplanation], frame: pd.DataFrame,
                   row_id: Hashable, *, stratum_column: str = "if_stratum") -> RowExplanation:
    """SHAP waterfall for one row, returning the context needed to drill in further."""
    label = frame.loc[row_id, stratum_column]
    s = shap_by_stratum[label]
    row_pos = s.X.index.get_loc(row_id)

    explanation = shap.Explanation(
        values=s.shap_frame.to_numpy()[row_pos],
        base_values=0.0,   # the additivity constant is in h(x) units, not if_score units
        data=s.X.to_numpy()[row_pos],
        feature_names=list(s.partition.columns),
    )
    shap.plots.waterfall(explanation, show=True)
    return RowExplanation(row_id, label, s.partition, s.X, s.shap_frame, row_pos)


def print_motif_length_dominance(shap_by_stratum: dict[str, StratumExplanation], *,
                                 feature: str = "log_motif_length",
                                 threshold: float = 0.5) -> None:
    """Print :func:`~.calibration.motif_length_dominance` for every stratum that has it."""
    from .calibration import motif_length_dominance
    from .columns import shap_column

    column = shap_column(feature)
    for label, s in shap_by_stratum.items():
        if column not in s.shap_frame.columns:
            continue
        result = motif_length_dominance(s.shap_frame, feature=feature, threshold=threshold)
        print(f"{label:<20} share={result['share']:.3f}  dominant={result['dominant']}")


def plot_isolation_tree(model: IsolationForest, trace: PathTrace, columns, *, ax=None):
    """Draw one tree's branching structure, with ``trace``'s path highlighted in red.

    Sibling branches not taken are drawn in gray so the real shape of the tree is
    visible -- a single split in isolation (the common case for the *shortest*
    tree in an ensemble) reads as "one deciding factor" when it's really one
    tree's contribution to an average over ``model.n_estimators`` of them. Off-path
    subtrees are truncated at the path's own depth (drawn as ``"..."``) so the
    figure stays a manageable size even for a tree near the ``max_depth=8`` cap
    that ``model.py`` pins ``max_samples`` to.

    This is the graphical counterpart to ``explain_paths``'s numeric trace, not
    a replacement -- pick ``trace`` from that function's output (by
    ``tree_index``, or by whatever selection criterion the caller wants; the
    *shortest* trace makes a good illustration of ``h(x)``, a *typical*-length
    one usually makes a better illustration of the branching structure itself).
    """
    import matplotlib.pyplot as plt

    columns = list(columns)
    estimator = model.estimators_[trace.tree_index]
    feature_index = model.estimators_features_[trace.tree_index]
    t = estimator.tree_
    path_depth = len(trace.steps)

    path_nodes = [0]
    node = 0
    for step in trace.steps:
        node = t.children_left[node] if step.direction == "left" else t.children_right[node]
        path_nodes.append(node)
    path_nodes_set = set(path_nodes)
    path_values = {path_nodes[i]: trace.steps[i].value for i in range(len(trace.steps))}

    positions: dict[int, tuple[float, int]] = {}
    next_x = [0]

    def layout(n: int, depth: int) -> float:
        if depth > path_depth or t.feature[n] < 0:
            positions[n] = (float(next_x[0]), depth)
            next_x[0] += 1
            return positions[n][0]
        lx = layout(t.children_left[n], depth + 1)
        rx = layout(t.children_right[n], depth + 1)
        positions[n] = ((lx + rx) / 2, depth)
        return positions[n][0]

    layout(0, 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2.2 * (path_depth + 1)))

    for n, (xp, depth) in positions.items():
        if t.feature[n] >= 0 and depth <= path_depth:
            for child in (t.children_left[n], t.children_right[n]):
                cx, cy = positions[child]
                on_edge = n in path_nodes_set and child in path_nodes_set
                ax.plot([xp, cx], [-depth, -cy],
                       color="crimson" if on_edge else "lightgray",
                       lw=2.5 if on_edge else 1, zorder=1)

    for n, (xp, depth) in positions.items():
        on_path = n in path_nodes_set
        if t.feature[n] >= 0 and depth <= path_depth:
            original_feature = feature_index[t.feature[n]]
            text = f"{columns[original_feature]}\n<= {t.threshold[n]:.3g}"
            if on_path:
                text += f"\n(value={path_values[n]:.3g})"
            box = {"boxstyle": "round,pad=0.3",
                   "fc": "#fde0e0" if on_path else "white",
                   "ec": "crimson" if on_path else "lightgray", "lw": 2 if on_path else 1}
        else:
            text = f"leaf\nn={t.n_node_samples[n]}" if t.feature[n] < 0 else "..."
            box = {"boxstyle": "round,pad=0.3",
                   "fc": "#e0e0fd" if on_path else "white",
                   "ec": "navy" if on_path else "lightgray", "lw": 2 if on_path else 1}
        ax.annotate(text, xy=(xp, -depth), ha="center", va="center", bbox=box,
                   fontsize=8 if on_path else 6.5,
                   color="black" if on_path else "gray", zorder=2)

    ax.axis("off")
    return ax

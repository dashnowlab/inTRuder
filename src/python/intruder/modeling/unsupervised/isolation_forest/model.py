"""Fit, persist, and score an :class:`~sklearn.ensemble.IsolationForest` per stratum.

One model per motif-length x flank-presence stratum (see ``features.py``), so
that a homopolymer stratum and a long-VNTR stratum are never forced through
the same split thresholds. Two knobs are pinned rather than left at sklearn's
defaults, both source-verified against ``sklearn.ensemble._iforest``:

* ``max_samples=256`` (never ``"auto"``/``1.0``): sklearn caps
  ``max_depth = ceil(log2(max_samples))`` at 8, which is what keeps TreeSHAP
  tractable in ``explain.py``. Raising ``max_samples`` would make SHAP
  infeasible, not just slower.
* ``contamination="auto"``: ``"auto"`` fixes ``offset_ = -0.5``, while a
  numeric value derives ``offset_`` from a training-data percentile -- with no
  labelled anomaly rate, a data-dependent offset is unjustifiable and would
  shift every score whenever the cohort changes size. The operating threshold
  is chosen downstream in ``calibration.py`` instead.

``n_jobs`` does not change the fitted trees (``BaseBagging`` pre-draws
per-estimator seeds), so it is safe to override at score time.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from tqdm import tqdm

from .columns import FEATURE_BLOCKS, FeatureBlock, feature_columns
from .features import FeatureTable, assert_no_missing, audit_correlations

#: Below this many rows a partition is too thin to fit its own model reliably
#: -- it is exactly `max_samples`, the point at which every tree would just be
#: resampling (with replacement disabled) the whole partition every time.
MIN_PARTITION_ROWS = 256

#: The stratum label a thin partition's rows are scored under instead.
POOLED_STRATUM = "pooled"

MODEL_PARAMS: dict[str, object] = {
    "n_estimators": 100,
    "max_samples": 256,
    "bootstrap": False,
    "max_features": 1.0,
    "random_state": 0,
    "contamination": "auto",
}


def average_path_length(model: IsolationForest) -> float:
    """``c(n)`` for this model's actual ``max_samples_`` -- the SHAP additivity constant."""
    return float(_average_path_length(np.array([model.max_samples_]))[0])


def _drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A constant column wastes every split and distorts SHAP attribution."""
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), constant


def _fit_one(X: pd.DataFrame, *, n_jobs: int) -> IsolationForest:
    model = IsolationForest(n_jobs=n_jobs, **MODEL_PARAMS)
    model.fit(X.to_numpy(dtype="float64"))
    return model


def _fit_partition_job(label: str, rows: pd.DataFrame, *, n_jobs: int) -> tuple[str, PartitionFit]:
    """One stratum's fit, as a unit of work dispatchable to a thread pool.

    Tree building releases the GIL (it's Cython/numpy underneath), so running
    several of these concurrently on threads gets real parallelism without
    the pickling cost a process pool would pay to ship ``rows`` around.
    """
    X, dropped = _drop_constant_columns(rows)
    if dropped:
        print(f"[isolation-forest] stratum {label!r}: dropping constant feature(s) "
              f"{dropped}", file=sys.stderr)
    assert_no_missing(X, X.columns)
    model = _fit_one(X, n_jobs=n_jobs)
    return label, PartitionFit(label, tuple(X.columns), tuple(dropped), model, False, len(X))


@dataclass
class PartitionFit:
    """One stratum's fitted model, and which columns it actually saw."""

    stratum: str
    columns: tuple[str, ...]
    dropped_constant: tuple[str, ...]
    model: IsolationForest
    fallback: bool
    n_rows: int


@dataclass
class IsolationForestModel:
    """Every stratum's :class:`PartitionFit`, plus what produced them."""

    partitions: dict[str, PartitionFit]
    blocks: tuple[str, ...]
    strata_bounds: tuple[int, ...]
    n_jobs: int
    mode: str = "intrinsic"

    def sidecar(self) -> dict:
        return {
            "blocks": list(self.blocks),
            "strata_bounds": list(self.strata_bounds),
            "n_jobs": self.n_jobs,
            "mode": self.mode,
            "model_params": dict(MODEL_PARAMS),
            "min_partition_rows": MIN_PARTITION_ROWS,
            "sklearn_version": sklearn.__version__,
            "partitions": {
                stratum: {
                    "columns": list(p.columns),
                    "dropped_constant": list(p.dropped_constant),
                    "fallback": p.fallback,
                    "n_rows": p.n_rows,
                }
                for stratum, p in self.partitions.items()
            },
        }


def fit(features: pd.DataFrame, stratum: pd.Series, columns, *,
        blocks: tuple[str, ...], strata_bounds: tuple[int, ...],
        n_jobs: int = 1, strata_jobs: int = 1, audit: str | None = "warn",
        mode: str = "intrinsic") -> IsolationForestModel:
    """Fit one :class:`~sklearn.ensemble.IsolationForest` per stratum in ``stratum``.

    ``audit`` runs the pairwise Spearman correlation check (see
    ``features.audit_correlations``); ``"warn"`` (default) prints and
    continues, ``"error"`` fails the fit, ``None`` skips it.

    ``n_jobs`` is sklearn's per-model tree parallelism; ``strata_jobs``
    (default 1, sequential) is separate and controls how many strata are fit
    *concurrently* -- each stratum's fit is independent of every other's, so
    with several strata this is usually the bigger lever, especially since
    ``max_samples=256`` keeps every individual fit's trees too small for
    ``n_jobs`` to have much to parallelize.
    """
    columns = list(columns)
    if audit:
        audit_correlations(features, columns, mode=audit)

    rows_by_label = {label: features.loc[stratum.index[stratum == label], columns]
                     for label in pd.unique(stratum)}
    thin_labels = [label for label, rows in rows_by_label.items()
                  if len(rows) < MIN_PARTITION_ROWS]
    full_labels = [label for label in rows_by_label if label not in thin_labels]

    partitions: dict[str, PartitionFit] = {}

    if thin_labels:
        X, dropped = _drop_constant_columns(features[columns])
        assert_no_missing(X, X.columns)
        model = _fit_one(X, n_jobs=n_jobs)
        pooled = PartitionFit(POOLED_STRATUM, tuple(X.columns), tuple(dropped),
                              model, True, len(X))
        for label in thin_labels:
            print(f"[isolation-forest] stratum {label!r}: {len(rows_by_label[label])} row(s) "
                  f"below the {MIN_PARTITION_ROWS}-row floor; falling back to the "
                  f"pooled fit", file=sys.stderr)
            partitions[label] = PartitionFit(label, pooled.columns, pooled.dropped_constant,
                                             pooled.model, True, len(rows_by_label[label]))

    if full_labels:
        # `return_as="generator"` (joblib >=1.3) yields each result as its
        # stratum finishes, rather than only after every stratum is done, so
        # the bar advances with real progress instead of jumping to 100% at
        # the end. `disable=None` mutes it automatically off a TTY (CI logs).
        jobs = Parallel(n_jobs=strata_jobs, backend="threading", return_as="generator")(
            delayed(_fit_partition_job)(label, rows_by_label[label], n_jobs=n_jobs)
            for label in full_labels)
        for label, pf in tqdm(jobs, total=len(full_labels), disable=None,
                              desc="[isolation-forest] fitting strata", unit="stratum"):
            partitions[label] = pf

    return IsolationForestModel(partitions=partitions, blocks=tuple(blocks),
                        strata_bounds=tuple(strata_bounds), n_jobs=n_jobs, mode=mode)


def save(model: IsolationForestModel, path: str | os.PathLike[str]) -> None:
    """Write the fitted estimators (joblib) and a human-readable sidecar (JSON).

    The sidecar records the feature column order per partition -- SHAP columns
    are positional, so this is what lets ``annotate`` and the notebook line
    them back up with names.
    """
    path = Path(path)
    joblib.dump({stratum: p.model for stratum, p in model.partitions.items()}, path)
    path.with_name(path.name + ".json").write_text(json.dumps(model.sidecar(), indent=2))


def load(path: str | os.PathLike[str]) -> IsolationForestModel:
    path = Path(path)
    estimators = joblib.load(path)
    sidecar = json.loads(path.with_name(path.name + ".json").read_text())
    partitions = {
        stratum: PartitionFit(stratum, tuple(meta["columns"]),
                              tuple(meta["dropped_constant"]),
                              estimators[stratum], meta["fallback"], meta["n_rows"])
        for stratum, meta in sidecar["partitions"].items()
    }
    return IsolationForestModel(partitions=partitions, blocks=tuple(sidecar["blocks"]),
                        strata_bounds=tuple(sidecar["strata_bounds"]),
                        n_jobs=sidecar["n_jobs"], mode=sidecar.get("mode", "intrinsic"))


def _partition_for(model: IsolationForestModel, label: str) -> PartitionFit:
    partition = model.partitions.get(label)
    if partition is not None:
        return partition
    fallback = next((p for p in model.partitions.values() if p.fallback), None)
    if fallback is None:
        raise KeyError(f"stratum {label!r} has no fitted model and no pooled "
                       f"fallback to score it under")
    return fallback


def score(model: IsolationForestModel, features: pd.DataFrame, stratum: pd.Series, *,
         n_jobs: int | None = None) -> pd.DataFrame:
    """``if_score``, ``if_h`` and ``if_percentile`` for every row.

    A stratum absent from the fitted model (seen at fit time in too few rows
    to clear the floor at score time only, or genuinely new) falls back to
    whichever partition was fit pooled, mirroring how it would have been
    scored during ``fit``.
    """
    out = pd.DataFrame(index=features.index,
                       columns=["if_score", "if_h", "if_percentile"], dtype="float64")
    for label in pd.unique(stratum):
        idx = stratum.index[stratum == label]
        partition = _partition_for(model, label)
        if n_jobs is not None:
            partition.model.n_jobs = n_jobs
        X = features.loc[idx, list(partition.columns)]
        assert_no_missing(X, X.columns)
        s = partition.model.score_samples(X.to_numpy(dtype="float64"))
        c = average_path_length(partition.model)
        h = -c * np.log2(-s)
        out.loc[idx, "if_score"] = s
        out.loc[idx, "if_h"] = h
        out.loc[idx, "if_percentile"] = pd.Series(s, index=idx).rank(pct=True) * 100
    return out


def _fit_optional_vaf(features: pd.DataFrame, stratum: pd.Series, columns: list[str],
                      has_vaf: pd.Series | None, *, fit_kwargs: dict) -> dict[str, PartitionFit]:
    """Fit ``columns`` (+ ``"vaf"`` where present), splitting only if ``has_vaf`` is mixed.

    A uniform ``has_vaf`` (every row here has it, or none do -- the common
    case, since no table in this repo mixes depth formats) needs exactly one
    :func:`fit` call with ``vaf`` included or not. A genuinely mixed
    partition needs two, one per column set, the same way the flank split
    above does.
    """
    if has_vaf is None:
        return fit(features, stratum, columns, **fit_kwargs).partitions
    if has_vaf.nunique() <= 1:
        cols = columns + ["vaf"] if has_vaf.any() else columns
        return fit(features, stratum, cols, **fit_kwargs).partitions

    partitions: dict[str, PartitionFit] = {}
    novaf_idx = has_vaf.index[~has_vaf]
    vaf_idx = has_vaf.index[has_vaf]
    if len(novaf_idx):
        partitions.update(fit(features.loc[novaf_idx], stratum.loc[novaf_idx],
                              columns, **fit_kwargs).partitions)
    if len(vaf_idx):
        partitions.update(fit(features.loc[vaf_idx], stratum.loc[vaf_idx],
                              columns + ["vaf"], **fit_kwargs).partitions)
    return partitions


def fit_table(table: FeatureTable, blocks: tuple[FeatureBlock, ...], *,
             strata_bounds: tuple[int, ...], n_jobs: int = 1, strata_jobs: int = 1,
             audit: str | None = "warn", mode: str = "intrinsic") -> IsolationForestModel:
    """Fit :class:`IsolationForestModel` from a :class:`~features.FeatureTable`.

    Handles two structural-missingness partitions, each independent of the
    motif-length stratum:

    * ``flank``: when it is one of ``blocks``, unflanked rows
      (``table.has_flank`` false) are fit on ``intrinsic`` alone and flanked
      rows on ``intrinsic + flank`` -- two calls to :func:`fit`, merged.
    * ``vaf``: computed by ``features.build_features`` whenever the input's
      ``depth`` column supports it (see ``columns.VAF_FEATURE``); split the
      same way, but only when ``table.has_vaf`` actually varies within the
      rows being fit -- see :func:`_fit_optional_vaf`.

    Neither partition needs the other's cooperation: this fits each flank
    branch (or the whole table, if ``flank`` was not requested) and then
    resolves the vaf split within it, reusing the motif-length stratification
    machinery rather than adding new ones.
    """
    block_names = tuple(b.name for b in blocks)
    all_columns = list(feature_columns(blocks))
    fit_kwargs = {"blocks": block_names, "strata_bounds": strata_bounds,
                  "n_jobs": n_jobs, "strata_jobs": strata_jobs, "audit": audit, "mode": mode}

    if "flank" not in block_names or table.has_flank is None:
        partitions = _fit_optional_vaf(table.features, table.stratum, all_columns,
                                       table.has_vaf, fit_kwargs=fit_kwargs)
        return IsolationForestModel(partitions=partitions, blocks=block_names,
                            strata_bounds=tuple(strata_bounds), n_jobs=n_jobs, mode=mode)

    flank_columns = set(FEATURE_BLOCKS["flank"].columns)
    unflanked_columns = [c for c in all_columns if c not in flank_columns]
    unflanked_idx = table.has_flank.index[~table.has_flank]
    flanked_idx = table.has_flank.index[table.has_flank]

    partitions: dict[str, PartitionFit] = {}
    if len(unflanked_idx):
        sub_vaf = table.has_vaf.loc[unflanked_idx] if table.has_vaf is not None else None
        partitions.update(_fit_optional_vaf(
            table.features.loc[unflanked_idx], table.stratum.loc[unflanked_idx],
            unflanked_columns, sub_vaf, fit_kwargs=fit_kwargs))
    if len(flanked_idx):
        sub_vaf = table.has_vaf.loc[flanked_idx] if table.has_vaf is not None else None
        partitions.update(_fit_optional_vaf(
            table.features.loc[flanked_idx], table.stratum.loc[flanked_idx],
            all_columns, sub_vaf, fit_kwargs=fit_kwargs))
    return IsolationForestModel(partitions=partitions, blocks=block_names,
                        strata_bounds=tuple(strata_bounds), n_jobs=n_jobs, mode=mode)

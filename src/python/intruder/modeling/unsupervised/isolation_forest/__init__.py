"""Isolation-forest anomaly scoring + SHAP explanation for novel-TR candidates.

A QC/confidence score that ranks and explains itself -- **not** a replacement
for the catalog-based novelty verdict in ``pipeline.novelty``. Fits an
``IsolationForest`` over intrinsic sequence/call features, emits a per-candidate
anomaly score plus a full per-feature SHAP attribution matrix, and validates
against STRchive disease loci as a known-real-TR sanity set.

    columns       feature-block registry, output column names
    features      load, normalize dtype quirks, build feature blocks, assign strata
    flanks        re-read ALT sequences from the VCF, compute flank features
    model         fit / persist / score per stratum
    explain       SHAP matrix via TreeExplainer
    paths         explain_paths() per-candidate split traces
    calibration   STRchive recall gate, KS diagnostic, stratification checks
    visualize     notebook plotting: per-stratum SHAP, beeswarms, waterfalls, tree diagram
    cli           the `isolation-forest` command line
"""

from .calibration import (
    CalibrationReport,
    calibrate,
    motif_length_dominance,
    stratum_boundary_shift,
)
from .columns import (
    COHORT_FEATURES,
    DEFAULT_BLOCKS,
    FEATURE_BLOCKS,
    FLANK_FEATURES,
    FLANKED_BLOCKS,
    INTRINSIC_FEATURES,
    OUTPUT_META_COLUMNS,
    UNFLANKED_BLOCKS,
    FeatureBlock,
    feature_columns,
    resolve_blocks,
    shap_column,
)
from .explain import base_value, explain
from .features import (
    REQUIRED_INPUT_COLUMNS,
    FeatureTable,
    assert_no_missing,
    audit_correlations,
    build_features,
    compute_vaf,
    motif_strata,
    normalize_depth_total,
    normalize_rep_units,
)
from .flanks import compute_flank_features
from .model import (
    MIN_PARTITION_ROWS,
    MODEL_PARAMS,
    IsolationForestModel,
    PartitionFit,
    average_path_length,
    fit,
    fit_table,
    load,
    save,
    score,
)
from .paths import PathTrace, SplitStep, explain_paths, mean_path_length
from .visualize import (
    RowExplanation,
    StratumExplanation,
    explain_by_stratum,
    plot_beeswarms,
    plot_isolation_tree,
    plot_waterfall,
    print_motif_length_dominance,
)

__all__ = [
    "COHORT_FEATURES",
    "DEFAULT_BLOCKS",
    "FEATURE_BLOCKS",
    "FLANKED_BLOCKS",
    "FLANK_FEATURES",
    "INTRINSIC_FEATURES",
    "MIN_PARTITION_ROWS",
    "MODEL_PARAMS",
    "OUTPUT_META_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "UNFLANKED_BLOCKS",
    "CalibrationReport",
    "FeatureBlock",
    "FeatureTable",
    "IsolationForestModel",
    "PartitionFit",
    "PathTrace",
    "RowExplanation",
    "SplitStep",
    "StratumExplanation",
    "assert_no_missing",
    "audit_correlations",
    "average_path_length",
    "base_value",
    "build_features",
    "calibrate",
    "compute_flank_features",
    "compute_vaf",
    "explain",
    "explain_by_stratum",
    "explain_paths",
    "feature_columns",
    "fit",
    "fit_table",
    "load",
    "mean_path_length",
    "motif_length_dominance",
    "motif_strata",
    "normalize_depth_total",
    "normalize_rep_units",
    "plot_beeswarms",
    "plot_isolation_tree",
    "plot_waterfall",
    "print_motif_length_dominance",
    "resolve_blocks",
    "save",
    "score",
    "shap_column",
    "stratum_boundary_shift",
]

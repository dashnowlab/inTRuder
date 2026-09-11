"""Command line for isolation-forest anomaly scoring of novel-TR candidates.

``uv sync --group modeling`` installs this as the ``isolation-forest`` command.

    # inspect the feature table without fitting anything
    uv run isolation-forest features candidates.tsv features.tsv

    # fit one model per motif-length x flank-presence stratum
    uv run isolation-forest fit candidates.tsv --model-out model.joblib

    # score a table (any table -- this one need not be the training set)
    uv run isolation-forest annotate candidates.tsv scored.tsv --model-in model.joblib

    # sanity-check against STRchive disease loci
    uv run isolation-forest calibrate scored.tsv --metrics calibration.tsv

This is a QC/confidence score, not a replacement for the catalog-based novelty
verdict in ``pipeline.novelty`` -- see the package docstring.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from intruder.pipeline.strchive.catalog import BUILDS, Catalog

from .calibration import calibrate
from .columns import FEATURE_BLOCKS, feature_columns, resolve_blocks
from .explain import explain
from .features import audit_correlations, build_features
from .flanks import compute_flank_features
from .model import load, save

_STRATA_BOUNDS_DEFAULT = "6"


def _blocks_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _default_blocks(mode: str) -> list[str]:
    return ["intrinsic", "cohort"] if mode == "cohort" else ["intrinsic"]


def _needs_flank(blocks) -> bool:
    return any(b.name == "flank" for b in blocks)


def _load_input(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _resolve_flanks(frame: pd.DataFrame, blocks, vcf_path: str | None) -> pd.DataFrame | None:
    if not _needs_flank(blocks):
        return None
    if not vcf_path:
        raise SystemExit("error: the 'flank' block needs --vcf to re-read ALT sequences")
    return compute_flank_features(frame, vcf_path)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def _cmd_features(args: argparse.Namespace) -> int:
    frame = _load_input(args.input)
    blocks = resolve_blocks(args.blocks)
    flanks = _resolve_flanks(frame, blocks, args.vcf)
    table = build_features(frame, blocks, strata_bounds=args.strata_bounds, flanks=flanks,
                           drop_vaf=args.drop_vaf)

    if args.audit_correlations != "off":
        audit_correlations(table.features, feature_columns(blocks), mode=args.audit_correlations)

    out = pd.concat([frame, table.features, table.stratum], axis=1)
    out.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    print(f"[isolation-forest] {args.output}: {len(out):,} row(s), "
          f"blocks={','.join(b.name for b in blocks)}", file=sys.stderr)
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    frame = _load_input(args.input)
    block_names = args.blocks or _default_blocks(args.mode)
    blocks = resolve_blocks(block_names)
    flanks = _resolve_flanks(frame, blocks, args.vcf)
    table = build_features(frame, blocks, strata_bounds=args.strata_bounds, flanks=flanks,
                           drop_vaf=args.drop_vaf)

    from .model import fit_table
    model = fit_table(table, blocks, strata_bounds=tuple(args.strata_bounds),
                      n_jobs=args.n_jobs, strata_jobs=args.strata_jobs,
                      audit=args.audit_correlations, mode=args.mode)
    save(model, args.model_out)

    print(f"[isolation-forest] {args.model_out}: {len(model.partitions)} partition(s), "
          f"mode={args.mode}, blocks={','.join(block_names)}", file=sys.stderr)
    for stratum, partition in model.partitions.items():
        tag = " (pooled fallback)" if partition.fallback else ""
        print(f"[isolation-forest]   {stratum:<24} n={partition.n_rows:<8}{tag}", file=sys.stderr)
    return 0


def _partition_for_label(model, label: str):
    partition = model.partitions.get(label)
    if partition is not None:
        return partition
    return next(p for p in model.partitions.values() if p.fallback)


def _blocks_for_partition(model, partition) -> str:
    active = [name for name in model.blocks
             if any(c in partition.columns for c in FEATURE_BLOCKS[name].columns)]
    return "+".join(active) if active else "none"


def _cmd_annotate(args: argparse.Namespace) -> int:
    from .model import score as score_table

    frame = _load_input(args.input)
    model = load(args.model_in)
    blocks = resolve_blocks(model.blocks)
    flanks = _resolve_flanks(frame, blocks, args.vcf)
    table = build_features(frame, blocks, strata_bounds=model.strata_bounds, flanks=flanks)

    scores = score_table(model, table.features, table.stratum, n_jobs=args.n_jobs)

    shap_chunks = []
    if_blocks = pd.Series(index=frame.index, dtype=object)
    for label in pd.unique(table.stratum):
        idx = table.stratum.index[table.stratum == label]
        partition = _partition_for_label(model, label)
        X = table.features.loc[idx, list(partition.columns)]
        shap_chunks.append(explain(partition.model, X, n_jobs=args.n_jobs or 1,
                                   chunk_size=args.shap_chunk_size))
        if_blocks.loc[idx] = _blocks_for_partition(model, partition)

    shap_frame = pd.concat(shap_chunks).reindex(frame.index)
    # %.5g keeps the TSV from tripling in size; applied only to the columns
    # this step adds, so the caller's own columns round-trip unformatted.
    shap_text = shap_frame.map(lambda v: "" if pd.isna(v) else f"{v:.5g}")

    extra = pd.DataFrame(index=frame.index)
    extra["if_stratum"] = table.stratum
    extra["if_score"] = scores["if_score"].map(lambda v: "" if pd.isna(v) else f"{v:.5g}")
    extra["if_h"] = scores["if_h"].map(lambda v: "" if pd.isna(v) else f"{v:.5g}")
    extra["if_percentile"] = scores["if_percentile"].map(
        lambda v: "" if pd.isna(v) else f"{v:.5g}")
    extra = pd.concat([extra, shap_text], axis=1)
    extra["if_mode"] = model.mode
    extra["if_blocks"] = if_blocks

    out = pd.concat([frame, extra], axis=1)
    out.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    print(f"[isolation-forest] {args.output}: {len(out):,} row(s) scored across "
          f"{len(model.partitions)} partition(s)", file=sys.stderr)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    frame = _load_input(args.input)
    catalog = Catalog.load(build=args.build, cache_dir=args.cache_dir,
                           verbose=not args.quiet)
    report = calibrate(frame, catalog, score_col=args.score_col,
                       chrom_col=args.chrom_col, pos_col=args.pos_col,
                       coord_base=args.coord_base, window=args.window,
                       min_recall=args.min_recall, threshold=args.threshold)

    pd.DataFrame([report.as_row()]).to_csv(args.metrics, sep="\t", index=False)
    status = "PASS" if report.passed else "FAIL"
    print(f"[isolation-forest] recall gate: {status}  recall={report.recall:.3f} "
          f"(need >={report.min_recall}) at threshold={report.threshold:.4g}  "
          f"n_overlapping={report.n_overlapping}/{report.n_strchive_loci} loci",
          file=sys.stderr)
    print(f"[isolation-forest] KS(overlap, background): statistic={report.ks_statistic:.4f} "
          f"p={report.ks_pvalue:.4g}", file=sys.stderr)
    if report.dropped_loci:
        shown = ", ".join(report.dropped_loci[:5])
        print(f"[isolation-forest] {len(report.dropped_loci)} STRchive locus/loci with no "
              f"overlapping candidate: {shown}"
              + ("..." if len(report.dropped_loci) > 5 else ""), file=sys.stderr)
    return 0 if report.passed else 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def _add_block_args(parser: argparse.ArgumentParser, *, default: str | None) -> None:
    parser.add_argument("--blocks", type=_blocks_list, default=default,
                        metavar="NAME[,...]",
                        help=f"feature block(s) to compute: {', '.join(FEATURE_BLOCKS)} "
                             f"(default: derived from --mode)")
    parser.add_argument("--vcf", metavar="PATH",
                        help="VCF the candidates came from; needed for the 'flank' block")
    parser.add_argument("--strata-bounds", type=_int_list, default=_int_list(_STRATA_BOUNDS_DEFAULT),
                        metavar="N[,...]",
                        help="motif-length stratum boundaries, upper edge inclusive "
                             "(default: %(default)s)")
    parser.add_argument("--drop-vaf", action="store_true",
                        help="never compute 'vaf' (DV/(DV+DR)), even when the input's "
                             "'depth' column still has read-support to split "
                             "(default: include it whenever it's available)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isolation-forest",
        description="Isolation-forest anomaly scoring + SHAP explanation for "
                    "novel-TR candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[-1],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    features = sub.add_parser("features", help="compute the feature table without fitting")
    features.add_argument("input", help="TSV of TR candidates")
    features.add_argument("output", help="feature TSV to write")
    _add_block_args(features, default=["intrinsic"])
    features.add_argument("--audit-correlations", choices=("off", "warn", "error"),
                          default="off",
                          help="pairwise Spearman correlation check on the "
                               "computed features (default: %(default)s)")
    features.set_defaults(func=_cmd_features)

    fit = sub.add_parser("fit", help="fit one IsolationForest per stratum")
    fit.add_argument("input", help="TSV of TR candidates")
    fit.add_argument("--model-out", required=True, metavar="PATH",
                     help="where to write the fitted model (+ a PATH.json sidecar)")
    fit.add_argument("--mode", choices=("intrinsic", "cohort"), default="intrinsic",
                     help="intrinsic: single-sample capable (default). cohort: adds "
                          "recurrence features, needs multi-sample input")
    _add_block_args(fit, default=None)
    fit.add_argument("--n-jobs", type=int, default=1,
                     help="parallel jobs for sklearn's bagging fit, per stratum "
                          "(default: %(default)s)")
    fit.add_argument("--strata-jobs", type=int, default=1,
                     help="how many strata to fit concurrently -- usually the "
                          "bigger lever than --n-jobs since each stratum's fit "
                          "is independent (default: %(default)s)")
    fit.add_argument("--audit-correlations", choices=("warn", "error"), default="warn",
                     help="what to do about a correlated feature pair above 0.95 "
                          "Spearman (default: %(default)s)")
    fit.set_defaults(func=_cmd_fit)

    annotate = sub.add_parser("annotate", help="score a table with a fitted model")
    annotate.add_argument("input", help="TSV of TR candidates")
    annotate.add_argument("output", help="annotated TSV to write")
    annotate.add_argument("--model-in", required=True, metavar="PATH",
                          help="model written by `fit`")
    annotate.add_argument("--vcf", metavar="PATH",
                          help="VCF the candidates came from; needed if the model "
                               "used the 'flank' block")
    annotate.add_argument("--n-jobs", type=int, default=1,
                          help="parallel jobs for scoring and SHAP row-chunking "
                               "(default: %(default)s)")
    annotate.add_argument("--shap-chunk-size", type=int, default=None, metavar="N",
                          help="row-chunk size for parallel SHAP; only takes effect "
                               "with --n-jobs != 1 (default: no chunking)")
    annotate.set_defaults(func=_cmd_annotate)

    calibrate_cmd = sub.add_parser("calibrate",
                                   help="STRchive recall gate + KS diagnostic on a scored table")
    calibrate_cmd.add_argument("input", help="TSV written by `annotate`")
    calibrate_cmd.add_argument("--metrics", required=True, metavar="PATH",
                               help="one-row metrics TSV to write")
    calibrate_cmd.add_argument("--score-col", default="if_score")
    calibrate_cmd.add_argument("--chrom-col", default="chrom")
    calibrate_cmd.add_argument("--pos-col", default="ins_coord")
    calibrate_cmd.add_argument("--coord-base", type=int, choices=(0, 1), default=1)
    calibrate_cmd.add_argument("--window", type=int, default=0, metavar="BP",
                               help="how far a candidate may sit from a STRchive "
                                    "locus and still count as overlapping "
                                    "(default: %(default)s)")
    calibrate_cmd.add_argument("--min-recall", type=float, default=0.95)
    calibrate_cmd.add_argument("--threshold", type=float, default=None,
                               help="operating threshold on --score-col (default: "
                                    "the loosest one clearing --min-recall)")
    calibrate_cmd.add_argument("--build", choices=sorted(BUILDS), default="hg38")
    calibrate_cmd.add_argument("--cache-dir", metavar="PATH", default=None)
    calibrate_cmd.add_argument("--quiet", action="store_true")
    calibrate_cmd.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError, NotImplementedError) as exc:
        print(f"[isolation-forest] error: {exc}", file=sys.stderr)
        return 1

# Isolation forest anomaly scoring (`isolation-forest`)

**Fits an** [`IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
**over intrinsic sequence/call features and scores each novel-TR candidate for**
**how anomalous it looks, with a full SHAP attribution matrix explaining why.**

This is a QC/confidence score, and it does not replace the
catalog-based novelty screen— a candidate can be
`novel_locus` there and still score unremarkable here, or vice versa. What it
adds is a second, independent read on each candidate: does this call look like
the rest of the population, or does it stand out on the features that went
into the fit? [STRchive's](https://strchive.org) 82 curated, disease-biased loci are never
training data; `calibrate` checks whether they still land on the "normal" side of a
chosen threshold.

## uv setup

```bash
uv sync --group modeling      # installs shap, scikit-learn, scipy, matplotlib
```

## CLI

Four registered commands:

```bash
# 1. inspect the feature table without fitting anything
uv run isolation-forest features candidates.tsv features.tsv

# 2. fit one model per motif-length x flank-presence stratum
uv run isolation-forest fit candidates.tsv --model-out model.joblib

# 3. score a table (any table -- this one need not be the training set)
uv run isolation-forest annotate candidates.tsv scored.tsv --model-in model.joblib

# 4. sanity-check against STRchive disease loci
uv run isolation-forest calibrate scored.tsv --metrics calibration.tsv
```


| command                                   | what it does                                                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `features IN.tsv OUT.tsv`                 | computes the feature table and stratum labels, without fitting a model. For inspecting what the model would see                    |
| `fit IN.tsv --model-out PATH`             | fits one `IsolationForest` per stratum, writes `PATH` (joblib) and `PATH.json` (a sidecar recording column order and fit metadata) |
| `annotate IN.tsv OUT.tsv --model-in PATH` | scores every row with a fitted model, appending the score, stratum, and per-feature SHAP columns                                   |
| `calibrate IN.tsv --metrics PATH`         | STRchive recall gate plus a KS diagnostic on an already-scored table, writing a one-row metrics TSV                                |


### Shared flags (`features`, `fit`)


| flag                      | default                                                  | what it does                                                                                                                 |
| ------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--blocks NAME[,...]`     | `intrinsic` (`intrinsic,cohort` for `fit --mode cohort`) | feature block(s) to compute — see [Feature blocks](#feature-blocks)                                                          |
| `--vcf PATH`              | —                                                        | the VCF the candidates came from; required if `--blocks` includes `flank`                                                    |
| `--strata-bounds N[,...]` | `6`                                                      | motif-length stratum boundaries, upper edge inclusive                                                                        |
| `--drop-vaf`              | off                                                      | never compute `vaf` (`DV/(DV+DR)`), even when the input's `depth` column still has the read-support split to compute it from |


### `fit`


| flag                                | default     | what it does                                                                                                                                                                                                  |
| ----------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model-out PATH`                  | required    | where to write the fitted model                                                                                                                                                                               |
| `--mode {intrinsic,cohort}`         | `intrinsic` | `intrinsic` needs only a single sample; `cohort` adds recurrence features and needs multi-sample input                                                                                                        |
| `--n-jobs N`                        | `1`         | parallel jobs for sklearn's bagging fit, per stratum                                                                                                                                                          |
| `--strata-jobs N`                   | `1`         | how many strata to fit concurrently — usually the bigger lever than `--n-jobs`, since each stratum's fit is independent and `max_samples=256` keeps any one fit's trees too small for `--n-jobs` to help much |
| `--audit-correlations {warn,error}` | `warn`      | what to do about a feature pair above 0.95 Spearman correlation                                                                                                                                               |


### `annotate`


| flag                  | default  | what it does                                                             |
| --------------------- | -------- | ------------------------------------------------------------------------ |
| `--model-in PATH`     | required | model written by `fit`                                                   |
| `--vcf PATH`          | —        | required if the model used the `flank` block                             |
| `--n-jobs N`          | `1`      | parallel jobs for scoring and SHAP row-chunking                          |
| `--shap-chunk-size N` | none     | row-chunk size for parallel SHAP; only takes effect with `--n-jobs != 1` |


### `calibrate`


| flag                       | default              | what it does                                                                                |
| -------------------------- | -------------------- | ------------------------------------------------------------------------------------------- |
| `--metrics PATH`           | required             | one-row metrics TSV to write                                                                |
| `--score-col`              | `if_score`           | column `annotate` wrote the anomaly score to                                                |
| `--chrom-col`, `--pos-col` | `chrom`, `ins_coord` | input columns identifying each candidate's position                                         |
| `--coord-base {0,1}`       | `1`                  | whether `--pos-col` counts from 0 or 1                                                      |
| `--window BP`              | `0`                  | how far a candidate may sit from a STRchive locus and still count as overlapping            |
| `--min-recall`             | `0.95`               | required recall of STRchive-overlapping candidates at the chosen threshold                  |
| `--threshold`              | none                 | operating threshold on `--score-col`; default is the loosest one that clears `--min-recall` |
| `--build`                  | `hg38`               | assembly STRchive loci are pulled for                                                       |
| `--cache-dir`              | —                    | where the STRchive catalog is cached                                                        |


Exit code is `0` when the recall gate passes, `1` otherwise — the same
convention `novelty`'s `sweep --objective` uses, so it slots into a CI check.

### Feature blocks


| block         | needs                                 | columns                                                                                                           |
| ------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `intrinsic`   | nothing beyond the input table        | `purity`, `log_motif_length`, `log_rep_length`, `repeat_coverage`, `log_depth_total`, `motif_gc`, `motif_entropy` |
| `flank`       | `--vcf`                               | `flank_5p_identity`, `flank_3p_identity`, `flank_5p_len`, `flank_3p_len`                                          |
| `cohort`      | multi-sample input, grouped by `SVID` | `n_carriers`, `rep_length_cv`, `n_distinct_canonical_motifs`                                                      |
| `mappability` | —                                     | registered as a stub; `--blocks mappability` fails with "not implemented" until reference tracks exist            |


`vaf` (`DV/(DV+DR)`) rides along with `intrinsic` rather than being its own
block: it needs no external resource, just a `depth` column that still has
the read-support split. It is included whenever that split survives in the
input and left out, never imputed, once something upstream has already
collapsed `depth` to a bare total.

## Module layout

```
src/python/intruder/modeling/unsupervised/isolation_forest/
├── columns.py       feature-block registry, output column names
├── features.py      load, normalize dtype quirks, build feature blocks, assign strata
├── flanks.py        re-read ALT sequences from the VCF, compute flank features
├── model.py         fit / persist / score per stratum
├── explain.py       SHAP matrix via TreeExplainer
├── paths.py         explain_paths() per-candidate split traces
├── calibration.py   STRchive recall gate, KS diagnostic, stratification checks
├── visualize.py     notebook plotting: per-stratum SHAP, beeswarms, waterfalls, tree diagram
└── cli.py           the isolation-forest command line
```


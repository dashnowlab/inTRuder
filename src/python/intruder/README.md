# novelty
 
Finds tandem repeats inside SV insertions, drops the weak calls, and screens the
rest against reference repeat catalogues. Every surviving repeat gets a verdict:
`known`, or one of the `novel_*` kinds.
 
```
SV VCF ──► trfcaller ──► filter ──► catalog screen ──► TSV
           pytrf finds   drops weak  verdict per catalogue,
           the repeats   calls       then combined
```
 
One command, `detect`, runs all three in-process. The rules themselves are
explained in the docstring at the top of each file; this README covers usage and
behavior.
 
## Quick start
 
```bash
uv sync        # the code imports pytrf, cyvcf2, tqdm, pandas, numpy
 
# one catalogue, downloaded on first use
uv run python -m intruder.pipeline.novelty detect sv.vcf out.tsv \
    --repeats ucsc=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz \
    --format ucsc=ucsc
 
# two local catalogues, accepting motifs one edit apart
uv run python -m intruder.pipeline.novelty detect sv.vcf out.tsv \
    --repeats ucsc=simpleRepeat.txt.gz --format ucsc=ucsc \
    --repeats trex=trexplorer.bed.gz \
    --max-motif-edits 1
```
 
## Files
 
| File | Role |
|---|---|
| `cli.py` | the `detect` command: arguments, step order, verdict combining, run report |
| `trfcaller.py` | `run_trf_on_insertions`: VCF → table of repeat calls (pytrf, cyvcf2) |
| `filter.py` | `filter_trf`: quality filters on that table |
| `platforms.py` | catalogue readers (`bed`, `ucsc`, `strchive`) and URL download |
| `catalog.py` | `RepeatCatalog`: the interval index and the known/novel decision |
| `motifs.py` | when two motifs count as the same |
| `coords.py`, `fetch.py`, `paths.py` | coordinate conversion; download with TLS fallback; repo-root lookup |
 
## Input
 
- A VCF. Only records with `SVTYPE=INS`, an `SVLEN` and an ALT sequence are used;
  anything else is skipped without a message. Only the first ALT allele is searched.
- Each sample needs `DR` and `DV` FORMAT fields.
- Samples are searched one by one, and within a sample every non-reference allele
  of the first two genotype entries. A homozygous-alt sample therefore produces
  two identical rows.
## Output
 
A TSV with one row per repeat call that passed the filters.
 
| Columns | Meaning |
|---|---|
| `chrom`, `ins_coord` | insertion site: VCF `CHROM` and `POS` (1-based) |
| `SVID` | VCF ID. Not unique across samples |
| `depth` | `"<variant>,<reference>"` read counts for the sample (`DV`, `DR`) |
| `insert_size` | `SVLEN` |
| `sample`, `allele` | sample name; genotype allele index |
| `rep_start`, `rep_end` | repeat position **inside the insertion**. Not a genome coordinate, never used for screening |
| `motif`, `motif_length`, `rep_length`, `rep_units` | repeat unit pytrf reported; its length; repeat length; copy count |
| `purity` | pytrf identity on a 0–1 scale |
| `repeat_coverage` | `rep_length / insert_size`, added by the filter |
| `novelty` | combined verdict across all catalogues |
| `<name>_*` | one block per `--repeats NAME=…`, below |
 
**Per-catalogue block** (prefixed with the catalogue's `NAME`):
 
| Column | Meaning |
|---|---|
| `novelty` | this catalogue's verdict |
| `n_nearby` | reference repeats within `--window` of the insertion site |
| `start`, `end`, `distance` | the best-matching reference repeat, or the nearest one if none match: coordinates in the `--coord-base` convention, and bp from the site (0 = overlapping) |
| `motif`, `canonical` | its motif, and the canonical form used for comparison |
| `motif_edits`, `match` | edits used, and the rule that accepted it: `exact`, `wildcard`, `fuzzy`, `vntr`, `subrepeat`. `match` is empty when nothing matched |
| `period`, `copy_num`, … | catalogue annotations, only where the format has them (`ucsc`: all five; `strchive`: `period`, `copy_num`; `bed`: none) |
 
The reference-repeat columns are `NA` when nothing was found nearby.
 
**Verdicts**
 
| Verdict | Meaning |
|---|---|
| `known` | a reference repeat within the window has a matching motif |
| `novel_motif` | reference repeats are there, but none with a matching motif |
| `novel_locus` | nothing in the window |
| `unscreened` | the catalogue has no entries on this contig at all |
 
The combined `novelty` takes the most conservative verdict, in the order above:
one catalogue saying `known` makes the row `known`, and `unscreened` only wins if
every catalogue lacks the contig.
 
The run report on stderr counts verdicts per row and per locus. A locus is
`chrom` + `ins_coord`, so samples and repeated calls at one site count once, with
their most conservative verdict.
 
## Options
 
`--coord-base {0,1}` (default `1`) is a top-level option and goes **before**
`detect`. It sets the base of the position column and of the `<name>_start/_end`
output columns. Leave it at `1` for VCF input.
 
`detect INPUT OUTPUT`: `INPUT` is the VCF, `OUTPUT` the TSV.
`--chrom-col`, `--pos-col`, `--motif-col` name the table columns used for
screening. Their defaults (`chrom`, `ins_coord`, `motif`) match what `trfcaller`
writes; don't point `--pos-col` at `rep_start`/`rep_end`.
 
**Catalogues**
 
| Flag | Effect |
|---|---|
| `--repeats NAME=PATH\|URL` | required, repeatable. `NAME` must be unique and prefixes that catalogue's output columns |
| `--format NAME=FMT` | `bed` (default, BED4), `ucsc` (raw `simpleRepeat.txt(.gz)` dump) or `strchive` (STRchive JSON). Never sniffed from the file |
 
**TRF**
 
| Flag | Default | Effect |
|---|---|---|
| `--min-sv-length`, `--max-sv-length` | 50, 10000 | skip insertions whose `SVLEN` is outside this range |
| `--min-motif`, `--max-motif` | 1, 500 | repeat unit length range |
| `--min-identity` | 0.8 | minimum identity as a 0–1 fraction (multiplied by 100 before it reaches pytrf, which takes a percentage) |
| `--min-rep-units` | 3 | minimum copies in the seed |
| `--min-rep-length` | 10 | minimum seed length, bp |
| `--max-rep-length` | 10000 | maximum extension length, bp (pytrf `max_extend`) |
 
**Filters** (starting points, tune to your data). They run in this order, each on
what the previous one left, and act on repeat calls, so an insertion can keep
some calls and lose others.
 
| Flag | Default | Keeps rows where |
|---|---|---|
| `--exclude-motif-sizes` | none | `motif_length` is not in the comma list |
| `--keep-motif-sizes` | none | `motif_length` is in the comma list (after the exclusion) |
| `--min-repeat-coverage` | 0.8 | `repeat_coverage` ≥ value |
| `--min-purity` | 0.7 | `purity` ≥ value |
| `--max-insert-size` | 10000 | `insert_size` ≤ value |
| `--min-depth` | 30 | **both** `depth` numbers ≥ value |
| `-s`, `--stats-output` | | accepted, not used |
 
**Screening**
 
| Flag | Default | Effect |
|---|---|---|
| `--window` | 10 | bp from the insertion site within which a reference repeat counts as "there" |
| `--max-motif-edits` | 0 | accept a reference motif within N edits, at any length |
| `--max-motif-edit-fraction` | off | also accept F × motif length edits, for long motifs only (`STR_MAX_MOTIF` in `motifs.py`) |
| `--min-subrepeat-motif` | off | accept a motif that the other one tiles, tiling unit ≥ BP long |
| `--circular` / `--no-circular` | on | treat rotations as the same motif |
 
## Behavior
 
**Matching**
- Repeat copies collapse before comparing (`CAGCAG` = `CAG`). With `--circular`,
  rotations are equal too (`CAG` = `AGC`). Strand is not considered (`CAG` ≠ `CTG`).
- An `N` in a reference motif matches any base of a same-length query motif
  (`NAC` accepts `CAC`, `GAC`, …). Always on; reported as `match = wildcard`.
- With `--min-subrepeat-motif`, a shorter motif can tile a longer one (`ACC`
  against `ACCATC`). An `N` on either side is free while tiling, so `AC` matches
  `ACAN` with no edit budget. Other mismatches need `--max-motif-edits` or
  `--max-motif-edit-fraction`.
- Without any edit option, only the rules above apply: `CAG` never matches `CAT`.
**Catalogues**
- A URL is downloaded once and reused afterwards (delete the file to refresh).
  It lands in `<repo>/data/novelty/` inside a checkout (a directory holding
  `pyproject.toml` and `data/`), otherwise `~/.cache/novelty/`. If Python's TLS
  verification fails, the download retries with the system `curl`.
- BED and UCSC coordinates are read as 0-based half-open; there is no option for
  1-based files. Gzip is detected automatically.
- Each run re-reads and re-indexes every catalogue. There is no index cache.
**Filtering**
- `repeat_coverage` is 0.0 when `insert_size` is not positive.
- `--min-depth` needs the reference-supporting count to pass too, so calls with
  low reference depth (for example homozygous-alt insertions) are dropped. Use
  `--min-depth 0` to disable it, for instance when `DV`/`DR` are all zero in
  your VCF.
**Errors**
- `ValueError`, `FileNotFoundError` and `RuntimeError` print `error: …` and exit 1.
  Anything else is a plain traceback.
## Known limitations
 
- If pytrf finds no repeats at all, the empty table has no columns and
  `filter_trf` fails with `KeyError: 'insert_size'`.
- A VCF record without `DR` or `DV` raises a `TypeError` once a sample carries
  the insertion.
- `--stats-output` does nothing: no stats file is written.
## Extending
 
- **New filter:** add an `apply_step(...)` in `run_filters` (`filter.py`) and a
  flag in `_add_filter_args` (`cli.py`).
- **New catalogue format:** add a `read_<fmt>` in `platforms.py` that returns
  `chrom, start, end, motif` (plus any annotation columns), branch on it in
  `read_catalog`, and list it in the `--format` help.
- **New matching rule:** `MotifTolerance.compare` in `motifs.py`; expose it
  through `_add_screen_args` and `_tolerance` in `cli.py`.

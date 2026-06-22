# CLIP Analysis Scripts

Scripts for downloading, processing, and visualizing CLIP (Cross-Linking and Immunoprecipitation) data from Flow.bio.

## Prerequisites

1. **Conda environment** with required packages:
   ```bash
   conda activate bioinf
   ```

2. **API credentials**: Create `credentials.json` in this directory:
   ```json
   {
     "username": "your_email@example.com",
     "password": "your_password"
   }
   ```

## Data Files

The analysis requires three types of data files:

| File Type | Directory | Description |
|-----------|-----------|-------------|
| `*_UMICollapse.log` | `data/` | PCR duplication stats and crosslink counts |
| `*.summary_subtype_premapadjusted.tsv` | `data_subtype/` | Regional distribution (CDS, UTR, intron, etc.) |
| `*.genome_5mer_distribution_genome.tsv` | `data_peka/` | PEKA motif enrichment scores |

### File Naming Convention

**All downloaded files are prefixed with their Flow.bio sample_id for traceability:**

```
{sample_id}_{original_filename}
```

Example: `12345_HNRNPC.unique_genome.dedup_UMICollapse.log`

This ensures every file can be traced back to its source sample in the Flow.bio database.

## Updating the Plot

### Step 1: Download Main Data Files

This downloads `UMICollapse.log` files and creates `filtered_data.json` with basic sample info:

```bash
# Fresh download (re-fetches all public samples from API)
python download_files.py --fresh

# Or use cached sample list (faster, uses existing filtered_data.json)
python download_files.py

# Download only human samples
python download_files.py --fresh --organism Human

# Download from a private project with a specific pipeline
python download_files.py --project <PROJECT_ID> --pipeline "hanalysis clipseq"
```

**Note**: The `--fresh` flag fetches all public samples from the API, which can take 10-30 minutes depending on the number of samples.

### Step 2: Enrich Sample Metadata

This fetches detailed metadata (purification target, experimental method, cell type, etc.) for each sample:

```bash
python enrich_sample_metadata.py
```

### Step 3: Download Regional Distribution Files

```bash
python download_additional_files.py \
    --regex ".*summary_subtype_premapadjusted\.tsv" \
    --dir data_subtype
```

### Step 4: Download PEKA Motif Enrichment Files

```bash
python download_additional_files.py \
    --regex ".*genome_5mer_distribution_genome\.tsv" \
    --dir data_peka
```

### Step 5: Generate the Circos Plot

```bash
python plot_circos.py
```

This creates `regional_distribution_circos_motif.png` with:
- Hierarchical clustering based on PEKA motif enrichment
- Regional distribution (stacked bars)
- PCR duplication ratio
- Crosslink counts
- Dinucleotide sequence specificity

## Quick Update (All Steps)

Run all download and plot generation steps:

```bash
# Activate environment
conda activate bioinf

# 1. Download main files (UMICollapse.log)
python download_files.py --fresh

# 2. Enrich metadata with full sample details
python enrich_sample_metadata.py

# 3. Download regional distribution files
python download_additional_files.py -r ".*summary_subtype_premapadjusted\.tsv" -d data_subtype

# 4. Download PEKA files
python download_additional_files.py -r ".*genome_5mer_distribution_genome\.tsv" -d data_peka

# 5. Generate plot
python plot_circos.py
```

## Script Reference

### `download_files.py`

Downloads main data files (UMICollapse.log) and updates metadata.

```
Options:
  --fresh              Re-fetch all public samples from API (slower but complete)
  --regex              Filename pattern to match (default: UMICollapse.log)
  --dir                Output directory (default: data)
  --json               Output JSON file (default: filtered_data.json)
  --slurm              Generate SLURM job scripts instead of downloading
  --slurm-dir          Directory for SLURM job scripts (default: slurm_jobs)
  --no-sample-id-prefix  Don't prefix filenames with sample_id (not recommended)
  --organism           Filter samples by organism (e.g. 'Human', 'Mouse'). Case-insensitive.
  --project            Fetch samples from a specific project ID instead of public samples
                       (includes private projects you have access to)
  --pipeline           Only keep data records from a given pipeline (e.g. 'hanalysis clipseq').
                       Case-insensitive substring match on the pipeline_name field.
```

### `download_additional_files.py`

Fast download of additional file types using existing sample IDs.

```
Options:
  --regex, -r    Filename regex pattern (required)
  --dir, -d      Output directory (required)
  --source, -s   Source JSON with sample IDs (default: filtered_data.json)
  --workers, -w  Parallel workers (default: 8)
  --slurm        Generate SLURM job scripts instead of downloading
  --slurm-dir    Directory for SLURM job scripts (default: slurm_jobs)
  --no-sample-id-prefix  Don't prefix filenames with sample_id
  --pipeline     Only keep data records from this pipeline (e.g. 'hanalysis clipseq').
                 Case-insensitive substring match.
  --latest-per-sample
                 Keep only the most recent matched file per sample_id, even
                 when several different filenames match the regex.
```

## Full CLIP Sample Status Pull (bedgraph + UMICollapse)

Refreshes the master spreadsheet of every public CLIP sample on Flow.bio with
its latest CLIP-seq v1.7 (and v1.6 fallback) `genome.xl.bedgraph`,
`UMICollapse.log`, and PEKA whole-gene 5mer distribution
(`genome_5mer_distribution_whole_gene.tsv`) status. Downloads the matched
UMICollapse logs and computes per-sample PCR duplication / crosslink count QC.
PEKA files are tracked for presence + data ID only (not downloaded — too
large); fetch them on demand with `download_additional_files.py`.

Driver script: [scripts/build_clipseq_v17_sample_status.py](scripts/build_clipseq_v17_sample_status.py).

### Run it

```bash
conda activate bioinf

python scripts/build_clipseq_v17_sample_status.py \
    --timestamp-output \
    --download-umicollapse-logs \
    --slim-output-csv scripts/clipseq_v17_sample_status_slim.csv \
    --workers 24
```

`--timestamp-output` appends `_YYYY-MM-DD_HHMMSS` to the output stem, so each
run is preserved. With the defaults, output is:

| Path | Contents |
|------|----------|
| `scripts/clipseq_v17_sample_status_<ts>.csv` | One row per CLIP sample, ~100 columns: bedgraph/UMI presence, execution IDs, retry chains, QC metrics, full sample metadata. |
| `scripts/clipseq_v17_sample_status_<ts>_no_clipseq_execution.csv` | Samples with no v1.7 or v1.6 clipseq execution (chase-up list). |
| `scripts/clipseq_v17_sample_status_slim.csv` | Compact metrics-focused subset. |
| `scripts/data_v17_umicollapse/<sample_id>_<filename>` | Downloaded UMICollapse logs (one per sample). |

### Key flags

- `--pipeline-version 1.7` / `--fallback-pipeline-version 1.6` — change which pipeline versions are accepted. Pass empty string to disable fallback.
- `--workers N` — parallelism for API calls (default 24). Lower if the API rate-limits.
- `--max-samples N` — smoketest a small batch first.
- `--sample-source samples_search|project_shard` — `project_shard` is slower but more thorough for catching samples missed by the global list.
- `--download-umicollapse-logs` — required to populate `umicollapse_input_reads`, `umicollapse_dedup_reads`, `pcr_duplication_rate`, `millions_of_crosslinks`.
- `--peka-regex` — override the PEKA filename pattern (default `genome_5mer_distribution_whole_gene.tsv$`).

### PEKA columns

For each sample the CSV contains:

- `has_peka_whole_gene`, `latest_used_peka_whole_gene_filename`, `latest_used_peka_whole_gene_data_id`, `latest_used_peka_whole_gene_created`, `latest_used_peka_whole_gene_pipeline_name` — the "used" rollup (whichever pipeline version was selected).
- `has_v17_peka_whole_gene`, `latest_v17_peka_whole_gene_{filename,data_id,created,pipeline_name}`, `used_v17_retry_chain_for_peka` — v1.7-specific.
- Equivalent `v16` columns for the v1.6 fallback.

### Status string

One column. Two shapes:

- **`has_bedgraph_and_umicollapse_and_peka`** — all three resolved. (No matter which fallback layer found them; the layer is recorded in the per-artifact `used_v17_*_fallback_for_*` columns for diagnostics.)
- **`missing_<what>[_because_<why>]`** — at least one artifact missing. `<what>` is the missing artifacts joined by `_and_`. `_because_<why>` is appended only when a known systemic reason explains the absence:
  - `_because_no_clipseq_execution` — no v1.7 or v1.6 execution exists.
  - `_because_clipseq_execution_has_no_sample_records` — the execution exists but the per-sample data join is empty and no fallback rescued anything.

Examples seen in practice:

| Status | Meaning |
|---|---|
| `has_bedgraph_and_umicollapse_and_peka` | ready for downstream analysis |
| `missing_peka` | PEKA didn't run / wasn't produced |
| `missing_umicollapse` | UMI log unavailable; bedgraph and PEKA present |
| `missing_bedgraph_and_peka` | only UMI was produced |
| `missing_bedgraph_and_umicollapse_and_peka_because_no_clipseq_execution` | sample has never run through CLIP-Seq |
| `missing_bedgraph_and_umicollapse_and_peka_because_clipseq_execution_has_no_sample_records` | execution exists but the data join is broken on the platform side |

To check which fallback layer found a given artifact (when status is `has_*`), look at `used_v17_retry_chain_for_<artifact>`, `used_v17_alt_ok_execution_for_<artifact>`, `used_v17_execution_filename_fallback_for_<artifact>` (and the v16 equivalents). The actual pipeline version used per row is in `pipeline_version_used`.

### Fallback layers for finding outputs

The script picks the **latest matching execution** per sample (v1.7 first, v1.6 fallback) and looks for bedgraph / UMI / PEKA among that execution's downstream data. When the strict pass misses an artifact, it tries each fallback in turn:

1. **Retry chain** — primary execution + any executions linked via `retries`. Status column suffix: `_with_*_retry_chain_fallback`. Flag: `used_v17_retry_chain_for_*`.
2. **Alt-OK execution pool** — other OK executions at the same pipeline version, not in the retry chain. Catches the case where the latest OK run didn't actually produce outputs for this sample but an earlier OK run did. Status: `_with_*_alt_ok_execution_fallback`. Flags: `used_v17_alt_ok_execution_for_*`, `v17_alt_ok_execution_ids`.
3. **Execution filename-prefix scan** — scans the pooled execution downstream (retry chain ∪ alt-OK) for filenames starting with `<sample_name>[._]` and matching the artifact regex. Catches multi-sample executions where `samples/{id}/data` doesn't back-link the outputs. Status: `_with_*_execution_filename_fallback`. Flag: `used_v17_execution_filename_fallback_for_*`.

The summary at the end of a run prints how many samples needed each fallback layer.

### Smoketest first

The full pull hits the API thousands of times and takes a few hours. Before a full run, sanity-check with:

```bash
python scripts/build_clipseq_v17_sample_status.py \
    --max-samples 50 \
    --download-umicollapse-logs \
    --timestamp-output \
    -o scripts/clipseq_v17_sample_status_smoketest.csv
```

### Refresh slim metrics only

If you only need to regenerate the slim CSV from an existing full CSV (no API
calls):

```bash
python scripts/export_clipseq_v17_slim_metrics.py \
    --input scripts/clipseq_v17_sample_status_<ts>.csv \
    --output scripts/clipseq_v17_sample_status_slim.csv
```

### Rewrite status column without re-pulling

After a change to the status-building logic, rewrite the `status` column on
an existing CSV (and regenerate slim) without re-querying the API:

```bash
python scripts/recompute_status_with_peka.py \
    --input-csv scripts/clipseq_v17_sample_status_<ts>.csv
```

Writes `<ts>_status_recomputed.csv` next to the input, and overwrites
`scripts/clipseq_v17_sample_status_slim.csv`.

## Lambert Paper May 28 2026 Peak BED Download

The Lambert paper sample list is in `lambert_paper_may2826_METADATA.csv`.
For Flow.bio downloads, use the execution-aware helper
`scripts/build_lambert_clipseq_peak_metadata.py`. It reuses the CLIP-seq
v1.7/v1.6 execution lookup code from `scripts/build_clipseq_v17_sample_status.py`
so peak candidates are restricted to downstream outputs from CLIP-seq pipeline
versions 1.7 or 1.6. The current source list is
`lambert_paper_may2826_sample_ids.json`; use it rather than reparsing the CSV
directly, because the CSV has one sample ID rendered in scientific notation
(`iCLIP_TDP43_HUVEC_2` / `677452933401`).

To refresh the latest CLIP-seq v1.6/v1.7 peak BEDs and supplementary metadata:

```bash
python scripts/build_lambert_clipseq_peak_metadata.py \
    --workers 8
```

This mirrors the bedgraph retrieval pattern, but with the stricter execution
lineage gate: resolve the allowed CLIP-seq executions, collect their
`downstream_data` IDs, filter those records by `*minGeneCount5_Peaks.bed`, and
select the latest peak by `created` timestamp. The v1.6/v1.7-only output
directory currently contains 144 non-empty `*minGeneCount5_Peaks.bed` files for
144 unique sample IDs: 100 selected from CLIP-seq v1.7 and 44 selected from
CLIP-seq v1.6.

For manuscript supplementary information, the metadata table with the latest
CLIP-seq v1.6/v1.7 peak BED Flow data IDs is:

```text
lambert_paper_may2826_METADATA_with_clipseq_v16_v17_peak_bed_ids.csv
```

### `download_data_objects.py`

Downloads data objects produced by a specific pipeline process within a project. This handles multi-sample integration outputs (e.g. TETRANSCRIPTS) that aren't linked to individual samples.

It works by navigating: **project → executions → process steps → downstream_data**.

**File naming**: output filenames are prefixed with their execution ID and datetime (`{execution_id}_{YYYY-MM-DD_HHmmss}_{filename}`) since each execution produces identically-named files (e.g. `control.cntTable`). This ensures outputs from different executions don't overwrite each other and makes it easy to identify when each execution was run.

```
Options:
  --project            Project ID to search within (required)
  --process, -p        Process name substring to match (required, e.g. TETRANSCRIPTS)
  --regex, -r          Regex to further filter output filenames (optional)
  --dir, -d            Output directory (required)
  --slurm              Generate SLURM job scripts instead of downloading
  --slurm-dir          Directory for SLURM job scripts (default: slurm_jobs)
  --workers, -w        Parallel download workers (default: 4)
  --json               Save matched data records to a JSON file
```

```bash
# Download all TETRANSCRIPTS outputs from the RBP ENCODE Data project
python download_data_objects.py \
    --project 523943332699993118 \
    --process TETRANSCRIPTS \
    --dir data_te

# Filter by filename pattern
python download_data_objects.py \
    --project 523943332699993118 \
    --process TETRANSCRIPTS \
    --regex ".*\.cntTable" \
    --dir data_te

# SLURM mode
python download_data_objects.py \
    --project 523943332699993118 \
    --process TETRANSCRIPTS \
    --dir data_te --slurm
```

### `generate_sample_metrics.py`

Generates a CSV with sample metrics from downloaded UMICollapse log files and PEKA motif enrichment data.

```bash
python generate_sample_metrics.py --output sample_metrics.csv --peka-dir data_peka
```

Output columns:
- `sample_id` - Flow.bio sample identifier
- `pcr_duplication_rate` - Ratio of input reads to deduplicated reads  
- `millions_of_crosslinks` - Number of deduplicated reads / 1,000,000
- `similarity_score` - Mean overlap ratio of top 50 k-mers vs other samples (see below)

#### Similarity Score

The **similarity score** measures how similar a CLIP dataset's top motifs are compared to all other datasets. It is calculated as follows:

1. For each sample, rank all 5-mers by their PEKA enrichment score and select the **top 50 k-mers**.
2. For every other sample (excluding samples targeting the **same protein**), calculate the **overlap ratio**: the number of shared k-mers in both top-50 sets, divided by 50.
3. The similarity score is the **mean** of all pairwise overlap ratios.

**Interpretation:**
- A score of **0** means the sample's top 50 k-mers were not found in any other dataset's top 50.
- **Higher scores** indicate that the sample's enriched motifs are commonly enriched across many other datasets (e.g. general RNA-binding motifs).
- **Lower scores** indicate more unique or protein-specific binding preferences.

Same-protein comparisons are excluded so that similarity reflects cross-protein overlap rather than replicate consistency.

### `plot_circos.py`

Generates the circos visualization.

```python
from plot_circos import create_circos_plot

# Default settings
create_circos_plot()

# Custom filtering
create_circos_plot(
    output_file='my_plot.png',
    min_crosslinks=150000,  # Minimum deduplicated reads
    max_pcr_ratio=5         # Maximum PCR duplication ratio
)
```

### `enrich_sample_metadata.py`

Enriches `filtered_data.json` with detailed sample metadata from the API.

```bash
python enrich_sample_metadata.py
```

## Filtering Criteria

The circos plot applies these quality filters:
- **Minimum crosslinks**: 150,000 deduplicated reads
- **Maximum PCR ratio**: 5 (input reads / deduplicated reads)

## Output Files

| File | Description |
|------|-------------|
| `filtered_data.json` | Sample metadata and file information |
| `sample_metrics.csv` | Sample ID, PCR duplication rate, crosslink counts, similarity score |
| `regional_distribution_circos_motif.png` | Main circos visualization |
| `pcr_duplication_histogram.png` | PCR duplication ratio distribution |
| `pcr_duplication_by_target.png` | Box plot by purification target |

## SLURM Cluster Download

For large downloads on HPC clusters, use the `--slurm` flag to generate job scripts instead of downloading directly:

```bash
# 1. Generate SLURM job scripts (no downloads yet)
python download_files.py --slurm --slurm-dir slurm_logs

# Or with filters (human-only, specific project/pipeline)
python download_files.py --slurm --slurm-dir slurm_logs --organism Human
python download_files.py --slurm --slurm-dir slurm_logs --project <PROJECT_ID> --pipeline "hanalysis clipseq"

# 2. Submit as array job (efficient)
sbatch slurm_logs/submit_array.sh

# 3. Monitor progress
squeue -u $USER

# 4. Once complete, generate metrics (including similarity scores from PEKA data)
python generate_sample_metrics.py --output sample_metrics.csv --peka-dir data_peka
```

For additional file types:

```bash
# Generate SLURM jobs for PEKA files
python download_additional_files.py \
    --regex ".*genome_5mer_distribution_genome\.tsv" \
    --dir data_peka \
    --slurm --slurm-dir slurm_peka

# Submit
sbatch slurm_peka/submit_array.sh
```

The SLURM scripts include:
- Individual job scripts (`dl_00000.sh`, etc.)
- Array job script (`submit_array.sh`) - submits up to 50 jobs at a time
- Log directory for stdout/stderr

## Troubleshooting

### API Rate Limiting
If downloads are slow or failing, the API may be rate-limiting requests. Wait a few minutes and try again, or reduce the number of workers:
```bash
python download_additional_files.py -r "..." -d ... --workers 2
```

### Missing Files
If some samples are missing data files, they may not have been processed yet on Flow.bio. Re-run the download scripts to fetch newly available files.

### Conda Activation Issues
If `conda activate` fails in scripts, try:
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bioinf
```

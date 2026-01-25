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
```

### `generate_sample_metrics.py`

Generates a CSV with sample metrics from downloaded UMICollapse log files.

```bash
python generate_sample_metrics.py --output sample_metrics.csv
```

Output columns:
- `sample_id` - Flow.bio sample identifier
- `pcr_duplication_rate` - Ratio of input reads to deduplicated reads  
- `millions_of_crosslinks` - Number of deduplicated reads / 1,000,000

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
| `sample_metrics.csv` | Sample ID, PCR duplication rate, crosslink counts |
| `regional_distribution_circos_motif.png` | Main circos visualization |
| `pcr_duplication_histogram.png` | PCR duplication ratio distribution |
| `pcr_duplication_by_target.png` | Box plot by purification target |

## SLURM Cluster Download

For large downloads on HPC clusters, use the `--slurm` flag to generate job scripts instead of downloading directly:

```bash
# 1. Generate SLURM job scripts (no downloads yet)
python download_files.py --slurm --slurm-dir slurm_logs

# 2. Submit as array job (efficient)
sbatch slurm_logs/submit_array.sh

# 3. Monitor progress
squeue -u $USER

# 4. Once complete, generate metrics
python generate_sample_metrics.py --output sample_metrics.csv
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

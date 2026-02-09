#!/usr/bin/env python3
"""
Download CLIP sample data files from Flow.bio.

Usage:
    python download_files.py                           # Download UMICollapse logs (default)
    python download_files.py --debug                   # Show first sample's data records
    python download_files.py --regex ".*subtype.*tsv" --dir data_subtype --json subtype_data.json
    
Examples:
    # Download UMICollapse log files (default)
    python download_files.py
    
    # Download subtype TSV files
    python download_files.py --regex ".*summary_subtype_premapadjusted\\.tsv" --dir data_subtype --json subtype_data.json
    
    # Download gene TSV files  
    python download_files.py --regex ".*gene_premapadjusted\\.tsv" --dir data_gene --json gene_data.json
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import (
    load_credentials,
    get_access_token,
    get_all_public_samples,
    get_all_sample_data,
    filter_samples_by_type,
    compile_filename_regexes,
    filter_by_regex,
    dedupe_latest_by_filename,
    download_file,
    flatten_data_dir,
)


# =============================================================================
# Configuration (defaults, can be overridden via command line)
# =============================================================================

BASE_URL = "https://api.flow.bio"

# Default filename regex
DEFAULT_REGEX = r"(.*unique_genome.dedup_UMICollapse.log)"

# Default output paths
DEFAULT_JSON = "filtered_data.json"
DEFAULT_DATA_DIR = "data"

# Processing settings
MAX_WORKERS = 8          # Parallel sample processing
MAX_DOWNLOAD_WORKERS = 4  # Parallel downloads

# Desired sample type
SAMPLE_TYPE = "CLIP"

# Credentials file path
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download CLIP sample data files from Flow.bio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download UMICollapse log files (default)
  python download_files.py
  
  # Download subtype TSV files
  python download_files.py --regex ".*summary_subtype_premapadjusted\\.tsv" --dir data_subtype --json subtype_data.json
  
  # Download gene TSV files
  python download_files.py --regex ".*gene_premapadjusted\\.tsv" --dir data_gene --json gene_data.json
  
  # Force fresh API fetch (ignore cached JSON)
  python download_files.py --fresh
        """
    )
    parser.add_argument(
        "--regex", "-r",
        default=DEFAULT_REGEX,
        help=f"Filename regex pattern to match (default: {DEFAULT_REGEX})"
    )
    parser.add_argument(
        "--dir", "-d",
        default=DEFAULT_DATA_DIR,
        help=f"Output directory for downloaded files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--json", "-j",
        default=DEFAULT_JSON,
        help=f"JSON file for caching file metadata (default: {DEFAULT_JSON})"
    )
    parser.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="Force fresh API fetch, ignore cached JSON"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all data records from the first CLIP sample"
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate SLURM job scripts instead of downloading directly"
    )
    parser.add_argument(
        "--slurm-dir",
        default="slurm_jobs",
        help="Directory for SLURM job scripts (default: slurm_jobs)"
    )
    parser.add_argument(
        "--no-sample-id-prefix",
        action="store_true",
        help="Don't prefix filenames with sample_id (not recommended)"
    )
    return parser.parse_args()


# =============================================================================
# Sample Processing
# =============================================================================

def process_sample(
    session: requests.Session,
    compiled_patterns: List,
    sample: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Process a single sample: fetch data, filter, and return matching files."""
    sample_id = str(
        sample.get("id")
        or sample.get("sample_id")
        or sample.get("uid")
        or sample.get("uuid")
        or ""
    )
    sample_name = sample.get("name") or ""
    
    if not sample_id:
        return ("", [])
    
    try:
        data_items = get_all_sample_data(session, sample_id)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status is not None and 500 <= status < 600:
            return (f"Skipping sample {sample_id}: server error {status}", [])
        raise

    total_count = len(data_items)
    if total_count < 10:
        return (f"Skipping sample {sample_id}: only {total_count} data records (<10)", [])

    filtered_items = filter_by_regex(data_items, compiled_patterns)
    unique_names = set((it.get("filename") or it.get("name") or "") for it in filtered_items)
    
    if len(unique_names) < len(filtered_items):
        kept_items = dedupe_latest_by_filename(filtered_items)
    else:
        kept_items = filtered_items

    results = [
        {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "file": d,
        }
        for d in kept_items
    ]
    
    msg = f"Completed sample {sample_id} ({sample_name}): total={total_count}, matched={len(filtered_items)}, kept={len(kept_items)}"
    return (msg, results)


def debug_first_sample(session: requests.Session) -> None:
    """Debug function to show all data records from the first CLIP sample."""
    print("=== DEBUG: First CLIP sample data records ===")
    
    samples = get_all_public_samples(session, sample_type=SAMPLE_TYPE)
    samples = filter_samples_by_type(samples, SAMPLE_TYPE)
    
    if not samples:
        print("No CLIP samples found")
        return
    
    first_sample = samples[0]
    sample_id = str(
        first_sample.get("id")
        or first_sample.get("sample_id")
        or first_sample.get("uid")
        or first_sample.get("uuid")
        or ""
    )
    sample_name = first_sample.get("name") or ""
    
    print(f"Sample ID: {sample_id}")
    print(f"Sample Name: {sample_name}")
    print()
    
    try:
        data_items = get_all_sample_data(session, sample_id)
        print(f"Found {len(data_items)} data records")
        print()
        
        for item in data_items:
            print(f"filename: {item.get('filename')}")
    except Exception as e:
        print(f"Error fetching data for sample {sample_id}: {e}")


# =============================================================================
# SLURM job generation
# =============================================================================

def generate_slurm_jobs(all_data: List[Dict[str, Any]], data_dir: str, 
                        slurm_dir: str, include_sample_id: bool = True) -> None:
    """
    Generate SLURM job scripts for downloading files on a cluster.
    
    Creates:
    - Individual job scripts for each download
    - A master script to submit all jobs
    - A download helper script used by each job
    """
    os.makedirs(slurm_dir, exist_ok=True)
    
    # Create the download helper script
    helper_script = os.path.join(slurm_dir, "download_single.py")
    with open(helper_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""Helper script to download a single file from Flow.bio."""
import argparse
import os
import sys
import time
import random

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

def download_file(url, dest_path, max_retries=5):
    """Download a file with retries."""
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=300) as resp:
                if 200 <= resp.status_code < 300:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                    return True
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(backoff + random.uniform(0, backoff/2))
                    backoff = min(backoff * 2, 30)
                    continue
                print(f"HTTP {resp.status_code}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    
    if os.path.exists(args.dest) and os.path.getsize(args.dest) > 0:
        print(f"Exists: {args.dest}")
        sys.exit(0)
    
    if download_file(args.url, args.dest):
        print(f"Downloaded: {args.dest}")
        sys.exit(0)
    else:
        print(f"Failed: {args.dest}", file=sys.stderr)
        sys.exit(1)
''')
    os.chmod(helper_script, 0o755)
    
    # Generate individual job scripts
    job_files = []
    from urllib.parse import quote
    
    for i, record in enumerate(all_data):
        file_obj = record.get("file", {})
        sample_id = str(record.get("sample_id") or "").strip()
        data_id = str(file_obj.get("id") or "").strip()
        original_filename = str(file_obj.get("filename") or file_obj.get("name") or "").strip()
        
        if not data_id or not original_filename:
            continue
        
        if include_sample_id and sample_id:
            filename = f"{sample_id}_{os.path.basename(original_filename)}"
        else:
            filename = os.path.basename(original_filename)
        
        url = f"https://app.flow.bio/files/downloads/{quote(data_id)}/{quote(original_filename)}"
        dest_path = os.path.join(data_dir, filename)
        
        job_name = f"dl_{i:05d}"
        job_script = os.path.join(slurm_dir, f"{job_name}.sh")
        
        with open(job_script, "w") as f:
            f.write(f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_dir}/logs/{job_name}.out
#SBATCH --error={slurm_dir}/logs/{job_name}.err
#SBATCH --time=00:30:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

python3 {os.path.abspath(helper_script)} \\
    --url "{url}" \\
    --dest "{os.path.abspath(dest_path)}"
''')
        job_files.append(job_script)
    
    # Create logs directory
    os.makedirs(os.path.join(slurm_dir, "logs"), exist_ok=True)
    
    # Create master submission script
    master_script = os.path.join(slurm_dir, "submit_all.sh")
    with open(master_script, "w") as f:
        f.write(f'''#!/bin/bash
# Submit all download jobs to SLURM
# Generated by download_files.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting {len(job_files)} download jobs..."

for job in dl_*.sh; do
    sbatch "$job"
    sleep 0.1  # Small delay to avoid overwhelming scheduler
done

echo "All jobs submitted. Monitor with: squeue -u $USER"
''')
    os.chmod(master_script, 0o755)
    
    # Create array job scripts (split into batches to respect job limits)
    max_array = 500  # Safe default below most admin limits
    n_jobs = len(job_files)
    n_batches = (n_jobs + max_array - 1) // max_array
    
    batch_scripts = []
    for batch in range(n_batches):
        start = batch * max_array
        end = min(start + max_array, n_jobs) - 1
        batch_script = os.path.join(slurm_dir, f"submit_batch_{batch}.sh")
        
        with open(batch_script, "w") as f:
            f.write(f'''#!/bin/bash
#SBATCH --job-name=flow_dl_{batch}
#SBATCH --output={slurm_dir}/logs/array_%A_%a.out
#SBATCH --error={slurm_dir}/logs/array_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{end - start}%50

# Batch {batch + 1}/{n_batches} (jobs {start}-{end})
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IDX=$(( SLURM_ARRAY_TASK_ID + {start} ))
JOB_SCRIPT=$(ls dl_*.sh | sed -n "$(( IDX + 1 ))p")

if [ -n "$JOB_SCRIPT" ]; then
    bash "$JOB_SCRIPT"
else
    echo "No job script found for index $IDX"
    exit 1
fi
''')
        os.chmod(batch_script, 0o755)
        batch_scripts.append(batch_script)
    
    # Create wrapper script that submits batches with dependencies
    submit_script = os.path.join(slurm_dir, "submit_all_batches.sh")
    with open(submit_script, "w") as f:
        f.write(f'''#!/bin/bash
# Submit all {n_batches} batch(es) of download jobs
# Each batch waits for the previous one to complete
# Total: {n_jobs} downloads in batches of {max_array}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PREV_JOB=""
''')
        for i, bs in enumerate(batch_scripts):
            bs_name = os.path.basename(bs)
            f.write(f'''
echo "Submitting batch {i + 1}/{n_batches}..."
if [ -z "$PREV_JOB" ]; then
    PREV_JOB=$(sbatch --parsable {bs_name})
else
    PREV_JOB=$(sbatch --parsable --dependency=afterany:$PREV_JOB {bs_name})
fi
echo "  Job ID: $PREV_JOB"
''')
        f.write(f'''
echo ""
echo "All {n_batches} batch(es) submitted. Monitor with: squeue -u $USER"
''')
    os.chmod(submit_script, 0o755)
    
    # Also keep a single-batch script for small jobs
    if n_batches == 1:
        # Symlink for convenience
        array_script = os.path.join(slurm_dir, "submit_array.sh")
        import shutil
        shutil.copy2(batch_scripts[0], array_script)
    
    print(f"\nGenerated {n_jobs} SLURM job scripts in '{slurm_dir}/'")
    print(f"Helper script: {helper_script}")
    if n_batches > 1:
        print(f"Split into {n_batches} batches of up to {max_array} jobs each")
        print(f"Submit all batches: bash {submit_script}")
    else:
        print(f"Submit with: sbatch {os.path.join(slurm_dir, 'submit_array.sh')}")
    print(f"Or submit individual jobs: bash {master_script}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    
    # Use command-line arguments
    filename_regex = args.regex
    data_dir = args.dir
    json_file = args.json
    
    username, password = load_credentials(CREDENTIALS_PATH)
    compiled_patterns = compile_filename_regexes(filename_regex)

    access_token = get_access_token(username, password)
    print("Authenticated and obtained access token")

    # Debug mode
    if args.debug:
        with requests.Session() as session:
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            debug_first_sample(session)
        return

    print(f"\nConfiguration:")
    print(f"  Regex:  {filename_regex}")
    print(f"  Dir:    {data_dir}")
    print(f"  JSON:   {json_file}")
    print()

    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        all_data: List[Dict[str, Any]] = []
        
        # Try loading from existing JSON first (unless --fresh)
        if not args.fresh and json_file and os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    all_data = loaded
                    print(f"Loaded {len(all_data)} records from {json_file}; skipping API fetch")
                    print("(use --fresh to force new API fetch)")
                else:
                    print(f"Input JSON {json_file} is not a list; ignoring")
            except Exception as e:
                print(f"Failed to read {json_file}: {e}; proceeding with API fetch")
        
        samples = None
        if not all_data:
            print("Fetching public samples...")
            samples = get_all_public_samples(session, sample_type=SAMPLE_TYPE)
            samples = filter_samples_by_type(samples, SAMPLE_TYPE)
            print(f"Found {len(samples)} public samples of type {SAMPLE_TYPE}")

            # Process samples in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(process_sample, session, compiled_patterns, s)
                    for s in samples
                ]
                for fut in as_completed(futures):
                    msg, results = fut.result()
                    if msg:
                        print(msg)
                    if results:
                        all_data.extend(results)

        # Summary
        if samples is None:
            sample_ids = {str(r.get("sample_id") or "") for r in all_data}
            samples_count = len([sid for sid in sample_ids if sid])
        else:
            samples_count = len(samples)
        
        print(f"Fetched {len(all_data)} total data files across {samples_count} samples after filtering")
        print(f"Filename regex(es): {filename_regex}")

        # Save records to JSON
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(all_data)} records to {json_file}")
        except Exception as e:
            print(f"Failed to write output JSON to {json_file}: {e}")
        
        # Handle SLURM mode or direct download
        include_sample_id = not args.no_sample_id_prefix
        
        if args.slurm:
            # Generate SLURM job scripts instead of downloading
            generate_slurm_jobs(all_data, data_dir, args.slurm_dir, include_sample_id)
        else:
            # Download files directly
            print(f"Starting downloads to '{data_dir}'...")
            os.makedirs(data_dir, exist_ok=True)
            success = 0
            attempted = 0
            
            with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
                futures = [
                    executor.submit(download_file, session, r, data_dir, 
                                   include_sample_id=include_sample_id)
                    for r in all_data
                ]
                for fut in as_completed(futures):
                    msg, ok = fut.result()
                    attempted += 1
                    if ok:
                        success += 1
                    if not ok or (attempted % 25 == 0):
                        print(msg)
            
            print(f"Downloads complete: {success}/{attempted} succeeded")

        # Flatten nested directories
        moved, skipped = flatten_data_dir(data_dir)
        print(f"Flattened data dir: moved {moved} files, skipped {skipped}")


if __name__ == "__main__":
    main()

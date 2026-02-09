#!/usr/bin/env python3
"""
Fast download of additional files for existing samples.

Uses sample IDs from filtered_data.json (already fetched) to download
additional file types without re-fetching all public samples.

Usage:
    python download_additional_files.py --regex ".*summary_subtype_premapadjusted\.tsv" --dir data_subtype
"""

import argparse
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import (
    load_credentials,
    get_access_token,
    get_all_sample_data,
    compile_filename_regexes,
    filter_by_regex,
    dedupe_latest_by_filename,
    download_file,
)

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
SOURCE_JSON = "filtered_data.json"


def generate_slurm_jobs(all_files, data_dir, slurm_dir, include_sample_id=True):
    """Generate SLURM job scripts for downloading files."""
    from urllib.parse import quote
    
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(os.path.join(slurm_dir, "logs"), exist_ok=True)
    
    job_files = []
    for i, record in enumerate(all_files):
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

curl -L -o "{os.path.abspath(dest_path)}" "{url}" || exit 1
echo "Downloaded: {dest_path}"
''')
        job_files.append(job_script)
    
    # Create array job scripts (split into batches to respect job limits)
    max_array = 500
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
#SBATCH --array=0-{end - start}%50

# Batch {batch + 1}/{n_batches} (jobs {start}-{end})
cd "$(dirname "$0")"
IDX=$(( SLURM_ARRAY_TASK_ID + {start} ))
JOB_SCRIPT=$(ls dl_*.sh | sed -n "$(( IDX + 1 ))p")
[ -n "$JOB_SCRIPT" ] && bash "$JOB_SCRIPT"
''')
        os.chmod(batch_script, 0o755)
        batch_scripts.append(batch_script)
    
    # Create wrapper to submit batches with dependencies
    submit_script = os.path.join(slurm_dir, "submit_all_batches.sh")
    with open(submit_script, "w") as f:
        f.write(f'''#!/bin/bash
# Submit {n_batches} batch(es) of downloads ({n_jobs} total, max {max_array} per batch)
cd "$(dirname "$0")"
PREV=""
''')
        for i, bs in enumerate(batch_scripts):
            bs_name = os.path.basename(bs)
            f.write(f'''echo "Batch {i + 1}/{n_batches}..."
if [ -z "$PREV" ]; then PREV=$(sbatch --parsable {bs_name})
else PREV=$(sbatch --parsable --dependency=afterany:$PREV {bs_name}); fi
echo "  Job: $PREV"
''')
        f.write(f'echo "All {n_batches} batch(es) submitted."\n')
    os.chmod(submit_script, 0o755)
    
    if n_batches == 1:
        import shutil
        shutil.copy2(batch_scripts[0], os.path.join(slurm_dir, "submit_array.sh"))
    
    print(f"Generated {n_jobs} SLURM job scripts in '{slurm_dir}/'")
    if n_batches > 1:
        print(f"Split into {n_batches} batches of up to {max_array}. Submit: bash {submit_script}")
    else:
        print(f"Submit with: sbatch {os.path.join(slurm_dir, 'submit_array.sh')}")


def main():
    parser = argparse.ArgumentParser(description="Download additional files for existing samples")
    parser.add_argument("--regex", "-r", required=True, help="Filename regex pattern")
    parser.add_argument("--dir", "-d", required=True, help="Output directory")
    parser.add_argument("--source", "-s", default=SOURCE_JSON, help="Source JSON with sample IDs")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers")
    parser.add_argument("--slurm", action="store_true", help="Generate SLURM jobs instead of downloading")
    parser.add_argument("--slurm-dir", default="slurm_jobs", help="SLURM job directory")
    parser.add_argument("--no-sample-id-prefix", action="store_true", help="Don't prefix filenames with sample_id")
    args = parser.parse_args()
    
    # Load existing sample IDs
    if not os.path.exists(args.source):
        print(f"Source file not found: {args.source}")
        sys.exit(1)
    
    with open(args.source) as f:
        data = json.load(f)
    
    sample_ids = list(set(str(r.get("sample_id", "")) for r in data if r.get("sample_id")))
    print(f"Found {len(sample_ids)} unique sample IDs in {args.source}")
    
    # Authenticate
    username, password = load_credentials(CREDENTIALS_PATH)
    access_token = get_access_token(username, password)
    print("Authenticated successfully")
    
    compiled_patterns = compile_filename_regexes(args.regex)
    print(f"Regex: {args.regex}")
    print(f"Output: {args.dir}")
    
    os.makedirs(args.dir, exist_ok=True)
    
    # Fetch and filter files for each sample
    all_files = []
    
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})
        
        def process_sample(sample_id):
            try:
                data_items = get_all_sample_data(session, sample_id)
                filtered = filter_by_regex(data_items, compiled_patterns)
                if len(set(f.get("filename") for f in filtered)) < len(filtered):
                    filtered = dedupe_latest_by_filename(filtered)
                return sample_id, filtered
            except Exception as e:
                return sample_id, []
        
        print(f"\nFetching file lists for {len(sample_ids)} samples...")
        completed = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, sid): sid for sid in sample_ids}
            for future in as_completed(futures):
                sample_id, files = future.result()
                completed += 1
                if files:
                    for f in files:
                        all_files.append({"sample_id": sample_id, "file": f})
                if completed % 100 == 0:
                    print(f"  Processed {completed}/{len(sample_ids)} samples, found {len(all_files)} files...")
        
        print(f"\nFound {len(all_files)} matching files")
        
        if not all_files:
            print("No files matched the regex pattern!")
            return
        
        include_sample_id = not args.no_sample_id_prefix
        
        if args.slurm:
            # Generate SLURM job scripts
            generate_slurm_jobs(all_files, args.dir, args.slurm_dir, include_sample_id)
        else:
            # Download files directly
            print(f"\nDownloading to {args.dir}...")
            success = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(download_file, session, r, args.dir, 
                                   include_sample_id=include_sample_id) 
                    for r in all_files
                ]
                for i, future in enumerate(as_completed(futures), 1):
                    msg, ok = future.result()
                    if ok:
                        success += 1
                    if not ok or i % 50 == 0:
                        print(f"  [{i}/{len(all_files)}] {msg}")
            
            print(f"\nDone! Downloaded {success}/{len(all_files)} files to {args.dir}")


if __name__ == "__main__":
    main()

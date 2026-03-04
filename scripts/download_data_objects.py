#!/usr/bin/env python3
"""
Download data objects produced by a specific pipeline process from Flow.bio.

This script finds outputs from multi-sample integration processes like
TETRANSCRIPTS by navigating:
  project → executions → process_executions → downstream_data

This is the right approach for data objects that aren't linked to individual
samples but are produced as part of a pipeline execution within a project.

Output filenames are prefixed with their execution ID to avoid collisions,
since each execution produces identically-named files.

Usage:
    # Download all TETRANSCRIPTS outputs from the RBP ENCODE Data project
    python download_data_objects.py \\
        --project 523943332699993118 \\
        --process TETRANSCRIPTS \\
        --dir data_te

    # Filter by filename pattern
    python download_data_objects.py \\
        --project 523943332699993118 \\
        --process TETRANSCRIPTS \\
        --regex ".*\\.cntTable" \\
        --dir data_te

    # SLURM mode
    python download_data_objects.py \\
        --project 523943332699993118 \\
        --process TETRANSCRIPTS \\
        --dir data_te --slurm
"""

import argparse
import os
import sys
import json
import re
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import (
    load_credentials,
    get_access_token,
    download_file,
)

BASE_URL = "https://app.flow.bio/api"
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
PER_PAGE = 100
MAX_WORKERS = 4
REQUEST_DELAY_SEC = 0.1


# ── Pagination ──────────────────────────────────────────────────────────────

def paginate_items(session, url, params=None, items_key=None):
    """Paginate through API results with progress logging."""
    params = dict(params or {})
    page = 1
    total_yielded = 0
    per_page = params.pop("count", PER_PAGE)
    while True:
        resp = session.get(url, params={**params, "page": page, "count": per_page}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if items_key and items_key in data:
            items = data[items_key]
        else:
            items = (
                data.get("items")
                or data.get("results")
                or data.get("data")
                or data.get("executions")
                or data.get("samples")
                or []
            )
        if not items:
            break
        yield from items
        total_yielded += len(items)
        if page % 5 == 0:
            print(f"  ... fetched {total_yielded} items (page {page})")
        page += 1
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)


# ── Core: find process outputs via project executions ───────────────────────

def find_process_outputs(session, project_id, process_name_filter):
    """
    Find all data objects produced by a specific process across all executions
    in a project.

    Path: /projects/{id}/executions → /executions/{id} → process_executions
          → filter by process_name → downstream_data
    """
    process_filter_lower = process_name_filter.lower()

    # 1. List all executions in the project
    print(f"Listing executions for project {project_id}...")
    executions = list(paginate_items(
        session,
        f"{BASE_URL}/projects/{project_id}/executions",
        items_key="executions",
    ))
    print(f"Found {len(executions)} executions")

    # 2. For each execution, fetch details and find matching process steps
    all_outputs = []
    for i, ex_summary in enumerate(executions, 1):
        exec_id = ex_summary.get("id")
        if not exec_id:
            continue

        if i % 10 == 0 or i == 1:
            print(f"  Checking execution {i}/{len(executions)} (id={exec_id})...")

        try:
            resp = session.get(f"{BASE_URL}/executions/{exec_id}", timeout=120)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status and 500 <= status < 600:
                print(f"  Skipping execution {exec_id}: server error {status}")
                continue
            raise
        except requests.exceptions.RequestException as e:
            print(f"  Skipping execution {exec_id}: {e}")
            continue

        edata = resp.json()
        proc_execs = edata.get("process_executions") or []

        for proc in proc_execs:
            proc_name = proc.get("process_name") or proc.get("name") or ""
            if process_filter_lower not in proc_name.lower():
                continue

            downstream = proc.get("downstream_data") or []
            for data_obj in downstream:
                # Attach execution context for traceability
                data_obj["_execution_id"] = exec_id
                data_obj["_execution_created"] = ex_summary.get("created")
                data_obj["_execution_pipeline"] = ex_summary.get("pipeline_name", "")
                data_obj["_process_name"] = proc_name
                all_outputs.append(data_obj)

        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)

    return all_outputs


# ── Record formatting ──────────────────────────────────────────────────────

def format_records(items):
    """Convert data objects into download records compatible with flow_api.download_file.

    Filenames are prefixed with execution ID and date for uniqueness and
    traceability: {execution_id}_{YYYY-MM-DD}_{filename}
    """
    records = []
    for item in items:
        data_id = str(item.get("id") or "").strip()
        filename = str(item.get("filename") or "").strip()
        exec_id = str(item.get("_execution_id") or "").strip()
        if not data_id or not filename:
            continue
        # Build date string from execution created timestamp
        date_str = ""
        exec_created = item.get("_execution_created")
        if exec_created and isinstance(exec_created, (int, float)):
            try:
                date_str = datetime.fromtimestamp(exec_created, tz=timezone.utc).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                pass
        # Prefix with execution ID and date for uniqueness
        if exec_id and date_str:
            prefixed_filename = f"{exec_id}_{date_str}_{filename}"
        elif exec_id:
            prefixed_filename = f"{exec_id}_{filename}"
        else:
            prefixed_filename = filename
        records.append({
            "sample_id": "",  # these are standalone objects
            "sample_name": "",
            "file": {
                "id": data_id,
                "filename": prefixed_filename,
            },
            "_execution_id": exec_id,
            "_process_name": item.get("_process_name", ""),
        })
    return records


# ── SLURM generation ───────────────────────────────────────────────────────

def generate_slurm_jobs(records, data_dir, slurm_dir):
    """Generate SLURM job scripts for downloading data objects."""
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(os.path.join(slurm_dir, "logs"), exist_ok=True)

    job_files = []
    for i, record in enumerate(records):
        file_obj = record.get("file", {})
        data_id = str(file_obj.get("id") or "").strip()
        filename = str(file_obj.get("filename") or "").strip()  # already prefixed with exec id
        if not data_id or not filename:
            continue

        url = f"https://app.flow.bio/files/downloads/{quote(data_id)}/{quote(filename)}"
        dest_path = os.path.join(os.path.abspath(data_dir), filename)

        job_name = f"dl_{i:05d}"
        job_script = os.path.join(slurm_dir, f"{job_name}.sh")
        with open(job_script, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={os.path.abspath(slurm_dir)}/logs/{job_name}.out
#SBATCH --error={os.path.abspath(slurm_dir)}/logs/{job_name}.err
#SBATCH --time=00:30:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

curl -L -o "{dest_path}" "{url}" || exit 1
echo "Downloaded: {filename}"
""")
        job_files.append(job_script)

    n_jobs = len(job_files)
    if n_jobs == 0:
        print("No jobs to generate.")
        return

    abs_slurm_dir = os.path.abspath(slurm_dir)
    submit_script = os.path.join(slurm_dir, "submit_array.sh")
    with open(submit_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=flow_dl
#SBATCH --output={abs_slurm_dir}/logs/array_%A_%a.out
#SBATCH --error={abs_slurm_dir}/logs/array_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=1G
#SBATCH --array=0-{n_jobs - 1}%50

cd {abs_slurm_dir}
JOB_SCRIPT=$(ls dl_*.sh | sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p")
[ -n "$JOB_SCRIPT" ] && bash "$JOB_SCRIPT"
""")
    os.chmod(submit_script, 0o755)
    print(f"Generated {n_jobs} SLURM job scripts in '{slurm_dir}/'")
    print(f"Submit with: sbatch {submit_script}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download data objects from a Flow.bio project by process name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  1. Lists all pipeline executions in the project
  2. For each execution, finds process steps matching --process
  3. Collects their downstream_data (output files)
  4. Deduplicates by filename (keeps most recent)
  5. Downloads or generates SLURM jobs

Examples:
  python download_data_objects.py \\
      --project 523943332699993118 \\
      --process TETRANSCRIPTS \\
      --dir data_te

  python download_data_objects.py \\
      --project 523943332699993118 \\
      --process TETRANSCRIPTS \\
      --regex ".*\\.cntTable" \\
      --dir data_te --slurm
""",
    )
    parser.add_argument(
        "--project", required=True,
        help="Project ID to search within",
    )
    parser.add_argument(
        "--process", "-p", required=True,
        help="Process name substring to match (e.g. TETRANSCRIPTS). Case-insensitive.",
    )
    parser.add_argument(
        "--regex", "-r", default=None,
        help="Regex to further filter output filenames locally",
    )
    parser.add_argument(
        "--dir", "-d", required=True,
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help="Generate SLURM job scripts instead of downloading",
    )
    parser.add_argument(
        "--slurm-dir", default="slurm_jobs",
        help="Directory for SLURM job scripts (default: slurm_jobs)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=MAX_WORKERS,
        help=f"Parallel download workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--json", default=None,
        help="Save matched data records to a JSON file (optional)",
    )
    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Configuration:")
    print(f"  Project:   {args.project}")
    print(f"  Process:   {args.process}")
    if args.regex:
        print(f"  Regex:     {args.regex}")
    print(f"  Dir:       {args.dir}")
    print()

    # Authenticate
    username, password = load_credentials(CREDENTIALS_PATH)
    access_token = get_access_token(username, password)
    print("Authenticated successfully")

    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        # 1. Find process outputs
        outputs = find_process_outputs(session, args.project, args.process)
        print(f"\nFound {len(outputs)} output files from '{args.process}' processes")

        if not outputs:
            print("No outputs found. Check the project ID and process name.")
            return

        # 2. Apply regex filter
        if args.regex:
            pattern = re.compile(args.regex, re.IGNORECASE)
            outputs = [o for o in outputs if pattern.search(o.get("filename") or "")]
            print(f"After regex filter: {len(outputs)} files")

        # 3. Format and report
        records = format_records(outputs)
        print(f"\n{len(records)} files to download:")
        for r in records:
            print(f"  {r['file']['filename']}")

        # 4. Save JSON
        if args.json:
            with open(args.json, "w") as f:
                json.dump(records, f, indent=2)
            print(f"\nSaved records to {args.json}")

        # 5. Download or SLURM
        if args.slurm:
            generate_slurm_jobs(records, args.dir, args.slurm_dir)
        else:
            os.makedirs(args.dir, exist_ok=True)
            print(f"\nDownloading to {args.dir}...")
            success = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(download_file, session, r, args.dir,
                                    include_sample_id=False)
                    for r in records
                ]
                for i, fut in enumerate(as_completed(futures), 1):
                    msg, ok = fut.result()
                    if ok:
                        success += 1
                    print(f"  [{i}/{len(records)}] {msg}")

            print(f"\nDone! Downloaded {success}/{len(records)} files to {args.dir}")


if __name__ == "__main__":
    main()

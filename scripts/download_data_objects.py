#!/usr/bin/env python3
"""
Download data objects directly from the Flow.bio /data/search API.

Unlike download_files.py (which iterates samples), this script searches for
data objects by process_execution_name and/or filename pattern. This is useful
for downloading outputs from multi-sample integration processes (e.g.
TETRANSCRIPTS) that may not be directly linked to individual samples.

Deduplication: when the same sample has been run through a process multiple
times, multiple data objects with the same filename will exist. By default,
only the most recently created object per (sample_id, filename) pair is kept.

Usage:
    # Download all TETRANSCRIPTS outputs
    python download_data_objects.py --process TETRANSCRIPTS --dir data_te

    # Also filter by filename pattern
    python download_data_objects.py --process TETRANSCRIPTS --filename "counts" --dir data_te

    # Generate SLURM jobs instead of downloading directly
    python download_data_objects.py --process TETRANSCRIPTS --dir data_te --slurm

    # Keep all versions (skip deduplication)
    python download_data_objects.py --process TETRANSCRIPTS --dir data_te --keep-all
"""

import argparse
import os
import sys
import json
import time
import re
from typing import Dict, List, Any, Tuple
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

def paginate_items(session, url, params=None):
    """Paginate through API results."""
    params = dict(params or {})
    page = 1
    per_page = params.pop("count", PER_PAGE)
    while True:
        resp = session.get(url, params={**params, "page": page, "count": per_page}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        items = (
            data.get("items")
            or data.get("results")
            or data.get("data")
            or data.get("samples")
            or []
        )
        if not items:
            break
        yield from items
        page += 1
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)


# ── Data search ─────────────────────────────────────────────────────────────

def search_data_objects(session, process_name=None, filename=None):
    """Search /data/search with optional filters."""
    params = {}
    if process_name:
        params["process_execution_name"] = process_name
    if filename:
        params["filename"] = filename
    items = list(paginate_items(session, f"{BASE_URL}/data/search", params))
    return items


# ── Timestamp parsing ───────────────────────────────────────────────────────

def _to_timestamp(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime
            if value.endswith("Z"):
                value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return 0.0
    return 0.0


# ── Deduplication ───────────────────────────────────────────────────────────

def dedupe_by_sample_and_filename(items):
    """Keep only the most recently created item per (sample_id, filename).

    If a data object has no sample linkage, sample_id is treated as empty
    so all such objects are deduped by filename alone.
    """
    best = {}      # (sample_id, filename) -> item
    best_ts = {}   # (sample_id, filename) -> timestamp

    for item in items:
        fname = item.get("filename") or item.get("name") or ""
        if not fname:
            continue

        # Extract sample linkage
        sample = item.get("sample") or {}
        sample_id = str(
            item.get("sample_id")
            or (sample.get("id") if isinstance(sample, dict) else "")
            or ""
        ).strip()

        ts = _to_timestamp(
            item.get("created") or item.get("created_at") or item.get("timestamp")
        )

        key = (sample_id, fname)
        if ts >= best_ts.get(key, -1.0):
            best_ts[key] = ts
            best[key] = item

    return list(best.values())


def dedupe_by_filename_only(items):
    """Keep only the most recently created item per filename (ignoring sample)."""
    best = {}
    best_ts = {}
    for item in items:
        fname = item.get("filename") or item.get("name") or ""
        if not fname:
            continue
        ts = _to_timestamp(
            item.get("created") or item.get("created_at") or item.get("timestamp")
        )
        if ts >= best_ts.get(fname, -1.0):
            best_ts[fname] = ts
            best[fname] = item
    return list(best.values())


# ── Record formatting ──────────────────────────────────────────────────────

def format_records(items, include_sample_id=True):
    """Convert raw data objects into download records compatible with flow_api.download_file."""
    records = []
    for item in items:
        data_id = str(item.get("id") or "").strip()
        filename = str(item.get("filename") or item.get("name") or "").strip()
        if not data_id or not filename:
            continue

        sample = item.get("sample") or {}
        sample_id = str(
            item.get("sample_id")
            or (sample.get("id") if isinstance(sample, dict) else "")
            or ""
        ).strip()

        records.append({
            "sample_id": sample_id,
            "sample_name": sample.get("name", "") if isinstance(sample, dict) else "",
            "file": {
                "id": data_id,
                "filename": filename,
                "pipeline_name": item.get("pipeline_name", ""),
                "process_execution_name": item.get("process_execution_name", ""),
            },
        })
    return records


# ── SLURM generation ───────────────────────────────────────────────────────

def generate_slurm_jobs(records, data_dir, slurm_dir, include_sample_id=True):
    """Generate SLURM job scripts for downloading data objects."""
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(os.path.join(slurm_dir, "logs"), exist_ok=True)

    job_files = []
    for i, record in enumerate(records):
        file_obj = record.get("file", {})
        sample_id = str(record.get("sample_id") or "").strip()
        data_id = str(file_obj.get("id") or "").strip()
        original_filename = str(file_obj.get("filename") or "").strip()
        if not data_id or not original_filename:
            continue

        if include_sample_id and sample_id:
            filename = f"{sample_id}_{os.path.basename(original_filename)}"
        else:
            filename = os.path.basename(original_filename)

        url = f"https://app.flow.bio/files/downloads/{quote(data_id)}/{quote(original_filename)}"
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

    # Array job script
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
        description="Download data objects from Flow.bio by process execution name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all TETRANSCRIPTS outputs
  python download_data_objects.py --process TETRANSCRIPTS --dir data_te

  # Filter by filename substring too
  python download_data_objects.py --process TETRANSCRIPTS --filename counts --dir data_te

  # Filter results with a local regex on the filename
  python download_data_objects.py --process TETRANSCRIPTS --regex ".*\\.tsv" --dir data_te

  # Keep all versions (skip deduplication)
  python download_data_objects.py --process TETRANSCRIPTS --dir data_te --keep-all

  # Generate SLURM jobs
  python download_data_objects.py --process TETRANSCRIPTS --dir data_te --slurm
""",
    )
    parser.add_argument(
        "--process", "-p", required=True,
        help="process_execution_name to search for (e.g. TETRANSCRIPTS, CLIPSEQ:PEKA)",
    )
    parser.add_argument(
        "--filename", "-f", default=None,
        help="Server-side filename substring filter (sent to /data/search)",
    )
    parser.add_argument(
        "--regex", "-r", default=None,
        help="Local regex to further filter filenames after fetching from API",
    )
    parser.add_argument(
        "--dir", "-d", required=True,
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep all versions instead of deduplicating by (sample_id, filename)",
    )
    parser.add_argument(
        "--no-sample-id-prefix", action="store_true",
        help="Don't prefix filenames with sample_id",
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
    print(f"  Process:   {args.process}")
    if args.filename:
        print(f"  Filename:  {args.filename}")
    if args.regex:
        print(f"  Regex:     {args.regex}")
    print(f"  Dir:       {args.dir}")
    print(f"  Dedupe:    {'off (--keep-all)' if args.keep_all else 'most recent per (sample_id, filename)'}")
    print()

    # Authenticate
    username, password = load_credentials(CREDENTIALS_PATH)
    access_token = get_access_token(username, password)
    print("Authenticated successfully")

    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        # 1. Search for data objects
        print(f"\nSearching /data/search for process_execution_name='{args.process}'...")
        items = search_data_objects(session, process_name=args.process, filename=args.filename)
        print(f"Found {len(items)} data objects from API")

        if not items:
            print("No data objects found. Check the process_execution_name value.")
            return

        # 2. Apply local regex filter if specified
        if args.regex:
            pattern = re.compile(args.regex, re.IGNORECASE)
            items = [
                it for it in items
                if pattern.search(it.get("filename") or it.get("name") or "")
            ]
            print(f"After regex filter '{args.regex}': {len(items)} items")

        # 3. Deduplicate
        if not args.keep_all:
            before = len(items)
            items = dedupe_by_sample_and_filename(items)
            removed = before - len(items)
            if removed:
                print(f"Deduplication: kept {len(items)} (removed {removed} older duplicates)")
            else:
                print(f"Deduplication: no duplicates found ({len(items)} items)")

        # 4. Format records
        include_sample_id = not args.no_sample_id_prefix
        records = format_records(items, include_sample_id=include_sample_id)
        print(f"\n{len(records)} files to download")

        # Summarise sample linkage
        with_sample = sum(1 for r in records if r.get("sample_id"))
        without_sample = len(records) - with_sample
        if without_sample:
            print(f"  {with_sample} linked to a sample, {without_sample} standalone (no sample_id)")

        # 5. Save JSON if requested
        if args.json:
            with open(args.json, "w") as f:
                json.dump(records, f, indent=2)
            print(f"Saved records to {args.json}")

        # 6. Download or generate SLURM jobs
        if args.slurm:
            generate_slurm_jobs(records, args.dir, args.slurm_dir, include_sample_id)
        else:
            os.makedirs(args.dir, exist_ok=True)
            print(f"\nDownloading to {args.dir}...")
            success = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(download_file, session, r, args.dir,
                                    include_sample_id=include_sample_id)
                    for r in records
                ]
                for i, fut in enumerate(as_completed(futures), 1):
                    msg, ok = fut.result()
                    if ok:
                        success += 1
                    if not ok or i % 50 == 0:
                        print(f"  [{i}/{len(records)}] {msg}")

            print(f"\nDone! Downloaded {success}/{len(records)} files to {args.dir}")


if __name__ == "__main__":
    main()

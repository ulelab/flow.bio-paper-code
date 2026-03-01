#!/usr/bin/env python3
"""
Identify public CLIP samples on Flow.bio that have NO UMICollapse log file,
i.e. samples that were fetched by download_files.py but never made it into
filtered_data.json because:
  - The sample had fewer than 10 data records (pipeline barely ran)
  - No data record matched the UMICollapse regex (pipeline didn't produce one)
  - A server error occurred when fetching the sample's data

This script:
1. Fetches all public CLIP samples from the API
2. Compares against the sample IDs already in filtered_data.json
3. For each missing sample, queries the API for data records to diagnose WHY
4. Outputs a CSV + console summary with project, target, skip reason, etc.

Usage:
    python find_missing_samples.py
    python find_missing_samples.py --output missing_samples.csv --workers 8
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import (
    load_credentials,
    get_access_token,
    get_all_public_samples,
    get_all_sample_data,
    filter_samples_by_type,
    compile_filename_regexes,
    filter_by_regex,
)


# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "https://api.flow.bio"
SAMPLE_TYPE = "CLIP"
UMICOLLAPSE_REGEX = r"(.*unique_genome.dedup_UMICollapse.log)"
MIN_DATA_RECORDS = 10  # same threshold as download_files.py
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find public CLIP samples that have no UMICollapse log on Flow.bio"
    )
    parser.add_argument(
        "--json", "-j",
        default="filtered_data.json",
        help="Path to filtered_data.json (default: filtered_data.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="missing_samples.csv",
        help="Output CSV file (default: missing_samples.csv)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=8,
        help="Parallel workers for API queries (default: 8)"
    )
    return parser.parse_args()


# =============================================================================
# Helpers to extract fields from the public sample object
# =============================================================================

def get_sample_id(sample: Dict) -> str:
    return str(
        sample.get("id")
        or sample.get("sample_id")
        or sample.get("uid")
        or sample.get("uuid")
        or ""
    ).strip()


def get_sample_name(sample: Dict) -> str:
    return str(sample.get("name") or "")


def get_project(sample: Dict) -> Tuple[str, str]:
    """Return (project_name, project_id) from a public sample object."""
    proj = sample.get("project") or {}
    if isinstance(proj, dict):
        return (str(proj.get("name") or ""), str(proj.get("id") or ""))
    return ("", "")


def get_organism(sample: Dict) -> str:
    org = sample.get("organism") or {}
    if isinstance(org, dict):
        return str(org.get("name") or org.get("id") or "")
    return str(org) if org else ""


def get_purification_target(sample: Dict) -> str:
    """Extract purification target from sample metadata."""
    meta = sample.get("metadata") or {}
    if isinstance(meta, dict):
        pt = meta.get("purification_target") or {}
        if isinstance(pt, dict):
            val = pt.get("value") or pt.get("name") or ""
            if val:
                target = str(val).upper()
                return "TARDBP" if target == "TDP43" else target
        elif pt:
            target = str(pt).upper()
            return "TARDBP" if target == "TDP43" else target
    return ""


def get_experimental_method(sample: Dict) -> str:
    meta = sample.get("metadata") or {}
    if isinstance(meta, dict):
        em = meta.get("experimental_method") or {}
        if isinstance(em, dict):
            return str(em.get("value") or em.get("name") or "")
        return str(em) if em else ""
    return ""


def get_owner(sample: Dict) -> str:
    owner = sample.get("owner") or {}
    if isinstance(owner, dict):
        return str(owner.get("name") or owner.get("username") or "")
    return ""


# =============================================================================
# Diagnose why a sample has no UMICollapse log
# =============================================================================

def diagnose_sample(
    session: requests.Session,
    sample_id: str,
    compiled_patterns: List,
) -> Tuple[str, int, List[str]]:
    """
    Query the API for a sample's data records and figure out why there's
    no UMICollapse log.

    Returns:
        (reason, total_data_count, list_of_pipeline_names)
    """
    try:
        data_items = get_all_sample_data(session, sample_id)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status and 500 <= status < 600:
            return (f"server_error_{status}", 0, [])
        return (f"http_error_{status}", 0, [])
    except Exception as e:
        return (f"error: {str(e)[:80]}", 0, [])

    total_count = len(data_items)

    # Collect pipeline names for context
    pipelines = set()
    for item in data_items:
        pn = item.get("pipeline_name") or ""
        if pn:
            pipelines.add(pn)

    if total_count == 0:
        return ("no_data_records", total_count, sorted(pipelines))

    if total_count < MIN_DATA_RECORDS:
        return (f"too_few_data_records ({total_count}<{MIN_DATA_RECORDS})", total_count, sorted(pipelines))

    # Check if any match the UMICollapse regex
    matches = filter_by_regex(data_items, compiled_patterns)
    if not matches:
        # Check what files DO exist to give context
        filenames = [
            (item.get("filename") or item.get("name") or "")
            for item in data_items
        ]
        # Look for partial pipeline completion clues
        has_dedup = any("dedup" in f.lower() for f in filenames)
        has_clip = any("clip" in f.lower() or "unique_genome" in f.lower() for f in filenames)

        if has_dedup and not has_clip:
            return ("pipeline_incomplete (has dedup files but no UMICollapse)", total_count, sorted(pipelines))
        elif not has_dedup and not has_clip:
            return ("no_clip_pipeline_output", total_count, sorted(pipelines))
        else:
            return ("no_umicollapse_log (has other CLIP output)", total_count, sorted(pipelines))

    # This shouldn't happen if the sample truly isn't in filtered_data.json
    return (f"has_umicollapse_log ({len(matches)} found - check filtered_data.json)", total_count, sorted(pipelines))


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # --- Load filtered_data.json to get sample IDs already processed ---
    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.json, 'r', encoding='utf-8') as f:
        existing_records = json.load(f)

    existing_ids: Set[str] = set()
    for record in existing_records:
        sid = str(record.get("sample_id") or "").strip()
        if sid:
            existing_ids.add(sid)

    print(f"Samples in {args.json}: {len(existing_ids)}")

    # --- Authenticate and fetch all public CLIP samples ---
    username, password = load_credentials(CREDENTIALS_PATH)
    access_token = get_access_token(username, password)
    print("Authenticated successfully")

    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        print("Fetching all public CLIP samples...")
        all_samples = get_all_public_samples(session, sample_type=SAMPLE_TYPE)
        all_samples = filter_samples_by_type(all_samples, SAMPLE_TYPE)
        print(f"Total public CLIP samples: {len(all_samples)}")

        # --- Build lookup of public samples and find which are missing ---
        public_by_id: Dict[str, Dict] = {}
        for s in all_samples:
            sid = get_sample_id(s)
            if sid:
                public_by_id[sid] = s

        missing_ids = set(public_by_id.keys()) - existing_ids
        print(f"\nSamples WITHOUT UMICollapse log: {len(missing_ids)}")
        print(f"Samples WITH UMICollapse log:    {len(existing_ids)}")

        if not missing_ids:
            print("\nAll public samples have UMICollapse logs. Nothing missing.")
            return

        # --- Diagnose each missing sample (parallel API calls) ---
        print(f"\nDiagnosing {len(missing_ids)} missing samples (querying API for data records)...")
        compiled_patterns = compile_filename_regexes(UMICOLLAPSE_REGEX)

        diagnoses: Dict[str, Tuple[str, int, List[str]]] = {}
        done = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(diagnose_sample, session, sid, compiled_patterns): sid
                for sid in missing_ids
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    reason, count, pipelines = fut.result()
                    diagnoses[sid] = (reason, count, pipelines)
                except Exception as e:
                    diagnoses[sid] = (f"exception: {str(e)[:80]}", 0, [])
                done += 1
                if done % 50 == 0:
                    print(f"  Diagnosed {done}/{len(missing_ids)}...")

        print(f"  Diagnosed {done}/{len(missing_ids)} samples")

    # --- Build output rows ---
    rows = []
    for sid in sorted(missing_ids):
        sample = public_by_id[sid]
        reason, data_count, pipelines = diagnoses.get(sid, ("unknown", 0, []))

        target = get_purification_target(sample)
        project_name, project_id = get_project(sample)

        rows.append({
            "sample_id": sid,
            "sample_name": get_sample_name(sample),
            "purification_target": target,
            "project_name": project_name,
            "project_id": project_id,
            "organism": get_organism(sample),
            "method": get_experimental_method(sample),
            "owner": get_owner(sample),
            "data_record_count": data_count,
            "pipelines": "; ".join(pipelines),
            "skip_reason": reason,
            "flow_url": f"https://app.flow.bio/samples/{sid}",
        })

    # Sort by skip reason then sample name for readability
    rows.sort(key=lambda r: (r["skip_reason"], r["sample_name"]))

    # --- Write CSV ---
    fieldnames = [
        "sample_id", "sample_name", "purification_target",
        "project_name", "project_id", "organism", "method", "owner",
        "data_record_count", "pipelines", "skip_reason", "flow_url",
    ]

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} missing samples to {args.output}")

    # --- Console Summary ---
    print(f"\n{'='*70}")
    print("MISSING SAMPLES SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal public CLIP samples: {len(public_by_id)}")
    print(f"Have UMICollapse log:      {len(existing_ids)}")
    print(f"Missing UMICollapse log:   {len(rows)}")

    # By skip reason
    reason_counts = Counter(r["skip_reason"] for r in rows)
    print(f"\nBy skip reason:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")

    # By project
    project_counts = Counter(r["project_name"] or "UNKNOWN" for r in rows)
    print(f"\nBy project ({len(project_counts)} unique):")
    for proj, count in project_counts.most_common():
        print(f"  {proj}: {count}")

    # By purification target
    target_counts = Counter(r["purification_target"] or "UNKNOWN" for r in rows)
    print(f"\nBy purification target ({len(target_counts)} unique):")
    for target, count in target_counts.most_common(20):
        print(f"  {target}: {count}")
    if len(target_counts) > 20:
        remaining = len(target_counts) - 20
        print(f"  ... and {remaining} more")

    # By organism
    org_counts = Counter(r["organism"] or "UNKNOWN" for r in rows)
    print(f"\nBy organism:")
    for org, count in org_counts.most_common():
        print(f"  {org}: {count}")

    # By owner
    owner_counts = Counter(r["owner"] or "UNKNOWN" for r in rows)
    print(f"\nBy owner:")
    for owner, count in owner_counts.most_common():
        print(f"  {owner}: {count}")

    # List first few
    print(f"\nFirst 20 missing samples:")
    print(f"  {'Sample ID':<22} {'Target':<12} {'Project':<22} {'Reason'}")
    print(f"  {'-'*20:<22} {'-'*10:<12} {'-'*20:<22} {'-'*30}")
    for r in rows[:20]:
        print(f"  {r['sample_id']:<22} {r['purification_target'] or 'N/A':<12} "
              f"{(r['project_name'] or 'N/A')[:20]:<22} {r['skip_reason']}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more (see {args.output})")

    print(f"\nFull details: {args.output}")


if __name__ == "__main__":
    main()

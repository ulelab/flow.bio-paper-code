#!/usr/bin/env python3
"""
Identify samples in filtered_data.json that are missing UMICollapse log files.

Compares the sample IDs in filtered_data.json against actual downloaded files
in the data directory, and produces a summary of missing samples with metadata.

Usage:
    python find_missing_samples.py
    python find_missing_samples.py --json filtered_data.json --data-dir data --output missing_samples.csv

Also checks SLURM logs (if available) to diagnose download failures.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find samples missing UMICollapse log files"
    )
    parser.add_argument(
        "--json", "-j",
        default="filtered_data.json",
        help="Path to filtered_data.json (default: filtered_data.json)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Directory containing downloaded UMICollapse logs (default: data)"
    )
    parser.add_argument(
        "--slurm-dir", "-s",
        default=None,
        help="SLURM logs directory to check for error messages (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        default="missing_samples.csv",
        help="Output CSV file (default: missing_samples.csv)"
    )
    return parser.parse_args()


def extract_sample_id_from_filename(filename: str) -> Optional[str]:
    """Extract sample_id prefix from a downloaded file."""
    parts = filename.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[0]
    return None


def get_downloaded_sample_ids(data_dir: str) -> Set[str]:
    """Scan the data directory and return the set of sample IDs with files."""
    sample_ids = set()
    if not os.path.isdir(data_dir):
        print(f"Warning: data directory '{data_dir}' not found", file=sys.stderr)
        return sample_ids
    
    for fname in os.listdir(data_dir):
        sid = extract_sample_id_from_filename(fname)
        if sid:
            sample_ids.add(sid)
    return sample_ids


def extract_metadata_field(record: Dict, *keys) -> str:
    """Safely extract a nested metadata field, returning '' if missing."""
    metadata = record.get("sample_metadata", {}) or {}
    
    for key in keys:
        val = metadata.get(key)
        if val is not None:
            if isinstance(val, dict):
                return str(val.get("name") or val.get("identifier") or val.get("value") or val)
            return str(val)
    return ""


def extract_purification_target(record: Dict) -> str:
    """Extract purification target from nested metadata structure."""
    metadata = record.get("sample_metadata", {}) or {}
    
    # Try metadata.metadata.purification_target.value (enriched format)
    inner = metadata.get("metadata", {}) or {}
    if isinstance(inner, dict):
        pt = inner.get("purification_target", {})
        if isinstance(pt, dict):
            val = pt.get("value") or pt.get("name") or ""
            if val:
                return str(val).upper()
        elif pt:
            return str(pt).upper()
    
    # Try top-level purification_target
    pt = metadata.get("purification_target")
    if pt:
        if isinstance(pt, dict):
            return str(pt.get("value") or pt.get("name") or "").upper()
        return str(pt).upper()
    
    return ""


def check_slurm_logs(slurm_dir: str, sample_ids: Set[str]) -> Dict[str, str]:
    """
    Check SLURM log files for error messages related to missing samples.
    Returns dict mapping sample_id -> error summary.
    """
    errors: Dict[str, str] = {}
    if not slurm_dir or not os.path.isdir(slurm_dir):
        return errors
    
    logs_dir = os.path.join(slurm_dir, "logs")
    if not os.path.isdir(logs_dir):
        logs_dir = slurm_dir
    
    # Also check the job scripts to map job index -> sample_id
    job_sample_map: Dict[str, str] = {}  # job_index -> sample_id
    
    for fname in sorted(os.listdir(slurm_dir)):
        if not fname.endswith('.sh') or not fname.startswith('dl_'):
            continue
        job_path = os.path.join(slurm_dir, fname)
        try:
            with open(job_path, 'r') as f:
                content = f.read()
            # Extract sample_id from the dest path in the script
            match = re.search(r'--dest\s+"[^"]*?/(\d+)_', content)
            if match:
                job_idx = fname.replace('dl_', '').replace('.sh', '')
                job_sample_map[job_idx] = match.group(1)
        except Exception:
            pass
    
    # Check stderr logs for errors
    if os.path.isdir(logs_dir):
        for fname in os.listdir(logs_dir):
            if not fname.endswith('.err'):
                continue
            log_path = os.path.join(logs_dir, fname)
            try:
                with open(log_path, 'r') as f:
                    content = f.read().strip()
                if not content:
                    continue
                
                # Try to find which sample this log belongs to
                # Array logs: array_JOBID_INDEX.err
                match = re.search(r'_(\d+)\.err$', fname)
                if match:
                    idx = match.group(1).zfill(5)
                    sid = job_sample_map.get(idx)
                    if sid and sid in sample_ids:
                        # Summarize the error (first meaningful line)
                        err_lines = [l.strip() for l in content.split('\n') if l.strip()]
                        error_summary = err_lines[-1] if err_lines else "Unknown error"
                        errors[sid] = error_summary[:200]
            except Exception:
                pass
    
    return errors


def categorize_failure(record: Dict, slurm_error: str = "") -> str:
    """Try to categorize why a sample might be missing its file."""
    reasons = []
    
    file_obj = record.get("file", {})
    if not file_obj:
        reasons.append("no_file_record")
    
    data_id = str(file_obj.get("id") or "").strip() if file_obj else ""
    if not data_id:
        reasons.append("no_data_id")
    
    filename = str(file_obj.get("filename") or file_obj.get("name") or "").strip() if file_obj else ""
    if not filename:
        reasons.append("no_filename")
    
    if slurm_error:
        if "HTTP 404" in slurm_error or "Not Found" in slurm_error:
            reasons.append("file_not_found_404")
        elif "HTTP 403" in slurm_error or "Forbidden" in slurm_error:
            reasons.append("access_denied_403")
        elif "HTTP 5" in slurm_error:
            reasons.append("server_error_5xx")
        elif "Failed" in slurm_error:
            reasons.append("download_failed")
        elif "timeout" in slurm_error.lower():
            reasons.append("timeout")
        else:
            reasons.append(f"slurm_error")
    
    if not reasons:
        reasons.append("unknown")
    
    return "; ".join(reasons)


def main():
    args = parse_args()
    
    # --- Load filtered_data.json ---
    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(args.json, 'r', encoding='utf-8') as f:
        all_records = json.load(f)
    
    if not isinstance(all_records, list):
        print(f"Error: expected list in {args.json}", file=sys.stderr)
        sys.exit(1)
    
    # Build map: sample_id -> record (keep last if duplicates)
    records_by_sample: Dict[str, Dict] = {}
    for record in all_records:
        sid = str(record.get("sample_id") or "").strip()
        if sid:
            records_by_sample[sid] = record
    
    all_sample_ids = set(records_by_sample.keys())
    print(f"Total samples in {args.json}: {len(all_sample_ids)}")
    
    # --- Scan downloaded files ---
    downloaded_ids = get_downloaded_sample_ids(args.data_dir)
    print(f"Samples with files in {args.data_dir}/: {len(downloaded_ids)}")
    
    # --- Find missing ---
    missing_ids = all_sample_ids - downloaded_ids
    extra_ids = downloaded_ids - all_sample_ids  # files not in JSON (shouldn't happen)
    
    print(f"\nMissing samples (in JSON but no file): {len(missing_ids)}")
    if extra_ids:
        print(f"Extra files (in dir but not in JSON): {len(extra_ids)}")
    
    if not missing_ids:
        print("\nAll samples have downloaded files. Nothing to investigate.")
        return
    
    # --- Check SLURM logs if available ---
    slurm_errors: Dict[str, str] = {}
    if args.slurm_dir:
        slurm_errors = check_slurm_logs(args.slurm_dir, missing_ids)
        if slurm_errors:
            print(f"Found SLURM error logs for {len(slurm_errors)} missing samples")
    
    # --- Build output rows ---
    rows = []
    for sid in sorted(missing_ids, key=int):
        record = records_by_sample[sid]
        target = extract_purification_target(record)
        if target == "TDP43":
            target = "TARDBP"
        
        sample_name = record.get("sample_name", "")
        organism = extract_metadata_field(record, "organism")
        method = extract_metadata_field(record, "sample_method")
        
        file_obj = record.get("file", {}) or {}
        filename = file_obj.get("filename") or file_obj.get("name") or ""
        data_id = file_obj.get("id") or ""
        
        slurm_err = slurm_errors.get(sid, "")
        reason = categorize_failure(record, slurm_err)
        
        rows.append({
            "sample_id": sid,
            "sample_name": sample_name,
            "purification_target": target,
            "organism": organism,
            "method": method,
            "expected_filename": filename,
            "data_id": data_id,
            "failure_reason": reason,
            "slurm_error": slurm_err,
            "flow_url": f"https://app.flow.bio/samples/{sid}",
        })
    
    # --- Write CSV ---
    fieldnames = [
        "sample_id", "sample_name", "purification_target", "organism",
        "method", "expected_filename", "data_id", "failure_reason",
        "slurm_error", "flow_url",
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
    
    # By failure reason
    reason_counts = Counter(r["failure_reason"] for r in rows)
    print(f"\nBy failure reason:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")
    
    # By purification target
    target_counts = Counter(r["purification_target"] or "UNKNOWN" for r in rows)
    print(f"\nBy purification target ({len(target_counts)} unique):")
    for target, count in target_counts.most_common(20):
        print(f"  {target}: {count}")
    if len(target_counts) > 20:
        print(f"  ... and {len(target_counts) - 20} more")
    
    # By organism
    org_counts = Counter(r["organism"] or "UNKNOWN" for r in rows)
    if org_counts:
        print(f"\nBy organism:")
        for org, count in org_counts.most_common():
            print(f"  {org}: {count}")
    
    # By method
    method_counts = Counter(r["method"] or "UNKNOWN" for r in rows)
    if method_counts:
        print(f"\nBy experimental method:")
        for method, count in method_counts.most_common():
            print(f"  {method}: {count}")
    
    # List first few
    print(f"\nFirst 20 missing samples:")
    print(f"  {'Sample ID':<12} {'Target':<15} {'Name':<30} {'Reason'}")
    print(f"  {'-'*10:<12} {'-'*13:<15} {'-'*28:<30} {'-'*20}")
    for r in rows[:20]:
        print(f"  {r['sample_id']:<12} {r['purification_target'] or 'N/A':<15} "
              f"{r['sample_name'][:28]:<30} {r['failure_reason']}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more (see {args.output})")
    
    print(f"\nFull details: {args.output}")
    print(f"View on Flow.bio: https://app.flow.bio/samples/{{sample_id}}")


if __name__ == "__main__":
    main()

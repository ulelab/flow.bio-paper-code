#!/usr/bin/env python3
"""
Fill in Flow.bio project links for projects in UK DRI Data Catalogue CSV.
Also exports all sample metadata from matched projects.

Usage:
    python fill_project_links.py
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Any, Optional

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import load_credentials, get_access_token, paginate_items, DEFAULT_BASE_URL

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials_dri.json")
INPUT_CSV = os.path.join(os.path.dirname(__file__), "Ule Lab UK DRI Data Catalogue.xlsx - UK DRI Data Catalogue.csv")


def progress(current: int, total: int, msg: str = ""):
    """Print progress bar."""
    pct = (current / total * 100) if total > 0 else 0
    bar = "=" * int(pct // 5) + ">" + " " * (20 - int(pct // 5))
    print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) {msg[:40]:<40}", end="", flush=True)


def paginate_projects(session: requests.Session, url: str) -> List[Dict[str, Any]]:
    """Paginate through project endpoints (uses 'projects' key)."""
    results = []
    page = 1
    while True:
        resp = session.get(url, params={"page": page, "count": 100}, timeout=60)
        if resp.status_code != 200:
            break
        data = resp.json()
        projects = data.get("projects", [])
        if not projects:
            break
        results.extend(projects)
        if len(projects) < 100:
            break
        page += 1
    return results


def get_all_projects(session: requests.Session) -> Dict[str, Dict[str, Any]]:
    """Fetch all projects, indexed by name (lowercase for matching)."""
    project_ids = {}
    
    # Collect project IDs from all endpoints
    for endpoint in ["/projects", "/projects/owned", "/projects/shared", "/projects/public"]:
        for p in paginate_projects(session, f"{DEFAULT_BASE_URL}{endpoint}"):
            if p.get("id"):
                project_ids[p["id"]] = p
    
    # Fetch full details for each project
    projects_by_name = {}
    projects_by_name_normalized = {}
    total = len(project_ids)
    print(f"\nFetching details for {total} projects...")
    
    for i, (pid, p) in enumerate(project_ids.items()):
        progress(i + 1, total, p.get("name", "")[:30])
        try:
            resp = session.get(f"{DEFAULT_BASE_URL}/projects/{pid}", timeout=15)
            proj = resp.json() if resp.status_code == 200 else p
        except Exception:
            proj = p
        
        name = proj.get("name", "").strip()
        if name:
            # Store by lowercase name for matching
            projects_by_name[name.lower()] = proj
            # Also store normalized version
            projects_by_name_normalized[normalize_unicode(name.lower())] = proj
    
    print()
    return projects_by_name, projects_by_name_normalized


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters for matching."""
    import unicodedata
    # Normalize to NFKC form (compatibility decomposition)
    text = unicodedata.normalize('NFKC', text)
    # Replace common problematic characters
    replacements = {
        '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
        '′': "'", '″': '"', '‚': ',', '…': '...',
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
        '³': '3', '²': '2', '¹': '1', '⁰': '0',
        'œ±': 'alpha',  # Mojibake for α
        '‚Ä≤': "'", '‚Äô': "'",  # Various mojibake patterns
        '‚Äì': '-',
        # Specific mojibake sequences
        'rna‚äìprotein': 'rna-protein',
        '3‚ä≤': "3'",
        'trim25‚äôs': "trim25's",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# Manual mappings for severely corrupted project names
MANUAL_PROJECT_MAPPINGS = {
    "flash: ultra-fast protocol to identify rna‚äìprotein interactions in cells": 
        "flash: ultra-fast protocol to identify rna–protein interactions in cells",
    "fubp1 is a general splicing factor facilitating 3‚ä≤ splice site recognition and splicing of long introns":
        "fubp1 is a general splicing factor facilitating 3′ splice site recognition and splicing of long introns",
    "the molecular dissection of trim25‚äôs rna-binding mechanism provides key insights into its antiviral activity":
        "the molecular dissection of trim25's rna-binding mechanism provides key insights into its antiviral activity",
}


def extract_project_name(description: str) -> Optional[str]:
    """Extract project name from brief description field."""
    # Format: "Project Name (X samples) | Sources: ..."
    # Or: "Project Name (X samples)"
    if not description:
        return None
    
    # Try to match pattern with sample count
    match = re.match(r'^(.+?)\s*\(\d+\s*samples?\)', description)
    if match:
        return match.group(1).strip()
    
    # Try splitting on pipe
    parts = description.split("|")
    if parts:
        return parts[0].strip()
    
    return description.strip()


def get_project_samples_full(session: requests.Session, project_id: str, fetch_full: bool = True) -> List[Dict[str, Any]]:
    """Fetch all samples for a project with full metadata."""
    samples = list(paginate_items(session, f"{DEFAULT_BASE_URL}/projects/{project_id}/samples"))
    
    if not fetch_full:
        return samples
    
    # Fetch full details for all samples via /samples/{id} endpoint
    full_samples = []
    for s in samples:
        sid = s.get("id")
        if not sid:
            full_samples.append(s)
            continue
        try:
            resp = session.get(f"{DEFAULT_BASE_URL}/samples/{sid}", timeout=15)
            if resp.status_code == 200:
                full_samples.append(resp.json())
            else:
                full_samples.append(s)
        except Exception:
            full_samples.append(s)
    
    return full_samples


def extract_meta(sample: Dict[str, Any], key: str) -> str:
    """Extract value from sample metadata."""
    meta = sample.get("metadata", {})
    if isinstance(meta, dict) and key in meta:
        field = meta[key]
        return field.get("value", "") if isinstance(field, dict) else str(field or "")
    return ""


def flatten_value(val: Any) -> str:
    """Flatten a value to a string for CSV export."""
    if val is None:
        return ""
    if isinstance(val, dict):
        # For nested dicts like sample_type, organism, etc.
        return val.get("name") or val.get("identifier") or val.get("value") or str(val)
    if isinstance(val, list):
        return "; ".join(flatten_value(v) for v in val)
    return str(val)


def sample_to_full_row(sample: Dict[str, Any], project_name: str) -> Dict[str, str]:
    """Convert sample to metadata row with ALL fields."""
    row = {"project_name": project_name}
    
    # Add all top-level fields
    for key, val in sample.items():
        if key == "metadata":
            continue  # Handle separately
        row[key] = flatten_value(val)
    
    # Add all metadata fields
    meta = sample.get("metadata", {})
    if isinstance(meta, dict):
        for key, val in meta.items():
            meta_key = f"meta_{key}"
            if isinstance(val, dict):
                row[meta_key] = val.get("value", "") or flatten_value(val)
            else:
                row[meta_key] = flatten_value(val)
    
    return row


def main():
    parser = argparse.ArgumentParser(description="Fill project links and export sample metadata")
    parser.add_argument("--input", "-i", default=INPUT_CSV, help="Input CSV file")
    parser.add_argument("--output", "-o", default="UK_DRI_Catalogue_with_links.csv", help="Output CSV with links")
    parser.add_argument("--samples-output", "-s", default="all_sample_metadata.csv", help="Sample metadata CSV")
    args = parser.parse_args()
    
    # Read existing CSV
    print(f"Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Found {len(rows)} projects in CSV")
    
    # Extract project names from descriptions
    csv_projects = {}
    csv_projects_normalized = {}  # For fallback matching
    for i, row in enumerate(rows):
        desc = row.get("Brief dataset description", "")
        name = extract_project_name(desc)
        if name:
            csv_projects[name.lower()] = i  # Map lowercase name to row index
            csv_projects_normalized[normalize_unicode(name.lower())] = i
    
    print(f"Extracted {len(csv_projects)} project names")
    
    # Authenticate and fetch Flow.bio projects
    print("\nAuthenticating...")
    username, password = load_credentials(CREDENTIALS_PATH)
    token = get_access_token(username, password)
    
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {token}"})
        
        me = session.get(f"{DEFAULT_BASE_URL}/me", timeout=15).json()
        print(f"Logged in as: {me.get('username')}")
        
        # Get all projects from Flow.bio
        flow_projects, flow_projects_normalized = get_all_projects(session)
        print(f"Found {len(flow_projects)} projects in Flow.bio")
        
        # Match projects and collect sample metadata
        matched = 0
        unmatched = []
        all_samples = []
        
        # Add Project Link column if not present
        if "Project Link" not in fieldnames:
            fieldnames = list(fieldnames) + ["Project Link"]
        
        print("\nMatching projects and fetching samples...")
        total = len(csv_projects)
        
        for i, (csv_name_lower, row_idx) in enumerate(csv_projects.items()):
            progress(i + 1, total, csv_name_lower[:30])
            
            # Try exact match first
            flow_proj = flow_projects.get(csv_name_lower)
            
            # Try manual mapping for known mojibake
            if not flow_proj and csv_name_lower in MANUAL_PROJECT_MAPPINGS:
                mapped_name = MANUAL_PROJECT_MAPPINGS[csv_name_lower].lower()
                flow_proj = flow_projects.get(mapped_name)
            
            # Try normalized match
            if not flow_proj:
                csv_normalized = normalize_unicode(csv_name_lower)
                flow_proj = flow_projects_normalized.get(csv_normalized)
            
            # Try partial match if no exact match
            if not flow_proj:
                for flow_name, proj in flow_projects.items():
                    # Check if CSV name is contained in Flow name or vice versa
                    if csv_name_lower in flow_name or flow_name in csv_name_lower:
                        flow_proj = proj
                        break
            
            # Try partial match with normalized names
            if not flow_proj:
                csv_normalized = normalize_unicode(csv_name_lower)
                for flow_name_norm, proj in flow_projects_normalized.items():
                    if csv_normalized in flow_name_norm or flow_name_norm in csv_normalized:
                        flow_proj = proj
                        break
            
            if flow_proj:
                matched += 1
                project_id = flow_proj.get("id", "")
                is_private = flow_proj.get("private", True)
                
                # Add project link for public projects
                if not is_private and project_id:
                    rows[row_idx]["Project Link"] = f"https://app.flow.bio/projects/{project_id}/"
                else:
                    rows[row_idx]["Project Link"] = ""
                
                # Fetch samples for this project (with full details)
                samples = get_project_samples_full(session, project_id, fetch_full=True)
                project_name = flow_proj.get("name", "Unknown")
                
                for sample in samples:
                    all_samples.append(sample_to_full_row(sample, project_name))
            else:
                unmatched.append(csv_name_lower)
                rows[row_idx]["Project Link"] = ""
        
        print(f"\n\nMatched: {matched}/{len(csv_projects)} projects")
        
        if unmatched[:10]:
            print(f"\nUnmatched projects (first 10):")
            for name in unmatched[:10]:
                print(f"  - {name}")
        
        # Write updated CSV
        output_path = os.path.join(os.path.dirname(__file__), args.output)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Written {len(rows)} rows to {output_path}")
        
        # Write sample metadata CSV with ALL fields
        if all_samples:
            # Collect all unique fieldnames across all samples
            all_fieldnames = set()
            for s in all_samples:
                all_fieldnames.update(s.keys())
            
            # Order fieldnames: project_name first, then id/name fields, then alphabetical
            priority_fields = ["project_name", "id", "name", "sample_type", "organism", "owner_name", "private", "created"]
            meta_fields = sorted([f for f in all_fieldnames if f.startswith("meta_")])
            other_fields = sorted([f for f in all_fieldnames if f not in priority_fields and not f.startswith("meta_")])
            
            sample_fieldnames = [f for f in priority_fields if f in all_fieldnames] + other_fields + meta_fields
            
            samples_path = os.path.join(os.path.dirname(__file__), args.samples_output)
            with open(samples_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sample_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_samples)
            
            print(f"✓ Written {len(all_samples)} samples with {len(sample_fieldnames)} columns to {samples_path}")
        else:
            print("No samples found to export")


if __name__ == "__main__":
    main()

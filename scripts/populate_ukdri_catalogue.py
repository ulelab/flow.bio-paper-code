#!/usr/bin/env python3
"""
Populate UKDRI Data Catalogue CSV from Flow.bio projects.

Usage:
    python populate_ukdri_catalogue.py
    python populate_ukdri_catalogue.py --output catalogue.csv
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Any

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import load_credentials, get_access_token, paginate_items, DEFAULT_BASE_URL

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials_dri.json")
DEFAULT_CENTRE = "Kings"
DEFAULT_RESEARCH_DIVISION = "Genetics and Molecular Networks"
CONTACT_EMAIL = "jernej.ule@ukdri.ac.uk"


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


def get_all_projects(session: requests.Session) -> List[Dict[str, Any]]:
    """Fetch all projects with full details."""
    project_ids = {}
    
    # Collect project IDs from all endpoints
    for endpoint in ["/projects", "/projects/owned", "/projects/shared", "/projects/public"]:
        for p in paginate_projects(session, f"{DEFAULT_BASE_URL}{endpoint}"):
            if p.get("id"):
                project_ids[p["id"]] = p
    
    # Fetch full details for each project
    projects = []
    total = len(project_ids)
    print(f"\nFetching details for {total} projects...")
    
    for i, (pid, p) in enumerate(project_ids.items()):
        progress(i + 1, total, p.get("name", "")[:30])
        try:
            resp = session.get(f"{DEFAULT_BASE_URL}/projects/{pid}", timeout=15)
            projects.append(resp.json() if resp.status_code == 200 else p)
        except Exception:
            projects.append(p)
    
    print()  # Newline after progress
    return projects


def get_project_samples(session: requests.Session, project_id: str) -> List[Dict[str, Any]]:
    """Fetch samples for a project with metadata for first 5."""
    samples = list(paginate_items(session, f"{DEFAULT_BASE_URL}/projects/{project_id}/samples"))
    
    # Fetch full details for first 5 samples (for metadata)
    for i in range(min(5, len(samples))):
        try:
            resp = session.get(f"{DEFAULT_BASE_URL}/samples/{samples[i]['id']}", timeout=10)
            if resp.status_code == 200:
                samples[i] = resp.json()
        except Exception:
            pass
    
    return samples


def extract_meta(sample: Dict[str, Any], key: str) -> str:
    """Extract value from sample metadata."""
    meta = sample.get("metadata", {})
    if isinstance(meta, dict) and key in meta:
        field = meta[key]
        return field.get("value", "") if isinstance(field, dict) else str(field or "")
    return ""


def project_to_row(project: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """Convert project to catalogue row."""
    
    # Extract from samples
    pis, owners, scientists, sources = set(), set(), set(), set()
    sample_types, methods, organisms = set(), set(), set()
    has_geo, has_ena = False, False
    
    for s in samples:
        if pi := extract_meta(s, "pi"): pis.add(pi)
        if sci := extract_meta(s, "scientist"): scientists.add(sci)
        if src := extract_meta(s, "source"): sources.add(src)
        if method := extract_meta(s, "experimental_method"): methods.add(method)
        if owner := s.get("owner_name"): owners.add(owner)
        
        # Sample type
        st = s.get("sample_type")
        if isinstance(st, dict): st = st.get("identifier") or st.get("name")
        if st: sample_types.add(str(st))
        
        # Organism
        org = s.get("organism")
        if isinstance(org, dict): org = org.get("name")
        if org: organisms.add(org)
        
        # Check for repository IDs in sample name
        name = s.get("name", "")
        if any(x in name for x in ["GSM", "GSE", "SRR"]): has_geo = True
        if any(x in name for x in ["ERR", "ERX", "ERS"]): has_ena = True
    
    # Project owner
    proj_owner = project.get("owner", {})
    if isinstance(proj_owner, dict) and proj_owner.get("name"):
        owners.add(proj_owner["name"])
    
    # PI / Dataset Lead Name
    pi_parts = []
    if pis: pi_parts.append(f"PI: {', '.join(sorted(pis))}")
    if owners: pi_parts.append(f"Owner: {', '.join(sorted(owners))}")
    if scientists: pi_parts.append(f"Scientist: {', '.join(sorted(scientists))}")
    
    # Data types (sample types + methods)
    data_types = sample_types | methods
    
    # Publications
    papers = project.get("papers", [])
    pmids = [p.get("id") for p in papers if p.get("id")]
    
    # Repository
    repo = "NCBI GEO" if has_geo else ("ArrayExpress" if has_ena else "")
    
    # Project link (only for public projects)
    is_private = project.get("private", True)
    project_id = project.get("id", "")
    project_link = f"https://app.flow.bio/search/projects/{project_id}" if not is_private and project_id else ""
    
    return {
        "PI / Dataset Lead Name": "; ".join(pi_parts) if pi_parts else "Unknown",
        "Centre": DEFAULT_CENTRE,
        "Research Divisons": DEFAULT_RESEARCH_DIVISION,
        "Data type(s), select all that apply": "; ".join(sorted(data_types)) if data_types else "Unknown",
        "Brief dataset description": f"{project.get('name', 'Unknown')} ({len(samples)} samples)" + 
                                     (f" | Sources: {', '.join(sorted(sources))}" if sources else ""),
        "Is there documentation available describing the dataset?": "Yes, basic" if project.get("description") else "No",
        "Species / Source": "; ".join(sorted(organisms)) if organisms else "Unknown",
        "Is your data published?": "Yes" if pmids else "No",
        "If published, provide publication link/s": "; ".join(f"PMID:{p}" for p in pmids),
        "Has the data been deposited in a repository?": "Yes" if repo else "No",
        "Select repository": repo,
        "Is this dataset part of a larger consortium or initiative?": "Not part of a consortium",
        "Data Access Status": "Available on request" if is_private else "Openly available",
        "Email address for contact": CONTACT_EMAIL,
        "Project Link": project_link,
        "Please Specify": ""
    }


def main():
    parser = argparse.ArgumentParser(description="Populate UKDRI Data Catalogue from Flow.bio")
    parser.add_argument("--output", "-o", default="UKDRIDataCatalogue_filled.csv")
    args = parser.parse_args()
    
    print("Authenticating...")
    username, password = load_credentials(CREDENTIALS_PATH)
    token = get_access_token(username, password)
    
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {token}"})
        
        # Get user info
        me = session.get(f"{DEFAULT_BASE_URL}/me", timeout=15).json()
        print(f"Logged in as: {me.get('username')}")
        
        # Get all projects
        projects = get_all_projects(session)
        print(f"Found {len(projects)} projects")
        
        # Process each project
        print("\nProcessing projects...")
        rows = []
        total = len(projects)
        total_samples = 0
        
        for i, project in enumerate(sorted(projects, key=lambda p: p.get("name", ""))):
            pid = project.get("id")
            pname = project.get("name", "Unknown")
            progress(i + 1, total, pname[:30])
            
            samples = get_project_samples(session, pid) if pid else []
            total_samples += len(samples)
            rows.append(project_to_row(project, samples))
        
        print(f"\n\nTotal samples: {total_samples}")
        
        # Write CSV
        fieldnames = [
            "PI / Dataset Lead Name", "Centre", "Research Divisons",
            "Data type(s), select all that apply", "Brief dataset description",
            "Is there documentation available describing the dataset?", "Species / Source",
            "Is your data published?", "If published, provide publication link/s",
            "Has the data been deposited in a repository?", "Select repository",
            "Is this dataset part of a larger consortium or initiative?",
            "Data Access Status", "Email address for contact", "Project Link", "Please Specify"
        ]
        
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Written {len(rows)} entries to {args.output}")


if __name__ == "__main__":
    main()

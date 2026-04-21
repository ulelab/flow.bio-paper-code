#!/usr/bin/env python3
"""
Enrich filtered_data.json with full sample metadata from Flow.bio API.

This script:
1. Loads existing filtered_data.json
2. Gets unique sample IDs
3. Fetches full metadata for each sample from /samples/{sample_id}
4. Adds the metadata to each record in filtered_data.json
5. Saves the enriched data back
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

from flow_api import load_credentials, get_access_token

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "https://api.flow.bio"
DEFAULT_INPUT_JSON = "filtered_data.json"
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
REQUEST_DELAY = 0.25  # Delay between API calls to avoid rate limiting


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich filtered Flow file records with full sample metadata"
    )
    parser.add_argument(
        "--input-json", "-i",
        default=DEFAULT_INPUT_JSON,
        help=f"Input JSON file with sample_id fields (default: {DEFAULT_INPUT_JSON})"
    )
    parser.add_argument(
        "--output-json", "-o",
        default=None,
        help="Output JSON path (default: overwrite input JSON)"
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Append a human-readable timestamp to the output JSON filename"
    )
    return parser.parse_args()


def add_timestamp_to_filename(path: str) -> str:
    """Append a timestamp before the file extension."""
    base, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


# =============================================================================
# API Functions
# =============================================================================

def get_sample_metadata(
    session: requests.Session,
    sample_id: str,
    base_url: str = BASE_URL,
) -> Dict[str, Any]:
    """Fetch full metadata for a specific sample."""
    url = f"{base_url}/samples/{sample_id}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    input_json = args.input_json
    output_json = args.output_json or input_json
    if args.timestamp_output:
        output_json = add_timestamp_to_filename(output_json)

    # Load existing filtered_data.json
    if not os.path.exists(input_json):
        print(f"Input file not found: {input_json}", file=sys.stderr)
        sys.exit(1)
    
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Expected list in {input_json}, got {type(data)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(data)} records from {input_json}")
    
    # Get unique sample IDs
    sample_ids = list(set(
        str(record.get("sample_id") or "")
        for record in data
        if record.get("sample_id")
    ))
    print(f"Found {len(sample_ids)} unique sample IDs")
    
    # Authenticate
    username, password = load_credentials(CREDENTIALS_PATH)
    access_token = get_access_token(username, password)
    print("Authenticated successfully")
    
    # Fetch metadata for each sample
    sample_metadata: Dict[str, Dict[str, Any]] = {}
    
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})
        
        for i, sample_id in enumerate(sample_ids):
            try:
                metadata = get_sample_metadata(session, sample_id)
                sample_metadata[sample_id] = metadata
                
                if (i + 1) % 50 == 0:
                    print(f"Fetched metadata for {i + 1}/{len(sample_ids)} samples...")
                
                time.sleep(REQUEST_DELAY)
                
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                print(f"Error fetching sample {sample_id}: HTTP {status}")
                sample_metadata[sample_id] = {}
            except Exception as e:
                print(f"Error fetching sample {sample_id}: {e}")
                sample_metadata[sample_id] = {}
    
    print(f"Fetched metadata for {len(sample_metadata)} samples")
    
    # Show a sample of what metadata fields are available
    if sample_metadata:
        first_sample = next(iter(sample_metadata.values()))
        if first_sample:
            print("\nAvailable metadata fields:")
            for key in sorted(first_sample.keys()):
                value = first_sample[key]
                value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"  - {key}: {value_preview}")
    
    # Enrich the data records with sample metadata
    for record in data:
        sample_id = str(record.get("sample_id") or "")
        if sample_id and sample_id in sample_metadata:
            record["sample_metadata"] = sample_metadata[sample_id]
    
    # Save enriched data
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved enriched data to {output_json}")
    
    # Summary of key categorization fields
    print("\n=== Metadata Summary ===")
    
    # Collect unique values for common categorization fields
    fields_to_summarize = [
        "sample_method",
        "organism", 
        "cell_type",
        "tissue",
        "sample_type",
    ]
    
    for field in fields_to_summarize:
        values = set()
        for metadata in sample_metadata.values():
            if metadata:
                val = metadata.get(field)
                if val:
                    if isinstance(val, dict):
                        val = val.get("name") or val.get("identifier") or str(val)
                    values.add(str(val))
        
        if values:
            print(f"\n{field} ({len(values)} unique values):")
            for v in sorted(values)[:10]:  # Show first 10
                print(f"  - {v}")
            if len(values) > 10:
                print(f"  ... and {len(values) - 10} more")


if __name__ == "__main__":
    main()

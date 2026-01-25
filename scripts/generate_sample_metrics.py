#!/usr/bin/env python3
"""
Generate a CSV file with sample metrics from downloaded Flow.bio data.

Parses UMICollapse log files to extract:
- sample_id
- pcr_duplication_rate
- millions_of_crosslinks

Usage:
    python generate_sample_metrics.py
    python generate_sample_metrics.py --data-dir data --output sample_metrics.csv
"""

import argparse
import csv
import os
import re
import sys
import json
from typing import Dict, Optional, Tuple


def parse_umicollapse_log(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a UMICollapse log file to extract read counts.
    
    Returns:
        Tuple of (input_reads, dedup_reads) or (None, None) if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        input_match = re.search(r'Number of input reads\s+(\d+)', content)
        dedup_match = re.search(r'Number of reads after deduplicating\s+(\d+)', content)
        
        if input_match and dedup_match:
            return int(input_match.group(1)), int(dedup_match.group(1))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
    
    return None, None


def extract_sample_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract sample_id from a filename.
    
    Handles both old format (no sample_id prefix) and new format (sample_id_filename).
    For old format, returns None.
    """
    # New format: sample_id is first part before underscore, and it's numeric
    # e.g., "12345_sample.unique_genome.dedup_UMICollapse.log"
    parts = filename.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[0]
    return None


def load_sample_id_mapping(json_file: str) -> Dict[str, str]:
    """
    Load sample_id to filename mapping from filtered_data.json.
    
    Returns dict mapping filename (without sample_id prefix) to sample_id.
    """
    mapping = {}
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for record in data:
            sample_id = str(record.get("sample_id", ""))
            file_obj = record.get("file", {})
            filename = file_obj.get("filename", "")
            
            if sample_id and filename:
                # Store mapping from original filename to sample_id
                mapping[os.path.basename(filename)] = sample_id
    except Exception as e:
        print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample metrics CSV from UMICollapse logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output columns:
  sample_id             - Flow.bio sample identifier
  pcr_duplication_rate  - Ratio of input reads to deduplicated reads
  millions_of_crosslinks - Number of deduplicated reads / 1,000,000
"""
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Directory containing UMICollapse log files (default: data)"
    )
    parser.add_argument(
        "--json", "-j",
        default="filtered_data.json",
        help="JSON file with sample metadata for ID lookup (default: filtered_data.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="sample_metrics.csv",
        help="Output CSV file (default: sample_metrics.csv)"
    )
    args = parser.parse_args()
    
    # Load sample ID mapping from JSON
    sample_id_mapping = load_sample_id_mapping(args.json)
    print(f"Loaded {len(sample_id_mapping)} sample ID mappings from {args.json}")
    
    # Find all UMICollapse log files
    log_files = []
    if os.path.isdir(args.data_dir):
        for filename in os.listdir(args.data_dir):
            if filename.endswith('_UMICollapse.log'):
                log_files.append(os.path.join(args.data_dir, filename))
    
    if not log_files:
        print(f"No UMICollapse log files found in {args.data_dir}")
        sys.exit(1)
    
    print(f"Found {len(log_files)} UMICollapse log files")
    
    # Parse log files and collect metrics
    metrics = []
    for filepath in sorted(log_files):
        filename = os.path.basename(filepath)
        
        # Try to extract sample_id from filename (new format)
        sample_id = extract_sample_id_from_filename(filename)
        
        # If not in new format, look up from JSON mapping
        if sample_id is None:
            # Remove any existing sample_id prefix if present
            original_filename = filename
            sample_id = sample_id_mapping.get(original_filename, "")
        
        # Parse the log file
        input_reads, dedup_reads = parse_umicollapse_log(filepath)
        
        if input_reads is not None and dedup_reads is not None and dedup_reads > 0:
            pcr_rate = input_reads / dedup_reads
            millions_xl = dedup_reads / 1_000_000
            
            metrics.append({
                'sample_id': sample_id,
                'filename': filename,
                'pcr_duplication_rate': round(pcr_rate, 3),
                'millions_of_crosslinks': round(millions_xl, 3)
            })
        else:
            print(f"Warning: Could not parse {filename}", file=sys.stderr)
    
    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'pcr_duplication_rate', 'millions_of_crosslinks', 'filename'
        ])
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"\nWritten {len(metrics)} samples to {args.output}")
    
    # Summary statistics
    if metrics:
        pcr_rates = [m['pcr_duplication_rate'] for m in metrics]
        xl_counts = [m['millions_of_crosslinks'] for m in metrics]
        
        print(f"\nSummary:")
        print(f"  PCR duplication rate: min={min(pcr_rates):.2f}, max={max(pcr_rates):.2f}, "
              f"median={sorted(pcr_rates)[len(pcr_rates)//2]:.2f}")
        print(f"  Crosslinks (millions): min={min(xl_counts):.2f}, max={max(xl_counts):.2f}, "
              f"total={sum(xl_counts):.1f}")


if __name__ == "__main__":
    main()

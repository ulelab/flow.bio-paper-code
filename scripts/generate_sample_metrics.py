#!/usr/bin/env python3
"""
Generate a CSV file with sample metrics from downloaded Flow.bio data.

Parses UMICollapse log files to extract:
- sample_id
- pcr_duplication_rate
- millions_of_crosslinks

Calculates from PEKA 5-mer distribution files:
- similarity_score (mean overlap ratio of top 50 k-mers vs other samples,
  excluding same-protein comparisons)

Usage:
    python generate_sample_metrics.py
    python generate_sample_metrics.py --data-dir data --peka-dir data_peka --output sample_metrics.csv
"""

import argparse
import csv
import glob
import os
import re
import sys
import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


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


def load_metadata_targets(json_file: str) -> Dict[str, str]:
    """
    Load protein targets from filtered_data.json.
    
    Returns dict mapping sample filename stem to uppercase protein target.
    """
    targets = {}
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for r in data:
            fn = r.get('file', {}).get('filename', '').replace('.unique_genome.dedup_UMICollapse.log', '')
            t = r.get('sample_metadata', {}).get('metadata', {}).get('purification_target', {})
            t = t.get('value', '') if isinstance(t, dict) else ''
            if fn:
                target = t.upper() if t else 'UNKNOWN'
                if target == 'TDP43':
                    target = 'TARDBP'
                targets[fn] = target
    except Exception as e:
        print(f"Warning: Could not load targets from {json_file}: {e}", file=sys.stderr)
    
    return targets


def load_peka_data(data_dir: str = 'data_peka') -> Dict[str, Dict[str, float]]:
    """Load PEKA scores from 5-mer distribution TSV files."""
    peka_data = {}
    for f in glob.glob(os.path.join(data_dir, '*.tsv')):
        s = os.path.basename(f).replace('.genome_5mer_distribution_genome.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t')
            peka_data[s] = dict(zip(df.iloc[:, 0], df['PEKA-score']))
        except Exception:
            pass
    return peka_data


def get_top_kmers(peka_scores: Dict[str, float], n: int = 50) -> Set[str]:
    """Get top n k-mers by PEKA score (excluding non-finite values)."""
    v = {k: v for k, v in peka_scores.items() if np.isfinite(v)}
    if not v:
        return set()
    return set(k for k, _ in sorted(v.items(), key=lambda x: -x[1])[:n])


def calculate_similarity_scores(
    sample_names: List[str],
    peka_data: Dict[str, Dict[str, float]],
    targets: Dict[str, str],
    top_n: int = 50,
) -> Dict[str, float]:
    """
    Calculate similarity scores for all samples.
    
    Similarity score = mean overlap ratio of top 50 k-mers with all other 
    samples, excluding samples of the same protein target.
    
    - A score of 0 means the top 50 k-mers of a dataset were not ranked 
      among the top 50 in any other dataset.
    - Higher values indicate that top motifs overlap with many other datasets.
    
    Args:
        sample_names: List of sample filename stems
        peka_data: Dict mapping sample name to {kmer: PEKA-score}
        targets: Dict mapping sample name to protein target name
        top_n: Number of top k-mers to compare (default 50)
    
    Returns:
        Dict mapping sample name to similarity score
    """
    # Pre-compute top k-mers for all samples
    top_kmers = {s: get_top_kmers(peka_data.get(s, {}), top_n) for s in sample_names}
    
    scores = {}
    for i, si in enumerate(sample_names):
        if not top_kmers[si]:
            scores[si] = 0.0
            continue
        
        overlaps = []
        target_i = targets.get(si, 'UNKNOWN')
        
        for sj in sample_names:
            if si == sj:
                continue
            # Exclude same-protein comparisons
            if target_i == targets.get(sj, 'UNKNOWN') and target_i != 'UNKNOWN':
                continue
            if not top_kmers[sj]:
                continue
            
            overlap = len(top_kmers[si] & top_kmers[sj]) / top_n
            overlaps.append(overlap)
        
        scores[si] = float(np.mean(overlaps)) if overlaps else 0.0
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample metrics CSV from UMICollapse logs and PEKA data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output columns:
  sample_id               - Flow.bio sample identifier
  pcr_duplication_rate    - Ratio of input reads to deduplicated reads
  millions_of_crosslinks  - Number of deduplicated reads / 1,000,000
  similarity_score        - Mean overlap ratio of top 50 k-mers vs other samples
"""
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Directory containing UMICollapse log files (default: data)"
    )
    parser.add_argument(
        "--peka-dir", "-p",
        default="data_peka",
        help="Directory containing PEKA 5-mer distribution files (default: data_peka)"
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
    
    # Load sample ID mapping and targets from JSON
    sample_id_mapping = load_sample_id_mapping(args.json)
    print(f"Loaded {len(sample_id_mapping)} sample ID mappings from {args.json}")
    
    targets = load_metadata_targets(args.json)
    print(f"Loaded {len(targets)} protein targets from {args.json}")
    
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
    sample_stems = []  # Track sample stems for similarity calculation
    
    for filepath in sorted(log_files):
        filename = os.path.basename(filepath)
        
        # Try to extract sample_id from filename (new format)
        sample_id = extract_sample_id_from_filename(filename)
        
        # If not in new format, look up from JSON mapping
        if sample_id is None:
            original_filename = filename
            sample_id = sample_id_mapping.get(original_filename, "")
        
        # Parse the log file
        input_reads, dedup_reads = parse_umicollapse_log(filepath)
        
        if input_reads is not None and dedup_reads is not None and dedup_reads > 0:
            pcr_rate = input_reads / dedup_reads
            millions_xl = dedup_reads / 1_000_000
            
            # Derive sample stem (used as key for PEKA data matching)
            sample_stem = filename.replace('.unique_genome.dedup_UMICollapse.log', '')
            sample_stems.append(sample_stem)
            
            metrics.append({
                'sample_id': sample_id,
                'filename': filename,
                'sample_stem': sample_stem,
                'pcr_duplication_rate': round(pcr_rate, 3),
                'millions_of_crosslinks': round(millions_xl, 3),
                'similarity_score': None,  # Filled in below
            })
        else:
            print(f"Warning: Could not parse {filename}", file=sys.stderr)
    
    # Load PEKA data and calculate similarity scores
    if os.path.isdir(args.peka_dir):
        print(f"\nLoading PEKA data from {args.peka_dir}...")
        peka_data = load_peka_data(args.peka_dir)
        print(f"Loaded PEKA data for {len(peka_data)} samples")
        
        if peka_data:
            print("Calculating similarity scores...")
            all_stems = list(peka_data.keys())
            sim_scores = calculate_similarity_scores(all_stems, peka_data, targets)
            
            # Map similarity scores back to metrics
            for m in metrics:
                stem = m['sample_stem']
                if stem in sim_scores:
                    m['similarity_score'] = round(sim_scores[stem], 4)
            
            n_with_sim = sum(1 for m in metrics if m['similarity_score'] is not None)
            print(f"Similarity scores calculated for {n_with_sim}/{len(metrics)} samples")
    else:
        print(f"Warning: PEKA directory {args.peka_dir} not found, skipping similarity scores")
    
    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'pcr_duplication_rate', 'millions_of_crosslinks',
            'similarity_score', 'filename'
        ])
        writer.writeheader()
        for m in metrics:
            # Exclude internal key from output
            row = {k: v for k, v in m.items() if k != 'sample_stem'}
            writer.writerow(row)
    
    print(f"\nWritten {len(metrics)} samples to {args.output}")
    
    # Summary statistics
    if metrics:
        pcr_rates = [m['pcr_duplication_rate'] for m in metrics]
        xl_counts = [m['millions_of_crosslinks'] for m in metrics]
        sim_scores_list = [m['similarity_score'] for m in metrics if m['similarity_score'] is not None]
        
        print(f"\nSummary:")
        print(f"  PCR duplication rate: min={min(pcr_rates):.2f}, max={max(pcr_rates):.2f}, "
              f"median={sorted(pcr_rates)[len(pcr_rates)//2]:.2f}")
        print(f"  Crosslinks (millions): min={min(xl_counts):.2f}, max={max(xl_counts):.2f}, "
              f"total={sum(xl_counts):.1f}")
        if sim_scores_list:
            print(f"  Similarity score: min={min(sim_scores_list):.4f}, max={max(sim_scores_list):.4f}, "
                  f"median={sorted(sim_scores_list)[len(sim_scores_list)//2]:.4f}")


if __name__ == "__main__":
    main()

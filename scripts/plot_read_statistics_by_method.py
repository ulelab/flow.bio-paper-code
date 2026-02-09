#!/usr/bin/env python3
"""
Generate faceted violin plot showing read statistics by experimental method.

Shows three metrics:
- Total number of reads (from BOWTIE_ALIGN .out files)
- Total genome mapped reads (from UMICollapse logs)
- Total crosslinks / deduplicated reads (from UMICollapse logs)

Usage:
    python plot_read_statistics_by_method.py
    python plot_read_statistics_by_method.py --output read_stats_by_method.png
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_metadata(json_path='filtered_data.json') -> Dict:
    """Load sample metadata from filtered_data.json"""
    with open(json_path) as f:
        metadata = json.load(f)
    
    # Create mapping: sample_id -> experimental_method
    sample_info = {}
    for record in metadata:
        sample_id = str(record.get('sample_id', ''))
        if not sample_id:
            continue
        
        # Get experimental method from nested metadata
        sample_meta = record.get('sample_metadata', {})
        metadata_nested = sample_meta.get('metadata', {})
        exp_method = metadata_nested.get('experimental_method', '')
        
        # Handle dict format
        if isinstance(exp_method, dict):
            exp_method = exp_method.get('value') or exp_method.get('name') or exp_method.get('identifier') or ''
        
        exp_method = str(exp_method).strip()
        
        # Also get protein target for reference
        target = sample_meta.get('metadata', {}).get('purification_target', {})
        if isinstance(target, dict):
            target = target.get('value', '')
        target = str(target).strip()
        # Normalize case: always uppercase
        target_upper = target.upper() if target else 'UNKNOWN'
        # Map TDP43 to TARDBP
        if target_upper == 'TDP43':
            target_upper = 'TARDBP'
        # Map HNRNPC case-insensitively
        if target_upper =='HNRNPC':
            target_upper = 'HNRNPC'
        # Map ptbp1 case-insensitively
        if target_upper =='PTBP1':
            target_upper = 'PTBP1'
        sample_info[sample_id] = {
            'method': exp_method,
            'target': target_upper
        }
        
        sample_info[sample_id] = {
            'method': exp_method,
            'target': target
        }
    
    return sample_info


def extract_sample_id_from_filename(filename: str) -> Optional[str]:
    """Extract sample_id from filename with sample_id prefix"""
    parts = filename.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[0]
    return None


def parse_umicollapse_log(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse UMICollapse log to extract genome mapped reads and crosslinks.
    
    Returns:
        (input_reads, dedup_reads) or (None, None) if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Number of input reads (genome mapped reads)
        input_match = re.search(r'Number of input reads\s+(\d+)', content)
        # Number of reads after deduplicating (crosslinks)
        dedup_match = re.search(r'Number of reads after deduplicating\s+(\d+)', content)
        
        if input_match and dedup_match:
            return int(input_match.group(1)), int(dedup_match.group(1))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return None, None


def parse_trimming_report(filepath: str) -> Optional[int]:
    """
    Parse trimming_report.txt file to extract total number of reads.
    
    Looks for lines like:
    "Total reads processed:              12,345,678"
    
    Returns:
        Total number of reads or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Prefer 'Reads written (passing filters):' for total_reads
                if 'Reads written (passing filters):' in line:
                    match = re.search(r'Reads written \(passing filters\):\s*([\d,]+)', line)
                    if match:
                        num_str = match.group(1).replace(',', '')
                        return int(num_str)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return None


def collect_read_statistics(
    metadata: Dict,
    data_dir: str = 'data',
    data_trimming_dir: str = 'data_trimming'
) -> pd.DataFrame:
    """
    Collect read statistics from all available files.
    
    Returns DataFrame with columns:
    - sample_id
    - method
    - target
    - total_reads
    - genome_mapped_reads
    - crosslinks
    """
    results = []
    
    # Process UMICollapse logs
    umicollapse_files = glob.glob(os.path.join(data_dir, '*_UMICollapse.log'))
    print(f"Found {len(umicollapse_files)} UMICollapse log files")
    
    debug_count = 0
    for filepath in umicollapse_files:
        filename = os.path.basename(filepath)
        
        # Try to extract sample_id from filename (new format with prefix)
        sample_id = extract_sample_id_from_filename(filename)
        
        # For old files without sample_id prefix, try to match by looking up in trimming files
        # by the base filename pattern
        if not sample_id:
            # Old format: just the filename without sample_id prefix
            # Try to find a match in data_trimming by the base pattern
            base_name = filename.replace('.unique_genome.dedup_UMICollapse.log', '')
            # Look for any trimming file that contains this base pattern after the sample_id
            trimming_candidates = glob.glob(os.path.join(data_trimming_dir, f"*{base_name}*_trimming_report.txt"))
            if trimming_candidates:
                # Extract sample_id from the matching trimming file
                trimming_filename = os.path.basename(trimming_candidates[0])
                sample_id = extract_sample_id_from_filename(trimming_filename)
        
        if not sample_id or sample_id not in metadata:
            continue
        
        genome_mapped, crosslinks = parse_umicollapse_log(filepath)
        
        if genome_mapped is None or crosslinks is None:
            continue
        
        # Look for corresponding trimming report file
        trimming_pattern = os.path.join(data_trimming_dir, f"{sample_id}_*_trimming_report.txt")
        trimming_files = glob.glob(trimming_pattern)
        
        if debug_count < 5:
            print(f"  Debug sample {sample_id}: found {len(trimming_files)} trimming files")
            debug_count += 1
        
        total_reads = None
        if trimming_files:
            # Use the first matching file
            total_reads = parse_trimming_report(trimming_files[0])
        
        # Only include samples where we have all three metrics
        if total_reads is not None:
            results.append({
                'sample_id': sample_id,
                'method': metadata[sample_id]['method'],
                'target': metadata[sample_id]['target'],
                'total_reads': total_reads,
                'genome_mapped_reads': genome_mapped,
                'crosslinks': crosslinks
            })
    
    df = pd.DataFrame(results)
    print(f"Collected statistics for {len(df)} samples")
    
    return df


def plot_read_statistics_faceted(df: pd.DataFrame, output_file: str = 'read_stats_by_method.png'):
    """
    Create faceted violin plot showing read statistics by experimental method.
    
    Each method gets its own panel showing:
    - Total reads
    - Genome mapped reads  
    - Crosslinks (deduplicated)
    """
    # Filter out unknown/empty methods and low-count methods
    method_counts = df['method'].value_counts()
    methods_to_keep = method_counts[method_counts >= 3].index.tolist()
    # Remove empty method names
    methods_to_keep = [m for m in methods_to_keep if m.strip() != '']
    df_filtered = df[df['method'].isin(methods_to_keep)].copy()
    
    print(f"\nExperimental methods included (n >= 3):")
    for method in sorted(methods_to_keep):
        count = len(df_filtered[df_filtered['method'] == method])
        print(f"  {method}: {count} samples")
    
    if len(df_filtered) == 0:
        print("No data to plot after filtering")
        return
    
    # Create figure with subplots for each method
    n_methods = len(methods_to_keep)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color palette for the three metrics
    metric_colors = {
        'total_reads': '#3498db',           # Blue
        'genome_mapped_reads': '#2ecc71',   # Green
        'crosslinks': '#e74c3c'             # Red
    }
    
    metrics = [
        ('total_reads', 'Total\nReads'),
        ('genome_mapped_reads', 'Genome\nMapped'),
        ('crosslinks', 'Crosslinks')
    ]
    
    # Calculate global y-axis limits across all data
    all_values = []
    for metric_col, _ in metrics:
        all_values.extend(df_filtered[metric_col].values)
    y_min = min(all_values) * 0.5  # Add some padding
    y_max = max(all_values) * 2.0
    
    for method_idx, method in enumerate(sorted(methods_to_keep)):
        ax = axes[method_idx]
        method_data = df_filtered[df_filtered['method'] == method]
        n_samples = len(method_data)
        
        # Prepare data for violin plot
        plot_data = []
        plot_labels = []
        plot_colors = []
        
        for metric_col, metric_label in metrics:
            values = method_data[metric_col].values
            if len(values) > 0:
                plot_data.append(values)
                plot_labels.append(metric_label)
                plot_colors.append(metric_colors[metric_col])
        
        # Create violin plot
        parts = ax.violinplot(
            plot_data,
            positions=range(len(plot_data)),
            widths=0.6,
            showmeans=False,
            showmedians=True,
            showextrema=True,
            quantiles=[[0.25, 0.75]] * len(plot_data)  # Show IQR boundaries
        )
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Color the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cquantiles'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5 if partname == 'cmedians' else 1.0)
                if partname == 'cquantiles':
                    parts[partname].set_linestyle('--')
                    parts[partname].set_alpha(0.6)
        
        # Overlay individual points with jitter
        for i, (metric_col, _) in enumerate(metrics):
            values = method_data[metric_col].values
            x_jitter = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(
                x_jitter, values,
                alpha=0.3, s=15, color=plot_colors[i],
                edgecolors='white', linewidth=0.5, zorder=3
            )
        
        # Set y-axis to log scale with shared limits
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        
        # Add median percentage labels to each violin
        for i, (metric_col, metric_label) in enumerate(metrics):
            values = method_data[metric_col].values
            total_reads = method_data['total_reads'].values
            
            # Calculate median percentage of total reads
            if metric_col == 'total_reads':
                pct_label = '100%'
            else:
                percentages = (values / total_reads) * 100
                median_pct = np.median(percentages)
                pct_label = f'{median_pct:.1f}%'
            
            # Get the median value for positioning
            median_val = np.median(values)
            
            # Place text above the violin
            ax.text(i, median_val * 1.5, pct_label,
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        
        # Formatting
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=10)
        ax.set_ylabel('Count (log scale)', fontsize=11)
        ax.set_title(f'{method}\n(n={n_samples})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Read Statistics by Experimental Method',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_file}")
    plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for method in sorted(methods_to_keep):
        print(f"\n{method} (n={len(df_filtered[df_filtered['method'] == method])}):")
        for metric_col, metric_label in metrics:
            method_data = df_filtered[df_filtered['method'] == method][metric_col].values
            print(f"  {metric_label.replace(chr(10), ' '):20s}: median={np.median(method_data):.0f}, "
                  f"mean={np.mean(method_data):.0f}, "
                  f"range=[{np.min(method_data):.0f}, {np.max(method_data):.0f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate faceted violin plot of read statistics by experimental method",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output', '-o',
        default='read_stats_by_method.png',
        help='Output filename for plot (default: read_stats_by_method.png)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing UMICollapse log files (default: data)'
    )
    parser.add_argument(
        '--trimming-dir',
        default='data_trimming',
        help='Directory containing trimming_report.txt files (default: data_trimming)'
    )
    parser.add_argument(
        '--metadata',
        default='filtered_data.json',
        help='Path to metadata JSON file (default: filtered_data.json)'
    )
    
    args = parser.parse_args()
    
    print("Loading metadata...")
    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata for {len(metadata)} samples")
    
    print("\nCollecting read statistics...")
    df = collect_read_statistics(metadata, args.data_dir, args.trimming_dir)
    
    if len(df) == 0:
        print("No data collected. Make sure you have downloaded both:")
        print("  1. UMICollapse logs in data/")
        print("  2. Trimming report files in data_trimming/")
        return
    
    print("\nGenerating plot...")
    plot_read_statistics_faceted(df, args.output)

    # Save data to CSV for reference
    csv_file = args.output.replace('.png', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved data to {csv_file}")

    # --- Generate separate plots for TDP43, HNRNPC, PTBP1 ---
    # Use normalized names as in metadata: TARDBP, HNRNPC, PTBP1
    protein_targets = ["TARDBP", "HNRNPC", "PTBP1"]
    for prot in protein_targets:
        df_prot = df[df['target'] == prot].copy()
        if len(df_prot) == 0:
            print(f"No data for {prot}, skipping plot.")
            continue
        out_png = f"read_stats_by_method_{prot}.png"
        out_csv = f"read_stats_by_method_{prot}.csv"
        print(f"\nGenerating plot for {prot} ({len(df_prot)} samples)...")
        plot_read_statistics_faceted(df_prot, out_png)
        df_prot.to_csv(out_csv, index=False)
        print(f"Saved data to {out_csv}")


if __name__ == "__main__":
    main()

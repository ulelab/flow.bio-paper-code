#!/usr/bin/env python3
"""
Calculate PCR duplication ratios from UMICollapse log files.
Extracts "Number of input reads" and "Number of reads after deduplicating"
to compute the duplication ratio for each sample.

Can split histograms by experimental method (eCLIP, iCLIP, irCLIP, etc.)
when filtered_data.json with sample metadata is available.
"""

import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTLIER_THRESHOLD = 5.0
METADATA_JSON = "filtered_data.json"


def load_sample_metadata(json_path):
    """
    Load sample metadata from filtered_data.json.
    
    Returns a dict mapping log filename -> experimental_method
    """
    if not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata from {json_path}: {e}")
        return {}
    
    filename_to_method = {}
    for record in data:
        file_info = record.get('file', {})
        filename = file_info.get('filename', '')
        
        sample_meta = record.get('sample_metadata', {})
        nested_meta = sample_meta.get('metadata', {})
        
        # Get experimental method
        method_info = nested_meta.get('experimental_method', {})
        method = method_info.get('value', '') if isinstance(method_info, dict) else ''
        
        if filename and method:
            filename_to_method[filename] = method
    
    return filename_to_method


def parse_log_file(filepath):
    """
    Parse a UMICollapse log file and extract read counts.
    
    Returns:
        tuple: (input_reads, deduplicated_reads) or (None, None) if parsing fails
    """
    input_reads = None
    dedup_reads = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('Number of input reads'):
                    input_reads = int(line.split('\t')[1].strip())
                elif line.startswith('Number of reads after deduplicating'):
                    dedup_reads = int(line.split('\t')[1].strip())
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None
    
    return input_reads, dedup_reads


def calculate_duplication_ratio(input_reads, dedup_reads):
    """
    Calculate PCR duplication ratio.
    
    Ratio = input_reads / deduplicated_reads
    A ratio of 1 means no duplicates, higher values indicate more PCR duplicates.
    """
    if dedup_reads and dedup_reads > 0:
        return input_reads / dedup_reads
    return None


def main():
    # Get the data directory (same directory as script, in 'data' subfolder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Load sample metadata for experimental method info
    metadata_path = os.path.join(script_dir, METADATA_JSON)
    filename_to_method = load_sample_metadata(metadata_path)
    print(f"Loaded metadata for {len(filename_to_method)} files")
    
    # Find all UMICollapse log files
    log_files = glob.glob(os.path.join(data_dir, '*_UMICollapse.log'))
    
    print(f"Found {len(log_files)} log files in {data_dir}")
    
    # Parse each file and calculate ratios
    results = []
    for log_file in log_files:
        filename = os.path.basename(log_file)
        sample_name = filename.replace('.unique_genome.dedup_UMICollapse.log', '')
        input_reads, dedup_reads = parse_log_file(log_file)
        
        # Get experimental method from metadata
        method = filename_to_method.get(filename, 'Unknown')
        
        if input_reads is not None and dedup_reads is not None:
            ratio = calculate_duplication_ratio(input_reads, dedup_reads)
            if ratio is not None:
                results.append({
                    'sample': sample_name,
                    'input_reads': input_reads,
                    'dedup_reads': dedup_reads,
                    'duplication_ratio': ratio,
                    'method': method
                })
                print(f"{sample_name}: {input_reads:,} input -> {dedup_reads:,} dedup (ratio: {ratio:.2f}) [{method}]")
    
    print(f"\nSuccessfully processed {len(results)} samples")
    
    # Show method breakdown
    methods = {}
    for r in results:
        m = r['method']
        methods[m] = methods.get(m, 0) + 1
    print("\nSamples by experimental method:")
    for m, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {m}: {count}")
    
    if not results:
        print("No valid data to plot!")
        return
    
    # Get unique methods (excluding Unknown if there are known methods)
    unique_methods = sorted(set(r['method'] for r in results))
    known_methods = [m for m in unique_methods if m != 'Unknown']
    
    # Define colors for each method
    method_colors = {
        'eCLIP': '#2ecc71',      # Green
        'iCLIP': '#3498db',      # Blue
        'irCLIP': '#9b59b6',     # Purple
        'Re-CLIP': '#e74c3c',    # Red
        'Unknown': '#95a5a6',    # Gray
    }
    
    # Create bins from 1 to threshold with fine resolution
    bins_normal = np.linspace(1, OUTLIER_THRESHOLD, 41)  # 40 bins (0.1 resolution)
    
    # =========================================================================
    # Plot 1: Combined histogram (all methods)
    # =========================================================================
    ratios = [r['duplication_ratio'] for r in results]
    normal_ratios = [r for r in ratios if r <= OUTLIER_THRESHOLD]
    outlier_ratios = [r for r in ratios if r > OUTLIER_THRESHOLD]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(normal_ratios, bins=bins_normal, edgecolor='black', 
            alpha=0.7, color='steelblue', label='Ratio 1-5')
    
    if outlier_ratios:
        outlier_bar_pos = OUTLIER_THRESHOLD + 0.2
        ax.bar(outlier_bar_pos, len(outlier_ratios), width=0.3, 
               color='coral', edgecolor='black', alpha=0.8, label=f'Outliers (>{OUTLIER_THRESHOLD})')
        ax.annotate(f'n={len(outlier_ratios)}\n(max={max(outlier_ratios):.1f})', 
                    xy=(outlier_bar_pos, len(outlier_ratios)), 
                    xytext=(outlier_bar_pos + 0.3, len(outlier_ratios) + 10),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    
    if median_ratio <= OUTLIER_THRESHOLD:
        ax.axvline(median_ratio, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_ratio:.2f}')
    if mean_ratio <= OUTLIER_THRESHOLD:
        ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
    
    ax.set_xlabel('PCR Duplication Ratio (Input Reads / Deduplicated Reads)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Distribution of PCR Duplication Ratios - All Samples\n(High resolution 1-5, outliers grouped)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0.9, OUTLIER_THRESHOLD + 0.7)
    
    stats_text = (f'Total N = {len(ratios)}\n'
                  f'In range (1-5): {len(normal_ratios)}\n'
                  f'Outliers (>5): {len(outlier_ratios)}\n'
                  f'Overall Mean: {mean_ratio:.2f}\n'
                  f'Overall Median: {median_ratio:.2f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(script_dir, 'pcr_duplication_histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")
    plt.close()
    
    # =========================================================================
    # Plot 2: Split by experimental method (subplots)
    # =========================================================================
    if known_methods:
        # Include Unknown only if it has samples
        plot_methods = known_methods + (['Unknown'] if 'Unknown' in unique_methods else [])
        n_methods = len(plot_methods)
        
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods), sharex=True)
        if n_methods == 1:
            axes = [axes]
        
        for ax, method in zip(axes, plot_methods):
            method_results = [r for r in results if r['method'] == method]
            method_ratios = [r['duplication_ratio'] for r in method_results]
            
            normal = [r for r in method_ratios if r <= OUTLIER_THRESHOLD]
            outliers = [r for r in method_ratios if r > OUTLIER_THRESHOLD]
            
            color = method_colors.get(method, '#7f8c8d')
            
            ax.hist(normal, bins=bins_normal, edgecolor='black', 
                    alpha=0.7, color=color, label=f'{method} (1-5)')
            
            if outliers:
                outlier_bar_pos = OUTLIER_THRESHOLD + 0.2
                ax.bar(outlier_bar_pos, len(outliers), width=0.3, 
                       color='coral', edgecolor='black', alpha=0.8)
                ax.annotate(f'n={len(outliers)}', 
                            xy=(outlier_bar_pos, len(outliers)), 
                            xytext=(outlier_bar_pos + 0.15, len(outliers) + 1),
                            fontsize=8, ha='left')
            
            if method_ratios:
                mean_r = np.mean(method_ratios)
                median_r = np.median(method_ratios)
                
                if median_r <= OUTLIER_THRESHOLD:
                    ax.axvline(median_r, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
                if mean_r <= OUTLIER_THRESHOLD:
                    ax.axvline(mean_r, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                
                stats = f'{method}\nN={len(method_ratios)}, Med={median_r:.2f}, Mean={mean_r:.2f}'
            else:
                stats = f'{method}\nN=0'
            
            ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlim(0.9, OUTLIER_THRESHOLD + 0.7)
        
        axes[-1].set_xlabel('PCR Duplication Ratio (Input Reads / Deduplicated Reads)', fontsize=12)
        fig.suptitle('PCR Duplication Ratios by Experimental Method', fontsize=14, y=1.02)
        
        plt.tight_layout()
        output_path_split = os.path.join(script_dir, 'pcr_duplication_histogram_by_method.png')
        plt.savefig(output_path_split, dpi=150, bbox_inches='tight')
        print(f"Split histogram saved to: {output_path_split}")
        plt.close()
        
        # =========================================================================
        # Plot 3: Overlaid histogram (all methods on same axes)
        # =========================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method in known_methods:
            method_results = [r for r in results if r['method'] == method]
            method_ratios = [r['duplication_ratio'] for r in method_results]
            normal = [r for r in method_ratios if r <= OUTLIER_THRESHOLD]
            
            color = method_colors.get(method, '#7f8c8d')
            
            ax.hist(normal, bins=bins_normal, edgecolor='black', linewidth=0.5,
                    alpha=0.5, color=color, label=f'{method} (n={len(method_ratios)})')
        
        ax.set_xlabel('PCR Duplication Ratio (Input Reads / Deduplicated Reads)', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('PCR Duplication Ratios by Experimental Method (Overlaid)\n(Showing ratios 1-5 only)', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(0.9, OUTLIER_THRESHOLD)
        
        plt.tight_layout()
        output_path_overlay = os.path.join(script_dir, 'pcr_duplication_histogram_overlay.png')
        plt.savefig(output_path_overlay, dpi=150, bbox_inches='tight')
        print(f"Overlay histogram saved to: {output_path_overlay}")
        plt.close()
    
    # =========================================================================
    # Plot 4: Box plot by purification target
    # =========================================================================
    plot_by_purification_target(results, script_dir)


def load_purification_targets(json_path):
    """
    Load purification target info from filtered_data.json.
    
    Returns a dict mapping log filename -> purification_target
    """
    if not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return {}
    
    filename_to_target = {}
    for record in data:
        file_info = record.get('file', {})
        filename = file_info.get('filename', '')
        
        sample_meta = record.get('sample_metadata', {})
        nested_meta = sample_meta.get('metadata', {})
        
        target_info = nested_meta.get('purification_target', {})
        target = target_info.get('value', '') if isinstance(target_info, dict) else ''
        
        if filename:
            filename_to_target[filename] = target if target else 'Unknown'
    
    return filename_to_target


def plot_by_purification_target(results, script_dir, top_n=25):
    """
    Create box plot of PCR duplication ratios by purification target.
    """
    # Load purification target metadata
    metadata_path = os.path.join(script_dir, METADATA_JSON)
    filename_to_target = load_purification_targets(metadata_path)
    
    if not filename_to_target:
        print("No purification target metadata available, skipping box plot")
        return
    
    # Assign targets to results
    for r in results:
        filename = r['sample'] + '.unique_genome.dedup_UMICollapse.log'
        r['target'] = filename_to_target.get(filename, 'Unknown')
    
    # Count samples per target
    target_counts = {}
    for r in results:
        t = r['target']
        target_counts[t] = target_counts.get(t, 0) + 1
    
    # Get top N targets by sample count (exclude Unknown and controls)
    exclude = {'Unknown', 'GFP', 'IgG', 'input'}
    ranked_targets = sorted(
        [(t, c) for t, c in target_counts.items() if t not in exclude],
        key=lambda x: -x[1]
    )[:top_n]
    top_targets = [t for t, _ in ranked_targets]
    
    # Also add controls as a separate group
    controls = ['GFP', 'IgG', 'input']
    controls_present = [c for c in controls if c in target_counts]
    
    # Prepare data for box plot
    plot_targets = top_targets + controls_present
    data_by_target = {t: [] for t in plot_targets}
    
    for r in results:
        if r['target'] in data_by_target:
            data_by_target[r['target']].append(r['duplication_ratio'])
    
    # Sort by median ratio
    target_medians = [(t, np.median(data_by_target[t])) for t in plot_targets if data_by_target[t]]
    target_medians.sort(key=lambda x: x[1])
    sorted_targets = [t for t, _ in target_medians]
    
    # Prepare box plot data
    box_data = [data_by_target[t] for t in sorted_targets]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create box plot
    bp = ax.boxplot(box_data, vert=True, patch_artist=True, 
                     showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    
    # Color boxes
    control_color = '#e74c3c'  # Red for controls
    target_color = '#3498db'   # Blue for targets
    
    for i, (patch, target) in enumerate(zip(bp['boxes'], sorted_targets)):
        if target in controls:
            patch.set_facecolor(control_color)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(target_color)
            patch.set_alpha(0.7)
    
    # Labels
    labels = [f"{t}\n(n={len(data_by_target[t])})" for t in sorted_targets]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel('PCR Duplication Ratio', fontsize=12)
    ax.set_xlabel('Purification Target', fontsize=12)
    ax.set_title(f'PCR Duplication Ratio by Purification Target (Top {top_n} + Controls)\nSorted by median ratio', fontsize=14)
    
    # Add horizontal line at ratio=1 (no duplicates)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='No duplicates (ratio=1)')
    
    # Set y-axis limit to focus on main data (cap at 15 for visibility)
    ax.set_ylim(0.8, 15)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=target_color, alpha=0.7, label='RBP targets'),
        Patch(facecolor=control_color, alpha=0.7, label='Controls (GFP/IgG/input)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    output_path = os.path.join(script_dir, 'pcr_duplication_by_target.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Box plot by target saved to: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n=== PCR Duplication by Purification Target (sorted by median) ===")
    for t in sorted_targets:
        ratios = data_by_target[t]
        print(f"  {t}: n={len(ratios)}, median={np.median(ratios):.2f}, mean={np.mean(ratios):.2f}, max={max(ratios):.1f}")


if __name__ == '__main__':
    main()

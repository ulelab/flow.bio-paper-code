#!/usr/bin/env python3
"""
Condensate Clustering Analysis

This script analyzes whether proteins in the same condensate tend to cluster
together in the motif-based dendrogram (similar RNA-binding preferences).

Output:
- condensate_clustering.json: Data for circos highlighting
- condensate_clustering_analysis.png: Bar chart visualization

Metrics:
- Clustering score: How much more clustered than random (>1 = clustered)
- Contiguity: Fraction of proteins in largest contiguous block
"""

import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict
import os
import re


def load_data():
    """Load all required data."""
    # Metadata
    with open('filtered_data.json') as f:
        metadata = json.load(f)
    
    filename_info = {}
    for r in metadata:
        fn = r.get('file', {}).get('filename', '').replace('.unique_genome.dedup_UMICollapse.log', '')
        t = r.get('sample_metadata', {}).get('metadata', {}).get('purification_target', {})
        t = t.get('value', '') if isinstance(t, dict) else ''
        if fn:
            target = t.upper() if t else 'UNKNOWN'
            if target == 'TDP43':
                target = 'TARDBP'
            filename_info[fn] = target
    
    # PEKA data
    peka_data = {}
    for f in glob.glob('data_peka/*.tsv'):
        s = f.split('/')[-1].replace('.genome_5mer_distribution_genome.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t')
            peka_data[s] = dict(zip(df.iloc[:, 0], df['PEKA-score']))
        except:
            pass
    
    # Log data for filtering
    log_data = {}
    for fn in os.listdir('data'):
        if fn.endswith('_UMICollapse.log'):
            sample_name = fn.replace('.unique_genome.dedup_UMICollapse.log', '')
            with open(f'data/{fn}') as f:
                content = f.read()
            input_match = re.search(r'Number of input reads\s+(\d+)', content)
            dedup_match = re.search(r'Number of reads after deduplicating\s+(\d+)', content)
            if input_match and dedup_match:
                log_data[sample_name] = {
                    'pcr_ratio': int(input_match.group(1)) / int(dedup_match.group(1)),
                    'crosslinks': int(dedup_match.group(1))
                }
    
    # Condensate data
    cond_df = pd.read_csv('../data/protein2cdcode_v2.0.tsv', sep='\t')
    mapping_df = pd.read_csv('uniprot_to_gene.csv')
    uniprot_to_gene = dict(zip(mapping_df['uniprot_id'], mapping_df['gene_name']))
    
    protein_condensates = defaultdict(set)
    all_condensate_proteins = defaultdict(set)  # All proteins per condensate from database
    
    for _, row in cond_df.iterrows():
        cond_name = row['condensate_name']
        gene_raw = uniprot_to_gene.get(row['uniprotkb_ac'], '')
        
        # Track all proteins in each condensate (even if we don't have CLIP data)
        if not pd.isna(gene_raw) and gene_raw != '':
            gene = str(gene_raw).upper()
            all_condensate_proteins[cond_name].add(gene)
            protein_condensates[gene].add(cond_name)
    
    return filename_info, peka_data, log_data, protein_condensates, all_condensate_proteins


def compute_clustering_order(samples, peka_data, filename_info):
    """Compute hierarchical clustering order of proteins."""
    all_kmers = list(samples[0]['peka'].keys())
    
    # Compute mean PEKA vector per protein
    protein_vectors = {}
    proteins = list(set(s['target'] for s in samples))
    
    for p in proteins:
        p_samples = [s for s in samples if s['target'] == p]
        vectors = []
        for s in p_samples:
            vec = [s['peka'].get(k, 0) if np.isfinite(s['peka'].get(k, 0)) else 0 for k in all_kmers]
            vectors.append(vec)
        protein_vectors[p] = np.nan_to_num(np.nanmean(vectors, axis=0))
    
    # Hierarchical clustering
    protein_names = list(protein_vectors.keys())
    protein_matrix = np.array([protein_vectors[p] for p in protein_names])
    Z = linkage(protein_matrix, method='ward')
    dendro = dendrogram(Z, no_plot=True)
    protein_order = [protein_names[i] for i in dendro['leaves']]
    
    return protein_order


def calculate_condensate_stats(protein_order, protein_condensates, all_condensate_proteins):
    """Calculate clustering statistics for each condensate.
    
    Args:
        protein_order: List of proteins in clustering order
        protein_condensates: Dict mapping protein -> set of condensates (for proteins in our dataset)
        all_condensate_proteins: Dict mapping condensate -> set of all proteins (from database)
    """
    position = {p: i for i, p in enumerate(protein_order)}
    n = len(protein_order)
    
    condensate_stats = []
    all_condensates = set(c for conds in protein_condensates.values() for c in conds)
    
    for cond_name in all_condensates:
        # Get proteins in this condensate that are in our dataset
        members = [p for p in protein_order if cond_name in protein_condensates.get(p, set())]
        
        if len(members) < 3:  # Need at least 3 to be meaningful
            continue
        
        # Get positions in clustering
        positions = sorted([position[p] for p in members])
        
        # Calculate mean pairwise distance in cluster order
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(abs(positions[j] - positions[i]))
        mean_dist = np.mean(distances)
        
        # Expected mean distance if randomly distributed
        expected_dist = n / 3
        
        # Clustering score: how much more clustered than random
        clustering_score = expected_dist / mean_dist if mean_dist > 0 else 0
        
        # Calculate contiguity: fraction of members in largest contiguous block
        max_block = 1
        current_block = 1
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] <= 3:  # Allow small gaps
                current_block += 1
                max_block = max(max_block, current_block)
            else:
                current_block = 1
        contiguity = max_block / len(members)
        
        # Get total proteins in this condensate from database
        total_in_condensate = len(all_condensate_proteins.get(cond_name, set()))
        
        # Sort members by their position to get the most central proteins
        # (those in the middle of the largest cluster block)
        member_positions = [(p, position[p]) for p in members]
        member_positions.sort(key=lambda x: x[1])
        
        # Find the most tightly clustered subset (consecutive proteins)
        best_start = 0
        best_density = 0
        for start in range(len(member_positions)):
            for end in range(start + 2, min(start + 5, len(member_positions) + 1)):
                span = member_positions[end-1][1] - member_positions[start][1] + 1
                density = (end - start) / span if span > 0 else 0
                if density > best_density:
                    best_density = density
                    best_start = start
        
        # Get the core proteins (most tightly clustered)
        core_proteins = [p for p, _ in member_positions[best_start:best_start+3]]
        
        condensate_stats.append({
            'condensate': cond_name,
            'n_proteins': len(members),
            'total_proteins': total_in_condensate,
            'mean_distance': mean_dist,
            'clustering_score': clustering_score,
            'contiguity': contiguity,
            'proteins': members,
            'core_proteins': core_proteins  # Most tightly clustered proteins
        })
    
    # Filter out synthetic condensates
    condensate_stats = [s for s in condensate_stats if 'Synthetic' not in s['condensate']]
    
    # Sort by clustering score (descending)
    condensate_stats.sort(key=lambda x: -x['clustering_score'])
    
    return condensate_stats


def plot_clustering_analysis(condensate_stats, output_file='condensate_clustering_analysis.png'):
    """Create visualization of condensate clustering."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Ensure no overlap: take top 15 and bottom 15 from non-overlapping sets
    n_show = min(15, len(condensate_stats) // 2)
    top_n = condensate_stats[:n_show]
    bottom_n = condensate_stats[-n_show:][::-1]
    
    # Left: Top N most clustered
    ax1 = axes[0]
    
    scores = [s['clustering_score'] for s in top_n]
    colors = ['#e74c3c' if s['clustering_score'] > 1.5 else 
              '#f39c12' if s['clustering_score'] > 1.2 else '#3498db' 
              for s in top_n]
    
    # Check if we have a large outlier (first value much larger than second)
    has_outlier = len(scores) > 1 and scores[0] > scores[1] * 4
    
    if has_outlier:
        # Clip the outlier bar and add break marks
        break_point = scores[1] * 1.4  # Just above second highest
        
        # Plot all bars, clipping the first one
        for i, (stat, color) in enumerate(zip(top_n, colors)):
            score = min(stat['clustering_score'], break_point)
            bar = ax1.barh(i, score, color=color)
            
            # Add break marks to the clipped bar
            if i == 0:
                # Add diagonal break lines at the end of the clipped bar
                bar_right = break_point
                bar_y = i
                # Draw break marks
                ax1.plot([bar_right - 0.08, bar_right + 0.08], [bar_y - 0.25, bar_y + 0.25], 
                        color='white', lw=2, clip_on=True)
                ax1.plot([bar_right - 0.15, bar_right + 0.01], [bar_y - 0.25, bar_y + 0.25], 
                        color='white', lw=2, clip_on=True)
        
        ax1.set_xlim(0, break_point * 1.02)
        
        # Add annotation showing the actual value for the outlier
        ax1.annotate(f'→ {scores[0]:.1f}', 
                     xy=(break_point, 0), xytext=(5, 0),
                     textcoords='offset points', fontsize=10, va='center', 
                     fontweight='bold', color='#e74c3c')
        
        # Add protein labels starting from second entry (core proteins = most tightly clustered)
        for i, stat in enumerate(top_n[1:5], 1):
            ax1.annotate(', '.join(stat.get('core_proteins', stat['proteins'][:3])),
                         xy=(stat['clustering_score'], i),
                         xytext=(5, 0), textcoords='offset points',
                         fontsize=7, alpha=0.7, va='center')
    else:
        # No outlier - simple plot
        ax1.barh(range(len(top_n)), scores, color=colors)
        
        # Add protein labels for top 5 (core proteins = most tightly clustered)
        for i, stat in enumerate(top_n[:5]):
            ax1.annotate(', '.join(stat.get('core_proteins', stat['proteins'][:3])),
                         xy=(stat['clustering_score'], i),
                         xytext=(5, 0), textcoords='offset points',
                         fontsize=7, alpha=0.7, va='center')
    
    ax1.set_yticks(range(len(top_n)))
    ax1.set_yticklabels([f"{s['condensate']} ({s['n_proteins']}/{s['total_proteins']})" for s in top_n], fontsize=9)
    ax1.set_xlabel('Clustering Score\n(higher = proteins cluster together by motif)', fontsize=11)
    ax1.set_title('Most Clustered Condensates\n(Similar RNA-binding preferences)', 
                  fontsize=12, fontweight='bold')
    ax1.axvline(x=1, color='grey', linestyle='--', alpha=0.5, label='Random expectation')
    ax1.invert_yaxis()
    ax1.legend(loc='lower right')
    
    # Right: Bottom N (least clustered)
    ax2 = axes[1]
    colors = ['#2ecc71' if s['clustering_score'] < 0.9 else '#95a5a6' for s in bottom_n]
    
    ax2.barh(range(len(bottom_n)), [s['clustering_score'] for s in bottom_n], color=colors)
    ax2.set_yticks(range(len(bottom_n)))
    ax2.set_yticklabels([f"{s['condensate']} ({s['n_proteins']}/{s['total_proteins']})" for s in bottom_n], fontsize=9)
    ax2.set_xlabel('Clustering Score\n(lower = proteins scattered across dendrogram)', fontsize=11)
    ax2.set_title('Least Clustered Condensates\n(Diverse RNA-binding preferences)', 
                  fontsize=12, fontweight='bold')
    ax2.axvline(x=1, color='grey', linestyle='--', alpha=0.5, label='Random expectation')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("Loading data...")
    filename_info, peka_data, log_data, protein_condensates, all_condensate_proteins = load_data()
    
    # Build filtered dataset
    samples = []
    for sample, peka in peka_data.items():
        if sample not in log_data:
            continue
        if log_data[sample]['crosslinks'] < 150000 or log_data[sample]['pcr_ratio'] >= 5:
            continue
        samples.append({
            'sample': sample,
            'target': filename_info.get(sample, 'Unknown'),
            'peka': peka
        })
    
    print(f"Samples after filtering: {len(samples)}")
    
    # Get clustering order
    print("Computing hierarchical clustering...")
    protein_order = compute_clustering_order(samples, peka_data, filename_info)
    print(f"Proteins in clustering: {len(protein_order)}")
    
    # Calculate condensate stats
    print("Calculating condensate clustering statistics...")
    condensate_stats = calculate_condensate_stats(protein_order, protein_condensates, all_condensate_proteins)
    
    # Save results for circos
    output_data = {
        'top_clustered': [s['condensate'] for s in condensate_stats[:10]],
        'stats': {
            s['condensate']: {
                'score': s['clustering_score'],
                'n': s['n_proteins'],
                'contiguity': s['contiguity'],
                'proteins': s['proteins'][:10]  # First 10 proteins
            } for s in condensate_stats
        }
    }
    
    with open('condensate_clustering.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("Saved: condensate_clustering.json")
    
    # Create visualization
    print("Generating plot...")
    plot_clustering_analysis(condensate_stats)
    
    # Print summary
    print("\n" + "=" * 70)
    print("MOST CLUSTERED (proteins share RNA-binding preferences):")
    print("=" * 70)
    for stat in condensate_stats[:5]:
        print(f"  {stat['condensate']}: score={stat['clustering_score']:.2f}, "
              f"n={stat['n_proteins']}, contiguity={stat['contiguity']:.0%}")
    
    print("\n" + "=" * 70)
    print("LEAST CLUSTERED (diverse RNA-binding preferences):")
    print("=" * 70)
    for stat in condensate_stats[-5:]:
        print(f"  {stat['condensate']}: score={stat['clustering_score']:.2f}, "
              f"n={stat['n_proteins']}, contiguity={stat['contiguity']:.0%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze and plot binding specificity of RNA-binding proteins using PEKA scores.

Specificity is measured by "concentration" = proportion of total PEKA signal
contained in the top 5 motifs. Higher concentration = more specific binding.

Usage:
    python plot_binding_specificity.py
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(min_crosslinks=150000, max_pcr_ratio=5):
    """Load and filter all required data sources."""
    
    # Load metadata
    with open('filtered_data.json') as f:
        meta = json.load(f)
    
    fn_info = {}
    for r in meta:
        fn = r.get('file', {}).get('filename', '').replace('.unique_genome.dedup_UMICollapse.log', '')
        t = r.get('sample_metadata', {}).get('metadata', {}).get('purification_target', {})
        t = t.get('value', '') if isinstance(t, dict) else ''
        if fn:
            target = t.upper() if t else 'UNKNOWN'
            if target == 'TDP43':
                target = 'TARDBP'
            fn_info[fn] = target
    
    # Load log data (PCR ratio, crosslinks)
    log_data = {}
    for f in glob.glob('data/*_UMICollapse.log'):
        s = os.path.basename(f).replace('.unique_genome.dedup_UMICollapse.log', '')
        try:
            lines = open(f).readlines()
            inp = int([l for l in lines if 'input reads' in l][0].split('\t')[1])
            ded = int([l for l in lines if 'after deduplicating' in l][0].split('\t')[1])
            log_data[s] = {'pcr': inp / ded, 'xl': ded}
        except:
            pass
    
    # Load PEKA data
    peka = {}
    for f in glob.glob('data_peka/*.tsv'):
        s = os.path.basename(f).replace('.genome_5mer_distribution_genome.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t')
            peka[s] = dict(zip(df.iloc[:, 0], df['PEKA-score']))
        except:
            pass
    
    # Filter samples
    filtered_peka = {}
    filtered_targets = {}
    for s in peka:
        if s not in log_data:
            continue
        if log_data[s]['xl'] < min_crosslinks or log_data[s]['pcr'] >= max_pcr_ratio:
            continue
        filtered_peka[s] = peka[s]
        filtered_targets[s] = fn_info.get(s, 'UNKNOWN')
    
    return filtered_peka, filtered_targets


def calculate_concentration(scores, top_n=5):
    """
    Calculate concentration: proportion of total PEKA signal in top N motifs.
    
    Higher concentration = more specific binding (signal concentrated in few motifs)
    Lower concentration = promiscuous binding (signal spread across many motifs)
    """
    vals = np.array([v for v in scores.values() if np.isfinite(v) and v > 0])
    if len(vals) < top_n:
        return None
    vals_sorted = np.sort(vals)[::-1]
    return vals_sorted[:top_n].sum() / vals.sum()


def create_specificity_plots(output_file='binding_specificity_concentration.png',
                             min_samples=3):
    """Generate binding specificity analysis plots."""
    
    print("Loading data...")
    peka, targets = load_data()
    print(f"Samples after filtering: {len(peka)}")
    
    # Calculate concentration for each sample
    rows = []
    for s, scores in peka.items():
        conc = calculate_concentration(scores)
        if conc:
            rows.append({'sample': s, 'target': targets[s], 'concentration': conc})
    
    df = pd.DataFrame(rows)
    print(f"Samples with valid concentration: {len(df)}")
    print(f"Unique proteins: {df['target'].nunique()}")
    
    # Aggregate by protein
    protein_conc = df.groupby('target').agg({
        'concentration': ['mean', 'std'],
        'sample': 'count'
    })
    protein_conc.columns = ['conc_mean', 'conc_std', 'n_samples']
    protein_conc = protein_conc[protein_conc['n_samples'] >= min_samples]
    protein_conc = protein_conc.sort_values('conc_mean', ascending=False)
    
    print(f"\nProteins with >= {min_samples} samples: {len(protein_conc)}")
    
    # Print results
    print("\n=== Top 15 MOST SPECIFIC (high concentration) ===")
    print(protein_conc.head(15).to_string())
    
    print("\n=== Top 15 LEAST SPECIFIC (low concentration) ===")
    print(protein_conc.tail(15).to_string())
    
    # Proteins of interest
    proteins_of_interest = ['RBFOX2', 'FUS', 'PTBP1', 'HNRNPC', 'SRSF1', 'TARDBP', 'ELAVL1', 'U2AF2']
    print("\n=== Proteins of interest ===")
    for p in proteins_of_interest:
        if p in protein_conc.index:
            m = protein_conc.loc[p]
            print(f"{p}: concentration={m['conc_mean']:.3f} ± {m['conc_std']:.3f}, n={int(m['n_samples'])}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histogram of concentration
    ax = axes[0, 0]
    ax.hist(protein_conc['conc_mean'], bins=25, edgecolor='black', alpha=0.7, color='#3498db')
    for p, col in [('FUS', 'red'), ('RBFOX2', 'blue'), ('PTBP1', 'green')]:
        if p in protein_conc.index:
            ax.axvline(protein_conc.loc[p, 'conc_mean'], color=col, linestyle='--', lw=2, label=p)
    ax.set_xlabel('Concentration (top-5 / total)', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Distribution of Binding Specificity', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 2. Bar plot of top/bottom proteins
    ax = axes[0, 1]
    top10 = protein_conc.head(10)
    bot10 = protein_conc.tail(10)
    combined = pd.concat([top10, bot10])
    colors = ['#2ecc71'] * 10 + ['#e74c3c'] * 10
    y = range(len(combined))
    ax.barh(y, combined['conc_mean'], xerr=combined['conc_std'], color=colors, alpha=0.7, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(combined.index, fontsize=9)
    ax.set_xlabel('Concentration', fontsize=12)
    ax.set_title('Most vs Least Specific Proteins', fontsize=14, fontweight='bold')
    ax.axhline(9.5, color='black', linestyle='-', lw=1)
    
    # 3. PEKA distribution comparison
    ax = axes[1, 0]
    for p, col, label in [('PTBP1', '#2ecc71', 'PTBP1 (most specific)'),
                          ('RBFOX2', '#3498db', 'RBFOX2'),
                          ('FUS', '#e74c3c', 'FUS'),
                          ('SRSF1', '#f39c12', 'SRSF1 (least specific)')]:
        samples = df[df['target'] == p]['sample'].values
        if len(samples) > 0:
            scores = sorted([v for v in peka[samples[0]].values() 
                           if np.isfinite(v) and v > 0], reverse=True)
            ax.plot(range(min(100, len(scores))), scores[:100], color=col, label=label, lw=2)
    ax.set_xlabel('Motif Rank', fontsize=12)
    ax.set_ylabel('PEKA Score', fontsize=12)
    ax.set_title('PEKA Score Distribution (top 100 motifs)', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 4. Top motifs for specific vs non-specific
    ax = axes[1, 1]
    proteins_compare = ['PTBP1', 'RBFOX2', 'FUS', 'SRSF1']
    colors_dict = {'PTBP1': '#2ecc71', 'RBFOX2': '#3498db', 'FUS': '#e74c3c', 'SRSF1': '#f39c12'}
    width = 0.2
    y = np.arange(5)
    
    for i, p in enumerate(proteins_compare):
        samples = df[df['target'] == p]['sample'].values
        if len(samples) > 0:
            top5 = sorted(peka[samples[0]].items(), 
                         key=lambda x: -x[1] if np.isfinite(x[1]) else 0)[:5]
            scores = [x[1] for x in top5]
            ax.barh(y + i * width, scores, width, label=p, color=colors_dict[p], alpha=0.8)
    
    ax.set_yticks(y + 1.5 * width)
    ax.set_yticklabels(['#1', '#2', '#3', '#4', '#5'])
    ax.set_xlabel('PEKA Score', fontsize=12)
    ax.set_title('Top 5 Motifs by Protein', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    return df, protein_conc


if __name__ == '__main__':
    create_specificity_plots()

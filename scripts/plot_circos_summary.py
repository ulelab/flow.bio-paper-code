#!/usr/bin/env python3
"""
Compact circos summary - method, PCR duplication, crosslinks, plus similarity, specificity, recall
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from collections import Counter
import os
import re
import glob

def load_data():
    """Load all data"""
    with open('filtered_data.json') as f:
        metadata = json.load(f)
    
    filename_info = {}
    for r in metadata:
        fn = r.get('file', {}).get('filename', '').replace('.unique_genome.dedup_UMICollapse.log', '')
        meta = r.get('sample_metadata', {}).get('metadata', {})
        
        t = meta.get('purification_target', {})
        t = t.get('value', '') if isinstance(t, dict) else ''
        target = t.upper() if t else 'UNKNOWN'
        if target == 'TDP43':
            target = 'TARDBP'
        
        m = meta.get('experimental_method', {})
        method = m.get('value', '') if isinstance(m, dict) else ''
        
        if fn:
            filename_info[fn] = {'target': target, 'method': method}
    
    # Load log data
    log_data = {}
    for fn in os.listdir('data'):
        if fn.endswith('_UMICollapse.log'):
            sample_name = fn.replace('.unique_genome.dedup_UMICollapse.log', '')
            filepath = os.path.join('data', fn)
            with open(filepath) as f:
                content = f.read()
            input_match = re.search(r'Number of input reads\s+(\d+)', content)
            dedup_match = re.search(r'Number of reads after deduplicating\s+(\d+)', content)
            if input_match and dedup_match:
                input_reads = int(input_match.group(1))
                dedup_reads = int(dedup_match.group(1))
                ratio = input_reads / dedup_reads if dedup_reads > 0 else float('inf')
                log_data[sample_name] = {'pcr_ratio': ratio, 'crosslinks': dedup_reads}
    
    return filename_info, log_data


def load_peka_data(data_dir='data_peka'):
    """Load PEKA scores from 5-mer distribution TSV files"""
    peka_data = {}
    for f in glob.glob(os.path.join(data_dir, '*.tsv')):
        s = os.path.basename(f).replace('.genome_5mer_distribution_genome.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t')
            peka_data[s] = dict(zip(df.iloc[:, 0], df['PEKA-score']))
        except Exception:
            pass
    return peka_data


def load_recall_scores(csv_file='recall_by_method_results.csv'):
    """Load recall scores"""
    try:
        df = pd.read_csv(csv_file)
        recall_scores = {}
        for _, row in df.iterrows():
            peka_file = row['peka_file']
            sample = peka_file.replace('.genome_5mer_distribution_genome.tsv', '')
            recall_scores[sample] = row['method6_recall_0.5']
        return recall_scores
    except FileNotFoundError:
        return {}


def calculate_concentration(peka_scores, top_n=5):
    """Calculate concentration (specificity)"""
    vals = np.array([v for v in peka_scores.values() if np.isfinite(v) and v > 0])
    if len(vals) < top_n:
        return 0
    vals_sorted = np.sort(vals)[::-1]
    return vals_sorted[:top_n].sum() / vals.sum()


def get_top_kmers(peka_scores, n=50):
    """Get top n k-mers by PEKA score"""
    v = {k: v for k, v in peka_scores.items() if np.isfinite(v)}
    if not v:
        return set()
    return set(k for k, _ in sorted(v.items(), key=lambda x: -x[1])[:n])


def calculate_similarity_scores(samples_data):
    """Calculate similarity scores for samples"""
    # Pre-compute top 50 k-mers for all samples
    top_kmers = [get_top_kmers(s.get('peka_scores', {}), 50) for s in samples_data]
    targets = [s['target'] for s in samples_data]
    
    similarity_scores = []
    for i in range(len(samples_data)):
        if not top_kmers[i]:
            similarity_scores.append(0)
            continue
        
        overlaps = []
        for j in range(len(samples_data)):
            if i == j:
                continue
            if targets[i] == targets[j]:
                continue
            if not top_kmers[j]:
                continue
            overlap = len(top_kmers[i] & top_kmers[j]) / 50
            overlaps.append(overlap)
        
        similarity_scores.append(np.mean(overlaps) if overlaps else 0)
    
    return similarity_scores


def create_summary_circos():
    """Create compact summary circos with method, PCR dup, crosslinks, similarity, specificity, recall"""
    metadata, log_data = load_data()
    peka_data = load_peka_data()
    recall_data = load_recall_scores()
    
    # Filter samples and include all data
    samples = []
    for s in set(metadata.keys()) & set(log_data.keys()):
        if log_data[s]['crosslinks'] >= 150000 and log_data[s]['pcr_ratio'] < 5:
            samples.append({
                'name': s,
                'target': metadata[s]['target'],
                'method': metadata[s]['method'],
                'pcr_ratio': log_data[s]['pcr_ratio'],
                'crosslinks': log_data[s]['crosslinks'],
                'peka_scores': peka_data.get(s, {}),
                'recall': recall_data.get(s, np.nan),
            })
    
    # Calculate concentration (specificity) for each sample
    for s in samples:
        s['concentration'] = calculate_concentration(s['peka_scores']) if s['peka_scores'] else 0
    
    # Calculate similarity scores
    print("Calculating similarity scores...")
    sim_scores = calculate_similarity_scores(samples)
    for i, s in enumerate(samples):
        s['similarity'] = sim_scores[i]
    
    # Sort by method then crosslinks
    samples.sort(key=lambda x: (x['method'], -x['crosslinks']))
    
    n_samples = len(samples)
    n_proteins = len(set(s['target'] for s in samples))
    n_methods = len(set(s['method'] for s in samples))
    
    print(f"Samples: {n_samples}, Proteins: {n_proteins}, Methods: {n_methods}")
    
    # Method colors
    method_colors = {
        'eCLIP': '#58a6ff', 'iCLIP': '#f778ba', 'iCLIP2': '#ff7b72',
        'iiCLIP': '#ffa657', 'irCLIP': '#7ee787', 'FLASH': '#d2a8ff',
        'Re-CLIP': '#79c0ff'
    }
    
    # Colormaps
    pcr_cmap = cm.RdYlGn_r
    pcr_norm = Normalize(1, 5)
    xl_cmap = cm.Greys
    xl_norm = LogNorm(vmin=150000, vmax=max(s['crosslinks'] for s in samples))
    
    # New colormaps for similarity, specificity, recall
    sim_vals = [s['similarity'] for s in samples]
    sim_cmap = cm.viridis
    sim_norm = Normalize(min(sim_vals), max(sim_vals))
    
    conc_vals = [s['concentration'] for s in samples]
    conc_cmap = cm.Greys
    conc_norm = Normalize(min(conc_vals), max(conc_vals))
    
    recall_vals = [s['recall'] for s in samples if not np.isnan(s['recall'])]
    recall_cmap = cm.RdYlGn
    recall_norm = Normalize(min(recall_vals) if recall_vals else 0, max(recall_vals) if recall_vals else 1)
    
    # Create compact figure
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_axes([0.08, 0.08, 0.60, 0.84], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.axis('off')
    
    n = len(samples)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = 2*np.pi / n * 0.95
    
    # Ring heights and positions (6 rings now)
    h = 0.08
    r1 = 1.0    # Method (outermost)
    r2 = 0.91   # PCR Duplication
    r3 = 0.82   # Crosslinks
    r4 = 0.73   # Recall
    r5 = 0.64   # Specificity
    r6 = 0.55   # Similarity (innermost)
    
    # Ring 1: Method (outermost)
    for i, s in enumerate(samples):
        color = method_colors.get(s['method'], '#888888')
        ax.bar(angles[i], h, width=width, bottom=r1-h, color=color, edgecolor='none')
    
    # Ring 2: PCR Duplication
    for i, s in enumerate(samples):
        color = pcr_cmap(pcr_norm(s['pcr_ratio']))
        ax.bar(angles[i], h, width=width, bottom=r2-h, color=color, edgecolor='none')
    
    # Ring 3: Crosslinks
    for i, s in enumerate(samples):
        color = xl_cmap(xl_norm(s['crosslinks']))
        ax.bar(angles[i], h, width=width, bottom=r3-h, color=color, edgecolor='none')
    
    # Ring 4: Recall
    for i, s in enumerate(samples):
        if np.isnan(s['recall']):
            color = '#d0d0d0'
        else:
            color = recall_cmap(recall_norm(s['recall']))
        ax.bar(angles[i], h, width=width, bottom=r4-h, color=color, edgecolor='none')
    
    # Ring 5: Specificity (concentration)
    for i, s in enumerate(samples):
        color = conc_cmap(conc_norm(s['concentration']))
        ax.bar(angles[i], h, width=width, bottom=r5-h, color=color, edgecolor='none')
    
    # Ring 6: Similarity (innermost)
    for i, s in enumerate(samples):
        color = sim_cmap(sim_norm(s['similarity']))
        ax.bar(angles[i], h, width=width, bottom=r6-h, color=color, edgecolor='none')
    
    # Center text
    ax.text(0, 0, f"{n_samples}\nsamples\n\n{n_proteins}\nproteins\n\n{n_methods}\nmethods", 
            ha='center', va='center', fontsize=14, fontweight='bold', linespacing=1.3)
    
    ax.set_ylim(0, 1.05)
    
    # Legends on the right side - two columns
    # Method legend (top right)
    ax_method = fig.add_axes([0.72, 0.72, 0.15, 0.20])
    ax_method.axis('off')
    ax_method.set_title('Method', fontsize=10, fontweight='bold', loc='left')
    methods_used = sorted(set(s['method'] for s in samples))
    for i, m in enumerate(methods_used):
        y = 1 - (i + 0.5) / len(methods_used)
        ax_method.add_patch(plt.Rectangle((0, y - 0.06), 0.12, 0.08, 
                           facecolor=method_colors.get(m, '#888')))
        ax_method.text(0.16, y - 0.02, m, fontsize=9, va='center')
    ax_method.set_xlim(0, 1)
    ax_method.set_ylim(0, 1)
    
    # Colorbars in a 2x3 grid
    cb_width = 0.025
    cb_height = 0.12
    
    # Row 1: PCR, Crosslinks, Recall
    ax_pcr = fig.add_axes([0.72, 0.52, cb_width, cb_height])
    cb_pcr = plt.colorbar(cm.ScalarMappable(norm=pcr_norm, cmap=pcr_cmap), 
                          cax=ax_pcr, orientation='vertical')
    cb_pcr.set_label('PCR Dup', fontsize=9, fontweight='bold')
    cb_pcr.ax.tick_params(labelsize=8)
    
    ax_xl = fig.add_axes([0.80, 0.52, cb_width, cb_height])
    cb_xl = plt.colorbar(cm.ScalarMappable(norm=xl_norm, cmap=xl_cmap),
                         cax=ax_xl, orientation='vertical')
    cb_xl.set_label('Crosslinks', fontsize=9, fontweight='bold')
    cb_xl.ax.tick_params(labelsize=8)
    
    ax_recall = fig.add_axes([0.88, 0.52, cb_width, cb_height])
    cb_recall = plt.colorbar(cm.ScalarMappable(norm=recall_norm, cmap=recall_cmap),
                             cax=ax_recall, orientation='vertical')
    cb_recall.set_label('Recall', fontsize=9, fontweight='bold')
    cb_recall.ax.tick_params(labelsize=8)
    
    # Row 2: Specificity, Similarity
    ax_conc = fig.add_axes([0.72, 0.32, cb_width, cb_height])
    cb_conc = plt.colorbar(cm.ScalarMappable(norm=conc_norm, cmap=conc_cmap),
                           cax=ax_conc, orientation='vertical')
    cb_conc.set_label('Specificity', fontsize=9, fontweight='bold')
    cb_conc.ax.tick_params(labelsize=8)
    
    ax_sim = fig.add_axes([0.80, 0.32, cb_width, cb_height])
    cb_sim = plt.colorbar(cm.ScalarMappable(norm=sim_norm, cmap=sim_cmap),
                          cax=ax_sim, orientation='vertical')
    cb_sim.set_label('Similarity', fontsize=9, fontweight='bold')
    cb_sim.ax.tick_params(labelsize=8)
    
    # Ring labels
    fig.text(0.72, 0.20, 'Rings (outside→in):', fontsize=9, fontweight='bold')
    ring_labels = ['Method', 'PCR Dup', 'Crosslinks', 'Recall', 'Specificity', 'Similarity']
    for i, label in enumerate(ring_labels):
        fig.text(0.72, 0.17 - i*0.025, f'{i+1}. {label}', fontsize=8)
    
    plt.savefig('circos_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: circos_summary.png")

if __name__ == '__main__':
    create_summary_circos()

#!/usr/bin/env python3
"""
Generate a circos plot showing CLIP sample data with:
- Hierarchical clustering based on PEKA motif enrichment scores
- Regional distribution (stacked bars)
- Motif class enrichment (U-rich, G-rich, CG-rich clusters)
- Similarity scores (overlap of top k-mers with other samples)

Usage:
    python plot_circos.py
"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import Counter


def load_metadata(json_path='filtered_data.json'):
    """Load sample metadata from filtered_data.json"""
    with open(json_path) as f:
        metadata = json.load(f)
    
    filename_info = {}
    for r in metadata:
        fn = r.get('file', {}).get('filename', '').replace('.unique_genome.dedup_UMICollapse.log', '')
        t = r.get('sample_metadata', {}).get('metadata', {}).get('purification_target', {})
        t = t.get('value', '') if isinstance(t, dict) else ''
        # Normalize protein names: uppercase and TDP43 -> TARDBP
        if fn:
            target = t.upper() if t else 'UNKNOWN'
            if target == 'TDP43':
                target = 'TARDBP'
            filename_info[fn] = {'target': target}
    
    return filename_info


def load_log_data(data_dir='data'):
    """Load PCR duplication ratios and crosslink counts from UMICollapse log files"""
    log_data = {}
    for f in glob.glob(os.path.join(data_dir, '*_UMICollapse.log')):
        s = os.path.basename(f).replace('.unique_genome.dedup_UMICollapse.log', '')
        try:
            with open(f) as fh:
                inp, ded = None, None
                for l in fh:
                    if 'input reads' in l:
                        inp = int(l.split('\t')[1])
                    elif 'after deduplicating' in l:
                        ded = int(l.split('\t')[1])
                if inp and ded:
                    log_data[s] = {'pcr_ratio': inp / ded, 'crosslinks': ded}
        except Exception:
            pass
    return log_data


def load_regional_data(data_dir='data_subtype'):
    """Load regional distribution data from subtype TSV files"""
    region_cats = {
        'CDS': ['CDS'],
        'UTR3': ['UTR3'],
        'UTR5': ['UTR5'],
        'intron': ['intron'],
        'ncRNA': ['ncRNA', 'lncRNA', 'snRNA', 'snoRNA', 'miRNA'],
        'intergenic': ['intergenic'],
        'rRNA/tRNA': ['rRNA', 'tRNA', 'premapped']
    }
    
    regional_data = {}
    for f in glob.glob(os.path.join(data_dir, '*.tsv')):
        s = os.path.basename(f).replace('.summary_subtype_premapadjusted.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t')
            tots = {}
            for _, r in df.iterrows():
                for c, kws in region_cats.items():
                    if any(k.lower() in r['Subtype'].lower() for k in kws):
                        tots[c] = tots.get(c, 0) + r['cDNA %']
                        break
            regional_data[s] = tots
        except Exception:
            pass
    return regional_data


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


def get_dominant_dinucleotide(peka_scores):
    """Extract dominant dinucleotide from top 3 motifs"""
    v = {k: v for k, v in peka_scores.items() if np.isfinite(v)}
    if not v:
        return 'Other'
    
    t3 = sorted(v.items(), key=lambda x: -x[1])[:3]
    c = Counter()
    # Merge reciprocal dinucleotides
    norm = {'UA': 'AU', 'UG': 'GU', 'CA': 'AC', 'CG': 'GC', 'GA': 'AG', 'CU': 'UC'}
    for m, _ in t3:
        for i in range(len(m) - 1):
            di = m[i:i + 2]
            c[norm.get(di, di)] += 1
    
    return c.most_common(1)[0][0] if c else 'Other'


def load_motif_clusters(cluster_file='motif_clusters.json'):
    """Load the pre-computed motif clusters"""
    with open(cluster_file) as f:
        data = json.load(f)
    return data


def load_protein_functions(func_file='protein_functions.json'):
    """Load protein GO term-based functional categories"""
    try:
        with open(func_file) as f:
            data = json.load(f)
        return data.get('protein_functions', {}), data.get('function_colors', {})
    except FileNotFoundError:
        return {}, {}


def load_protein_domains(domain_file='protein_domains.json'):
    """Load protein RNA binding domain annotations"""
    try:
        with open(domain_file) as f:
            data = json.load(f)
        return data.get('protein_domain_proportions', {}), data.get('domain_colors', {})
    except FileNotFoundError:
        return {}, {}


def load_multivalency(mv_file='multivalency.json', mode='mixed'):
    """Load multivalency group annotations for proteins"""
    try:
        with open(mv_file) as f:
            data = json.load(f)
        return data.get(mode, {}), data.get('colors', {})
    except FileNotFoundError:
        return {}, {}


def load_rbp_family(csv_file='human_rbp_summary.csv'):
    """Load RBP family annotations from CISBP-RNA database summary"""
    try:
        df = pd.read_csv(csv_file)
        # Create protein -> family mapping (use primary family if multiple)
        protein_family = {}
        for _, row in df.iterrows():
            name = row['RBP_Name'].upper()
            family = row['Family']
            # Handle multiple families by taking the first one
            if ',' in str(family):
                family = family.split(',')[0].strip()
            protein_family[name] = family
        
        # Define colors for major families
        family_colors = {
            'RRM': '#3498db',      # Blue - most common RNA recognition motif
            'KH': '#e74c3c',       # Red - hnRNP K homology
            'CCCH ZF': '#2ecc71',  # Green - zinc finger CCCH type
            'CCHC ZF': '#9b59b6',  # Purple - zinc finger CCHC type
            'CSD': '#f39c12',      # Orange - cold shock domain
            'PUF': '#1abc9c',      # Teal - Pumilio
            'La': '#e67e22',       # Dark orange - La motif
            'NHL': '#8e44ad',      # Dark purple
            'S1': '#d35400',       # Burnt orange
            'RanBP ZF': '#16a085', # Dark teal
            'YTH': '#c0392b',      # Dark red
            'SAM': '#7f8c8d',      # Grey
            'C2H2 ZF': '#27ae60',  # Dark green
            'CCHH ZF': '#2980b9',  # Dark blue
            'Unknown': '#bdc3c7', # Light grey
        }
        
        return protein_family, family_colors
    except FileNotFoundError:
        return {}, {}


def load_recall_scores(csv_file='recall_by_method_results.csv', column='method6_recall_0.5'):
    """Load recall scores from the recall analysis results"""
    try:
        df = pd.read_csv(csv_file)
        # Create sample -> recall mapping using peka_file
        # Keep the _R1 suffix to match PEKA sample names
        recall_scores = {}
        for _, row in df.iterrows():
            # Extract sample name from peka_file (keep _R1)
            peka_file = row['peka_file']
            sample = peka_file.replace('.genome_5mer_distribution_genome.tsv', '')
            recall_scores[sample] = row[column]
        return recall_scores
    except FileNotFoundError:
        return {}


def load_condensate_data(condensate_file='../data/protein2cdcode_v2.0.tsv',
                         mapping_file='uniprot_to_gene.csv'):
    """Load condensate annotations for proteins"""
    try:
        # Load UniProt to gene name mapping
        mapping_df = pd.read_csv(mapping_file)
        uniprot_to_gene = dict(zip(mapping_df['uniprot_id'], mapping_df['gene_name']))
        
        # Load condensate data
        cond_df = pd.read_csv(condensate_file, sep='\t')
        
        # Map to gene names and aggregate condensates per protein
        protein_condensates = {}
        for _, row in cond_df.iterrows():
            gene_raw = uniprot_to_gene.get(row['uniprotkb_ac'], '')
            if pd.isna(gene_raw) or gene_raw == '':
                continue
            gene = str(gene_raw).upper()
            if gene:
                condensate = row['condensate_name']
                if gene not in protein_condensates:
                    protein_condensates[gene] = set()
                protein_condensates[gene].add(condensate)
        
        # Convert sets to lists
        protein_condensates = {k: list(v) for k, v in protein_condensates.items()}
        
        # Define colors for major condensates
        condensate_colors = {
            'Stress granule': '#e74c3c',      # Red
            'Stress Granule': '#e74c3c',      # Red (alternate spelling)
            'P-body': '#3498db',              # Blue
            'Nuclear speckle': '#2ecc71',     # Green
            'Paraspeckle': '#9b59b6',         # Purple
            'Nucleolus': '#f39c12',           # Orange
            'Cajal body': '#1abc9c',          # Teal
            'PML body': '#e67e22',            # Dark orange
            'P-granule': '#8e44ad',           # Dark purple
            'Transcriptional condensate': '#16a085',  # Dark teal
            'Centrosome': '#c0392b',          # Dark red
            'Chromatoid body': '#2980b9',     # Dark blue
            'mtRNA granule': '#7f8c8d',       # Grey-blue
            'Postsynaptic density': '#d35400', # Burnt orange
            'PcG body': '#27ae60',            # Darker green
            'Mitochondrial cloud': '#95a5a6', # Light grey
            'Unknown': '#bdc3c7',             # Light grey for unknown
            'Other': '#bdc3c7',               # Grey
        }
        
        return protein_condensates, condensate_colors
    except FileNotFoundError as e:
        print(f"Warning: Could not load condensate data: {e}")
        return {}, {}


def calculate_cluster_scores(peka_scores, cluster_data):
    """
    Calculate aggregated PEKA scores for each motif cluster.
    Returns dict with cluster scores (normalized to sum to 1).
    """
    cluster_motifs = cluster_data['cluster_motifs']
    cluster_names = cluster_data.get('cluster_names', 
                                      {'1': 'U-rich', '2': 'G-rich', '3': 'CG-rich'})
    
    scores = {}
    for cluster_id, motifs in cluster_motifs.items():
        # Sum PEKA scores for all motifs in this cluster
        total = sum(peka_scores.get(m, 0) for m in motifs 
                   if np.isfinite(peka_scores.get(m, 0)))
        scores[cluster_names.get(cluster_id, f'Cluster {cluster_id}')] = max(0, total)
    
    # Normalize to sum to 1
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    
    return scores


def get_top_kmers(peka_scores, n=50):
    """Get top n k-mers by PEKA score"""
    v = {k: v for k, v in peka_scores.items() if np.isfinite(v)}
    if not v:
        return set()
    return set(k for k, _ in sorted(v.items(), key=lambda x: -x[1])[:n])


def calculate_concentration(peka_scores, top_n=5):
    """
    Calculate concentration: proportion of total PEKA signal in top N motifs.
    Higher concentration = more specific binding.
    """
    vals = np.array([v for v in peka_scores.values() if np.isfinite(v) and v > 0])
    if len(vals) < top_n:
        return 0
    vals_sorted = np.sort(vals)[::-1]
    return vals_sorted[:top_n].sum() / vals.sum()


def calculate_similarity_scores(df, exclude_same_protein=True):
    """
    Calculate similarity scores for all samples.
    
    Similarity score = mean overlap ratio of top 50 k-mers with other samples
    (excluding samples of the same protein if exclude_same_protein=True)
    """
    # Pre-compute top 50 k-mers for all samples
    top_kmers = [get_top_kmers(row['peka_scores'], 50) for _, row in df.iterrows()]
    targets = df['target'].values
    
    similarity_scores = []
    for i in range(len(df)):
        if not top_kmers[i]:
            similarity_scores.append(0)
            continue
        
        overlaps = []
        for j in range(len(df)):
            if i == j:
                continue
            if exclude_same_protein and targets[i] == targets[j]:
                continue
            if not top_kmers[j]:
                continue
            
            # Calculate overlap ratio
            overlap = len(top_kmers[i] & top_kmers[j]) / 50
            overlaps.append(overlap)
        
        similarity_scores.append(np.mean(overlaps) if overlaps else 0)
    
    return similarity_scores


def create_circos_condensate(output_file='circos_condensate.png',
                              min_crosslinks=150000, max_pcr_ratio=5):
    """
    Generate circos plot with 5 condensate membership rings.
    Shows: Protein, Motif clusters, Regional distribution, and 5 condensate rings.
    """
    
    # Top 5 condensates: 4 major hubs + 1 clustered condensate
    # Major hubs (NOT clustered - diverse RNA preferences)
    # Paraspeckle IS clustered (proteins share motif preferences)
    top5_condensates = [
        ('Stress granule', '#e74c3c', False),   # Red - NOT clustered (diverse)
        ('Nucleolus', '#f39c12', False),         # Orange - NOT clustered
        ('P-body', '#3498db', False),            # Blue - NOT clustered
        ('Nuclear speckle', '#2ecc71', False),   # Green - NOT clustered  
        ('Paraspeckle', '#9b59b6', True),        # Purple - CLUSTERED! (score ~1.5)
    ]
    
    # Load clustering data to identify highly clustered proteins
    try:
        with open('condensate_clustering.json') as f:
            clustering_data = json.load(f)
        highly_clustered_condensates = set(clustering_data.get('top_clustered', [])[:5])
    except:
        highly_clustered_condensates = set()
    
    # Load all data
    print("Loading metadata...")
    filename_info = load_metadata()
    
    print("Loading log data...")
    log_data = load_log_data()
    
    print("Loading regional data...")
    regional_data = load_regional_data()
    
    print("Loading PEKA data...")
    peka_data = load_peka_data()
    
    print("Loading motif clusters...")
    cluster_data = load_motif_clusters()
    
    print("Loading condensate data...")
    protein_condensates, _ = load_condensate_data()
    
    # Combine data
    results = []
    for sample, peka_scores in peka_data.items():
        results.append({
            'sample': sample,
            'target': filename_info.get(sample, {}).get('target', 'Unknown'),
            'pcr_ratio': log_data.get(sample, {}).get('pcr_ratio', np.nan),
            'crosslinks': log_data.get(sample, {}).get('crosslinks', np.nan),
            'peka_scores': peka_scores,
            'regions': regional_data.get(sample, {})
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results[
        (df_results['crosslinks'] >= min_crosslinks) & 
        (df_results['pcr_ratio'] < max_pcr_ratio)
    ].reset_index(drop=True)
    
    print(f"Samples after filtering: {len(df_results)}")
    
    # Build PEKA matrix for clustering
    all_kmers = list(df_results.iloc[0]['peka_scores'].keys())
    peka_matrix = np.array([
        [r['peka_scores'].get(k, 0) if np.isfinite(r['peka_scores'].get(k, 0)) else 0 
         for k in all_kmers] 
        for _, r in df_results.iterrows()
    ])
    
    # Compute mean PEKA vector per protein for clustering
    protein_names, protein_vectors = [], []
    for t, g in df_results.groupby('target'):
        protein_names.append(t)
        protein_vectors.append(np.nan_to_num(
            np.nanmean(np.array([peka_matrix[i] for i in g.index]), axis=0)
        ))
    
    # Hierarchical clustering
    Z = linkage(np.array(protein_vectors), method='ward')
    dendro = dendrogram(Z, no_plot=True)
    protein_order = [protein_names[i] for i in dendro['leaves']]
    protein_counts = df_results['target'].value_counts()
    
    # Sort samples by protein cluster order
    df_results['ord'] = df_results['target'].map({p: i for i, p in enumerate(protein_order)})
    df_sorted = df_results.sort_values(['ord', 'target', 'pcr_ratio']).reset_index(drop=True)
    
    # Calculate motif cluster scores
    df_sorted['cluster_scores'] = df_sorted['peka_scores'].apply(
        lambda x: calculate_cluster_scores(x, cluster_data)
    )
    
    # Add condensate membership columns
    for cond_name, _, _ in top5_condensates:
        df_sorted[f'cond_{cond_name}'] = df_sorted['target'].apply(
            lambda t: cond_name in protein_condensates.get(t, []) or 
                     (cond_name == 'Stress granule' and 'Stress Granule' in protein_condensates.get(t, []))
        )
    
    # Check if protein is in any highly clustered condensate
    df_sorted['in_clustered_cond'] = df_sorted['target'].apply(
        lambda t: any(c in protein_condensates.get(t, []) for c in highly_clustered_condensates)
    )
    
    # Color schemes
    cluster_cols = {
        'U-rich': '#e74c3c', 'G-rich': '#3498db', 'CG-rich': '#2ecc71'
    }
    cluster_order = ['U-rich', 'G-rich', 'CG-rich']
    
    reg_cols = {
        'CDS': '#2ecc71', 'UTR3': '#e74c3c', 'UTR5': '#3498db',
        'intron': '#9b59b6', 'ncRNA': '#f39c12', 'intergenic': '#95a5a6',
        'rRNA/tRNA': '#1abc9c'
    }
    reg_ord = ['CDS', 'UTR5', 'UTR3', 'intron', 'ncRNA', 'rRNA/tRNA', 'intergenic']
    
    # Create figure
    fig = plt.figure(figsize=(48, 40))
    ax = fig.add_axes([0.02, 0.02, 0.70, 0.96], polar=True)
    
    # Setup rings (from outside to inside)
    n = len(df_sorted)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    w = 2 * np.pi / n
    
    # Heights and radii: protein, 5 condensates, motif, region
    h_prot = 0.05
    h_cond = 0.03  # Thinner for condensate rings
    h_motif = 0.06
    h_region = 0.12
    
    r_prot = 1.0
    r_cond = [0.95, 0.92, 0.89, 0.86, 0.83]  # 5 condensate rings
    r_motif = 0.80
    r_region = 0.74
    
    targs = df_sorted['target'].values
    tstarts = [(0, targs[0])] + [(i, targs[i]) for i in range(1, len(targs)) if targs[i] != targs[i-1]]
    
    # Draw protein ring (outermost)
    sh = True
    for i, (st, t) in enumerate(tstarts):
        en = tstarts[i + 1][0] if i + 1 < len(tstarts) else n
        for j in range(st, en):
            ax.bar(angles[j], h_prot, width=w, bottom=r_prot - h_prot, 
                   color='#404040' if sh else '#707070', lw=0)
        sh = not sh
    
    # Draw 5 condensate rings
    for ring_idx, (cond_name, cond_color, is_clustered) in enumerate(top5_condensates):
        r = r_cond[ring_idx]
        col_name = f'cond_{cond_name}'
        for i, belongs in enumerate(df_sorted[col_name]):
            if belongs:
                # Use darker outline for clustered condensates
                color = cond_color
                edge = 'white' if is_clustered else 'none'
                lw = 0.3 if is_clustered else 0
            else:
                color = '#f0f0f0'  # Light grey if not member
                edge = 'none'
                lw = 0
            ax.bar(angles[i], h_cond, width=w, bottom=r - h_cond, color=color, 
                   edgecolor=edge, linewidth=lw)
    
    # Draw motif cluster ring
    for i, cs in enumerate(df_sorted['cluster_scores']):
        if not cs or sum(cs.values()) == 0:
            ax.bar(angles[i], h_motif, width=w, bottom=r_motif - h_motif, color='#bdc3c7', lw=0)
            continue
        bot = r_motif - h_motif
        for cluster in cluster_order:
            ht = cs.get(cluster, 0) * h_motif
            if ht > 0:
                ax.bar(angles[i], ht, width=w, bottom=bot,
                       color=cluster_cols.get(cluster, '#bdc3c7'), lw=0)
                bot += ht
    
    # Draw regional distribution ring
    for i, regs in enumerate(df_sorted['regions']):
        if not regs or sum(regs.values()) == 0:
            ax.bar(angles[i], h_region, width=w, bottom=r_region - h_region, color='#bdc3c7', lw=0)
            continue
        tot = sum(regs.values())
        bot = r_region - h_region
        for rg in reg_ord + ['other']:
            ht = (regs.get(rg, 0) / tot) * h_region
            if ht > 0:
                ax.bar(angles[i], ht, width=w, bottom=bot, 
                       color=reg_cols.get(rg, '#bdc3c7'), lw=0)
                bot += ht
    
    # Draw dendrogram
    dendro_top = r_region - h_region
    dendro_bottom = 0.04
    pangs = {
        tstarts[i][1]: angles[(tstarts[i][0] + (tstarts[i + 1][0] if i + 1 < len(tstarts) else n)) // 2]
        for i in range(len(tstarts))
    }
    ic, dc = np.array(dendro['icoord']), np.array(dendro['dcoord'])
    xtp = {5 + i * 10: protein_names[idx] for i, idx in enumerate(dendro['leaves'])}
    
    for i in range(len(ic)):
        pts = []
        for x, y in zip(ic[i], dc[i]):
            if x in xtp:
                a = pangs.get(xtp[x], 0)
            else:
                xps = sorted(xtp.keys())
                a = 0
                for j in range(len(xps) - 1):
                    if xps[j] <= x <= xps[j + 1]:
                        t_ = (x - xps[j]) / (xps[j + 1] - xps[j])
                        a1, a2 = pangs.get(xtp[xps[j]], 0), pangs.get(xtp[xps[j + 1]], 0)
                        if abs(a2 - a1) > np.pi:
                            a1, a2 = (a1 + 2 * np.pi, a2) if a2 > a1 else (a1, a2 + 2 * np.pi)
                        a = a1 + t_ * (a2 - a1)
                        break
            r = dendro_top - (y / dc.max()) * (dendro_top - dendro_bottom)
            pts.append((a, r))
        
        for j in range(len(pts) - 1):
            a1, r1_, a2, r2_ = *pts[j], *pts[j + 1]
            if j == 1 and abs(a2 - a1) > 0.01:
                ax.plot(np.linspace(a1, a2, 20), [r1_] * 20, c='#444', lw=1.2, alpha=0.8)
            else:
                ax.plot([a1, a2], [r1_, r2_], c='#444', lw=1.2, alpha=0.8)
    
    # Draw protein labels
    protein_labels = sorted(
        [(t, pangs.get(t, 0), protein_counts.get(t, 0)) for t in set(x[1] for x in tstarts)],
        key=lambda x: x[1]
    )
    
    n_layers = 3
    layer_spacing = 0.05
    base_radius = 1.04
    min_angular_gap = 0.08
    
    layer_assignments = []
    last_angle_in_layer = [-999] * n_layers
    
    for t, a, c in protein_labels:
        best_layer = 0
        best_gap = 0
        for layer in range(n_layers):
            gap = abs(a - last_angle_in_layer[layer])
            if gap > np.pi:
                gap = 2 * np.pi - gap
            if gap > best_gap:
                best_gap = gap
                best_layer = layer
        
        if best_gap < min_angular_gap:
            best_layer = len(layer_assignments) % n_layers
        
        last_angle_in_layer[best_layer] = a
        layer_assignments.append((t, a, c, best_layer))
    
    control_proteins = {'GFP', 'IGG', 'INPUT', 'UNKNOWN', 'IgG'}
    
    for t, a, c, layer in layer_assignments:
        rl = base_radius + layer * layer_spacing
        is_control = t.upper() in {x.upper() for x in control_proteins}
        label_color = '#999999' if is_control else '#000000'
        line_color = '#cccccc' if is_control else '#999999'
        ax.plot([a, a], [1.01, rl - 0.005], c=line_color, lw=0.6, alpha=0.5)
        ax.annotate(t, xy=(a, rl), fontsize=13 if c < 3 else 14 if c < 10 else 15,
                    ha='center', va='center', color=label_color,
                    fontstyle='italic' if is_control else 'normal')
    
    # Configure axes
    ax.set_ylim(0, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Center text
    ax.text(0, 0.02, f'{n} samples\n{len(protein_names)} proteins',
            ha='center', va='center', fontsize=32, fontweight='bold',
            transform=ax.transData)
    
    # Draw legends
    leg_top = 0.88
    leg_height = 0.18
    leg_bottom = leg_top - leg_height
    col_x = [0.74, 0.80, 0.88, 0.94]
    box_size = 0.006
    
    # Condensate legend
    fig.text(col_x[0] + 0.02, leg_top + 0.015, 'Condensates',
             fontsize=18, fontweight='bold', ha='center')
    spacing = leg_height / len(top5_condensates)
    for i, (cond_name, cond_color, is_clustered) in enumerate(top5_condensates):
        y_pos = leg_top - (i + 0.5) * spacing
        rect = mpatches.FancyBboxPatch(
            (col_x[0], y_pos - box_size / 2), box_size, box_size,
            boxstyle="round,pad=0.001", facecolor=cond_color,
            edgecolor='black' if is_clustered else 'none',
            linewidth=1.5 if is_clustered else 0,
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(rect)
        # Add star for clustered condensates
        label = f"{cond_name} ★" if is_clustered else cond_name
        fig.text(col_x[0] + box_size + 0.003, y_pos, label, fontsize=11, va='center',
                fontweight='bold' if is_clustered else 'normal')
    # Legend note
    fig.text(col_x[0] + 0.02, leg_top - leg_height - 0.01, 
             '★ proteins cluster by motif', fontsize=9, ha='center', style='italic', color='#666666')
    
    # Region legend
    fig.text(col_x[1] + 0.015, leg_top + 0.015, 'Region',
             fontsize=18, fontweight='bold', ha='center')
    spacing = leg_height / len(reg_ord)
    for i, r in enumerate(reg_ord):
        y_pos = leg_top - (i + 0.5) * spacing
        rect = mpatches.FancyBboxPatch(
            (col_x[1], y_pos - box_size / 2), box_size, box_size,
            boxstyle="round,pad=0.001", facecolor=reg_cols[r],
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(rect)
        fig.text(col_x[1] + box_size + 0.002, y_pos, r, fontsize=13, va='center')
    
    # Motif cluster legend
    fig.text(col_x[2] + 0.012, leg_top + 0.015, 'Motif',
             fontsize=18, fontweight='bold', ha='center')
    spacing = leg_height / len(cluster_order)
    for i, cluster in enumerate(cluster_order):
        y_pos = leg_top - (i + 0.5) * spacing
        rect = mpatches.FancyBboxPatch(
            (col_x[2], y_pos - box_size / 2), box_size, box_size,
            boxstyle="round,pad=0.001", facecolor=cluster_cols[cluster],
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(rect)
        fig.text(col_x[2] + box_size + 0.002, y_pos, cluster, fontsize=13, va='center')
    
    # Save
    fig.patch.set_facecolor('white')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def create_circos_plot(output_file='regional_distribution_circos_motif.png',
                       min_crosslinks=150000, max_pcr_ratio=5,
                       use_recall=True):
    """Generate the circos plot"""
    
    # Load all data
    print("Loading metadata...")
    filename_info = load_metadata()
    
    print("Loading log data...")
    log_data = load_log_data()
    
    print("Loading regional data...")
    regional_data = load_regional_data()
    
    print("Loading PEKA data...")
    peka_data = load_peka_data()
    
    print("Loading motif clusters...")
    cluster_data = load_motif_clusters()
    
    # Load recall scores if requested
    if use_recall:
        print("Loading recall scores...")
        recall_data = load_recall_scores()
    
    # Load condensate data
    print("Loading condensate data...")
    protein_condensates, condensate_colors = load_condensate_data()
    
    # Combine data
    results = []
    for sample, peka_scores in peka_data.items():
        results.append({
            'sample': sample,
            'target': filename_info.get(sample, {}).get('target', 'Unknown'),
            'pcr_ratio': log_data.get(sample, {}).get('pcr_ratio', np.nan),
            'crosslinks': log_data.get(sample, {}).get('crosslinks', np.nan),
            'peka_scores': peka_scores,
            'regions': regional_data.get(sample, {})
        })
    
    df_results = pd.DataFrame(results)
    
    # Filter samples
    df_results = df_results[
        (df_results['crosslinks'] >= min_crosslinks) & 
        (df_results['pcr_ratio'] < max_pcr_ratio)
    ].reset_index(drop=True)
    
    print(f"Samples after filtering: {len(df_results)}")
    
    # Build PEKA matrix for clustering
    all_kmers = list(df_results.iloc[0]['peka_scores'].keys())
    peka_matrix = np.array([
        [r['peka_scores'].get(k, 0) if np.isfinite(r['peka_scores'].get(k, 0)) else 0 
         for k in all_kmers] 
        for _, r in df_results.iterrows()
    ])
    
    # Compute mean PEKA vector per protein for clustering
    protein_names, protein_vectors = [], []
    for t, g in df_results.groupby('target'):
        protein_names.append(t)
        protein_vectors.append(np.nan_to_num(
            np.nanmean(np.array([peka_matrix[i] for i in g.index]), axis=0)
        ))
    
    # Hierarchical clustering
    Z = linkage(np.array(protein_vectors), method='ward')
    dendro = dendrogram(Z, no_plot=True)
    protein_order = [protein_names[i] for i in dendro['leaves']]
    protein_counts = df_results['target'].value_counts()
    
    # Sort samples by protein cluster order
    df_results['ord'] = df_results['target'].map({p: i for i, p in enumerate(protein_order)})
    df_sorted = df_results.sort_values(['ord', 'target', 'pcr_ratio']).reset_index(drop=True)
    
    # Calculate dominant dinucleotide for each sample (kept for reference)
    df_sorted['dinuc'] = df_sorted['peka_scores'].apply(get_dominant_dinucleotide)
    
    # Calculate motif cluster scores for each sample
    df_sorted['cluster_scores'] = df_sorted['peka_scores'].apply(
        lambda x: calculate_cluster_scores(x, cluster_data)
    )
    
    # Calculate similarity scores (excluding same protein)
    print("Calculating similarity scores...")
    df_sorted['similarity'] = calculate_similarity_scores(df_sorted, exclude_same_protein=True)
    
    # Calculate concentration (binding specificity)
    print("Calculating concentration...")
    df_sorted['concentration'] = df_sorted['peka_scores'].apply(calculate_concentration)
    
    # Assign recall scores to samples
    if use_recall:
        df_sorted['recall'] = df_sorted['sample'].map(
            lambda s: recall_data.get(s, np.nan)
        )
    
    # Assign condensate data to samples (based on protein target)
    # Get primary condensate for each protein (most common or first relevant one)
    priority_condensates = ['Stress granule', 'Stress Granule', 'P-body', 'Nuclear speckle', 
                           'Paraspeckle', 'Nucleolus', 'Cajal body', 'P-granule']
    
    def get_primary_condensate(target):
        condensates = protein_condensates.get(target, [])
        if not condensates:
            return 'Unknown'
        # Check priority condensates first
        for pc in priority_condensates:
            if pc in condensates:
                return pc
        return condensates[0]  # Return first one if no priority match
    
    df_sorted['condensate'] = df_sorted['target'].map(get_primary_condensate)
    
    # Color schemes
    # Motif cluster colors (3 clusters)
    cluster_cols = {
        'U-rich': '#e74c3c',   # Red for U-rich (pyrimidine-tract binders)
        'G-rich': '#3498db',   # Blue for G-rich (G-quad binders)
        'CG-rich': '#2ecc71'   # Green for CG-rich (structured RNA binders)
    }
    cluster_order = ['U-rich', 'G-rich', 'CG-rich']
    
    # Legacy dinucleotide colors (kept for reference)
    di_cols = {
        'UU': '#e74c3c', 'CC': '#c0392b', 'UC': '#d35400', 'AA': '#3498db',
        'GG': '#2980b9', 'AG': '#1f618d', 'AU': '#9b59b6', 'GU': '#2ecc71',
        'AC': '#f39c12', 'GC': '#1abc9c'
    }
    reg_cols = {
        'CDS': '#2ecc71', 'UTR3': '#e74c3c', 'UTR5': '#3498db',
        'intron': '#9b59b6', 'ncRNA': '#f39c12', 'intergenic': '#95a5a6',
        'rRNA/tRNA': '#1abc9c'
    }
    reg_ord = ['CDS', 'UTR5', 'UTR3', 'intron', 'ncRNA', 'rRNA/tRNA', 'intergenic']
    
    xl_cm, xl_n = plt.cm.Greys, mcolors.LogNorm(1.5, 300)
    pcr_cm, pcr_n = plt.cm.RdYlGn_r, mcolors.Normalize(1, 5)
    sim_cm = plt.cm.viridis
    sim_min, sim_max = df_sorted['similarity'].min(), df_sorted['similarity'].max()
    sim_n = mcolors.Normalize(sim_min, sim_max)
    
    # Concentration colormap (greyscale)
    conc_cm = plt.cm.Greys
    conc_min, conc_max = df_sorted['concentration'].min(), df_sorted['concentration'].max()
    conc_n = mcolors.Normalize(conc_min, conc_max)
    
    # Recall colormap (green = high recall, red = low)
    if use_recall:
        recall_vals = df_sorted['recall'].dropna()
        if len(recall_vals) > 0:
            recall_min, recall_max = recall_vals.min(), recall_vals.max()
        else:
            recall_min, recall_max = 0, 1
        recall_cm = plt.cm.RdYlGn
        recall_n = mcolors.Normalize(recall_min, recall_max)
    
    # Create figure
    fig = plt.figure(figsize=(48, 40))
    ax = fig.add_axes([0.02, 0.02, 0.70, 0.96], polar=True)
    
    # Setup rings (from outside to inside) - simplified layout
    # Only: Protein, Motif, Region rings (removed condensate, recall, similarity, concentration)
    n = len(df_sorted)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    w = 2 * np.pi / n
    # Heights: protein, motif, gap, regional
    h1 = 0.06     # Protein ring height
    hm = 0.08     # Motif ring height  
    h_gap = 0.015 # White gap between motif and regional (original)
    h2 = 0.15     # Regional ring height
    r1 = 1.0      # Protein ring (outermost - for labels)
    r2 = 0.94     # Motif cluster ring
    r3 = 0.94 - hm - h_gap  # Regional distribution ring (after gap)
    
    targs = df_sorted['target'].values
    tstarts = [(0, targs[0])] + [(i, targs[i]) for i in range(1, len(targs)) if targs[i] != targs[i-1]]
    
    # Draw protein ring (outermost - for labels)
    sh = True
    for i, (st, t) in enumerate(tstarts):
        en = tstarts[i + 1][0] if i + 1 < len(tstarts) else n
        for j in range(st, en):
            ax.bar(angles[j], h1, width=w, bottom=r1 - h1, 
                   color='#404040' if sh else '#707070', lw=0)
        sh = not sh
    
    # Draw motif cluster ring (stacked bar)
    for i, cs in enumerate(df_sorted['cluster_scores']):
        if not cs or sum(cs.values()) == 0:
            ax.bar(angles[i], hm, width=w, bottom=r2 - hm, color='#bdc3c7', lw=0)
            continue
        bot = r2 - hm
        for cluster in cluster_order:
            ht = cs.get(cluster, 0) * hm
            if ht > 0:
                ax.bar(angles[i], ht, width=w, bottom=bot,
                       color=cluster_cols.get(cluster, '#bdc3c7'), lw=0)
                bot += ht
    
    # Draw regional distribution ring (stacked bars)
    for i, regs in enumerate(df_sorted['regions']):
        if not regs or sum(regs.values()) == 0:
            ax.bar(angles[i], h2, width=w, bottom=r3 - h2, color='#bdc3c7', lw=0)
            continue
        tot = sum(regs.values())
        bot = r3 - h2
        for rg in reg_ord + ['other']:
            ht = (regs.get(rg, 0) / tot) * h2
            if ht > 0:
                ax.bar(angles[i], ht, width=w, bottom=bot, 
                       color=reg_cols.get(rg, '#bdc3c7'), lw=0)
                bot += ht
    
    # Draw dendrogram - very compact in center (distributions more important)
    dendro_top = r3 - h2
    dendro_bottom = 0.25  # Very compressed dendrogram
    pangs = {
        tstarts[i][1]: angles[(tstarts[i][0] + (tstarts[i + 1][0] if i + 1 < len(tstarts) else n)) // 2]
        for i in range(len(tstarts))
    }
    ic, dc = np.array(dendro['icoord']), np.array(dendro['dcoord'])
    xtp = {5 + i * 10: protein_names[idx] for i, idx in enumerate(dendro['leaves'])}
    
    for i in range(len(ic)):
        pts = []
        for x, y in zip(ic[i], dc[i]):
            if x in xtp:
                a = pangs.get(xtp[x], 0)
            else:
                xps = sorted(xtp.keys())
                a = 0
                for j in range(len(xps) - 1):
                    if xps[j] <= x <= xps[j + 1]:
                        t_ = (x - xps[j]) / (xps[j + 1] - xps[j])
                        a1, a2 = pangs.get(xtp[xps[j]], 0), pangs.get(xtp[xps[j + 1]], 0)
                        if abs(a2 - a1) > np.pi:
                            a1, a2 = (a1 + 2 * np.pi, a2) if a2 > a1 else (a1, a2 + 2 * np.pi)
                        a = a1 + t_ * (a2 - a1)
                        break
            r = dendro_top - (y / dc.max()) * (dendro_top - dendro_bottom)
            pts.append((a, r))
        
        for j in range(len(pts) - 1):
            a1, r1_, a2, r2_ = *pts[j], *pts[j + 1]
            if j == 1 and abs(a2 - a1) > 0.01:
                ax.plot(np.linspace(a1, a2, 20), [r1_] * 20, c='#444', lw=1.2, alpha=0.8)
            else:
                ax.plot([a1, a2], [r1_, r2_], c='#444', lw=1.2, alpha=0.8)
    
    # Draw protein labels with smart layer assignment to avoid overlap
    protein_labels = sorted(
        [(t, pangs.get(t, 0), protein_counts.get(t, 0)) for t in set(x[1] for x in tstarts)],
        key=lambda x: x[1]
    )
    
    # Assign layers based on angular proximity
    n_layers = 3
    layer_spacing = 0.05
    base_radius = 1.04  # Outside the function ring
    min_angular_gap = 0.08  # Minimum gap before allowing same layer
    
    layer_assignments = []
    last_angle_in_layer = [-999] * n_layers  # Track last used angle per layer
    
    for t, a, c in protein_labels:
        # Find the best layer (one where we're far enough from the last label)
        best_layer = 0
        best_gap = 0
        for layer in range(n_layers):
            gap = abs(a - last_angle_in_layer[layer])
            # Handle wrap-around
            if gap > np.pi:
                gap = 2 * np.pi - gap
            if gap > best_gap:
                best_gap = gap
                best_layer = layer
        
        # If all layers are too close, use round-robin
        if best_gap < min_angular_gap:
            best_layer = len(layer_assignments) % n_layers
        
        last_angle_in_layer[best_layer] = a
        layer_assignments.append((t, a, c, best_layer))
    
    # Control proteins to highlight (grey out)
    control_proteins = {'GFP', 'IGG', 'INPUT', 'UNKNOWN', 'IgG'}
    
    # Draw labels with larger font
    for t, a, c, layer in layer_assignments:
        rl = base_radius + layer * layer_spacing
        is_control = t.upper() in {x.upper() for x in control_proteins}
        label_color = '#999999' if is_control else '#000000'
        line_color = '#cccccc' if is_control else '#999999'
        ax.plot([a, a], [1.01, rl - 0.005], c=line_color, lw=0.6, alpha=0.5)
        ax.annotate(t, xy=(a, rl), fontsize=16 if c < 3 else 17 if c < 10 else 18,
                    ha='center', va='center', color=label_color,
                    fontstyle='italic' if is_control else 'normal')
    
    # Configure axes - tighter layout
    ax.set_ylim(0, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Center text - positioned for compact layout, much larger font
    ax.text(0, 0.12, f'{n} samples\n{len(protein_names)} proteins',
            ha='center', va='center', fontsize=52, fontweight='bold',
            transform=ax.transData)
    
    # Draw legends - simplified: only Region and Motif, larger fonts
    leg_top = 0.88
    leg_height = 0.25
    leg_bottom = leg_top - leg_height
    col_x = [0.73, 0.82]  # Two columns only
    box_size = 0.012
    
    # Region legend
    fig.text(col_x[0] + 0.030, leg_top + 0.025, 'Region',
             fontsize=28, fontweight='bold', ha='center')
    spacing = leg_height / len(reg_ord)
    for i, r in enumerate(reg_ord):
        y_pos = leg_top - (i + 0.5) * spacing
        rect = mpatches.FancyBboxPatch(
            (col_x[0], y_pos - box_size / 2), box_size, box_size,
            boxstyle="round,pad=0.001", facecolor=reg_cols[r],
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(rect)
        fig.text(col_x[0] + box_size + 0.006, y_pos, r, fontsize=22, va='center')
    
    # Motif cluster legend
    fig.text(col_x[1] + 0.025, leg_top + 0.025, 'Motif',
             fontsize=28, fontweight='bold', ha='center')
    spacing = leg_height / len(cluster_order)
    for i, cluster in enumerate(cluster_order):
        y_pos = leg_top - (i + 0.5) * spacing
        rect = mpatches.FancyBboxPatch(
            (col_x[1], y_pos - box_size / 2), box_size, box_size,
            boxstyle="round,pad=0.001", facecolor=cluster_cols[cluster],
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(rect)
        fig.text(col_x[1] + box_size + 0.006, y_pos, cluster, fontsize=22, va='center')
    
    # Save
    fig.patch.set_facecolor('white')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_recall_by_method(output_file='recall_by_method.png',
                          top_n_proteins=10):
    """
    Generate a plot showing recall scores split by experimental method
    for proteins covered by the most methods.
    """
    
    # Load recall data
    print("Loading data for recall by method plot...")
    recall_csv = pd.read_csv('recall_by_method_results.csv')
    
    # Extract method info and recall scores
    results = []
    for _, row in recall_csv.iterrows():
        results.append({
            'sample': row['sample_name'],
            'target': row['rbp'].upper(),
            'method': row['method'],
            'recall': row['method6_recall_0.5'],
        })
    
    df_results = pd.DataFrame(results)
    
    # Filter out unknown methods
    df_results = df_results[df_results['method'] != 'Unknown'].reset_index(drop=True)
    
    print(f"Samples with recall data: {len(df_results)}")
    
    # Find proteins with the most methods
    protein_methods = df_results.groupby('target')['method'].nunique().sort_values(ascending=False)
    top_proteins = protein_methods.head(top_n_proteins).index.tolist()
    
    print(f"Top {top_n_proteins} proteins by method coverage:")
    for p in top_proteins:
        methods = df_results[df_results['target'] == p]['method'].unique()
        print(f"  {p}: {len(methods)} methods - {', '.join(methods)}")
    
    # Filter to top proteins
    df_top = df_results[df_results['target'].isin(top_proteins)].copy()
    
    # Create the plot
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()
    
    for i, protein in enumerate(top_proteins):
        ax = axes[i]
        df_prot = df_top[df_top['target'] == protein]
        
        # Group by method
        methods_data = df_prot.groupby('method')['recall'].apply(list).to_dict()
        labels = sorted(methods_data.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        
        # Plot each sample as a circle with jitter
        for j, method in enumerate(labels):
            values = methods_data[method]
            # Add jitter to x positions
            x_jitter = np.random.uniform(-0.2, 0.2, len(values))
            ax.scatter([j + 1 + x for x in x_jitter], values, 
                      c=[colors[j]], alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
            # Draw median line
            median = np.median(values)
            ax.hlines(median, j + 0.6, j + 1.4, colors='black', linewidth=2)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(protein, fontsize=14, fontweight='bold')
        ax.set_ylabel('Recall Score' if i % 5 == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.0)  # Recall is always 0-1
        ax.set_xlim(0.3, len(labels) + 0.7)
    
    plt.suptitle('Recall Score by Experimental Method\n(Top proteins by method coverage)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()
    
    return df_results


def plot_similarity_vs_specificity(output_file='similarity_vs_specificity.png',
                                    min_crosslinks=150000, max_pcr_ratio=5):
    """
    Generate a scatter plot showing the relationship between 
    similarity scores and specificity (concentration).
    """
    print("Generating similarity vs specificity plot...")
    
    # Load data
    filename_info = load_metadata()
    log_data = load_log_data()
    peka_data = load_peka_data()
    
    # Combine data
    results = []
    for sample, peka_scores in peka_data.items():
        results.append({
            'sample': sample,
            'target': filename_info.get(sample, {}).get('target', 'Unknown'),
            'pcr_ratio': log_data.get(sample, {}).get('pcr_ratio', np.nan),
            'crosslinks': log_data.get(sample, {}).get('crosslinks', np.nan),
            'peka_scores': peka_scores,
        })
    
    df = pd.DataFrame(results)
    df = df[(df['crosslinks'] >= min_crosslinks) & 
            (df['pcr_ratio'] < max_pcr_ratio)].reset_index(drop=True)
    
    # Calculate concentration (specificity)
    df['concentration'] = df['peka_scores'].apply(calculate_concentration)
    
    # Calculate similarity scores
    df['similarity'] = calculate_similarity_scores(df, exclude_same_protein=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by protein target
    unique_targets = df['target'].unique()
    n_targets = len(unique_targets)
    
    # Scatter plot
    scatter = ax.scatter(df['concentration'], df['similarity'], 
                        c=df['target'].astype('category').cat.codes,
                        cmap='tab20', alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    
    # Add correlation
    from scipy import stats
    r, p = stats.pearsonr(df['concentration'], df['similarity'])
    
    # Add trend line
    z = np.polyfit(df['concentration'], df['similarity'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df['concentration'].min(), df['concentration'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', alpha=0.8, linewidth=2,
            label=f'Trend (r={r:.3f}, p={p:.2e})')
    
    ax.set_xlabel('Specificity (Concentration)', fontsize=14)
    ax.set_ylabel('Similarity Score', fontsize=14)
    ax.set_title('Relationship between Binding Specificity and Motif Similarity\n'
                f'({len(df)} samples, {n_targets} proteins)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    
    # Add some annotations for interesting points
    # High specificity, low similarity = unique binders
    high_spec_low_sim = df[(df['concentration'] > df['concentration'].quantile(0.9)) & 
                           (df['similarity'] < df['similarity'].quantile(0.25))]
    for _, row in high_spec_low_sim.head(3).iterrows():
        ax.annotate(row['target'], (row['concentration'], row['similarity']),
                   fontsize=9, alpha=0.8)
    
    # Low specificity, high similarity = promiscuous binders
    low_spec_high_sim = df[(df['concentration'] < df['concentration'].quantile(0.25)) & 
                           (df['similarity'] > df['similarity'].quantile(0.9))]
    for _, row in low_spec_high_sim.head(3).iterrows():
        ax.annotate(row['target'], (row['concentration'], row['similarity']),
                   fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()
    
    return df


if __name__ == "__main__":
    # Main condensate circos (with 5 condensate rings)
    create_circos_condensate(
        output_file='circos_condensate.png'
    )
    
    plot_recall_by_method()
    plot_similarity_vs_specificity()
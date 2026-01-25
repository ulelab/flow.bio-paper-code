import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict
from itertools import product

# =============================================================================
# PATHS
# =============================================================================
dump_path = "extracted_db/home/albumiha/SQLArchive_cisbp_rna_2_00.dump"
pfm_dir = "../data/pwms_all_motifs"
peka_dir = "data_peka"
json_path = "filtered_data.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_all_inserts(table_name, content):
    pattern = rf"INSERT INTO `{table_name}` VALUES (.*?);"
    matches = re.findall(pattern, content, re.DOTALL)
    
    all_rows = []
    for values_str in matches:
        values_str = values_str.strip()
        if values_str.startswith('('):
            values_str = values_str[1:]
        if values_str.endswith(')'):
            values_str = values_str[:-1]
        
        raw_rows = re.split(r'\),\(', values_str)
        
        for raw_row in raw_rows:
            vals = []
            current = ''
            in_quotes = False
            for char in raw_row:
                if char == "'" and (not current or current[-1] != '\\'):
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    vals.append(current.strip().strip("'"))
                    current = ''
                    continue
                current += char
            vals.append(current.strip().strip("'"))
            all_rows.append(vals)
    
    return all_rows


def load_pfm(filepath):
    try:
        if os.path.getsize(filepath) == 0:
            return None
        df = pd.read_csv(filepath, sep='\t')
        if not all(col in df.columns for col in ['A', 'C', 'G', 'U']):
            return None
        matrix = df[['A', 'C', 'G', 'U']].values
        if matrix.shape[0] == 0:
            return None
        return matrix
    except:
        return None


def pfm_to_pwm(pfm, pseudocount=0.0001):
    background = np.array([0.25, 0.25, 0.25, 0.25])
    pfm_adj = pfm + pseudocount
    pfm_adj = pfm_adj / pfm_adj.sum(axis=1, keepdims=True)
    return np.log2(pfm_adj / background)


base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}


def method1_sliding_best(kmer, pwm):
    """Original: slide k-mer along PWM, take best score."""
    kmer = kmer.upper().replace('T', 'U')
    k = len(kmer)
    pwm_len = len(pwm)
    
    for base in kmer:
        if base not in base_to_idx:
            return None
    
    best_score = float('-inf')
    for start in range(pwm_len - k + 1):
        score = sum(pwm[start + i, base_to_idx[kmer[i]]] for i in range(k))
        best_score = max(best_score, score)
    return best_score


def generate_expected_kmers(pwm, k=5, n_top=30):
    """Generate top-scoring k-mers from the PWM."""
    pwm_len = len(pwm)
    bases = ['A', 'C', 'G', 'U']
    
    all_kmers = {}
    
    for start in range(max(1, pwm_len - k + 1)):
        window_pwm = pwm[start:start + k]
        if len(window_pwm) < k:
            continue
            
        for kmer_tuple in product(bases, repeat=k):
            kmer = ''.join(kmer_tuple)
            score = sum(window_pwm[i, base_to_idx[kmer[i]]] for i in range(k))
            if kmer not in all_kmers or score > all_kmers[kmer]:
                all_kmers[kmer] = score
    
    sorted_kmers = sorted(all_kmers.items(), key=lambda x: -x[1])
    return set(k for k, s in sorted_kmers[:n_top])


def method6_set_match(kmer, expected_set, allow_mismatch=1):
    """Check if k-mer matches expected set (with optional mismatch)."""
    kmer = kmer.upper().replace('T', 'U')
    
    if kmer in expected_set:
        return 1.0
    
    if allow_mismatch > 0:
        for expected in expected_set:
            mismatches = sum(1 for a, b in zip(kmer, expected) if a != b)
            if mismatches <= allow_mismatch:
                return 0.8
    
    return 0.0


def load_peka_top_kmers(filepath, n=50):
    df = pd.read_csv(filepath, sep='\t')
    kmer_col = df.columns[0]
    df_valid = df[pd.to_numeric(df['PEKA-score'], errors='coerce').notna()].copy()
    df_valid['PEKA-score'] = pd.to_numeric(df_valid['PEKA-score'])
    df_sorted = df_valid.sort_values('PEKA-score', ascending=False).head(n)
    return df_sorted[kmer_col].tolist()


def calculate_recall_multi_method(top_kmers, pwm):
    """Calculate recall using multiple methods."""
    if not top_kmers or pwm is None:
        return None
    
    k = len(top_kmers[0])
    
    # Method 1: Sliding window normalized
    m1_scores = []
    for kmer in top_kmers:
        score = method1_sliding_best(kmer, pwm)
        if score is not None:
            m1_scores.append(score)
    
    if not m1_scores:
        return None
    
    m1_min, m1_max = min(m1_scores), max(m1_scores)
    m1_norm = [(s - m1_min) / (m1_max - m1_min) if m1_max > m1_min else 0 for s in m1_scores]
    
    # Method 6: Set-based matching
    expected_set = generate_expected_kmers(pwm, k=k, n_top=30)
    m6_scores = [method6_set_match(kmer, expected_set, allow_mismatch=1) for kmer in top_kmers]
    
    return {
        'method1_recall_0.5': sum(1 for s in m1_norm if s >= 0.5) / len(m1_norm),
        'method1_recall_0.7': sum(1 for s in m1_norm if s >= 0.7) / len(m1_norm),
        'method1_mean_score': np.mean(m1_norm),
        'method6_recall_0.5': sum(1 for s in m6_scores if s >= 0.5) / len(m6_scores),
        'method6_recall_0.7': sum(1 for s in m6_scores if s >= 0.7) / len(m6_scores),
        'method6_mean_score': np.mean(m6_scores),
        'n_kmers': len(top_kmers),
    }


# =============================================================================
# BUILD MAPPING CHAIN
# =============================================================================
print("Building mapping chain...\n")

# 1. Load JSON and extract sample info including experimental method
with open(json_path, 'r') as f:
    filtered_data = json.load(f)

sample_info = {}
for item in filtered_data:
    sample_name = item.get('sample_name', '')
    metadata = item.get('sample_metadata', {}).get('metadata', {})
    
    purification_target = metadata.get('purification_target', {})
    target_value = purification_target.get('value') or purification_target.get('attribute_value') or ''
    
    experimental_method = metadata.get('experimental_method', {})
    method_value = experimental_method.get('value') or experimental_method.get('attribute_value') or 'Unknown'
    
    if sample_name and target_value:
        sample_info[sample_name] = {
            'target': target_value,
            'method': method_value
        }

print(f"Samples with metadata: {len(sample_info)}")

# Show unique methods
methods_found = set(info['method'] for info in sample_info.values())
print(f"Experimental methods found: {methods_found}")

# 2. Map PEKA files
peka_files = glob(os.path.join(peka_dir, "*5mer*.tsv"))
peka_to_info = {}

for peka_file in peka_files:
    basename = os.path.basename(peka_file)
    match = re.match(r'^(.+?)_R\d+\.genome_5mer', basename)
    if match:
        sample_name = match.group(1)
        if sample_name in sample_info:
            peka_to_info[peka_file] = {
                'sample_name': sample_name,
                'target': sample_info[sample_name]['target'],
                'method': sample_info[sample_name]['method']
            }

print(f"PEKA files with metadata: {len(peka_to_info)}")

# 3. Parse SQL dump
with open(dump_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

tfs_raw = parse_all_inserts('tfs', content)
name_to_tf = {}
for row in tfs_raw:
    if len(row) >= 6 and row[5] == 'Homo_sapiens':
        tf_info = {'tf_id': row[0], 'name': row[4]}
        name_to_tf[row[4]] = tf_info
        name_to_tf[row[4].upper()] = tf_info

motifs_raw = parse_all_inserts('motifs', content)
tf_to_motifs = defaultdict(list)
for row in motifs_raw:
    if len(row) >= 7:
        tf_to_motifs[row[1]].append(row[0])

# 4. PFM files
pfm_files_list = glob(os.path.join(pfm_dir, "*.txt"))
pfm_ids = set(os.path.basename(f).replace('.txt', '') for f in pfm_files_list)

# 5. Match targets to SQL
def normalize_name(name):
    return name.upper().replace('-', '').replace('_', '').replace(' ', '')

sql_name_normalized = {}
for name, tf_info in name_to_tf.items():
    norm = normalize_name(name)
    if norm not in sql_name_normalized:
        sql_name_normalized[norm] = tf_info

peka_targets = set(info['target'] for info in peka_to_info.values())
target_to_sql = {}

for target in peka_targets:
    norm_target = normalize_name(target)
    if norm_target in sql_name_normalized:
        target_to_sql[target] = sql_name_normalized[norm_target]
    elif target in name_to_tf:
        target_to_sql[target] = name_to_tf[target]
    elif target.upper() in name_to_tf:
        target_to_sql[target] = name_to_tf[target.upper()]

# 6. Build complete mappings with PFM validation
print("\nValidating PFMs and building mappings...")

target_to_pwm = {}
for target, tf_info in target_to_sql.items():
    motif_ids = tf_to_motifs.get(tf_info['tf_id'], [])
    for motif_id in motif_ids:
        if motif_id in pfm_ids:
            pfm_path = os.path.join(pfm_dir, f"{motif_id}.txt")
            pfm = load_pfm(pfm_path)
            if pfm is not None:
                target_to_pwm[target] = pfm_to_pwm(pfm)
                break

print(f"Targets with valid PWMs: {len(target_to_pwm)}")

# =============================================================================
# CALCULATE RECALL FOR ALL SAMPLES
# =============================================================================
print("\nCalculating recall for all samples...\n")

results = []

for peka_file, info in peka_to_info.items():
    target = info['target']
    
    if target not in target_to_pwm:
        continue
    
    pwm = target_to_pwm[target]
    
    try:
        top_kmers = load_peka_top_kmers(peka_file, n=50)
    except:
        continue
    
    if not top_kmers:
        continue
    
    recall_results = calculate_recall_multi_method(top_kmers, pwm)
    
    if recall_results:
        results.append({
            'rbp': target,
            'sample_name': info['sample_name'],
            'method': info['method'],
            'peka_file': os.path.basename(peka_file),
            **recall_results
        })

df = pd.DataFrame(results)
print(f"Total samples processed: {len(df)}")
print(f"Unique RBPs: {df['rbp'].nunique()}")
print(f"Unique methods: {df['method'].nunique()}")

# Save results
df.to_csv('recall_by_method_results.csv', index=False)

# =============================================================================
# SUMMARY BY EXPERIMENTAL METHOD
# =============================================================================
print("\n" + "="*80)
print("RECALL BY EXPERIMENTAL METHOD")
print("="*80)

method_summary = df.groupby('method').agg({
    'method1_recall_0.5': ['mean', 'std', 'count'],
    'method1_recall_0.7': ['mean'],
    'method6_recall_0.5': ['mean', 'std'],
    'method6_recall_0.7': ['mean'],
}).round(3)

method_summary.columns = ['m1_r0.5_mean', 'm1_r0.5_std', 'n_samples', 'm1_r0.7_mean', 
                          'm6_r0.5_mean', 'm6_r0.5_std', 'm6_r0.7_mean']

print(f"\n{method_summary.to_string()}")

# =============================================================================
# PLOTTING
# =============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Recall@0.5 by experimental method (Method 1 - sliding window)
ax1 = axes[0, 0]
method_order = df.groupby('method')['method1_recall_0.5'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='method', y='method1_recall_0.5', order=method_order, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax1.set_xlabel('Experimental Method')
ax1.set_ylabel('Recall @ 0.5')
ax1.set_title('Recall by Experimental Method\n(Sliding Window Scoring)')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Plot 2: Recall@0.5 by experimental method (Method 6 - set matching)
ax2 = axes[0, 1]
sns.boxplot(data=df, x='method', y='method6_recall_0.5', order=method_order, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax2.set_xlabel('Experimental Method')
ax2.set_ylabel('Recall @ 0.5')
ax2.set_title('Recall by Experimental Method\n(Set-Based Matching)')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Plot 3: Comparison of two scoring methods
ax3 = axes[0, 2]
ax3.scatter(df['method1_recall_0.5'], df['method6_recall_0.5'], 
            c=df['method'].astype('category').cat.codes, cmap='tab10', alpha=0.6)
ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)
ax3.set_xlabel('Method 1 Recall @ 0.5 (Sliding)')
ax3.set_ylabel('Method 6 Recall @ 0.5 (Set-Based)')
ax3.set_title('Scoring Method Comparison')

# Plot 4: Recall distribution by method (violin plot)
ax4 = axes[1, 0]
df_melt = df.melt(id_vars=['method'], value_vars=['method1_recall_0.5', 'method6_recall_0.5'],
                   var_name='scoring', value_name='recall')
df_melt['scoring'] = df_melt['scoring'].map({'method1_recall_0.5': 'Sliding Window', 
                                               'method6_recall_0.5': 'Set-Based'})
sns.violinplot(data=df_melt, x='method', y='recall', hue='scoring', split=True, ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax4.set_xlabel('Experimental Method')
ax4.set_ylabel('Recall @ 0.5')
ax4.set_title('Recall Distribution by Method & Scoring')
ax4.legend(title='Scoring', fontsize=8)

# Plot 5: Mean recall by RBP (colored by method)
ax5 = axes[1, 1]
rbp_summary = df.groupby(['rbp', 'method'])['method1_recall_0.5'].mean().reset_index()
rbp_order = rbp_summary.groupby('rbp')['method1_recall_0.5'].mean().sort_values(ascending=False).index

# Take top 20 RBPs
top_rbps = list(rbp_order[:20])
rbp_summary_top = rbp_summary[rbp_summary['rbp'].isin(top_rbps)]

sns.barplot(data=rbp_summary_top, x='rbp', y='method1_recall_0.5', hue='method', ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90, fontsize=8)
ax5.set_xlabel('RBP')
ax5.set_ylabel('Mean Recall @ 0.5')
ax5.set_title('Recall by RBP (Top 20, colored by method)')
ax5.legend(title='Method', fontsize=7, loc='upper right')
ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Plot 6: Sample counts by method
ax6 = axes[1, 2]
method_counts = df.groupby('method').size().sort_values(ascending=True)
ax6.barh(range(len(method_counts)), method_counts.values, color='steelblue')
ax6.set_yticks(range(len(method_counts)))
ax6.set_yticklabels(method_counts.index)
ax6.set_xlabel('Number of Samples')
ax6.set_title('Sample Count by Experimental Method')

for i, v in enumerate(method_counts.values):
    ax6.text(v + 1, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig('recall_by_experimental_method.png', dpi=150, bbox_inches='tight')
plt.savefig('recall_by_experimental_method.pdf', bbox_inches='tight')
print("Saved: recall_by_experimental_method.png, recall_by_experimental_method.pdf")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE BY EXPERIMENTAL METHOD")
print("="*80)

print(f"\n{'Method':<20} {'N':>6} {'M1 Recall@0.5':>15} {'M6 Recall@0.5':>15}")
print(f"{'':20} {'':>6} {'(mean ± std)':>15} {'(mean ± std)':>15}")
print("-"*60)

for method in method_order:
    subset = df[df['method'] == method]
    n = len(subset)
    m1_mean = subset['method1_recall_0.5'].mean()
    m1_std = subset['method1_recall_0.5'].std()
    m6_mean = subset['method6_recall_0.5'].mean()
    m6_std = subset['method6_recall_0.5'].std()
    
    print(f"{method:<20} {n:>6} {m1_mean:>7.3f} ± {m1_std:<6.3f} {m6_mean:>7.3f} ± {m6_std:<6.3f}")

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"Mean Method 1 Recall@0.5: {df['method1_recall_0.5'].mean():.3f} ± {df['method1_recall_0.5'].std():.3f}")
print(f"Mean Method 6 Recall@0.5: {df['method6_recall_0.5'].mean():.3f} ± {df['method6_recall_0.5'].std():.3f}")

# Save summary
method_summary.to_csv('recall_summary_by_experimental_method.csv')
print(f"\nResults saved to:")
print(f"  - recall_by_method_results.csv (per-sample)")
print(f"  - recall_summary_by_experimental_method.csv (summary)")
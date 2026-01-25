#!/usr/bin/env python3
"""
UMAP analysis of UTR3.tsv files based on PEKA-score values.
Reads all files ending with 'UTR3.tsv' from executions/data folder,
combines them by kmer row names, and creates UMAP visualization.
"""

import os
import sys
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import re

try:
    import umap
except ImportError:
    print("This script requires the 'umap-learn' package. Install with: pip install umap-learn", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.manifold import TSNE
except ImportError:
    print("This script requires the 'scikit-learn' package. Install with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot
except ImportError:
    print("This script requires the 'plotly' package. Install with: pip install plotly", file=sys.stderr)
    sys.exit(1)

# Configuration
DATA_DIR = "executions/data"
OUTPUT_DIR = "executions/analysis"
UTR3_PATTERN = "UTR3.tsv"
INTRON_PATTERN = "intron.tsv"
OTHER_EXON_PATTERN = "other_exon.tsv"
GENOME_PATTERN = "genome.tsv"
PEKA_COLUMN = "PEKA-score"
ANNOTATION_PATH = "/Users/capitac/data/RBP_annotations/human_protein_atlas_subcellular_location.tsv"
MULTIVALENCY_TABLES = {
    "mixed": {
        "path": "../data/multivalency-mixed.tsv",
        "label": "Multivalency Group (Mixed)",
        "column": "Multivalency_Mixed",
        "slug": "multivalency_mixed"
    },
    "strict": {
        "path": "../data/multivalency-strict.tsv",
        "label": "Multivalency Group (Strict)",
        "column": "Multivalency_Strict",
        "slug": "multivalency_strict"
    }
}

def find_matching_files(data_dir: str) -> tuple:
    """Find files for all genomic regions, return only samples that have all three types."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all files for each genomic region
    utr3_files = list(data_path.glob(f"*{UTR3_PATTERN}"))
    intron_files = list(data_path.glob(f"*{INTRON_PATTERN}"))
    other_exon_files = list(data_path.glob(f"*{OTHER_EXON_PATTERN}"))
    
    print(f"Found {len(utr3_files)} UTR3.tsv files")
    print(f"Found {len(intron_files)} intron.tsv files")
    print(f"Found {len(other_exon_files)} other_exon.tsv files")
    
    # Extract sample names (everything before the file type)
    utr3_samples = set()
    intron_samples = set()
    other_exon_samples = set()
    
    for file in utr3_files:
        sample_name = file.stem.replace(UTR3_PATTERN.replace('.tsv', ''), '').rstrip('_')
        utr3_samples.add(sample_name)
    
    for file in intron_files:
        sample_name = file.stem.replace(INTRON_PATTERN.replace('.tsv', ''), '').rstrip('_')
        intron_samples.add(sample_name)
    
    for file in other_exon_files:
        sample_name = file.stem.replace(OTHER_EXON_PATTERN.replace('.tsv', ''), '').rstrip('_')
        other_exon_samples.add(sample_name)
    
    
    # Find samples that have all three file types
    common_samples = utr3_samples.intersection(intron_samples).intersection(other_exon_samples)
    print(f"Found {len(common_samples)} samples with all three genomic region files")
    
    # Filter files to only include those with common samples
    utr3_matched = [f for f in utr3_files if any(sample in f.stem for sample in common_samples)]
    intron_matched = [f for f in intron_files if any(sample in f.stem for sample in common_samples)]
    other_exon_matched = [f for f in other_exon_files if any(sample in f.stem for sample in common_samples)]
    
    return utr3_matched, intron_matched, other_exon_matched, common_samples

def find_genome_files(data_dir: str) -> list:
    """Find all genome.tsv files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all genome files
    genome_files = list(data_path.glob(f"*{GENOME_PATTERN}"))
    print(f"Found {len(genome_files)} genome.tsv files")
    
    return genome_files

def read_genome_file(file_path: Path) -> pd.DataFrame:
    """Read a single genome.tsv file and return DataFrame with kmer as index."""
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)  # First column as index (kmers)
        
        if PEKA_COLUMN not in df.columns:
            print(f"Warning: {PEKA_COLUMN} column not found in {file_path.name}")
            return None
        
        # Extract PEKA-score column
        result = df[[PEKA_COLUMN]].copy()
        
        # Use sample name as column name (remove file type suffix)
        sample_name = file_path.stem.replace(GENOME_PATTERN.replace('.tsv', ''), '').rstrip('_')
        result.columns = [sample_name]
        
        print(f"Loaded {file_path.name}: {len(result)} kmers")
        return result
        
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

def read_file_with_prefix(file_path: Path, prefix: str) -> pd.DataFrame:
    """Read a single TSV file and return DataFrame with prefixed kmer as index."""
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)  # First column as index (kmers)
        
        # Check if PEKA-score column exists
        if PEKA_COLUMN not in df.columns:
            print(f"Warning: {PEKA_COLUMN} column not found in {file_path.name}")
            return None
            
        # Extract only the PEKA-score column and add filename info
        result = df[[PEKA_COLUMN]].copy()
        
        # Add prefix to kmer names
        result.index = [f"{prefix}_{kmer}" for kmer in result.index]
        
        # Use sample name as column name (remove file type suffix)
        sample_name = file_path.stem.replace(UTR3_PATTERN.replace('.tsv', ''), '').replace(INTRON_PATTERN.replace('.tsv', ''), '').replace(OTHER_EXON_PATTERN.replace('.tsv', ''), '').rstrip('_')
        result.columns = [sample_name]
        
        print(f"Loaded {file_path.name}: {len(result)} kmers with {prefix} prefix")
        return result
        
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

def combine_all_genomic_data(utr3_files: list, intron_files: list, other_exon_files: list) -> pd.DataFrame:
    """Combine all genomic region files by sample, with prefixed kmers as rows."""
    # Create a dictionary to store data for each sample
    sample_data = {}
    
    # Process UTR3 files
    for file_path in utr3_files:
        df = read_file_with_prefix(file_path, "3utr")
        if df is not None:
            sample_name = df.columns[0]  # Get the sample name
            if sample_name not in sample_data:
                sample_data[sample_name] = {}
            sample_data[sample_name]['utr3'] = df.iloc[:, 0]  # Get the PEKA scores as Series
    
    # Process intron files
    for file_path in intron_files:
        df = read_file_with_prefix(file_path, "intron")
        if df is not None:
            sample_name = df.columns[0]  # Get the sample name
            if sample_name not in sample_data:
                sample_data[sample_name] = {}
            sample_data[sample_name]['intron'] = df.iloc[:, 0]  # Get the PEKA scores as Series
    
    # Process other_exon files
    for file_path in other_exon_files:
        df = read_file_with_prefix(file_path, "other_exon")
        if df is not None:
            sample_name = df.columns[0]  # Get the sample name
            if sample_name not in sample_data:
                sample_data[sample_name] = {}
            sample_data[sample_name]['other_exon'] = df.iloc[:, 0]  # Get the PEKA scores as Series
    
    
    if not sample_data:
        raise ValueError("No valid genomic region files found")
    
    # Combine all genomic region data for each sample
    combined_data = {}
    for sample_name, data in sample_data.items():
        if all(region in data for region in ['utr3', 'intron', 'other_exon']):
            # Concatenate all genomic region kmers for this sample
            combined_series = pd.concat([data['utr3'], data['intron'], data['other_exon']])
            combined_data[sample_name] = combined_series
        else:
            missing_regions = [region for region in ['utr3', 'intron', 'other_exon'] if region not in data]
            print(f"Warning: Sample {sample_name} missing {missing_regions} data, skipping")
    
    if not combined_data:
        raise ValueError("No samples with all three genomic region data found")
    
    # Create final DataFrame with samples as columns and kmers as rows
    combined_df = pd.DataFrame(combined_data)
    
    # Fill missing values with 0 (kmers not present in all samples)
    combined_df = combined_df.fillna(0)
    
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Kmers: {len(combined_df)}")
    print(f"Samples: {len(combined_df.columns)}")
    
    return combined_df

def combine_genome_data(genome_files: list) -> pd.DataFrame:
    """Combine genome files by kmer index."""
    dataframes = []
    
    # Process genome files
    for file_path in genome_files:
        df = read_genome_file(file_path)
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No valid genome files found")
    
    # Combine all dataframes by index (kmer)
    combined_df = pd.concat(dataframes, axis=1, sort=True)
    
    # Fill missing values with 0 (kmers not present in all files)
    combined_df = combined_df.fillna(0)
    
    print(f"Combined genome data shape: {combined_df.shape}")
    print(f"Kmers: {len(combined_df)}")
    print(f"Samples: {len(combined_df.columns)}")
    
    return combined_df


def load_subcellular_annotations(annotation_path: str) -> dict:
    """Load subcellular localization annotations from TSV file."""
    if not annotation_path:
        return {}
    
    annotation_path = os.path.expanduser(annotation_path)
    
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found at {annotation_path}. Continuing without subcellular localization data.")
        return {}
    
    try:
        df = pd.read_csv(annotation_path, sep='\t')
    except Exception as e:
        print(f"Warning: Unable to read annotation file {annotation_path}: {e}")
        return {}
    
    if 'Gene name' not in df.columns:
        print("Warning: 'Gene name' column not found in annotation file. Skipping localization coloring.")
        return {}
    
    if 'Main location' not in df.columns:
        print("Warning: 'Main location' column not found in annotation file. Skipping localization coloring.")
        return {}
    
    annotations = {}
    for _, row in df.iterrows():
        gene_name = str(row['Gene name']).strip()
        if not gene_name:
            continue
        
        localization = row['Main location']
        if isinstance(localization, float) and np.isnan(localization):
            continue
        
        localization_str = str(localization).strip()
        if localization_str:
            annotations[gene_name] = localization_str
    
    print(f"Loaded subcellular localization annotations for {len(annotations)} proteins")
    return annotations


def extract_protein_name(sample_name: str) -> str:
    """Extract the protein name from a sample name (before the first underscore)."""
    if not sample_name:
        return ""
    if '_' in sample_name:
        return sample_name.split('_', 1)[0]
    return sample_name


def map_samples_to_localizations(sample_names: list, annotations: dict) -> tuple:
    """Map sample names to protein names and localization labels."""
    proteins = []
    localizations = []
    
    for sample in sample_names:
        protein = extract_protein_name(sample)
        proteins.append(protein)
        raw_localization = annotations.get(protein, "Unknown")
        normalized_localization = normalize_localization_label(raw_localization)
        localizations.append(normalized_localization)
    
    return proteins, localizations


def normalize_localization_label(localization: str) -> str:
    """Normalize localization string by applying business rules."""
    if not localization or localization == "Unknown":
        return "Unknown"
    
    if isinstance(localization, float) and np.isnan(localization):
        return "Unknown"
    
    localization_str = str(localization).strip()
    if not localization_str:
        return "Unknown"
    
    # Special case: Cytoplasm and Nuclear combination indicates whole cell
    if "Cytoplasm;Nucle" in localization_str:
        return "Whole Cell"
    
    # Otherwise take the first entry from semi-colon separated list
    entries = [entry.strip() for entry in localization_str.split(';') if entry.strip()]
    if entries:
        return entries[0]
    
    return "Unknown"


def update_localization_summary(summary_map: dict, sample_names: list, proteins: list, localizations: list, source: str):
    """Update the summary map with sample localization information."""
    for sample, protein, localization in zip(sample_names, proteins, localizations):
        if sample not in summary_map:
            summary_map[sample] = {
                'Sample': sample,
                'Protein': protein,
                'Localization': localization,
                'Sources': set()
            }
        else:
            # Update protein/localization if existing entry is unknown
            if summary_map[sample]['Localization'] == "Unknown" and localization != "Unknown":
                summary_map[sample]['Protein'] = protein
                summary_map[sample]['Localization'] = localization
        summary_map[sample]['Sources'].add(source)


def save_localization_summary(summary_map: dict, output_dir: str, filename: str = "sample_localization_summary.csv") -> str:
    """Save the localization summary map to CSV."""
    if not summary_map:
        print("No localization summary data to save.")
        return ""
    
    records = []
    for sample, info in summary_map.items():
        sources = sorted(info['Sources'])
        records.append({
            'Sample': info['Sample'],
            'Protein': info['Protein'],
            'Localization': info['Localization'],
            'Sources': ';'.join(sources)
        })
    
    summary_df = pd.DataFrame(records).sort_values('Sample')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    summary_df.to_csv(output_path, index=False)
    print(f"Sample localization summary saved to: {output_path}")
    return output_path


def update_multivalency_summary(
    summary_rows: List[Dict[str, str]],
    sample_names: List[str],
    protein_names: List[str],
    multivalency_configs: Optional[Dict[str, Dict[str, Any]]],
    source: str
) -> None:
    """Record per-sample multivalency matches for later inspection."""
    if not multivalency_configs:
        return
    
    for scheme_key, cfg in multivalency_configs.items():
        mapping = cfg.get("mapping") or {}
        scheme_label = cfg.get("label", scheme_key)
        for sample, protein in zip(sample_names, protein_names):
            normalized = normalize_identifier(protein)
            group = mapping.get(normalized)
            summary_rows.append({
                "Sample": sample,
                "Protein": protein,
                "IdentifierUsed": normalized,
                "Scheme": scheme_label,
                "Group": group if group else "Unknown",
                "Matched": "Yes" if group else "No",
                "Source": source
            })


def save_multivalency_summary(
    summary_rows: List[Dict[str, str]],
    output_dir: str,
    filename: str = "multivalency_matching_summary.csv"
) -> str:
    """Persist the multivalency matching diagnostics to CSV."""
    if not summary_rows:
        print("No multivalency summary data to save.")
        return ""
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.drop_duplicates().sort_values(["Scheme", "Sample"])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    summary_df.to_csv(output_path, index=False)
    print(f"Multivalency matching summary saved to: {output_path}")
    return output_path

def create_genome_visualization(
    data: pd.DataFrame,
    output_dir: str,
    annotations: dict,
    protein_names: Optional[List[str]] = None,
    localization_labels: Optional[List[str]] = None,
    multivalency_configs: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Create UMAP and t-SNE visualizations for genome data only."""
    print("Creating genome-only UMAP and t-SNE visualizations...")
    
    # Calculate z-scores per sample
    print("Calculating z-scores per sample...")
    zscore_data = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Compute UMAP embedding
    print("Computing UMAP embedding...")
    umap_reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        spread=1.0,
        metric='euclidean',
        random_state=42
    )
    umap_embedding = umap_reducer.fit_transform(zscore_data.T.values)
    
    # Compute t-SNE embedding
    print("Computing t-SNE embedding...")
    tsne_reducer = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate=200,
        metric='euclidean',
        random_state=42
    )
    tsne_embedding = tsne_reducer.fit_transform(zscore_data.T.values)
    
    # Analyze dinucleotide patterns from top 5 kmers
    print("Analyzing dinucleotide patterns from top 5 kmers (reverse dinucleotides combined)...")
    top_dinucleotides = get_top_dinucleotide_per_sample(data, top_n=5)
    
    # Create plot data
    plot_df = pd.DataFrame({
        'UMAP_1': umap_embedding[:, 0],
        'UMAP_2': umap_embedding[:, 1],
        'TSNE_1': tsne_embedding[:, 0],
        'TSNE_2': tsne_embedding[:, 1],
        'Sample': data.columns,
        'Mean_PEKA_Score': data.mean(),
        'Max_PEKA_Score': data.max(),
        'NonZero_Count': (data > 0).sum(),
        'Mean_ZScore': zscore_data.mean(),
        'Max_ZScore': zscore_data.max(),
        'Top_Dinucleotide': top_dinucleotides
    })
    
    # Map samples to protein names and localizations if not provided
    if protein_names is None or localization_labels is None:
        protein_names, localization_labels = map_samples_to_localizations(list(data.columns), annotations)
    plot_df['Protein'] = protein_names
    plot_df['Subcellular_Localization'] = localization_labels
    
    unique_localizations = plot_df['Subcellular_Localization'].unique()
    localization_color_map = get_category_color_map(unique_localizations)
    
    sample_count = len(data.columns)
    kmer_count = len(data)
    
    multivalency_columns = []
    if multivalency_configs:
        for key, cfg in multivalency_configs.items():
            mapping = cfg.get("mapping") or {}
            if not mapping:
                continue
            column_name = cfg["column"]
            groups = map_multivalency_groups(protein_names, mapping)
            plot_df[column_name] = groups
            multivalency_columns.append((column_name, cfg))
    
    # Get top 5 kmers for each sample for hover data
    top_5_kmers = []
    for sample in data.columns:
        sample_data = data[sample]
        if isinstance(sample_data, pd.DataFrame):
            sample_series = sample_data.iloc[:, 0]
        else:
            sample_series = sample_data
        top_kmers = sample_series.nlargest(5).index.tolist()
        top_5_kmers.append(', '.join(top_kmers))
    
    plot_df['Top_5_Kmers'] = top_5_kmers
    
    # Create interactive UMAP plot
    fig = px.scatter(
        plot_df, 
        x='UMAP_1', 
        y='UMAP_2',
        color='Mean_PEKA_Score',
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_5_Kmers'],
        title=f'Interactive UMAP Visualization of Genome Samples (Z-score normalized)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.7
    )
    
    # Update layout for better appearance
    fig.update_layout(
        width=1000,
        height=800,
        showlegend=False,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    
    # Save interactive UMAP
    umap_html_path = os.path.join(output_dir, "genome_umap_interactive.html")
    plot(fig, filename=umap_html_path, auto_open=False)
    print(f"Interactive UMAP visualization saved to: {umap_html_path}")
    
    # Create interactive t-SNE plot
    fig_tsne = px.scatter(
        plot_df, 
        x='TSNE_1', 
        y='TSNE_2',
        color='Mean_PEKA_Score',
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_5_Kmers'],
        title=f'Interactive t-SNE Visualization of Genome Samples (Z-score normalized)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.7
    )
    
    # Update layout for t-SNE
    fig_tsne.update_layout(
        width=1000,
        height=800,
        showlegend=False,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_tsne.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    
    # Save interactive t-SNE
    tsne_html_path = os.path.join(output_dir, "genome_tsne_interactive.html")
    plot(fig_tsne, filename=tsne_html_path, auto_open=False)
    print(f"Interactive t-SNE visualization saved to: {tsne_html_path}")
    
    # Create subcellular localization-colored interactive plots
    print("Creating subcellular localization-colored interactive plots...")
    fig_loc_umap = px.scatter(
        plot_df,
        x='UMAP_1',
        y='UMAP_2',
        color='Subcellular_Localization',
        color_discrete_map=localization_color_map,
        hover_data=['Sample', 'Protein', 'Subcellular_Localization', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Top_5_Kmers'],
        title=f'Interactive UMAP Colored by Subcellular Localization<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.8
    )
    fig_loc_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_loc_umap.update_traces(marker=dict(size=8))
    
    loc_umap_html_path = os.path.join(output_dir, "genome_umap_subcellular_colored.html")
    plot(fig_loc_umap, filename=loc_umap_html_path, auto_open=False)
    print(f"Subcellular localization-colored UMAP visualization saved to: {loc_umap_html_path}")
    
    fig_loc_tsne = px.scatter(
        plot_df,
        x='TSNE_1',
        y='TSNE_2',
        color='Subcellular_Localization',
        color_discrete_map=localization_color_map,
        hover_data=['Sample', 'Protein', 'Subcellular_Localization', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Top_5_Kmers'],
        title=f'Interactive t-SNE Colored by Subcellular Localization<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.8
    )
    fig_loc_tsne.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_loc_tsne.update_traces(marker=dict(size=8))
    
    loc_tsne_html_path = os.path.join(output_dir, "genome_tsne_subcellular_colored.html")
    plot(fig_loc_tsne, filename=loc_tsne_html_path, auto_open=False)
    print(f"Subcellular localization-colored t-SNE visualization saved to: {loc_tsne_html_path}")
    
    # Create static matplotlib versions
    # UMAP static plot
    plt.figure(figsize=(12, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], alpha=0.7, s=50)
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Visualization of Genome Samples (Z-score normalized)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save UMAP static versions
    umap_png_path = os.path.join(output_dir, "genome_umap_visualization.png")
    plt.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Static UMAP visualization saved to: {umap_png_path}")
    
    umap_pdf_path = os.path.join(output_dir, "genome_umap_visualization.pdf")
    plt.savefig(umap_pdf_path, bbox_inches='tight')
    print(f"Static UMAP visualization (PDF) saved to: {umap_pdf_path}")
    
    # t-SNE static plot
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], alpha=0.7, s=50)
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Visualization of Genome Samples (Z-score normalized)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save t-SNE static versions
    tsne_png_path = os.path.join(output_dir, "genome_tsne_visualization.png")
    plt.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Static t-SNE visualization saved to: {tsne_png_path}")
    
    tsne_pdf_path = os.path.join(output_dir, "genome_tsne_visualization.pdf")
    plt.savefig(tsne_pdf_path, bbox_inches='tight')
    print(f"Static t-SNE visualization (PDF) saved to: {tsne_pdf_path}")
    
    # Static localization-colored UMAP
    plt.figure(figsize=(14, 8))
    for localization in sorted(unique_localizations):
        mask = plot_df['Subcellular_Localization'] == localization
        plt.scatter(plot_df.loc[mask, 'UMAP_1'], plot_df.loc[mask, 'UMAP_2'],
                    label=localization, alpha=0.7, s=50, color=localization_color_map[localization])
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Colored by Subcellular Localization\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loc_umap_png_path = os.path.join(output_dir, "genome_umap_subcellular_colored.png")
    plt.savefig(loc_umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Subcellular localization-colored UMAP static visualization saved to: {loc_umap_png_path}")
    
    loc_umap_pdf_path = os.path.join(output_dir, "genome_umap_subcellular_colored.pdf")
    plt.savefig(loc_umap_pdf_path, bbox_inches='tight')
    print(f"Subcellular localization-colored UMAP static visualization (PDF) saved to: {loc_umap_pdf_path}")
    
    # Static localization-colored t-SNE
    plt.figure(figsize=(14, 8))
    for localization in sorted(unique_localizations):
        mask = plot_df['Subcellular_Localization'] == localization
        plt.scatter(plot_df.loc[mask, 'TSNE_1'], plot_df.loc[mask, 'TSNE_2'],
                    label=localization, alpha=0.7, s=50, color=localization_color_map[localization])
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Colored by Subcellular Localization\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loc_tsne_png_path = os.path.join(output_dir, "genome_tsne_subcellular_colored.png")
    plt.savefig(loc_tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Subcellular localization-colored t-SNE static visualization saved to: {loc_tsne_png_path}")
    
    loc_tsne_pdf_path = os.path.join(output_dir, "genome_tsne_subcellular_colored.pdf")
    plt.savefig(loc_tsne_pdf_path, bbox_inches='tight')
    print(f"Subcellular localization-colored t-SNE static visualization (PDF) saved to: {loc_tsne_pdf_path}")
    
    # Multivalency-colored plots
    for column_name, cfg in multivalency_columns:
        column_label = cfg.get("label", column_name)
        slug = cfg.get("slug", column_name.lower())
        create_category_colored_plots(
            plot_df,
            column_name,
            column_label,
            f"genome_{slug}",
            output_dir,
            sample_count,
            kmer_count
        )
    
    # Create dinucleotide-colored plots
    print("Creating dinucleotide-colored plots...")
    
    # Get unique dinucleotides for coloring
    unique_dinucleotides = plot_df['Top_Dinucleotide'].unique()
    print(f"Found {len(unique_dinucleotides)} unique dinucleotides: {sorted(unique_dinucleotides)}")
    
    # Create a consistent color map for dinucleotides
    color_map = get_dinucleotide_color_map(unique_dinucleotides)
    
    # Create interactive dinucleotide-colored UMAP plot
    fig_dinuc_umap = px.scatter(
        plot_df, 
        x='UMAP_1', 
        y='UMAP_2',
        color='Top_Dinucleotide',
        color_discrete_map=color_map,
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_5_Kmers'],
        title=f'Interactive UMAP Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.7
    )
    
    # Update layout
    fig_dinuc_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_dinuc_umap.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    
    # Save interactive dinucleotide-colored UMAP
    dinuc_umap_html_path = os.path.join(output_dir, "genome_umap_dinucleotide_colored.html")
    plot(fig_dinuc_umap, filename=dinuc_umap_html_path, auto_open=False)
    print(f"Interactive dinucleotide-colored UMAP visualization saved to: {dinuc_umap_html_path}")
    
    # Create interactive dinucleotide-colored t-SNE plot
    fig_dinuc_tsne = px.scatter(
        plot_df, 
        x='TSNE_1', 
        y='TSNE_2',
        color='Top_Dinucleotide',
        color_discrete_map=color_map,
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_5_Kmers'],
        title=f'Interactive t-SNE Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.7
    )
    
    # Update layout
    fig_dinuc_tsne.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_dinuc_tsne.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    
    # Save interactive dinucleotide-colored t-SNE
    dinuc_tsne_html_path = os.path.join(output_dir, "genome_tsne_dinucleotide_colored.html")
    plot(fig_dinuc_tsne, filename=dinuc_tsne_html_path, auto_open=False)
    print(f"Interactive dinucleotide-colored t-SNE visualization saved to: {dinuc_tsne_html_path}")
    
    # Create static dinucleotide-colored UMAP plot
    plt.figure(figsize=(14, 8))
    for dinucleotide in sorted(unique_dinucleotides):
        mask = plot_df['Top_Dinucleotide'] == dinucleotide
        plt.scatter(plot_df.loc[mask, 'UMAP_1'], plot_df.loc[mask, 'UMAP_2'], 
                   label=dinucleotide, alpha=0.7, s=50, color=color_map[dinucleotide])
    
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save dinucleotide-colored UMAP static versions
    dinuc_umap_png_path = os.path.join(output_dir, "genome_umap_dinucleotide_colored.png")
    plt.savefig(dinuc_umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Dinucleotide-colored UMAP static visualization saved to: {dinuc_umap_png_path}")
    
    dinuc_umap_pdf_path = os.path.join(output_dir, "genome_umap_dinucleotide_colored.pdf")
    plt.savefig(dinuc_umap_pdf_path, bbox_inches='tight')
    print(f"Dinucleotide-colored UMAP static visualization (PDF) saved to: {dinuc_umap_pdf_path}")
    
    # Create static dinucleotide-colored t-SNE plot
    plt.figure(figsize=(14, 8))
    for dinucleotide in sorted(unique_dinucleotides):
        mask = plot_df['Top_Dinucleotide'] == dinucleotide
        plt.scatter(plot_df.loc[mask, 'TSNE_1'], plot_df.loc[mask, 'TSNE_2'], 
                   label=dinucleotide, alpha=0.7, s=50, color=color_map[dinucleotide])
    
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save dinucleotide-colored t-SNE static versions
    dinuc_tsne_png_path = os.path.join(output_dir, "genome_tsne_dinucleotide_colored.png")
    plt.savefig(dinuc_tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Dinucleotide-colored t-SNE static visualization saved to: {dinuc_tsne_png_path}")
    
    dinuc_tsne_pdf_path = os.path.join(output_dir, "genome_tsne_dinucleotide_colored.pdf")
    plt.savefig(dinuc_tsne_pdf_path, bbox_inches='tight')
    print(f"Dinucleotide-colored t-SNE static visualization (PDF) saved to: {dinuc_tsne_pdf_path}")
    
    return umap_embedding, tsne_embedding


def normalize_dinucleotide(dinuc: str) -> str:
    """Normalize dinucleotide by taking the lexicographically smaller of the pair and its reverse."""
    if len(dinuc) != 2:
        return dinuc
    reverse = dinuc[1] + dinuc[0]  # Reverse the dinucleotide
    return min(dinuc, reverse)  # Return the lexicographically smaller one

def get_dinucleotide_color_map(dinucleotides: list) -> dict:
    """Create a consistent color mapping for dinucleotides across all plots."""
    # Sort dinucleotides for consistent ordering
    sorted_dinucleotides = sorted(dinucleotides)
    
    # Use a high-contrast color palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9896',  # light red
        '#98df8a',  # light green
        '#ffbb78',  # light orange
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d3',  # light pink
        '#c7c7c7',  # light gray
        '#dbdb8d',  # light olive
        '#9edae5',  # light cyan
        '#ff9896',  # light red
    ]
    
    # Create mapping, cycling through colors if needed
    color_map = {}
    for i, dinuc in enumerate(sorted_dinucleotides):
        color_map[dinuc] = colors[i % len(colors)]
    
    return color_map


def resolve_path(candidate_path: str) -> Path:
    """Resolve a path relative to script directory if needed."""
    path_obj = Path(candidate_path).expanduser()
    if path_obj.exists():
        return path_obj
    script_dir = Path(__file__).resolve().parent
    alt_path = script_dir / candidate_path
    if alt_path.exists():
        return alt_path
    return path_obj


def normalize_identifier(value: str) -> str:
    """Normalize identifiers (protein/gene names) for matching."""
    if value is None:
        return ""
    return str(value).strip().upper()


def load_multivalency_table(table_path: str) -> Dict[str, str]:
    """Load multivalency annotations from a TSV file."""
    resolved_path = resolve_path(table_path)
    if not resolved_path.exists():
        print(f"Warning: Multivalency table not found at {resolved_path}. Skipping.")
        return {}
    
    try:
        df = pd.read_csv(resolved_path, sep='\t')
    except Exception as e:
        print(f"Warning: Unable to read multivalency table {resolved_path}: {e}")
        return {}
    
    group_col = None
    for candidate in ["multivalency_group", "multivalency", "multivalency_type"]:
        if candidate in df.columns:
            group_col = candidate
            break
    if group_col is None:
        print(f"Warning: No 'multivalency_group' column in {resolved_path}. Columns: {list(df.columns)}")
        return {}
    
    id_columns = [
        "protein_id", "Protein ID", "protein", "Protein", "protein_name", "Protein Name",
        "gene_name", "Gene name", "gene", "Gene", "transcript_id", "Transcript ID"
    ]
    
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        group_value = row.get(group_col)
        if pd.isna(group_value):
            continue
        group = str(group_value).strip()
        if not group:
            continue
        
        identifiers = set()
        for col in id_columns:
            if col in df.columns:
                value = row.get(col)
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if value_str:
                    identifiers.add(value_str)
        
        for identifier in identifiers:
            mapping[normalize_identifier(identifier)] = group
    
    print(f"Loaded multivalency annotations from {resolved_path}: {len(mapping)} identifiers")
    return mapping


def load_multivalency_configs() -> Dict[str, Dict[str, Any]]:
    """Load all configured multivalency tables."""
    configs: Dict[str, Dict[str, Any]] = {}
    for key, cfg in MULTIVALENCY_TABLES.items():
        mapping = load_multivalency_table(cfg["path"])
        configs[key] = {
            **cfg,
            "mapping": mapping
        }
    return configs


def map_multivalency_groups(protein_names: List[str], mapping: Dict[str, str]) -> List[str]:
    """Map protein names to multivalency groups using normalized identifiers."""
    groups = []
    for protein in protein_names:
        normalized = normalize_identifier(protein)
        group = mapping.get(normalized, "Unknown")
        groups.append(group)
    return groups


def create_category_colored_plots(
    plot_df: pd.DataFrame,
    column_name: str,
    column_label: str,
    file_prefix: str,
    output_dir: str,
    sample_count: int,
    kmer_count: int
):
    """Create interactive and static plots colored by a categorical column."""
    if column_name not in plot_df.columns:
        return
    
    categories = plot_df[column_name].fillna("Unknown")
    unique_categories = sorted(categories.unique())
    color_map = get_category_color_map(unique_categories)
    
    hover_candidates = [
        'Sample', 'Protein', column_name,
        'Mean_PEKA_Score', 'Max_PEKA_Score',
        'NonZero_Count', 'Mean_ZScore', 'Max_ZScore',
        'Top_Dinucleotide', 'Top_Kmer', 'Top_5_Kmers'
    ]
    hover_data = [col for col in hover_candidates if col in plot_df.columns]
    
    title_suffix = f"<br>({sample_count} samples, {kmer_count} kmers)"
    
    # Interactive UMAP
    fig_umap = px.scatter(
        plot_df,
        x='UMAP_1',
        y='UMAP_2',
        color=column_name,
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=f"UMAP Colored by {column_label}{title_suffix}",
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.8
    )
    fig_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_umap.update_traces(marker=dict(size=8))
    
    umap_html_path = os.path.join(output_dir, f"{file_prefix}_umap.html")
    fig_umap.write_html(umap_html_path)
    print(f"{column_label} UMAP visualization saved to: {umap_html_path}")
    
    # Interactive t-SNE
    fig_tsne = px.scatter(
        plot_df,
        x='TSNE_1',
        y='TSNE_2',
        color=column_name,
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=f"t-SNE Colored by {column_label}{title_suffix}",
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.8
    )
    fig_tsne.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_tsne.update_traces(marker=dict(size=8))
    
    tsne_html_path = os.path.join(output_dir, f"{file_prefix}_tsne.html")
    fig_tsne.write_html(tsne_html_path)
    print(f"{column_label} t-SNE visualization saved to: {tsne_html_path}")
    
    # Static UMAP
    plt.figure(figsize=(14, 8))
    for category in unique_categories:
        mask = categories == category
        plt.scatter(
            plot_df.loc[mask, 'UMAP_1'],
            plot_df.loc[mask, 'UMAP_2'],
            label=category,
            alpha=0.7,
            s=50,
            color=color_map[category]
        )
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Colored by {column_label}\n({sample_count} samples, {kmer_count} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    umap_png_path = os.path.join(output_dir, f"{file_prefix}_umap.png")
    plt.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Static UMAP colored by {column_label} saved to: {umap_png_path}")
    
    umap_pdf_path = os.path.join(output_dir, f"{file_prefix}_umap.pdf")
    plt.savefig(umap_pdf_path, bbox_inches='tight')
    print(f"Static UMAP colored by {column_label} (PDF) saved to: {umap_pdf_path}")
    
    # Static t-SNE
    plt.figure(figsize=(14, 8))
    for category in unique_categories:
        mask = categories == category
        plt.scatter(
            plot_df.loc[mask, 'TSNE_1'],
            plot_df.loc[mask, 'TSNE_2'],
            label=category,
            alpha=0.7,
            s=50,
            color=color_map[category]
        )
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Colored by {column_label}\n({sample_count} samples, {kmer_count} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    tsne_png_path = os.path.join(output_dir, f"{file_prefix}_tsne.png")
    plt.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Static t-SNE colored by {column_label} saved to: {tsne_png_path}")
    
    tsne_pdf_path = os.path.join(output_dir, f"{file_prefix}_tsne.pdf")
    plt.savefig(tsne_pdf_path, bbox_inches='tight')
    print(f"Static t-SNE colored by {column_label} (PDF) saved to: {tsne_pdf_path}")


def get_category_color_map(categories: list) -> dict:
    """Create a color map for arbitrary categorical labels."""
    sorted_categories = sorted(categories)
    
    # High-contrast palette (extend Plotly qualitative colors)
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#a55194', '#6b6ecf', '#9c9ede', '#637939', '#b5cf6b',
        '#8c6d31', '#bd9e39', '#ad494a', '#d6616b', '#e7969c',
        '#7b4173', '#c7a9d6', '#3182bd', '#6baed6', '#9ecae1',
        '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2'
    ]
    
    color_map = {}
    for i, category in enumerate(sorted_categories):
        color_map[category] = palette[i % len(palette)]
    
    return color_map


def get_top_dinucleotide_per_sample(data: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """Get the most prevalent dinucleotide from the top N kmers for each sample.
    Reverse dinucleotides (e.g., GU and UG) are combined into a single group."""
    dinucleotides = []
    
    for sample in data.columns:
        # Get the top N kmers with highest PEKA scores for this sample
        sample_data = data[sample]
        
        # Handle case where sample_data might be a DataFrame (duplicate columns)
        if isinstance(sample_data, pd.DataFrame):
            # If it's a DataFrame, take the first column (should be the PEKA scores)
            sample_series = sample_data.iloc[:, 0]
        else:
            sample_series = sample_data
            
        top_kmers = sample_series.nlargest(top_n).index.tolist()
        
        # Extract all dinucleotides from all top kmers and normalize them
        all_dinucleotides = []
        for kmer in top_kmers:
            # Strip prefix (3utr_, intron_, or other_exon_) to get original kmer
            original_kmer = kmer.split('_', 1)[1] if '_' in kmer else kmer
            kmer_dinucleotides = [original_kmer[i:i+2] for i in range(len(original_kmer)-1)]
            # Normalize each dinucleotide to combine reverse pairs
            normalized_dinucleotides = [normalize_dinucleotide(dinuc) for dinuc in kmer_dinucleotides]
            all_dinucleotides.extend(normalized_dinucleotides)
        
        # Count dinucleotides across all top kmers and get the most common one
        dinucleotide_counts = Counter(all_dinucleotides)
        most_common_dinucleotide = dinucleotide_counts.most_common(1)[0][0]
        
        dinucleotides.append(most_common_dinucleotide)
    
    return pd.Series(dinucleotides, index=data.columns)

def create_umap_visualization(
    data: pd.DataFrame,
    output_dir: str,
    annotations: dict,
    protein_names: Optional[List[str]] = None,
    localization_labels: Optional[List[str]] = None,
    multivalency_configs: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Create interactive UMAP visualization where each dot represents a sample."""
    # Calculate z-scores per sample (across kmers)
    print("Calculating z-scores per sample...")
    data_zscored = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Handle any NaN values (if std=0 for a sample)
    data_zscored = data_zscored.fillna(0)
    
    # Transpose data so samples are rows and kmers are columns
    X = data_zscored.T.values  # Transpose: samples x kmers
    
    # Create UMAP embedding with parameters optimized for clustering
    print("Computing UMAP embedding...")
    reducer_umap = umap.UMAP(
        n_neighbors=50,  # Smaller for more local structure
        min_dist=0.01,  # Smaller for tighter clusters
        n_components=2,
        random_state=42,
        metric='cosine'  # Often better for biological data
    )
    
    umap_embedding = reducer_umap.fit_transform(X)
    
    # Create t-SNE embedding
    print("Computing t-SNE embedding...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, max(5, len(data.columns) // 3)),  # Adjust based on sample size
        max_iter=1000,
        learning_rate=200,
        metric='cosine'
    )
    
    tsne_embedding = tsne.fit_transform(X)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top dinucleotide for each sample (from top 5 kmers, reverse dinucleotides combined)
    print("Analyzing dinucleotide patterns from top 5 kmers (reverse dinucleotides combined)...")
    top_dinucleotides = get_top_dinucleotide_per_sample(data, top_n=5)
    
    # Create interactive plot with Plotly
    print("Creating interactive UMAP visualization...")
    
    # Create a DataFrame for the plots - each point is a sample
    plot_df = pd.DataFrame({
        'UMAP_1': umap_embedding[:, 0],
        'UMAP_2': umap_embedding[:, 1],
        'TSNE_1': tsne_embedding[:, 0],
        'TSNE_2': tsne_embedding[:, 1],
        'Sample': data.columns,  # Sample names
        'Mean_PEKA_Score': data.mean(axis=0).values,  # Mean PEKA score across all kmers for this sample
        'Max_PEKA_Score': data.max(axis=0).values,    # Max PEKA score for this sample
        'NonZero_Count': (data > 0).sum(axis=0).values,  # Number of kmers with non-zero PEKA scores
        'Mean_ZScore': data_zscored.mean(axis=0).values,  # Mean z-score for this sample
        'Max_ZScore': data_zscored.max(axis=0).values,    # Max z-score for this sample
        'Top_Dinucleotide': top_dinucleotides.values,     # Most prevalent dinucleotide in top kmer
        'Top_Kmer': [data[sample].idxmax() for sample in data.columns],  # The actual top kmer
        'Top_5_Kmers': [', '.join(data[sample].nlargest(5).index.tolist()) for sample in data.columns]  # Top 5 kmers
    })
    
    # Map samples to protein/localization annotations if not provided
    if protein_names is None or localization_labels is None:
        protein_names, localization_labels = map_samples_to_localizations(list(data.columns), annotations)
    plot_df['Protein'] = protein_names
    plot_df['Subcellular_Localization'] = localization_labels
    
    localization_categories = plot_df['Subcellular_Localization'].unique()
    localization_color_map = get_category_color_map(localization_categories)
    
    sample_count = len(data.columns)
    kmer_count = len(data)
    
    multivalency_columns = []
    if multivalency_configs:
        for key, cfg in multivalency_configs.items():
            mapping = cfg.get("mapping") or {}
            if not mapping:
                continue
            column_name = cfg["column"]
            groups = map_multivalency_groups(protein_names, mapping)
            plot_df[column_name] = groups
            multivalency_columns.append((column_name, cfg))
    
    # Create interactive scatter plot
    fig = px.scatter(
        plot_df, 
        x='UMAP_1', 
        y='UMAP_2',
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_Kmer', 'Top_5_Kmers'],
        title=f'Interactive UMAP Visualization of UTR3 Samples (Z-score normalized)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.7
    )
    
    # Update layout for better appearance
    fig.update_layout(
        width=1000,
        height=800,
        showlegend=False,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    
    # Save interactive UMAP HTML
    html_path = os.path.join(output_dir, "utr3_umap_interactive.html")
    fig.write_html(html_path)
    print(f"Interactive UMAP visualization saved to: {html_path}")
    
    # Create interactive t-SNE plot
    print("Creating interactive t-SNE visualization...")
    fig_tsne = px.scatter(
        plot_df, 
        x='TSNE_1', 
        y='TSNE_2',
        hover_data=['Sample', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Mean_ZScore', 'Max_ZScore', 'Top_Dinucleotide', 'Top_Kmer', 'Top_5_Kmers'],
        title=f'Interactive t-SNE Visualization of UTR3 Samples (Z-score normalized)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.7
    )
    
    # Update layout for t-SNE
    fig_tsne.update_layout(
        width=1000,
        height=800,
        showlegend=False,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_tsne.update_traces(marker=dict(size=8))
    
    # Save interactive t-SNE HTML
    tsne_html_path = os.path.join(output_dir, "utr3_tsne_interactive.html")
    fig_tsne.write_html(tsne_html_path)
    print(f"Interactive t-SNE visualization saved to: {tsne_html_path}")
    
    # Create subcellular localization-colored interactive plots
    print("Creating subcellular localization-colored UMAP/t-SNE visualizations...")
    fig_loc_umap = px.scatter(
        plot_df,
        x='UMAP_1',
        y='UMAP_2',
        color='Subcellular_Localization',
        color_discrete_map=localization_color_map,
        hover_data=['Sample', 'Protein', 'Subcellular_Localization', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Top_Kmer', 'Top_5_Kmers'],
        title=f'UMAP Colored by Subcellular Localization<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.8
    )
    fig_loc_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_loc_umap.update_traces(marker=dict(size=8))
    
    loc_umap_html_path = os.path.join(output_dir, "utr3_umap_subcellular_colored.html")
    fig_loc_umap.write_html(loc_umap_html_path)
    print(f"Subcellular localization-colored UMAP visualization saved to: {loc_umap_html_path}")
    
    fig_loc_tsne = px.scatter(
        plot_df,
        x='TSNE_1',
        y='TSNE_2',
        color='Subcellular_Localization',
        color_discrete_map=localization_color_map,
        hover_data=['Sample', 'Protein', 'Subcellular_Localization', 'Mean_PEKA_Score', 'Max_PEKA_Score', 'NonZero_Count', 'Top_Kmer', 'Top_5_Kmers'],
        title=f't-SNE Colored by Subcellular Localization<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.8
    )
    fig_loc_tsne.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    fig_loc_tsne.update_traces(marker=dict(size=8))
    
    loc_tsne_html_path = os.path.join(output_dir, "utr3_tsne_subcellular_colored.html")
    fig_loc_tsne.write_html(loc_tsne_html_path)
    print(f"Subcellular localization-colored t-SNE visualization saved to: {loc_tsne_html_path}")
    
    # Create dinucleotide-colored UMAP plot
    print("Creating dinucleotide-colored UMAP visualization...")
    
    # Create a consistent color map for dinucleotides
    unique_dinucleotides = plot_df['Top_Dinucleotide'].unique()
    color_map = get_dinucleotide_color_map(unique_dinucleotides)
    
    # Create the dinucleotide-colored UMAP plot
    fig_dinuc_umap = px.scatter(
        plot_df, 
        x='UMAP_1', 
        y='UMAP_2',
        color='Top_Dinucleotide',
        hover_data=['Sample', 'Top_Kmer', 'Top_Dinucleotide', 'Max_PEKA_Score', 'Top_5_Kmers'],
        title=f'UMAP Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2'},
        opacity=0.8
    )
    
    # Update layout
    fig_dinuc_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_dinuc_umap.update_traces(marker=dict(size=8))
    
    # Save dinucleotide-colored UMAP interactive HTML
    dinuc_umap_html_path = os.path.join(output_dir, "utr3_umap_dinucleotide_colored.html")
    fig_dinuc_umap.write_html(dinuc_umap_html_path)
    print(f"Dinucleotide-colored UMAP visualization saved to: {dinuc_umap_html_path}")
    
    # Create dinucleotide-colored t-SNE plot
    print("Creating dinucleotide-colored t-SNE visualization...")
    fig_dinuc_tsne = px.scatter(
        plot_df, 
        x='TSNE_1', 
        y='TSNE_2',
        color='Top_Dinucleotide',
        hover_data=['Sample', 'Top_Kmer', 'Top_Dinucleotide', 'Max_PEKA_Score', 'Top_5_Kmers'],
        title=f't-SNE Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)<br>({len(data.columns)} samples, {len(data)} kmers)',
        labels={'TSNE_1': 't-SNE 1', 'TSNE_2': 't-SNE 2'},
        opacity=0.8
    )
    
    # Update layout
    fig_dinuc_tsne.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(size=14),
        title_font_size=18,
        xaxis=dict(title_font_size=16, tickfont_size=14),
        yaxis=dict(title_font_size=16, tickfont_size=14)
    )
    
    # Update traces for better point visibility
    fig_dinuc_tsne.update_traces(marker=dict(size=8))
    
    # Save dinucleotide-colored t-SNE interactive HTML
    dinuc_tsne_html_path = os.path.join(output_dir, "utr3_tsne_dinucleotide_colored.html")
    fig_dinuc_tsne.write_html(dinuc_tsne_html_path)
    print(f"Dinucleotide-colored t-SNE visualization saved to: {dinuc_tsne_html_path}")
    
    # Create static matplotlib versions
    # UMAP static plot
    plt.figure(figsize=(12, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], alpha=0.7, s=50)
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Visualization of UTR3 Samples (Z-score normalized)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save UMAP static versions
    umap_png_path = os.path.join(output_dir, "utr3_umap_visualization.png")
    plt.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Static UMAP visualization saved to: {umap_png_path}")
    
    umap_pdf_path = os.path.join(output_dir, "utr3_umap_visualization.pdf")
    plt.savefig(umap_pdf_path, bbox_inches='tight')
    print(f"Static UMAP visualization (PDF) saved to: {umap_pdf_path}")
    
    # t-SNE static plot
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], alpha=0.7, s=50)
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Visualization of UTR3 Samples (Z-score normalized)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save t-SNE static versions
    tsne_png_path = os.path.join(output_dir, "utr3_tsne_visualization.png")
    plt.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Static t-SNE visualization saved to: {tsne_png_path}")
    
    tsne_pdf_path = os.path.join(output_dir, "utr3_tsne_visualization.pdf")
    plt.savefig(tsne_pdf_path, bbox_inches='tight')
    print(f"Static t-SNE visualization (PDF) saved to: {tsne_pdf_path}")
    
    # Static subcellular localization-colored plots
    plt.figure(figsize=(14, 8))
    for localization in sorted(localization_categories):
        mask = plot_df['Subcellular_Localization'] == localization
        plt.scatter(plot_df.loc[mask, 'UMAP_1'], plot_df.loc[mask, 'UMAP_2'],
                    label=localization, alpha=0.7, s=50, color=localization_color_map[localization])
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Colored by Subcellular Localization\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loc_umap_png_path = os.path.join(output_dir, "utr3_umap_subcellular_colored.png")
    plt.savefig(loc_umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Subcellular localization-colored UMAP static visualization saved to: {loc_umap_png_path}")
    
    loc_umap_pdf_path = os.path.join(output_dir, "utr3_umap_subcellular_colored.pdf")
    plt.savefig(loc_umap_pdf_path, bbox_inches='tight')
    print(f"Subcellular localization-colored UMAP static visualization (PDF) saved to: {loc_umap_pdf_path}")
    
    plt.figure(figsize=(14, 8))
    for localization in sorted(localization_categories):
        mask = plot_df['Subcellular_Localization'] == localization
        plt.scatter(plot_df.loc[mask, 'TSNE_1'], plot_df.loc[mask, 'TSNE_2'],
                    label=localization, alpha=0.7, s=50, color=localization_color_map[localization])
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Colored by Subcellular Localization\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loc_tsne_png_path = os.path.join(output_dir, "utr3_tsne_subcellular_colored.png")
    plt.savefig(loc_tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Subcellular localization-colored t-SNE static visualization saved to: {loc_tsne_png_path}")
    
    loc_tsne_pdf_path = os.path.join(output_dir, "utr3_tsne_subcellular_colored.pdf")
    plt.savefig(loc_tsne_pdf_path, bbox_inches='tight')
    print(f"Subcellular localization-colored t-SNE static visualization (PDF) saved to: {loc_tsne_pdf_path}")
    
    for column_name, cfg in multivalency_columns:
        column_label = cfg.get("label", column_name)
        slug = cfg.get("slug", column_name.lower())
        create_category_colored_plots(
            plot_df,
            column_name,
            column_label,
            f"utr3_{slug}",
            output_dir,
            sample_count,
            kmer_count
        )
    
    # Create static dinucleotide-colored UMAP plot
    plt.figure(figsize=(14, 8))
    for dinucleotide in sorted(unique_dinucleotides):
        mask = plot_df['Top_Dinucleotide'] == dinucleotide
        plt.scatter(plot_df.loc[mask, 'UMAP_1'], plot_df.loc[mask, 'UMAP_2'], 
                   label=dinucleotide, alpha=0.7, s=50, color=color_map[dinucleotide])
    
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.title(f'UMAP Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save dinucleotide-colored UMAP static versions
    dinuc_umap_png_path = os.path.join(output_dir, "utr3_umap_dinucleotide_colored.png")
    plt.savefig(dinuc_umap_png_path, dpi=300, bbox_inches='tight')
    print(f"Dinucleotide-colored UMAP static visualization saved to: {dinuc_umap_png_path}")
    
    dinuc_umap_pdf_path = os.path.join(output_dir, "utr3_umap_dinucleotide_colored.pdf")
    plt.savefig(dinuc_umap_pdf_path, bbox_inches='tight')
    print(f"Dinucleotide-colored UMAP static visualization (PDF) saved to: {dinuc_umap_pdf_path}")
    
    # Create static dinucleotide-colored t-SNE plot
    plt.figure(figsize=(14, 8))
    for dinucleotide in sorted(unique_dinucleotides):
        mask = plot_df['Top_Dinucleotide'] == dinucleotide
        plt.scatter(plot_df.loc[mask, 'TSNE_1'], plot_df.loc[mask, 'TSNE_2'], 
                   label=dinucleotide, alpha=0.7, s=50, color=color_map[dinucleotide])
    
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.title(f't-SNE Visualization Colored by Top Dinucleotide (from top 5 kmers, reverse combined)\n({len(data.columns)} samples, {len(data)} kmers)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save dinucleotide-colored t-SNE static versions
    dinuc_tsne_png_path = os.path.join(output_dir, "utr3_tsne_dinucleotide_colored.png")
    plt.savefig(dinuc_tsne_png_path, dpi=300, bbox_inches='tight')
    print(f"Dinucleotide-colored t-SNE static visualization saved to: {dinuc_tsne_png_path}")
    
    dinuc_tsne_pdf_path = os.path.join(output_dir, "utr3_tsne_dinucleotide_colored.pdf")
    plt.savefig(dinuc_tsne_pdf_path, bbox_inches='tight')
    print(f"Dinucleotide-colored t-SNE static visualization (PDF) saved to: {dinuc_tsne_pdf_path}")
    
    return umap_embedding, tsne_embedding


def save_combined_data(data: pd.DataFrame, output_dir: str):
    """Save the combined data to CSV."""
    output_path = os.path.join(output_dir, "combined_utr3_data.csv")
    data.to_csv(output_path)
    print(f"Combined data saved to: {output_path}")

def main():
    """Main function to run the genomic region UMAP analysis."""
    print("All Genomic Regions PEKA-score UMAP Analysis")
    print("=" * 50)
    
    try:
        # Load subcellular localization annotations
        annotations = load_subcellular_annotations(ANNOTATION_PATH)
        localization_summary = {}
        multivalency_summary: List[Dict[str, str]] = []
        multivalency_configs = load_multivalency_configs()
        
        # Find matching files for all genomic regions
        utr3_files, intron_files, other_exon_files, common_samples = find_matching_files(DATA_DIR)
        
        if not utr3_files or not intron_files or not other_exon_files:
            print(f"No matching files found for all three genomic regions in {DATA_DIR}")
            return
        
        # Combine data
        print("\nCombining all genomic region data...")
        combined_data = combine_all_genomic_data(utr3_files, intron_files, other_exon_files)
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save combined data
        save_combined_data(combined_data, OUTPUT_DIR)
        
        # Create UMAP visualization
        print("\nCreating UMAP visualization...")
        combined_samples = list(combined_data.columns)
        combined_proteins, combined_localizations = map_samples_to_localizations(combined_samples, annotations)
        update_localization_summary(localization_summary, combined_samples, combined_proteins, combined_localizations, "Combined Genomic Regions")
        update_multivalency_summary(multivalency_summary, combined_samples, combined_proteins, multivalency_configs, "Combined Genomic Regions")
        
        embedding = create_umap_visualization(
            combined_data,
            OUTPUT_DIR,
            annotations,
            protein_names=combined_proteins,
            localization_labels=combined_localizations,
            multivalency_configs=multivalency_configs
        )
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total kmers: {len(combined_data)}")
        print(f"Total samples: {len(combined_data.columns)}")
        print(f"Mean PEKA-score: {combined_data.values.mean():.4f}")
        print(f"Std PEKA-score: {combined_data.values.std():.4f}")
        
        # Count kmer types
        utr3_kmers = len([k for k in combined_data.index if k.startswith('3utr_')])
        intron_kmers = len([k for k in combined_data.index if k.startswith('intron_')])
        other_exon_kmers = len([k for k in combined_data.index if k.startswith('other_exon_')])
        print(f"3UTR kmers: {utr3_kmers}")
        print(f"Intron kmers: {intron_kmers}")
        print(f"Other exon kmers: {other_exon_kmers}")
        
        # Create genome-only analysis
        print("\n" + "="*60)
        print("CREATING GENOME-ONLY ANALYSIS")
        print("="*60)
        
        # Find genome files
        genome_files = find_genome_files(DATA_DIR)
        
        if genome_files:
            # Combine genome data
            print("\nCombining genome data...")
            genome_data = combine_genome_data(genome_files)
            
            # Save genome data
            genome_csv_path = os.path.join(OUTPUT_DIR, "combined_genome_data.csv")
            genome_data.to_csv(genome_csv_path)
            print(f"Genome data saved to: {genome_csv_path}")
            
            # Create genome visualization
            print("\nCreating genome-only UMAP and t-SNE visualizations...")
            genome_samples = list(genome_data.columns)
            genome_proteins, genome_localizations = map_samples_to_localizations(genome_samples, annotations)
            update_localization_summary(localization_summary, genome_samples, genome_proteins, genome_localizations, "Genome")
            update_multivalency_summary(multivalency_summary, genome_samples, genome_proteins, multivalency_configs, "Genome")
            
            create_genome_visualization(
                genome_data,
                OUTPUT_DIR,
                annotations,
                protein_names=genome_proteins,
                localization_labels=genome_localizations,
                multivalency_configs=multivalency_configs
            )
            
            print(f"\nGenome analysis complete! Results saved to: {OUTPUT_DIR}")
        else:
            print("No genome.tsv files found, skipping genome-only analysis")
        
        # Save summaries
        save_localization_summary(localization_summary, OUTPUT_DIR)
        save_multivalency_summary(multivalency_summary, OUTPUT_DIR)
        
        print(f"\nAll analyses complete! Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

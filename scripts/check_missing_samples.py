#!/usr/bin/env python3
"""
Script to identify samples that are missing UTR3, intron, or other_exon files.
"""

import os
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_DIR = "executions/data"
UTR3_PATTERN = "UTR3.tsv"
INTRON_PATTERN = "intron.tsv"
OTHER_EXON_PATTERN = "other_exon.tsv"

def find_sample_files(data_dir: str) -> dict:
    """Find all files for each genomic region and group by sample name."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Dictionary to store files by sample name
    sample_files = defaultdict(dict)
    
    # Find UTR3 files
    utr3_files = list(data_path.glob(f"*{UTR3_PATTERN}"))
    print(f"Found {len(utr3_files)} UTR3.tsv files")
    
    for file in utr3_files:
        sample_name = file.stem.replace(UTR3_PATTERN.replace('.tsv', ''), '').rstrip('_')
        sample_files[sample_name]['utr3'] = file
    
    # Find intron files
    intron_files = list(data_path.glob(f"*{INTRON_PATTERN}"))
    print(f"Found {len(intron_files)} intron.tsv files")
    
    for file in intron_files:
        sample_name = file.stem.replace(INTRON_PATTERN.replace('.tsv', ''), '').rstrip('_')
        sample_files[sample_name]['intron'] = file
    
    # Find other_exon files
    other_exon_files = list(data_path.glob(f"*{OTHER_EXON_PATTERN}"))
    print(f"Found {len(other_exon_files)} other_exon.tsv files")
    
    for file in other_exon_files:
        sample_name = file.stem.replace(OTHER_EXON_PATTERN.replace('.tsv', ''), '').rstrip('_')
        sample_files[sample_name]['other_exon'] = file
    
    return dict(sample_files)

def analyze_missing_files(sample_files: dict) -> None:
    """Analyze which samples are missing which file types."""
    required_types = ['utr3', 'intron', 'other_exon']
    
    # Count samples with different combinations
    complete_samples = []
    missing_samples = defaultdict(list)
    
    for sample_name, files in sample_files.items():
        missing_types = [file_type for file_type in required_types if file_type not in files]
        
        if not missing_types:
            complete_samples.append(sample_name)
        else:
            for missing_type in missing_types:
                missing_samples[missing_type].append(sample_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SAMPLE FILE AVAILABILITY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nComplete samples (have all three file types): {len(complete_samples)}")
    if complete_samples:
        print("Complete samples:")
        for sample in sorted(complete_samples):
            print(f"  - {sample}")
    
    print(f"\nSamples missing files: {len(set().union(*missing_samples.values()))}")
    
    # Print missing by file type
    for file_type in required_types:
        missing_count = len(missing_samples[file_type])
        print(f"\nSamples missing {file_type.upper()} files: {missing_count}")
        if missing_samples[file_type]:
            print(f"Missing {file_type}:")
            for sample in sorted(missing_samples[file_type]):
                print(f"  - {sample}")
    
    # Print samples missing multiple file types
    multi_missing = []
    for sample_name, files in sample_files.items():
        missing_types = [file_type for file_type in required_types if file_type not in files]
        if len(missing_types) > 1:
            multi_missing.append((sample_name, missing_types))
    
    if multi_missing:
        print(f"\nSamples missing multiple file types: {len(multi_missing)}")
        for sample_name, missing_types in multi_missing:
            print(f"  - {sample_name}: missing {', '.join(missing_types)}")
    
    # Print samples with only one file type
    single_type = []
    for sample_name, files in sample_files.items():
        present_types = [file_type for file_type in required_types if file_type in files]
        if len(present_types) == 1:
            single_type.append((sample_name, present_types[0]))
    
    if single_type:
        print(f"\nSamples with only one file type: {len(single_type)}")
        for sample_name, file_type in single_type:
            print(f"  - {sample_name}: only has {file_type}")

def main():
    """Main function to analyze missing sample files."""
    print("Analyzing sample file availability...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Required file types: UTR3, intron, other_exon")
    
    try:
        # Find all sample files
        sample_files = find_sample_files(DATA_DIR)
        
        if not sample_files:
            print("No sample files found!")
            return
        
        print(f"\nTotal unique samples found: {len(sample_files)}")
        
        # Analyze missing files
        analyze_missing_files(sample_files)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

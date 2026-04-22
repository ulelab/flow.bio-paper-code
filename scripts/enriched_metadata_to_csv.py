#!/usr/bin/env python3
"""
Convert enriched Flow sample metadata JSON to CSV.

Expected input format is the enriched list produced by enrich_sample_metadata.py,
where each record includes sample_id, sample_name, and sample_metadata.

Output column order:
1. name
2. id
3. organism_name
4. sample_type
5. project_name
6. fileset_filenames
7+. metadata columns (one per metadata key)
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Set

DEFAULT_INPUT_JSON = "filtered_data.json"
DEFAULT_OUTPUT_CSV = "enriched_sample_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform enriched metadata JSON into a flat CSV"
    )
    parser.add_argument(
        "--input-json", "-i",
        default=DEFAULT_INPUT_JSON,
        help=f"Input enriched JSON file (default: {DEFAULT_INPUT_JSON})"
    )
    parser.add_argument(
        "--output-csv", "-o",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})"
    )
    parser.add_argument(
        "--exclude-metadata-columns",
        default="",
        help="Comma-separated metadata keys to exclude (e.g. comments,geo)"
    )
    return parser.parse_args()


def flatten_value(value: Any) -> str:
    """Convert nested values into a readable string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value and value.get("value") is not None:
            return flatten_value(value.get("value"))
        if "name" in value and value.get("name") is not None:
            return flatten_value(value.get("name"))
        if "identifier" in value and value.get("identifier") is not None:
            return flatten_value(value.get("identifier"))
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = [flatten_value(v) for v in value]
        return "; ".join([p for p in parts if p])
    return str(value)


def add_unique_value(existing: str, new_value: str) -> str:
    """Merge semicolon-separated values while preserving order and uniqueness."""
    if not new_value:
        return existing
    if not existing:
        return new_value

    seen: Set[str] = set()
    merged: List[str] = []

    for part in existing.split("; "):
        if part and part not in seen:
            seen.add(part)
            merged.append(part)

    for part in new_value.split("; "):
        if part and part not in seen:
            seen.add(part)
            merged.append(part)

    return "; ".join(merged)


def get_project_name(sample_metadata: Dict[str, Any]) -> str:
    project = sample_metadata.get("project")
    if isinstance(project, dict):
        return str(project.get("name") or "")
    return flatten_value(project)


def get_organism_name(sample_metadata: Dict[str, Any]) -> str:
    organism = sample_metadata.get("organism")
    if isinstance(organism, dict):
        return str(organism.get("name") or organism.get("identifier") or "")
    return flatten_value(organism)


def get_fileset_filenames(sample_metadata: Dict[str, Any]) -> str:
    filenames: List[str] = []
    seen: Set[str] = set()

    for fileset in sample_metadata.get("filesets", []) or []:
        if not isinstance(fileset, dict):
            continue
        for data_item in fileset.get("data", []) or []:
            if not isinstance(data_item, dict):
                continue
            name = str(data_item.get("filename") or data_item.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                filenames.append(name)

    return "; ".join(filenames)


def parse_excluded_columns(raw: str) -> Set[str]:
    excluded = set()
    for item in raw.split(","):
        key = item.strip()
        if key:
            excluded.add(key)
    return excluded


def main() -> None:
    args = parse_args()
    excluded_meta_keys = parse_excluded_columns(args.exclude_metadata_columns)

    if not os.path.exists(args.input_json):
        print(f"Input file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Expected a list in {args.input_json}, got {type(data)}", file=sys.stderr)
        sys.exit(1)

    rows_by_sample_id: Dict[str, Dict[str, str]] = {}
    metadata_keys: Set[str] = set()

    for record in data:
        if not isinstance(record, dict):
            continue

        sample_metadata = record.get("sample_metadata") or {}
        if not isinstance(sample_metadata, dict):
            sample_metadata = {}

        sample_id = str(sample_metadata.get("id") or record.get("sample_id") or "").strip()
        if not sample_id:
            continue

        if sample_id not in rows_by_sample_id:
            rows_by_sample_id[sample_id] = {
                "name": str(sample_metadata.get("name") or record.get("sample_name") or ""),
                "id": sample_id,
                "organism_name": get_organism_name(sample_metadata),
                "sample_type": str(sample_metadata.get("sample_type") or ""),
                "project_name": get_project_name(sample_metadata),
                "fileset_filenames": get_fileset_filenames(sample_metadata),
            }
        else:
            row = rows_by_sample_id[sample_id]
            if not row.get("name"):
                row["name"] = str(sample_metadata.get("name") or record.get("sample_name") or "")
            if not row.get("organism_name"):
                row["organism_name"] = get_organism_name(sample_metadata)
            if not row.get("sample_type"):
                row["sample_type"] = str(sample_metadata.get("sample_type") or "")
            if not row.get("project_name"):
                row["project_name"] = get_project_name(sample_metadata)
            row["fileset_filenames"] = add_unique_value(
                row.get("fileset_filenames", ""),
                get_fileset_filenames(sample_metadata),
            )

        metadata = sample_metadata.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        row = rows_by_sample_id[sample_id]
        for meta_key, meta_value in metadata.items():
            if meta_key in excluded_meta_keys:
                continue
            metadata_keys.add(meta_key)
            row[meta_key] = add_unique_value(row.get(meta_key, ""), flatten_value(meta_value))

    fixed_columns = [
        "name",
        "id",
        "organism_name",
        "sample_type",
        "project_name",
        "fileset_filenames",
    ]
    metadata_columns = sorted(metadata_keys)
    fieldnames = fixed_columns + metadata_columns

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_by_sample_id.values():
            writer.writerow(row)

    print(f"Wrote {len(rows_by_sample_id)} samples to {args.output_csv}")
    print(f"Metadata columns included: {len(metadata_columns)}")
    if excluded_meta_keys:
        print(f"Metadata columns excluded: {', '.join(sorted(excluded_meta_keys))}")


if __name__ == "__main__":
    main()

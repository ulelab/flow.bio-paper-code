import os
import sys
import time
import random
import json
import csv
from typing import Dict, List, Any, Iterable, Tuple
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


BASE_URL = "https://app.flow.bio/api"

# Direct filename search - no regex needed since we can search server-side
FILENAME_SEARCH = "genome_5mer_distribution_genome.tsv"

# Set your output JSON file path here
OUTPUT_JSON = "filtered_data.json"
INPUT_JSON = ""  # Optional: path to an existing filtered_data.json to skip API fetch
SAMPLE_METADATA_TSV = "sample_metadata.tsv"
DATA_METADATA_DEBUG_LIMIT = 5  # Set to 0 to disable debug print
DATA_METADATA_DEBUG_PATH = "data_metadata_debug.json"

# Concurrency and paging settings
MAX_WORKERS = 1
PER_PAGE_COUNT = 100
MAX_DOWNLOAD_WORKERS = 1
DATA_DIR = "executions/data"
MAX_DOWNLOAD_RETRIES = 5
INITIAL_BACKOFF_SEC = 1.0
REQUEST_DELAY_SEC = 0.25

# Data search filters
UPSTREAM_PROCESS = "CLIPSEQ:PEKA"

# Path to credentials file (JSON with keys: username, password)
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")


def get_access_token(username: str, password: str) -> str:
    login_resp = requests.post(
        f"{BASE_URL}/login",
        headers={"Content-Type": "application/json"},
        json={"username": username, "password": password},
        timeout=30,
    )
    login_resp.raise_for_status()
    refresh_token = login_resp.json().get("token")
    if not refresh_token:
        raise RuntimeError("No refresh token in login response")

    token_resp = requests.get(
        f"{BASE_URL}/token",
        headers={"Content-Type": "application/json"},
        cookies={"flow_refresh_token": refresh_token},
        timeout=30,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("token")
    if not access_token:
        raise RuntimeError("No access token in token response")
    return access_token


def paginate_items(session: requests.Session, url: str, base_params: Dict[str, Any] | None = None) -> Iterable[Dict[str, Any]]:
    params = dict(base_params or {})
    page = 1
    per_page = int(params.pop("count", PER_PAGE_COUNT))
    while True:
        params_with_page = {**params, "page": page, "count": per_page}
        resp = session.get(url, params=params_with_page, timeout=60)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            status = resp.status_code if resp is not None else None
            if status is not None and 500 <= status < 600:
                print(f"Server error {status} for {resp.url}; skipping this resource")
                return
            raise
        data = resp.json()
        items = (
            data.get("items")
            or data.get("results")
            or data.get("data")
            or data.get("samples")
            or []
        )
        # Quiet: avoid per-page logging for performance/readability
        if not items:
            break
        for item in items:
            yield item
        page += 1
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)


def get_all_data_with_filters(session: requests.Session) -> List[Dict[str, Any]]:
    """Search data API endpoint directly with filters"""
    params = {
        "filename": FILENAME_SEARCH,
        "process_execution_name": UPSTREAM_PROCESS
    }
    return list(paginate_items(session, f"{BASE_URL}/data/search", params))




def _to_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try ISO8601 parsing
        try:
            from datetime import datetime
            if value.endswith("Z"):
                value = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except Exception:
            # Fallback: not parseable
            return 0.0
    return 0.0


def dedupe_latest_by_filename(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_name: Dict[str, Dict[str, Any]] = {}
    best_ts_by_name: Dict[str, float] = {}
    for it in items:
        fname = it.get("filename") or it.get("name")
        if not fname:
            continue
        created_raw = it.get("created") or it.get("created_at") or it.get("timestamp")
        ts = _to_timestamp(created_raw)
        prev_ts = best_ts_by_name.get(fname, -1.0)
        if ts >= prev_ts:
            best_ts_by_name[fname] = ts
            best_by_name[fname] = it
    return list(best_by_name.values())


def process_data_items(data_items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Process data items with deduplication"""
    total_count = len(data_items)
    print(f"Processing {total_count} data items from API")
    
    # Deduplicate by filename, keeping most recent
    unique_names = set((it.get("filename") or it.get("name") or "") for it in data_items)
    if len(unique_names) < len(data_items):
        kept_items = dedupe_latest_by_filename(data_items)
        print(f"After deduplication: {len(kept_items)} items")
    else:
        kept_items = data_items
    
    # Format results for consistency with original structure
    results = [
        {
            "data_id": str(item.get("id") or ""),
            "filename": item.get("filename") or item.get("name") or "",
            "file": item,
        }
        for item in kept_items
    ]
    
    msg = f"Processed {total_count} data items: kept={len(kept_items)}"
    return (msg, results)


def extract_sample_id_from_record(record: Dict[str, Any]) -> str:
    """Best effort extraction of sample ID from a data record."""
    file_obj = record.get("file") or record
    if not isinstance(file_obj, dict):
        return ""
    
    sample_id = (
        file_obj.get("sample_id")
        or file_obj.get("sampleId")
        or (file_obj.get("sample") or {}).get("id")
        or file_obj.get("sampleID")
    )
    
    if not sample_id and "sample" in file_obj and isinstance(file_obj["sample"], str):
        sample_id = file_obj["sample"]
    
    return str(sample_id or "").strip()


def fetch_sample_metadata(session: requests.Session, sample_id: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    """Fetch sample metadata from the API with caching."""
    if not sample_id:
        return None
    if sample_id in cache:
        return cache[sample_id]
    
    url = f"{BASE_URL}/samples/{quote(sample_id)}"
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        sample_data = resp.json()
        cache[sample_id] = sample_data
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)
        return sample_data
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            print(f"Sample {sample_id} not found (404); continuing without metadata")
            cache[sample_id] = None
            return None
        print(f"Error fetching sample {sample_id}: {e}")
        cache[sample_id] = None
        return None
    except Exception as e:
        print(f"Unexpected error fetching sample {sample_id}: {e}")
        return None


def fetch_data_metadata(session: requests.Session, data_id: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    """Fetch full data metadata from the API with caching."""
    if not data_id:
        return None
    if data_id in cache:
        return cache[data_id]
    
    url = f"{BASE_URL}/data/{quote(data_id)}"
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        cache[data_id] = data
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)
        return data
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            print(f"Data object {data_id} not found (404); continuing without metadata")
            cache[data_id] = None
            return None
        print(f"Error fetching data metadata for {data_id}: {e}")
        cache[data_id] = None
        return None
    except Exception as e:
        print(f"Unexpected error fetching data metadata for {data_id}: {e}")
        return None


def collect_sample_metadata_rows(session: requests.Session, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect sample metadata rows keyed by data filename.
    
    For each data record we:
      1. Fetch the full data metadata (/data/{id}) to obtain sample_id and other attributes.
      2. Fetch the sample metadata (/samples/{sample_id}) for richer information.
    """
    rows: List[Dict[str, Any]] = []
    data_cache: Dict[str, Dict[str, Any]] = {}
    sample_cache: Dict[str, Dict[str, Any]] = {}
    
    for record in data_records:
        file_obj = record.get("file") or {}
        filename = record.get("filename") or record.get("name") or file_obj.get("filename") or ""
        data_id = record.get("data_id") or file_obj.get("id") or ""
        
        data_metadata = fetch_data_metadata(session, data_id, data_cache) if data_id else None
        sample_id = ""
        sample_name_hint = ""
        sample_type_hint = ""
        
        if isinstance(data_metadata, dict):
            sample_id = (
                data_metadata.get("sample_id")
                or (data_metadata.get("sample") or {}).get("id")
                or data_metadata.get("sampleId")
            )
            sample_name_hint = str(data_metadata.get("sample_name") or "")
            sample_type_hint = str(data_metadata.get("sample_type") or "")
        else:
            sample_id = extract_sample_id_from_record(record)
            sample_name_hint = str(file_obj.get("sample_name") or "")
            sample_type_hint = str(file_obj.get("sample_type") or "")
        
        sample_id = str(sample_id or "").strip()
        sample_data = fetch_sample_metadata(session, sample_id, sample_cache) if sample_id else None
        
        row = {
            "data_filename": filename,
            "data_id": data_id,
            "sample_id": sample_id,
            "sample_name": "",
            "sample_type": "",
            "sample_subtype": "",
            "sample_created": "",
            "sample_updated": "",
            "sample_metadata_json": "",
            "data_metadata_json": "",
        }
        
        if data_metadata is not None:
            try:
                row["data_metadata_json"] = json.dumps(data_metadata, ensure_ascii=False)
            except TypeError:
                row["data_metadata_json"] = str(data_metadata)
        
        if isinstance(sample_data, dict):
            row["sample_name"] = str(sample_data.get("name") or sample_name_hint)
            row["sample_type"] = str(sample_data.get("sample_type") or sample_data.get("type") or sample_type_hint)
            row["sample_subtype"] = str(sample_data.get("sample_subtype") or "")
            row["sample_created"] = str(sample_data.get("created") or sample_data.get("created_at") or "")
            row["sample_updated"] = str(sample_data.get("updated") or sample_data.get("updated_at") or "")
            metadata_field = sample_data.get("metadata") or sample_data.get("sample_metadata")
            if metadata_field is not None:
                try:
                    row["sample_metadata_json"] = json.dumps(metadata_field, ensure_ascii=False)
                except TypeError:
                    row["sample_metadata_json"] = str(metadata_field)
        else:
            row["sample_name"] = sample_name_hint
            row["sample_type"] = sample_type_hint
        
        rows.append(row)
    
    return rows


def save_sample_metadata_tsv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Save sample metadata rows to TSV."""
    if not rows:
        print("No sample metadata rows to save.")
        return
    
    fieldnames = [
        "data_filename",
        "data_id",
        "sample_id",
        "sample_name",
        "sample_type",
        "sample_subtype",
        "sample_created",
        "sample_updated",
        "sample_metadata_json",
        "data_metadata_json",
    ]
    
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Sample metadata TSV saved to: {output_path}")
    except Exception as e:
        print(f"Failed to write sample metadata TSV to {output_path}: {e}")


def debug_print_data_metadata(
    records: List[Dict[str, Any]],
    limit: int = DATA_METADATA_DEBUG_LIMIT,
    output_path: str | None = DATA_METADATA_DEBUG_PATH,
) -> None:
    """Print and optionally save detailed metadata for a subset of data objects."""
    if not records:
        print("No data records available for metadata debug.")
        return
    if limit <= 0:
        return
    
    print("=" * 80)
    print(f"DATA OBJECT METADATA DEBUG (showing first {min(limit, len(records))} records)")
    print("=" * 80)
    debug_records = []
    for idx, record in enumerate(records):
        if idx >= limit:
            break
        filename = record.get("filename") or record.get("name") or "(unknown filename)"
        data_id = record.get("data_id") or (record.get("file") or {}).get("id")
        print(f"\nRecord #{idx+1}: filename='{filename}', data_id='{data_id}'")
        print("Top-level keys:", sorted(record.keys()))
        try:
            print("Full record JSON:")
            print(json.dumps(record, indent=2, ensure_ascii=False))
        except TypeError:
            print("Record contains non-serializable fields; showing str(record) instead.")
            print(str(record))
        
        file_obj = record.get("file")
        if file_obj:
            print("\n  Nested 'file' object keys:", sorted(file_obj.keys()))
            try:
                print("  'file' object JSON:")
                print(json.dumps(file_obj, indent=2, ensure_ascii=False))
            except TypeError:
                print("  'file' object not fully serializable; showing str(file_obj).")
                print(str(file_obj))
        else:
            print("No nested 'file' object present on this record.")
        
        debug_records.append(
            {
                "index": idx + 1,
                "filename": filename,
                "data_id": data_id,
                "record": record,
                "file_object": record.get("file"),
            }
        )
    print("=" * 80)
    
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(debug_records, f, ensure_ascii=False, indent=2)
            print(f"Data metadata debug details saved to: {output_path}")
        except Exception as e:
            print(f"Failed to write data metadata debug file '{output_path}': {e}")


def _safe_join_filename(directory: str, filename: str) -> str:
    base = os.path.basename(filename)
    return os.path.join(directory, base)


def download_file(session: requests.Session, record: Dict[str, Any]) -> Tuple[str, bool]:
    file_obj = record.get("file", {})
    data_id = str(file_obj.get("id") or "").strip()
    filename = str(file_obj.get("filename") or file_obj.get("name") or "").strip()
    if not data_id or not filename:
        return ("Missing id/filename; skipping", False)
    # Construct download URL
    url = f"https://app.flow.bio/files/downloads/{quote(data_id)}/{quote(filename)}"
    # Determine destination path
    dest_path = _safe_join_filename(DATA_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return (f"Exists: {dest_path}", True)
    backoff = INITIAL_BACKOFF_SEC
    attempts = 0
    while attempts < MAX_DOWNLOAD_RETRIES:
        attempts += 1
        try:
            with session.get(url, stream=True, timeout=300) as resp:
                if 200 <= resp.status_code < 300:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    # brief delay after a successful download
                    if REQUEST_DELAY_SEC:
                        time.sleep(REQUEST_DELAY_SEC)
                    return (f"Saved {dest_path}", True)

                status = resp.status_code
                # Retry on 429/5xx with backoff
                if status == 429 or 500 <= status < 600:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                            time.sleep(max(0.0, sleep_s))
                        except Exception:
                            pass
                    else:
                        time.sleep(backoff + random.uniform(0, backoff / 2))
                        backoff = min(backoff * 2, 30)
                    continue

                # Non-retryable HTTP status
                return (f"HTTP {status} for {url}; skipped", False)
        except requests.exceptions.RequestException as e:
            # Network-level error; retry with backoff
            time.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 30)
            last_err = str(e)
            continue
        except Exception as e:
            return (f"Error downloading {url}: {e}", False)

    # Exhausted retries
    msg = f"Failed after {MAX_DOWNLOAD_RETRIES} attempts: {url}"
    try:
        msg += f" (last error: {last_err})"
    except NameError:
        pass
    return (msg, False)


def _unique_dest_path(directory: str, filename: str) -> str:
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    candidate = os.path.join(directory, base)
    idx = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{name} ({idx}){ext}")
        idx += 1
    return candidate


def flatten_data_dir(root_dir: str) -> Tuple[int, int]:
    moved = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root itself for moving
        if os.path.abspath(dirpath) == os.path.abspath(root_dir):
            continue
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            dest = os.path.join(root_dir, os.path.basename(fname))
            if os.path.abspath(src) == os.path.abspath(dest):
                skipped += 1
                continue
            if os.path.exists(dest):
                dest = _unique_dest_path(root_dir, fname)
            try:
                os.replace(src, dest)
                moved += 1
            except Exception:
                skipped += 1
        # Attempt to clean up empty directories
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except Exception:
            pass
    return moved, skipped


def debug_data_search(session: requests.Session) -> None:
    """Debug function to show data records from direct data search"""
    print("=== DEBUG: Data search results ===")
    
    try:
        data_items = get_all_data_with_filters(session)
        print(f"Found {len(data_items)} data records")
        print()
        
        for i, item in enumerate(data_items[:10]):  # Show first 10 records
            print(f"Record {i+1}:")
            print(f"  Keys: {list(item.keys())}")
            print(f"  filename: {item.get('filename')}")
            print(f"  process_execution_name: {item.get('process_execution_name')}")
            print()
            
        if len(data_items) > 10:
            print(f"... and {len(data_items) - 10} more records")
            
    except Exception as e:
        print(f"Error fetching data: {e}")


def main() -> None:
    # Load credentials from file
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Credentials file not found: {CREDENTIALS_PATH}", file=sys.stderr)
        print('Create it with JSON: {"username": "...", "password": "..."}', file=sys.stderr)
        sys.exit(1)
    try:
        import json
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            creds = json.load(f)
        username = str(creds.get("username") or "").strip()
        password = str(creds.get("password") or "").strip()
    except Exception as e:
        print(f"Failed to read credentials from {CREDENTIALS_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    if not username or not password:
        print("Missing username/password in credentials file", file=sys.stderr)
        sys.exit(1)

    access_token = get_access_token(username, password)
    print("Authenticated and obtained access token")

    # Debug mode: show data search results
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        with requests.Session() as session:
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            debug_data_search(session)
        return

    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        all_data: List[Dict[str, Any]] = []
        if INPUT_JSON and os.path.exists(INPUT_JSON):
            try:
                import json
                with open(INPUT_JSON, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    all_data = loaded
                    print(f"Loaded {len(all_data)} records from {INPUT_JSON}; skipping API fetch")
                else:
                    print(f"Input JSON {INPUT_JSON} is not a list; ignoring and proceeding with API fetch")
            except Exception as e:
                print(f"Failed to read {INPUT_JSON}: {e}; proceeding with API fetch")
        
        if not all_data:
            print(f"Searching data API with filters: filename={FILENAME_SEARCH}, process_execution_name={UPSTREAM_PROCESS}")
            data_items = get_all_data_with_filters(session)
            print(f"Found {len(data_items)} data records from API")
            
            # Process data items with deduplication
            msg, all_data = process_data_items(data_items)
            print(msg)

        print(f"Final result: {len(all_data)} data files after deduplication")
        print(f"Filename search: {FILENAME_SEARCH}")
        
        # Debug: print metadata for a subset of data objects
        debug_print_data_metadata(all_data, DATA_METADATA_DEBUG_LIMIT)

        # Save final records to a JSON file
        output_path = OUTPUT_JSON
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(all_data)} records to {output_path}")
        except Exception as e:
            print(f"Failed to write output JSON to {output_path}: {e}")
        
        # Fetch and save sample metadata keyed by data filename
        try:
            metadata_rows = collect_sample_metadata_rows(session, all_data)
            if metadata_rows:
                save_sample_metadata_tsv(metadata_rows, SAMPLE_METADATA_TSV)
        except Exception as e:
            print(f"Failed to generate sample metadata TSV: {e}")
        
        # Download files concurrently
        print(f"Starting downloads to '{DATA_DIR}'...")
        os.makedirs(DATA_DIR, exist_ok=True)
        success = 0
        attempted = 0
        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            futures = [executor.submit(download_file, session, r) for r in all_data]
            for fut in as_completed(futures):
                msg, ok = fut.result()
                attempted += 1
                if ok:
                    success += 1
                # Keep download output minimal; show only failures or every N successes
                if not ok or (attempted % 25 == 0):
                    print(msg)
        print(f"Downloads complete: {success}/{attempted} succeeded")

        # Flatten any nested data directories (id/name structures) into DATA_DIR
        moved, skipped = flatten_data_dir(DATA_DIR)
        print(f"Flattened data dir: moved {moved} files, skipped {skipped}")

        # Final summary only (omit individual filenames for brevity)


if __name__ == "__main__":
    main()
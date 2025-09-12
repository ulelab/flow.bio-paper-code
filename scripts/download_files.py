import os
import sys
import re
import time
import random
from typing import Dict, List, Any, Iterable, Tuple
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


BASE_URL = "https://api.flow.bio"

# Set your filename regex here (comma-separated to support multiple patterns)
# Example: r"(gene_premapadjusted\.tsv$)|(subtype_premapadjusted\.tsv$)"
FILENAME_REGEX = r"(gene_premapadjusted\.tsv$)|(subtype_premapadjusted\.tsv$)"

# Set your output JSON file path here
OUTPUT_JSON = "filtered_data.json"
INPUT_JSON = "filtered_data.json"  # Optional: path to an existing filtered_data.json to skip API fetch

# Concurrency and paging settings
MAX_WORKERS = 1
PER_PAGE_COUNT = 100
MAX_DOWNLOAD_WORKERS = 1
DATA_DIR = "data"
MAX_DOWNLOAD_RETRIES = 5
INITIAL_BACKOFF_SEC = 1.0
REQUEST_DELAY_SEC = 0.25

# Desired sample type to process
SAMPLE_TYPE = "CLIP"

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


def get_all_public_samples(session: requests.Session, sample_type: str | None = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if sample_type:
        params["sample_type"] = sample_type
    return list(paginate_items(session, f"{BASE_URL}/samples/public", params))


def get_all_sample_data(session: requests.Session, sample_id: str) -> List[Dict[str, Any]]:
    return list(paginate_items(session, f"{BASE_URL}/samples/{sample_id}/data"))


def filter_by_suffix(items: List[Dict[str, Any]], suffixes: List[str]) -> List[Dict[str, Any]]:
    lowered = [s.lower() for s in suffixes]
    result: List[Dict[str, Any]] = []
    for item in items:
        name = (item.get("filename") or item.get("name") or "").lower()
        if any(name.endswith(suf) for suf in lowered):
            result.append(item)
    return result


def compile_filename_regexes(regex_env: str, suffixes: List[str]) -> List[re.Pattern[str]]:
    patterns: List[re.Pattern[str]] = []
    parts = [p.strip() for p in (regex_env or "").split(",") if p.strip()]
    if parts:
        for p in parts:
            patterns.append(re.compile(p, re.IGNORECASE))
    else:
        if suffixes:
            escaped = [re.escape(s) for s in suffixes]
            patterns.append(re.compile(r"(" + "|".join(escaped) + r")$", re.IGNORECASE))
    return patterns


def filter_by_regex(items: List[Dict[str, Any]], patterns: List[re.Pattern[str]]) -> List[Dict[str, Any]]:
    if not patterns:
        return items
    result: List[Dict[str, Any]] = []
    for item in items:
        name = (item.get("filename") or item.get("name") or "")
        for pat in patterns:
            if pat.search(name):
                result.append(item)
                break
    return result


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


def process_sample(session: requests.Session, compiled_patterns: List[re.Pattern[str]], sample: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    sample_id = str(sample.get("id") or sample.get("sample_id") or sample.get("uid") or sample.get("uuid") or "")
    sample_name = sample.get("name") or ""
    if not sample_id:
        return ("", [])
    try:
        data_items = get_all_sample_data(session, sample_id)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status is not None and 500 <= status < 600:
            return (f"Skipping sample {sample_id}: server error {status}", [])
        raise

    total_count = len(data_items)
    if total_count < 10:
        return (f"Skipping sample {sample_id}: only {total_count} data records (<10)", [])

    filtered_items = filter_by_regex(data_items, compiled_patterns)
    unique_names = set((it.get("filename") or it.get("name") or "") for it in filtered_items)
    if len(unique_names) < len(filtered_items):
        kept_items = dedupe_latest_by_filename(filtered_items)
    else:
        kept_items = filtered_items

    results = [
        {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "file": d,
        }
        for d in kept_items
    ]
    msg = f"Completed sample {sample_id} ({sample_name}): total={total_count}, matched={len(filtered_items)}, kept={len(kept_items)}"
    return (msg, results)


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

    suffix_env = os.environ.get("FLOWBIO_SUFFIXES", "")
    filter_suffixes = [s.strip() for s in suffix_env.split(",") if s.strip()] or [
        "gene_premapadjusted.tsv",
        "subtype_premapadjusted.tsv"
    ]
    # Prefer in-script regex setting; if blank, fallback to suffixes
    regex_env = FILENAME_REGEX
    compiled_patterns = compile_filename_regexes(regex_env, filter_suffixes)

    access_token = get_access_token(username, password)
    print("Authenticated and obtained access token")

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
        samples = None
        if not all_data:
            print("Fetching public samples...")
            samples = get_all_public_samples(session, sample_type=SAMPLE_TYPE)
            # Client-side filter by sample_type/type to be safe
            desired = SAMPLE_TYPE.upper()
            samples = [s for s in samples if (str(s.get("sample_type") or s.get("type") or "").upper() == desired)]
            print(f"Found {len(samples)} public samples of type {SAMPLE_TYPE}")

            # Parallelize per-sample processing
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_sample, session, compiled_patterns, s) for s in samples]
                for fut in as_completed(futures):
                    msg, results = fut.result()
                    if msg:
                        print(msg)
                    if results:
                        all_data.extend(results)

        # Compute a safe sample count for summary
        if samples is None:
            try:
                sample_ids = {str(r.get("sample_id") or "") for r in all_data}
                samples_count = len([sid for sid in sample_ids if sid])
            except Exception:
                samples_count = 0
        else:
            samples_count = len(samples)
        print(f"Fetched {len(all_data)} total data files across {samples_count} samples after filtering")
        if regex_env.strip():
            print(f"Filename regex(es): {regex_env}")
        else:
            print(f"Filename suffixes: {', '.join(filter_suffixes)}")

        # Save final records to a JSON file
        output_path = OUTPUT_JSON
        try:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(all_data)} records to {output_path}")
        except Exception as e:
            print(f"Failed to write output JSON to {output_path}: {e}")
        
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
"""
Flow.bio API client utilities.

Provides authentication, sample fetching, and file download functionality
for the Flow.bio platform.
"""

import json
import os
import re
import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Iterable, Tuple
from urllib.parse import quote

import requests


DEFAULT_BASE_URL = "https://api.flow.bio"
DEFAULT_PER_PAGE = 100
DEFAULT_REQUEST_DELAY = 0.1


def load_credentials(credentials_path: str) -> Tuple[str, str]:
    """
    Load username and password from a JSON credentials file.
    
    The file should contain: {"username": "...", "password": "..."}
    Exits with an error message if credentials cannot be loaded.
    """
    if not os.path.exists(credentials_path):
        print(f"Credentials file not found: {credentials_path}", file=sys.stderr)
        print('Create it with JSON: {"username": "...", "password": "..."}', file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            creds = json.load(f)
        username = str(creds.get("username") or "").strip()
        password = str(creds.get("password") or "").strip()
    except Exception as e:
        print(f"Failed to read credentials from {credentials_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not username or not password:
        print("Missing username/password in credentials file", file=sys.stderr)
        sys.exit(1)
    
    return username, password


def get_access_token(
    username: str,
    password: str,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Authenticate with Flow.bio and return an access token."""
    login_resp = requests.post(
        f"{base_url}/login",
        headers={"Content-Type": "application/json"},
        json={"username": username, "password": password},
        timeout=30,
    )
    login_resp.raise_for_status()
    refresh_token = login_resp.json().get("token")
    if not refresh_token:
        raise RuntimeError("No refresh token in login response")

    token_resp = requests.get(
        f"{base_url}/token",
        headers={"Content-Type": "application/json"},
        cookies={"flow_refresh_token": refresh_token},
        timeout=30,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("token")
    if not access_token:
        raise RuntimeError("No access token in token response")
    return access_token


def paginate_items(
    session: requests.Session,
    url: str,
    base_params: Dict[str, Any] | None = None,
    per_page: int = DEFAULT_PER_PAGE,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> Iterable[Dict[str, Any]]:
    """Iterate through paginated API results."""
    params = dict(base_params or {})
    page = 1
    per_page = int(params.pop("count", per_page))
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
        if not items:
            break
        for item in items:
            yield item
        page += 1
        if request_delay:
            time.sleep(request_delay)


def get_all_public_samples(
    session: requests.Session,
    sample_type: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, Any]]:
    """Fetch all public samples, optionally filtered by sample type."""
    params: Dict[str, Any] = {}
    if sample_type:
        params["sample_type"] = sample_type
    return list(paginate_items(session, f"{base_url}/samples/public", params))


def get_all_sample_data(
    session: requests.Session,
    sample_id: str,
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, Any]]:
    """Fetch all data records for a specific sample."""
    return list(paginate_items(session, f"{base_url}/samples/{sample_id}/data"))


def get_sample_type_str(sample: Dict[str, Any]) -> str:
    """
    Extract the sample type as an uppercase string.
    
    Handles both plain string types and dict types like:
    {'identifier': 'CLIP', 'name': 'CLIP'}
    """
    st = sample.get("sample_type") or sample.get("type") or ""
    if isinstance(st, dict):
        return str(st.get("identifier") or st.get("name") or "").upper()
    return str(st).upper()


def filter_samples_by_type(
    samples: List[Dict[str, Any]],
    sample_type: str,
) -> List[Dict[str, Any]]:
    """Filter samples to only include those matching the given type."""
    desired = sample_type.upper()
    return [s for s in samples if get_sample_type_str(s) == desired]


# --- Filename filtering utilities ---

def compile_filename_regexes(regex_str: str) -> List[re.Pattern[str]]:
    """Compile comma-separated regex patterns for filename matching."""
    patterns: List[re.Pattern[str]] = []
    parts = [p.strip() for p in (regex_str or "").split(",") if p.strip()]
    for p in parts:
        patterns.append(re.compile(p, re.IGNORECASE))
    return patterns


def filter_by_regex(
    items: List[Dict[str, Any]],
    patterns: List[re.Pattern[str]],
) -> List[Dict[str, Any]]:
    """Filter items to those whose filename matches any of the patterns."""
    if not patterns:
        return items
    result: List[Dict[str, Any]] = []
    for item in items:
        name = item.get("filename") or item.get("name") or ""
        for pat in patterns:
            if pat.search(name):
                result.append(item)
                break
    return result


def _to_timestamp(value: Any) -> float:
    """Convert various date representations to a Unix timestamp."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                value = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def dedupe_latest_by_filename(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the latest version of each file (by created timestamp)."""
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


# --- File download utilities ---

def download_file(
    session: requests.Session,
    record: Dict[str, Any],
    data_dir: str = "data",
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    include_sample_id: bool = True,
) -> Tuple[str, bool]:
    """
    Download a file from Flow.bio.
    
    Args:
        session: Authenticated requests session
        record: Dict containing 'file' and optionally 'sample_id'
        data_dir: Directory to save files
        max_retries: Number of retry attempts
        initial_backoff: Initial backoff time in seconds
        request_delay: Delay between requests
        include_sample_id: If True, prefix filename with sample_id for traceability
    
    Returns a tuple of (message, success).
    """
    file_obj = record.get("file", {})
    sample_id = str(record.get("sample_id") or "").strip()
    data_id = str(file_obj.get("id") or "").strip()
    original_filename = str(file_obj.get("filename") or file_obj.get("name") or "").strip()
    
    if not data_id or not original_filename:
        return ("Missing id/filename; skipping", False)
    
    # Prefix filename with sample_id for traceability
    if include_sample_id and sample_id:
        filename = f"{sample_id}_{os.path.basename(original_filename)}"
    else:
        filename = os.path.basename(original_filename)
    
    url = f"https://app.flow.bio/files/downloads/{quote(data_id)}/{quote(original_filename)}"
    dest_path = os.path.join(data_dir, filename)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return (f"Exists: {dest_path}", True)
    
    backoff = initial_backoff
    attempts = 0
    last_err = ""
    
    while attempts < max_retries:
        attempts += 1
        try:
            with session.get(url, stream=True, timeout=300) as resp:
                if 200 <= resp.status_code < 300:
                    os.makedirs(data_dir, exist_ok=True)
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    if request_delay:
                        time.sleep(request_delay)
                    return (f"Saved {dest_path}", True)

                status = resp.status_code
                if status == 429 or 500 <= status < 600:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(max(0.0, float(retry_after)))
                        except Exception:
                            pass
                    else:
                        time.sleep(backoff + random.uniform(0, backoff / 2))
                        backoff = min(backoff * 2, 30)
                    continue

                return (f"HTTP {status} for {url}; skipped", False)
        except requests.exceptions.RequestException as e:
            time.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 30)
            last_err = str(e)
            continue
        except Exception as e:
            return (f"Error downloading {url}: {e}", False)

    msg = f"Failed after {max_retries} attempts: {url}"
    if last_err:
        msg += f" (last error: {last_err})"
    return (msg, False)


# --- Directory utilities ---

def flatten_data_dir(root_dir: str) -> Tuple[int, int]:
    """
    Move all files from subdirectories into the root directory.
    
    Returns (moved_count, skipped_count).
    """
    moved = 0
    skipped = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
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
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except Exception:
            pass
    
    return moved, skipped


def _unique_dest_path(directory: str, filename: str) -> str:
    """Generate a unique destination path by appending (1), (2), etc."""
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    candidate = os.path.join(directory, base)
    idx = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{name} ({idx}){ext}")
        idx += 1
    return candidate

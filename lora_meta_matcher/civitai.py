import json
import os
import random
import threading
import time
import datetime

import requests

from .db import get_loras_without_triggers_but_have_hash, upsert_lora

CIVITAI_API_URL = "https://civitai.com/api/v1/model-versions/by-hash/"
CIVITAI_API_VERSION_URL = "https://civitai.com/api/v1/model-versions/"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRY_ATTEMPTS = 4
MIN_REQUEST_DELAY_SECONDS = 3.0
MAX_RETRY_BACKOFF_SECONDS = 90.0
JITTER_SECONDS = 0.35
RETRYABLE_STATUS_CODES = {408, 425, 500, 502, 503, 504}

_REQUEST_SESSION = requests.Session()
_RATE_LIMIT_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0
_COOLDOWN_UNTIL = 0.0

API_LOG_HISTORY = []
MAX_API_LOGS = 100

def _log_api_tx(url, result_status, reason=""):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"[{timestamp}] {result_status} - {url}"
    if reason:
        msg += f" ({reason})"
    API_LOG_HISTORY.append(msg)
    if len(API_LOG_HISTORY) > MAX_API_LOGS:
        API_LOG_HISTORY.pop(0)

def get_api_logs():
    return "\n".join(API_LOG_HISTORY) if API_LOG_HISTORY else "No API transactions recorded."


def _normalize_hash(value):
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


def _hash_signature_from_values(autov2_hash=None, autov3_hash=None, sha256_hash=None):
    parts = [
        _normalize_hash(autov2_hash),
        _normalize_hash(autov3_hash),
        _normalize_hash(sha256_hash),
    ]
    return "|".join(p if p else "" for p in parts)


def _parse_retry_after_seconds(value):
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _sleep_interruptible(seconds, halt_check):
    if seconds <= 0:
        return False
    # Keep sleeps interruptible for UI halt responsiveness.
    deadline = time.monotonic() + seconds
    while True:
        if halt_check and halt_check():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.5, remaining))


def _reserve_request_slot(min_delay, halt_check=None):
    global _NEXT_REQUEST_AT
    global _COOLDOWN_UNTIL

    if min_delay is not None:
        effective_delay = float(min_delay)
    else:
        effective_delay = MIN_REQUEST_DELAY_SECONDS

    while True:
        if halt_check and halt_check():
            return True
        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            wait_seconds = max(_COOLDOWN_UNTIL, _NEXT_REQUEST_AT) - now
            if wait_seconds <= 0:
                _NEXT_REQUEST_AT = now + effective_delay
                return False
        if _sleep_interruptible(wait_seconds, halt_check):
            return True


def _set_global_cooldown(seconds):
    global _COOLDOWN_UNTIL
    if seconds <= 0:
        return
    with _RATE_LIMIT_LOCK:
        candidate = time.monotonic() + seconds
        if candidate > _COOLDOWN_UNTIL:
            _COOLDOWN_UNTIL = candidate


def _build_hash_candidates(autov2_hash=None, autov3_hash=None, sha256_hash=None):
    """
    Build normalized hash candidates in order of precision:
    SHA256 -> AutoV3 -> AutoV2(12 chars). De-duplicates values.
    """
    candidates = []
    seen = set()

    def add_hash(value):
        if value and value not in seen:
            candidates.append(value)
            seen.add(value)

    normalized_sha256 = _normalize_hash(sha256_hash)
    if normalized_sha256:
        add_hash(normalized_sha256)

    normalized_autov3 = _normalize_hash(autov3_hash)
    if normalized_autov3:
        add_hash(normalized_autov3)

    normalized_autov2 = _normalize_hash(autov2_hash)
    if normalized_autov2:
        add_hash(normalized_autov2)
        if len(normalized_autov2) >= 12:
            add_hash(normalized_autov2[:12])

    # If AutoV2 was missing, derive short candidate from SHA256.
    if not normalized_autov2 and normalized_sha256 and len(normalized_sha256) >= 12:
        add_hash(normalized_sha256[:12])

    return candidates


def _safe_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_hashes_from_files(data):
    api_autov2 = None
    api_autov3 = None
    api_sha256 = None
    files = data.get("files")
    if not isinstance(files, list):
        return api_autov2, api_autov3, api_sha256

    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        hashes = file_info.get("hashes")
        if not isinstance(hashes, dict):
            continue
        if not api_autov2 and hashes.get("AutoV2"):
            api_autov2 = _normalize_hash(hashes.get("AutoV2"))
        if not api_autov3 and hashes.get("AutoV3"):
            api_autov3 = _normalize_hash(hashes.get("AutoV3"))
        if not api_sha256 and hashes.get("SHA256"):
            api_sha256 = _normalize_hash(hashes.get("SHA256"))
        if api_autov2 or api_autov3 or api_sha256:
            break

    return api_autov2, api_autov3, api_sha256


def _extract_api_metadata(data):
    """
    Parse and validate useful metadata from CivitAI API payload.
    Returns None for malformed/empty payloads.
    """
    if not isinstance(data, dict):
        return None

    api_autov2, api_autov3, api_sha256 = _extract_hashes_from_files(data)

    trigger_words = None
    raw_words = data.get("trainedWords")
    if isinstance(raw_words, list):
        cleaned = [str(w).strip() for w in raw_words if isinstance(w, str) and w.strip()]
        if cleaned:
            trigger_words = ", ".join(cleaned)

    base_model = data.get("baseModel") if isinstance(data.get("baseModel"), str) else None
    civitai_version_id = _safe_int(data.get("id"))

    loraname = None
    model = data.get("model")
    if isinstance(model, dict) and isinstance(model.get("name"), str) and model.get("name").strip():
        m_name = model["name"].strip()
        v_name = data.get("name") if isinstance(data.get("name"), str) and data.get("name").strip() else None
        loraname = f"{m_name} ({v_name})" if v_name else m_name
    elif isinstance(data.get("name"), str) and data.get("name").strip():
        loraname = data.get("name").strip()

    has_useful = any(
        [
            trigger_words,
            base_model,
            civitai_version_id,
            loraname,
            api_autov2,
            api_autov3,
            api_sha256,
        ]
    )
    if not has_useful:
        return None

    return {
        "autov2_hash": api_autov2,
        "autov3_hash": api_autov3,
        "sha256_hash": api_sha256,
        "trigger_words": trigger_words,
        "base_model": base_model,
        "civitai_version_id": civitai_version_id,
        "loraname": loraname,
    }


def _atomic_write_json(path, payload):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        return False, f"parent directory does not exist: {parent}"
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2)
        os.replace(tmp_path, path)
        return True, None
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False, str(exc)


def _safe_update_civitai_info(filepath, api_data, parsed_metadata):
    """
    Update sidecar metadata conservatively:
    - Keep existing metadata values when already present.
    - Fill missing values from API response and parsed fields.
    """
    if not parsed_metadata:
        return False

    orig_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    info_path = os.path.join(orig_dir, f"{base_name}.civitai.info")

    if not os.path.exists(orig_dir):
        return False, f"directory not found: {orig_dir}"

    try:
        existing = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing = loaded
            except Exception:
                existing = {}

        data = dict(existing)

        hashes = {}
        if parsed_metadata.get("autov2_hash"):
            hashes["AutoV2"] = parsed_metadata["autov2_hash"]
            data.setdefault("autov2", parsed_metadata["autov2_hash"])
        if parsed_metadata.get("autov3_hash"):
            hashes["AutoV3"] = parsed_metadata["autov3_hash"]
            data.setdefault("autov3", parsed_metadata["autov3_hash"])
        if parsed_metadata.get("sha256_hash"):
            hashes["SHA256"] = parsed_metadata["sha256_hash"]
            data.setdefault("sha256", parsed_metadata["sha256_hash"])

        if hashes:
            files = data.get("files")
            if not isinstance(files, list) or not files:
                data["files"] = [{"hashes": hashes}]
            else:
                if not isinstance(files[0], dict):
                    files[0] = {}
                first_hashes = files[0].get("hashes")
                if not isinstance(first_hashes, dict):
                    first_hashes = {}
                    files[0]["hashes"] = first_hashes
                for key, value in hashes.items():
                    first_hashes.setdefault(key, value)

        if parsed_metadata.get("base_model"):
            data.setdefault("baseModel", parsed_metadata["base_model"])
        if parsed_metadata.get("civitai_version_id"):
            data.setdefault("id", parsed_metadata["civitai_version_id"])
        if parsed_metadata.get("trigger_words"):
            words = [w.strip() for w in parsed_metadata["trigger_words"].split(",") if w.strip()]
            if words:
                data.setdefault("trainedWords", words)
        if parsed_metadata.get("loraname"):
            data.setdefault("name", parsed_metadata["loraname"])

        # Keep API model block if present and sidecar is missing it.
        if isinstance(api_data, dict) and isinstance(api_data.get("model"), dict):
            data.setdefault("model", api_data["model"])

        ok, write_error = _atomic_write_json(info_path, data)
        if ok:
            return True, None
        return False, f"{info_path}: {write_error}"
    except Exception as exc:
        return False, f"{info_path}: {exc}"


def _request_json(url, token=None, delay=MIN_REQUEST_DELAY_SECONDS, halt_check=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_status = 0
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        if halt_check and halt_check():
            _log_api_tx(url, "HALTED", "user requested halt")
            return None, 0, "halted"
        if _reserve_request_slot(delay, halt_check=halt_check):
            _log_api_tx(url, "HALTED", "user requested halt during delay")
            return None, 0, "halted"

        try:
            _log_api_tx(url, "REQ", f"att {attempt}")
            response = _REQUEST_SESSION.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            last_status = response.status_code

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    _log_api_tx(url, last_status, "invalid_json")
                    return None, 200, "invalid_json"
                _log_api_tx(url, last_status, "ok")
                return payload, 200, "ok"

            if response.status_code == 429:
                retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                cooldown = retry_after if retry_after is not None else min(60, 10 * attempt)
                _set_global_cooldown(cooldown)
                _log_api_tx(url, last_status, f"cooldown_{cooldown}s")
                return None, 429, f"rate_limited_cooldown_{cooldown}s"

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRY_ATTEMPTS:
                backoff = min(MAX_RETRY_BACKOFF_SECONDS, (2 ** (attempt - 1)) + random.uniform(0.0, JITTER_SECONDS))
                _log_api_tx(url, last_status, f"retry_{backoff:.1f}s")
                if _sleep_interruptible(backoff, halt_check):
                    return None, 0, "halted"
                continue
            
            _log_api_tx(url, last_status, "http_error")
            return None, response.status_code, "http_error"
        except requests.RequestException as e:
            if attempt >= MAX_RETRY_ATTEMPTS:
                _log_api_tx(url, "ERR", f"exception maxed: {str(e)[:20]}")
                break
            backoff = min(MAX_RETRY_BACKOFF_SECONDS, (2 ** (attempt - 1)) + random.uniform(0.0, JITTER_SECONDS))
            _log_api_tx(url, "ERR", f"retry_{backoff:.1f}s: {str(e)[:20]}")
            if _sleep_interruptible(backoff, halt_check):
                return None, 0, "halted"

    _log_api_tx(url, "FAIL", "retry_exhausted")
    return None, last_status, "retry_exhausted"


def fetch_civitai_version_info(version_id, token=None, delay=None):
    """
    Fetch model version info by CivitAI version id with shared throttling/retry logic.
    """
    url = f"{CIVITAI_API_VERSION_URL}{version_id}"
    req_delay = delay if delay is not None else MIN_REQUEST_DELAY_SECONDS
    data, status, reason = _request_json(url, token=token, delay=req_delay)
    if status == 429:
        print(f"Rate limited by CivitAI while fetching version {version_id}. Reason: {reason}")
    elif status not in (0, 200, 404):
        print(f"Failed to fetch version {version_id}. Status: {status} Reason: {reason}")
    return data, status


def fetch_civitai_info(hash_value, token=None, delay=None):
    """
    Fetch model version info by hash with shared throttling/retry logic.
    """
    url = f"{CIVITAI_API_URL}{hash_value}"
    req_delay = delay if delay is not None else MIN_REQUEST_DELAY_SECONDS
    data, status, reason = _request_json(url, token=token, delay=req_delay)
    if status == 429:
        print(f"Rate limited by CivitAI while fetching hash {hash_value}. Reason: {reason}")
    elif status not in (0, 200, 404):
        print(f"Failed to fetch hash {hash_value}. Status: {status} Reason: {reason}")
    return data, status


def search_civitai_model_by_name(model_name, token=None, delay=None):
    """
    Search for models on CivitAI by query/name.
    Returns (list_of_model_data, status)
    """
    import urllib.parse
    encoded_name = urllib.parse.quote(model_name)
    url = f"https://civitai.com/api/v1/models?query={encoded_name}&types=LORA&nsfw=true&limit=5"
    req_delay = delay if delay is not None else MIN_REQUEST_DELAY_SECONDS
    data, status, reason = _request_json(url, token=token, delay=req_delay)
    if status == 429:
        print(f"Rate limited by CivitAI while searching model '{model_name}'. Reason: {reason}")
    elif status not in (0, 200, 404):
        print(f"Failed to search model '{model_name}'. Status: {status} Reason: {reason}")
        
    if status == 200 and data and isinstance(data.get("items"), list):
        return data["items"], 200
    return None, status



def process_missing_civitai_metadata(token=None, delay=MIN_REQUEST_DELAY_SECONDS, halt_check=None):
    """
    Finds loras in the DB with a hash but no trigger_words and fetches from CivitAI.
    Requests are serialized and throttled to reduce API-ban risk.
    Yields (summary_text, log_text) tuples.
    """
    loras = get_loras_without_triggers_but_have_hash()
    total_files = len(loras)

    if not total_files:
        yield "All hashed Loras already have metadata.", "Check complete: No missing metadata found."
        return

    yield (
        f"Preparing fetching for {total_files} Loras...",
        f"Found {total_files} Loras missing metadata. Starting CivitAI fetch process (delay >= {max(delay, MIN_REQUEST_DELAY_SECONDS):.1f}s)...",
    )

    request_cache = {}
    processed = 0
    successful = 0
    not_found = 0
    transient_errors = 0
    bad_payload = 0
    halted_due_to_rate_limit = False

    for lora in loras:
        if halt_check and halt_check():
            yield (
                f"Halted at {processed} / {total_files}",
                f"User requested halt. Stopped after processing {processed} files.",
            )
            break

        filepath = lora["filepath"]
        filename = os.path.basename(filepath)
        hashes_to_try = _build_hash_candidates(
            autov2_hash=lora.get("autov2_hash"),
            autov3_hash=lora.get("autov3_hash"),
            sha256_hash=lora.get("sha256_hash"),
        )
        request_signature = _hash_signature_from_values(
            autov2_hash=lora.get("autov2_hash"),
            autov3_hash=lora.get("autov3_hash"),
            sha256_hash=lora.get("sha256_hash"),
        )

        processed += 1
        msg_sum = f"Processed {processed} / {total_files} API requests ({int((processed / total_files) * 100)}%)"

        if not hashes_to_try:
            transient_errors += 1
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1,
                metadata_fetch_hash_signature=request_signature,
            )
            msg_log = f"[SKIP] No valid hashes - '{filename}'"
            print(msg_log)
            yield msg_sum, msg_log
            continue

        best_data = None
        best_status = 0
        best_reason = "no_hash_tried"
        saw_not_found = False

        for try_hash in hashes_to_try:
            if try_hash in request_cache:
                data, status_code, reason = request_cache[try_hash]
            else:
                url = f"{CIVITAI_API_URL}{try_hash}"
                data, status_code, reason = _request_json(url, token=token, delay=delay, halt_check=halt_check)
                request_cache[try_hash] = (data, status_code, reason)

            if status_code == 200 and data:
                best_data = data
                best_status = status_code
                best_reason = reason
                break

            if status_code == 404:
                saw_not_found = True
                best_status = 404
                best_reason = "not_found"
                continue

            best_status = status_code
            best_reason = reason
            if status_code == 429:
                halted_due_to_rate_limit = True
                break
            if status_code in (401, 403):
                break

        if halted_due_to_rate_limit:
            msg_log = "CivitAI Rate Limit Exceeded (HTTP 429). Entered cooldown and halted API fetches for safety."
            print(msg_log)
            yield msg_sum, msg_log
            break

        if best_status in (401, 403):
            transient_errors += 1
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1,
                metadata_fetch_hash_signature=request_signature,
            )
            msg_log = f"[{best_status}] Authorization failed - '{filename}'. Stopping API fetches."
            print(msg_log)
            yield msg_sum, msg_log
            break

        if best_data and best_status == 200:
            parsed = _extract_api_metadata(best_data)
            if not parsed:
                bad_payload += 1
                upsert_lora(
                    filename=filename,
                    filepath=filepath,
                    metadata_fetch_attempted=1,
                    metadata_fetch_hash_signature=request_signature,
                )
                msg_log = f"[200] Invalid/empty payload - '{filename}' (DB unchanged)"
                print(msg_log)
                yield msg_sum, msg_log
                continue

            # Preserve existing values via DB COALESCE behavior.
            upsert_lora(
                filename=filename,
                filepath=filepath,
                autov2_hash=parsed.get("autov2_hash"),
                autov3_hash=parsed.get("autov3_hash"),
                sha256_hash=parsed.get("sha256_hash"),
                trigger_words=parsed.get("trigger_words"),
                base_model=parsed.get("base_model"),
                metadata_fetch_attempted=1,
                civitai_version_id=parsed.get("civitai_version_id"),
                loraname=parsed.get("loraname"),
                metadata_fetch_hash_signature=request_signature,
            )

            wrote_info, write_error = _safe_update_civitai_info(filepath, best_data, parsed)
            if not wrote_info:
                warn = (
                    f"Warning: DB updated but could not safely write .civitai.info for '{filename}'. "
                    f"Reason: {write_error}"
                )
                print(warn)
                yield msg_sum, warn

            successful += 1
            msg_log = f"[200] OK - '{filename}'"
            print(msg_log)
            yield msg_sum, msg_log
            continue

        if saw_not_found or best_status == 404:
            # Terminal case: prevent endless retries for models not known by CivitAI.
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1,
                metadata_fetch_hash_signature=request_signature,
            )
            not_found += 1
            msg_log = f"[404] Not Found - '{filename}' (all hash candidates exhausted)"
            print(msg_log)
            yield msg_sum, msg_log
            continue

        if best_reason != "halted":
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1,
                metadata_fetch_hash_signature=request_signature,
            )
        transient_errors += 1
        msg_log = f"[{best_status}] API error ({best_reason}) - '{filename}'"
        print(msg_log)
        yield msg_sum, msg_log

    final_summary = f"API fetching complete. Processed {processed} files."
    final_log = (
        f"Finished CivitAI metadata fetches. Processed: {processed}/{total_files}. "
        f"Success: {successful}. Not found: {not_found}. "
        f"Invalid payloads: {bad_payload}. Transient/other errors: {transient_errors}."
    )
    if halted_due_to_rate_limit:
        final_log += " Halted early due to HTTP 429 cooldown safety stop."
    yield final_summary, final_log

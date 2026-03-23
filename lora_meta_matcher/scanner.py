import os
import json
from .db import upsert_lora, get_lora_by_path

def _normalize_hash(value):
    if not value or not isinstance(value, str):
        return None
    return value.strip().upper()

def _derive_autov2_from_sha256(sha256_hash):
    normalized = _normalize_hash(sha256_hash)
    if normalized and len(normalized) >= 10:
        return normalized[:10]
    return None


def _hash_signature_from_row(row):
    """
    Stable signature of currently-known hashes for API retry gating.
    """
    if not row:
        return None
    parts = [
        _normalize_hash(row.get("autov2_hash")),
        _normalize_hash(row.get("autov3_hash")),
        _normalize_hash(row.get("sha256_hash")),
    ]
    return "|".join(p if p else "" for p in parts)

def _is_fully_populated_lora(row):
    """
    Decide whether scanning can safely skip this row.
    We require core metadata fields and at least one usable hash.
    """
    if not row:
        return False
    has_any_hash = bool(row.get("autov2_hash") or row.get("autov3_hash") or row.get("sha256_hash"))
    return (
        has_any_hash and
        bool(row.get("trigger_words")) and
        bool(row.get("base_model")) and
        bool(row.get("civitai_version_id")) and
        bool(row.get("loraname"))
    )

def _has_value(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    return True

def _metadata_would_improve_row(existing, metadata):
    """
    Return True if metadata can populate at least one currently-missing DB field.
    """
    if not metadata:
        return False
    if not existing:
        return True
    field_map = {
        "autov2_hash": "autov2_hash",
        "autov3_hash": "autov3_hash",
        "sha256_hash": "sha256_hash",
        "trigger_words": "trigger_words",
        "base_model": "base_model",
        "civitai_version_id": "civitai_version_id",
        "loraname": "loraname",
    }
    for meta_key, row_key in field_map.items():
        if _has_value(metadata.get(meta_key)) and not _has_value(existing.get(row_key)):
            return True
    return False

def parse_metadata_file(filepath):
    """
    Attempts to read a .civitai.info, .civitai_info, or .json file (CivitAI format).
    Returns a dict with autov2_hash, trigger_words, base_model, civitai_version_id, etc., or None on error.
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        source = data.get("civitai") if isinstance(data, dict) and isinstance(data.get("civitai"), dict) else data

        autov2_hash = None
        autov3_hash = None
        sha256_hash = None
        trigger_words = None
        base_model = None
        loraname = None
        
        # Base Model
        if isinstance(source, dict) and "baseModel" in source:
            base_model = source["baseModel"]
            
        # Trigger Words
        if isinstance(source, dict) and "trainedWords" in source and isinstance(source["trainedWords"], list):
            trigger_words = ", ".join(source["trainedWords"])

        # Hashes - CivitAI format
        if isinstance(source, dict) and "files" in source and isinstance(source["files"], list):
            for file_info in source["files"]:
                # Often the primary file or the first file has the hashes
                if "hashes" in file_info and isinstance(file_info["hashes"], dict):
                    hashes = file_info["hashes"]
                    if "AutoV2" in hashes:
                        autov2_hash = hashes["AutoV2"]
                    if "AutoV3" in hashes:
                        autov3_hash = hashes["AutoV3"]
                    if "SHA256" in hashes:
                        sha256_hash = hashes["SHA256"]
                
                # If we found any hash, we can probably stop searching for it,
                # but let's just make sure we check the first file mostly.
                if autov2_hash or sha256_hash:
                    break

        if isinstance(data, dict):
            if not sha256_hash and "sha256" in data:
                sha256_hash = data["sha256"]
            if not autov2_hash and "autov2" in data:
                autov2_hash = data["autov2"]
            if not autov3_hash and "autov3" in data:
                autov3_hash = data["autov3"]
        if isinstance(source, dict):
            if not sha256_hash and "sha256" in source:
                sha256_hash = source["sha256"]
            if not autov2_hash and "autov2" in source:
                autov2_hash = source["autov2"]
            if not autov3_hash and "autov3" in source:
                autov3_hash = source["autov3"]

        # Normalize hashes and derive AutoV2 from SHA256 when missing.
        sha256_hash = _normalize_hash(sha256_hash)
        autov2_hash = _normalize_hash(autov2_hash) or _derive_autov2_from_sha256(sha256_hash)
        autov3_hash = _normalize_hash(autov3_hash)
                
        # Civitai Version ID (API may return int or string)
        civitai_version_id = None
        if isinstance(source, dict) and "id" in source:
            raw_id = source["id"]
            if isinstance(raw_id, int):
                civitai_version_id = raw_id
            elif isinstance(raw_id, str) and raw_id.isdigit():
                civitai_version_id = int(raw_id)
            
        # Lora Name
        if isinstance(source, dict) and "model" in source and isinstance(source["model"], dict) and "name" in source["model"]:
            m_name = source["model"]["name"]
            v_name = source.get("name")
            loraname = f"{m_name} ({v_name})" if v_name else m_name
        elif isinstance(source, dict) and "name" in source:
            loraname = source["name"]
        elif isinstance(data, dict) and "name" in data:
            loraname = data["name"]

        # Ignore JSON files that parse but do not contain any useful metadata.
        has_useful_metadata = any([
            autov2_hash,
            autov3_hash,
            sha256_hash,
            trigger_words,
            base_model,
            civitai_version_id,
            loraname,
        ])
        if not has_useful_metadata:
            return None
            
        return {
            "autov2_hash": autov2_hash,
            "autov3_hash": autov3_hash,
            "sha256_hash": sha256_hash,
            "trigger_words": trigger_words,
            "base_model": base_model,
            "civitai_version_id": civitai_version_id,
            "loraname": loraname
        }
    except Exception as e:
        print(f"Error parsing metadata file {filepath}: {e}")
        return None

def _civitai_info_key(root, base_name):
    """Normalize (root, base_name) for lookup: realpath + normpath for dir, lowercase for base_name so matching is case-insensitive."""
    try:
        canonical_root = os.path.realpath(os.path.normpath(root))
    except OSError:
        canonical_root = os.path.normpath(root)
    return (canonical_root, (base_name or "").lower())

def _gather_json_fallback_candidates(root, base_name):
    """
    Build an ordered list of JSON metadata candidates for a safetensors file.
    Priority:
      1) exact sidecar files next to the model
    """
    exact_candidates = [
        os.path.join(root, f"{base_name}.civitai.info"),
        os.path.join(root, f"{base_name}.civitai_info"),
        os.path.join(root, f"{base_name}.metadata.json"),
        os.path.join(root, f"{base_name}.json"),
    ]

    candidates = []
    seen = set()
    for candidate in exact_candidates:
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    return candidates

def _sync_civitai_info_from_row(filepath, row):
    """
    Ensure a .civitai.info exists next to filepath and includes known DB metadata/hashes.
    This fills missing fields but does not remove existing ones.
    """
    if not row:
        return False

    orig_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    info_path = os.path.join(orig_dir, f"{base_name}.civitai.info")
    if not os.path.exists(orig_dir):
        return False

    try:
        data = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception:
                data = {}

        # Hashes (CivitAI style + flat fallback keys)
        hashes = {}
        if row.get("autov2_hash"):
            hashes["AutoV2"] = row["autov2_hash"]
            data.setdefault("autov2", row["autov2_hash"])
        if row.get("autov3_hash"):
            hashes["AutoV3"] = row["autov3_hash"]
            data.setdefault("autov3", row["autov3_hash"])
        if row.get("sha256_hash"):
            hashes["SHA256"] = row["sha256_hash"]
            data.setdefault("sha256", row["sha256_hash"])

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
                for k, v in hashes.items():
                    first_hashes.setdefault(k, v)

        # Common metadata fields
        if row.get("base_model"):
            data.setdefault("baseModel", row["base_model"])
        if row.get("civitai_version_id"):
            data.setdefault("id", row["civitai_version_id"])

        if row.get("trigger_words"):
            trained_words = [w.strip() for w in str(row["trigger_words"]).split(",") if w.strip()]
            if trained_words:
                data.setdefault("trainedWords", trained_words)

        # Keep a readable name if no richer model/version structure exists
        if row.get("loraname"):
            data.setdefault("name", row["loraname"])

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def scan_directory(directory_path):
    """
    Generator that recursively scans a directory for .safetensors files.
    First loads all .civitai.info / .civitai_info files to populate metadata, then
    processes each .safetensors and matches to that metadata (or falls back to
    per-file .civitai.info / .json next to the safetensors). This ensures existing
    .civitai.info files are used to populate the database before any hash/API work.
    Yields (summary_text, log_text) tuples.
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        yield f"Error: Directory not found", f"Error: Directory not found: {directory_path}"
        return

    yield "Loading .civitai.info files first...", f"Scanning '{directory_path}' for .civitai.info and .civitai_info files..."

    # Phase 1: Discover and parse all .civitai.info / .civitai_info files first
    civitai_info_map = {}  # (root_normpath, base_name) -> metadata dict
    civitai_info_count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            base_name = None
            if file.lower().endswith(".civitai.info"):
                base_name = file[:-len(".civitai.info")]
            elif file.lower().endswith(".civitai_info"):
                base_name = file[:-len(".civitai_info")]
            if base_name is not None:
                path = os.path.join(root, file)
                meta = parse_metadata_file(path)
                if meta is not None:
                    key = _civitai_info_key(root, base_name)
                    civitai_info_map[key] = meta
                    civitai_info_count += 1

    yield "Locating .safetensors...", f"Found {civitai_info_count} .civitai.info/.civitai_info files with valid metadata. Locating .safetensors files..."

    safetensors_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".safetensors"):
                safetensors_files.append(os.path.join(root, file))

    total_files = len(safetensors_files)
    if total_files == 0:
        yield "Found 0 files.", "No .safetensors files found in the directory."
        return

    yield f"Processing {total_files} files...", f"Found {total_files} .safetensors files. Matching to .civitai.info metadata and extracting."

    processed = 0
    from_sidecar_metadata = 0
    db_updated_files = 0
    skipped_existing = 0
    no_metadata_found = 0
    for filepath in safetensors_files:
        filename = os.path.basename(filepath)
        root = os.path.dirname(filepath)
        base_name = os.path.splitext(filename)[0]
        processed += 1
        updated_this_file = False

        # If existing row has SHA256 but no AutoV2, fill it immediately.
        existing = get_lora_by_path(filepath)
        if existing and existing.get("sha256_hash") and not existing.get("autov2_hash"):
            derived_autov2 = _derive_autov2_from_sha256(existing.get("sha256_hash"))
            if derived_autov2 and derived_autov2 != existing.get("autov2_hash"):
                upsert_lora(
                    filename=filename,
                    filepath=filepath,
                    autov2_hash=derived_autov2,
                )
                updated_this_file = True
                existing = get_lora_by_path(filepath)

        # Skip only if core metadata columns are already populated.
        if _is_fully_populated_lora(existing):
            _sync_civitai_info_from_row(filepath, existing)
            skipped_existing += 1
            msg = f"Skipping {filename} (Already fully populated in DB)"
            yield f"Processed {processed} / {total_files} files ({int((processed/total_files)*100)}%)", msg
            continue

        # Prefer metadata from phase 1 (.civitai.info we already loaded)
        key = _civitai_info_key(root, base_name)
        metadata = civitai_info_map.get(key)

        # Fallback: look for metadata in sidecar JSON files next to this .safetensors
        if metadata is None:
            for candidate in _gather_json_fallback_candidates(root, base_name):
                if os.path.exists(candidate):
                    metadata = parse_metadata_file(candidate)
                    # Stop at first parseable metadata source.
                    if metadata:
                        break

        if metadata:
            from_sidecar_metadata += 1
            metadata_improves = _metadata_would_improve_row(existing, metadata)
            if metadata_improves:
                upsert_lora(
                    filename=filename,
                    filepath=filepath,
                    autov2_hash=metadata.get("autov2_hash"),
                    autov3_hash=metadata.get("autov3_hash"),
                    sha256_hash=metadata.get("sha256_hash"),
                    trigger_words=metadata.get("trigger_words"),
                    base_model=metadata.get("base_model"),
                    civitai_version_id=metadata.get("civitai_version_id"),
                    loraname=metadata.get("loraname"),
                    # Sidecar import is not an API attempt; keep this fetch-eligible if still incomplete.
                    metadata_fetch_attempted=0,
                )
                updated_this_file = True
            refreshed = get_lora_by_path(filepath)
            _sync_civitai_info_from_row(filepath, refreshed)
            msg = f"Found {filename} (Extracted metadata from .civitai.info)"
        else:
            # Ensure a row exists in DB, but don't count as an update if it already existed.
            if not existing:
                upsert_lora(filename=filename, filepath=filepath)
                updated_this_file = True
            no_metadata_found += 1
            msg = f"Found {filename} (No metadata found)"

        if updated_this_file:
            db_updated_files += 1

        # Re-enable API fetch only when incomplete rows have changed hash identity, or if
        # legacy rows never recorded a prior fetch signature.
        refreshed_for_gate = get_lora_by_path(filepath)
        if refreshed_for_gate and not _is_fully_populated_lora(refreshed_for_gate):
            attempted = int(refreshed_for_gate.get("metadata_fetch_attempted") or 0)
            current_sig = _hash_signature_from_row(refreshed_for_gate)
            previous_sig = refreshed_for_gate.get("metadata_fetch_hash_signature")
            should_requeue = attempted == 0 or (attempted == 1 and (not previous_sig or previous_sig != current_sig))
            if should_requeue and attempted != 0:
                upsert_lora(
                    filename=filename,
                    filepath=filepath,
                    metadata_fetch_attempted=0,
                )

        yield f"Processed {processed} / {total_files} files ({int((processed/total_files)*100)}%)", msg

    missing_hash_files = []
    missing_some_metadata_count = 0
    for scanned_path in safetensors_files:
        row = get_lora_by_path(scanned_path)
        if not row:
            continue
        if not _is_fully_populated_lora(row):
            missing_some_metadata_count += 1
        has_autov2 = bool(row.get("autov2_hash"))
        has_sha256 = bool(row.get("sha256_hash"))
        if not has_autov2 and not has_sha256:
            missing_hash_files.append(scanned_path)
    unhashed_count = len(missing_hash_files)
    no_meta_count = no_metadata_found

    summary = (
        f"Scan complete: Database updated for {db_updated_files} loras. "
        f"{missing_some_metadata_count} loras missing some metadata. "
        f"{unhashed_count} loras require hash calculation."
    )
    log = (
        f"Finished scanning {total_files} Lora files. "
        f"Database updated for: {db_updated_files}. "
        f"Loaded from sidecar metadata: {from_sidecar_metadata}. "
        f"Skipped (already in DB): {skipped_existing}. "
        f"No metadata found: {no_meta_count}. "
        f"Missing some metadata: {missing_some_metadata_count}. "
        f"Missing AutoV2 hash: {unhashed_count}."
    )
    if missing_hash_files:
        preview_limit = 50
        preview_items = missing_hash_files[:preview_limit]
        preview_block = "\n".join(f"- {p}" for p in preview_items)
        remaining = unhashed_count - len(preview_items)
        if remaining > 0:
            preview_block += f"\n... and {remaining} more"
        log += f"\n\nMissing hash files (preview):\n{preview_block}"
    yield summary, log

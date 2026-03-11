import os
import json
from .db import upsert_lora, get_loras_without_hash, get_lora_by_path

def parse_metadata_file(filepath):
    """
    Attempts to read a .civitai.info or .json file sitting next to a safetensors file.
    Returns a dict with autov2_hash, trigger_words, and base_model if found.
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        autov2_hash = None
        trigger_words = None
        base_model = None
        
        # Base Model
        if "baseModel" in data:
            base_model = data["baseModel"]
            
        # Trigger Words
        if "trainedWords" in data and isinstance(data["trainedWords"], list):
            trigger_words = ", ".join(data["trainedWords"])

        # AutoV2 Hash - CivitAI format
        if "files" in data and isinstance(data["files"], list):
            for file_info in data["files"]:
                # Often the primary file or the first file has the hashes
                if "hashes" in file_info and isinstance(file_info["hashes"], dict):
                    hashes = file_info["hashes"]
                    if "AutoV2" in hashes:
                        autov2_hash = hashes["AutoV2"]
                    elif "SHA256" in hashes:
                        autov2_hash = hashes["SHA256"]
                
                # If we found a hash, we can stop searching for it
                if autov2_hash:
                    break

        if not autov2_hash:
            if "sha256" in data:
                autov2_hash = data["sha256"]
            elif "autov2" in data:
                autov2_hash = data["autov2"]
                
        return {
            "autov2_hash": autov2_hash,
            "trigger_words": trigger_words,
            "base_model": base_model
        }
    except Exception as e:
        print(f"Error parsing metadata file {filepath}: {e}")
        return None

def scan_directory(directory_path):
    """
    Generator that recursively scans a directory for .safetensors files.
    First counts total files, then processes them.
    Yields (summary_text, log_text) tuples.
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        yield f"Error: Directory not found", f"Error: Directory not found: {directory_path}"
        return

    yield "Scanning filesystem...", f"Locating .safetensors files in '{directory_path}'..."
    
    safetensors_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.safetensors'):
                safetensors_files.append(os.path.join(root, file))
                
    total_files = len(safetensors_files)
    if total_files == 0:
        yield "Found 0 files.", "No .safetensors files found in the directory."
        return
        
    yield f"Preparing to process {total_files} files...", f"Found {total_files} .safetensors files. Starting metadata extraction."

    processed = 0
    for filepath in safetensors_files:
        filename = os.path.basename(filepath)
        root = os.path.dirname(filepath)
                
        processed += 1
        
        # Check if already processed
        existing = get_lora_by_path(filepath)
        if existing:
            # Skip metadata extraction since it's already in the DB
            msg = f"Skipping {filename} (Already in database)"
            yield f"Processed {processed} / {total_files} files ({int((processed/total_files)*100)}%)", msg
            continue
            
        # Look for metadata files
        base_name = os.path.splitext(filename)[0]
        civitai_info_path = os.path.join(root, f"{base_name}.civitai.info")
        json_path = os.path.join(root, f"{base_name}.json")
        
        metadata = None
        if os.path.exists(civitai_info_path):
            metadata = parse_metadata_file(civitai_info_path)
        elif os.path.exists(json_path):
            metadata = parse_metadata_file(json_path)
            
        if metadata:
            upsert_lora(
                filename=filename,
                filepath=filepath,
                autov2_hash=metadata.get("autov2_hash"),
                trigger_words=metadata.get("trigger_words"),
                base_model=metadata.get("base_model"),
                metadata_fetch_attempted=1
            )
            msg = f"Found {filename} (Extracted metadata)"
        else:
            upsert_lora(
                filename=filename,
                filepath=filepath
            )
            msg = f"Found {filename} (No metadata found)"
            
        yield f"Processed {processed} / {total_files} files ({int((processed/total_files)*100)}%)", msg

    unhashed_count = len(get_loras_without_hash())
    
    yield f"Scan complete. {unhashed_count} files require hash calculation.", f"Finished scanning {total_files} Lora files. {unhashed_count} files in the database are currently missing an AutoV2 Hash."

import os
import json
import time
import requests
from .db import upsert_lora, get_loras_without_triggers_but_have_hash

CIVITAI_API_URL = "https://civitai.com/api/v1/model-versions/by-hash/"
CIVITAI_API_VERSION_URL = "https://civitai.com/api/v1/model-versions/"

def fetch_civitai_version_info(version_id, token=None):
    """
    Fetches the model version info from CivitAI using the version ID.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    url = f"{CIVITAI_API_VERSION_URL}{version_id}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json(), 200
        elif response.status_code == 404:
            return None, 404
        elif response.status_code == 429:
            print(f"Rate limited by CivitAI.")
            return None, 429
        else:
            print(f"Failed to fetch for version {version_id}. Status: {response.status_code}")
            return None, response.status_code
    except Exception as e:
        print(f"Error fetching from CivitAI API for version {version_id}: {e}")
        return None, 0

def fetch_civitai_info(autov2_hash, token=None):
    """
    Fetches the model version info from CivitAI using the AutoV2 or full SHA256 hash.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    url = f"{CIVITAI_API_URL}{autov2_hash}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json(), 200
        elif response.status_code == 404:
            return None, 404
        elif response.status_code == 429:
            print(f"Rate limited by CivitAI.")
            return None, 429
        else:
            print(f"Failed to fetch for hash {autov2_hash}. Status: {response.status_code}")
            return None, response.status_code
    except Exception as e:
        print(f"Error fetching from CivitAI API for hash {autov2_hash}: {e}")
        return None, 0

def process_missing_civitai_metadata(token=None, delay=2.0, halt_check=None):
    """
    Finds loras in the DB with a hash but no trigger_words,
    makes API requests to CivitAI, and saves the result to a .civitai.info file next to the lora.
    Yields (summary_text, log_text) tuples.
    """
    loras = get_loras_without_triggers_but_have_hash()
    total_files = len(loras)
    
    if not total_files:
        yield "All hashed Loras already have metadata.", "Check complete: No missing metadata found."
        return

    yield f"Preparing fetching for {total_files} Loras...", f"Found {total_files} Loras missing metadata. Starting CivitAI fetch process..."
    
    for count, lora in enumerate(loras):
        if halt_check and halt_check():
            yield f"Halted at {count} / {total_files}", f"User requested halt. Stopped after processing {count} requests."
            break
            
        filepath = lora["filepath"]
        autov2_hash = lora.get("autov2_hash")
        autov3_hash = lora.get("autov3_hash")
        sha256_hash = lora.get("sha256_hash")
        filename = os.path.basename(filepath)
        
        # We know from the DB query at least one is NOT None
        # Let's prefer the longest one usually, but CivitAI accepts AutoV2 (10/12char) or SHA256 (64char).
        # We'll prioritize autov2 logic first since it's the A1111 standard we were just dealing with.
        hashes_to_try = []
        if autov2_hash:
            hashes_to_try.append(autov2_hash)
            if len(autov2_hash) > 10:
                hashes_to_try.append(autov2_hash[:10])
        if sha256_hash:
            hashes_to_try.append(sha256_hash)
        if autov3_hash:
            hashes_to_try.append(autov3_hash)
            
        msg_sum = f"Processed {count+1} / {total_files} API requests ({int(((count+1)/total_files)*100)}%)"
        
        data = None
        status_code = None
        
        for try_hash in hashes_to_try:
            data, status_code = fetch_civitai_info(try_hash, token)
            
            if status_code == 200:
                break
            elif status_code == 429:
                break
                
        if status_code == 429:
            yield msg_sum, "CivitAI Rate Limit Exceeded (HTTP 429). Halting API fetches."
            break # Halt immediately
            
        if data:
            # Save the .civitai.info file next to the original file
            base_name = os.path.splitext(filepath)[0]
            info_path = f"{base_name}.civitai.info"
            
            orig_dir = os.path.dirname(filepath)
            if os.path.exists(orig_dir):
                try:
                    with open(info_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    yield msg_sum, f"Error saving info file for {filepath}: {e}"
                    continue
            else:
                yield msg_sum, f"Warning: Directory {orig_dir} not found. DB will be updated, but .info file skipped."
                
            trigger_words = None
            base_model = None
            civitai_version_id = None
            loraname = None
            api_autov2 = None
            api_autov3 = None
            api_sha256 = None
            
            if "trainedWords" in data and isinstance(data["trainedWords"], list):
                trigger_words = ", ".join(data["trainedWords"])
            if "baseModel" in data:
                base_model = data["baseModel"]
            if "id" in data and isinstance(data["id"], int):
                civitai_version_id = data["id"]
                
            if "model" in data and isinstance(data["model"], dict) and "name" in data["model"]:
                m_name = data["model"]["name"]
                v_name = data.get("name")
                loraname = f"{m_name} ({v_name})" if v_name else m_name
            elif "name" in data:
                loraname = data["name"]
                
            if "files" in data and isinstance(data["files"], list):
                for file_info in data["files"]:
                    if "hashes" in file_info and isinstance(file_info["hashes"], dict):
                        hashes = file_info["hashes"]
                        if "AutoV2" in hashes:
                            api_autov2 = hashes["AutoV2"]
                        if "AutoV3" in hashes:
                            api_autov3 = hashes["AutoV3"]
                        if "SHA256" in hashes:
                            api_sha256 = hashes["SHA256"]
                    
                    if api_autov2 or api_sha256:
                        break
                
            # Upsert partial data to populate triggers
            upsert_lora(
                filename=filename,
                filepath=filepath,
                autov2_hash=api_autov2 if api_autov2 else None,
                autov3_hash=api_autov3 if api_autov3 else None,
                sha256_hash=api_sha256 if api_sha256 else None,
                trigger_words=trigger_words,
                base_model=base_model,
                metadata_fetch_attempted=1,
                civitai_version_id=civitai_version_id,
                loraname=loraname
            )
            msg_log = f"[{status_code}] OK - '{filename}'"
            print(msg_log)
            yield msg_sum, msg_log
        elif status_code == 404:
            # Model missing from CivitAI; mark attempted to prevent infinite refetch polling
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1
            )
            msg_log = f"[404] Not Found - '{filename}' (Failed all available hashes)"
            print(msg_log)
            yield msg_sum, msg_log
        else:
            msg_log = f"[{status_code}] Error - '{filename}'"
            print(msg_log)
            yield msg_sum, msg_log
            
        time.sleep(delay) # Rate limiting
            
    yield f"API fetching complete. Processed {total_files} files.", f"Finished fetching CivitAI metadata for {total_files} Loras."

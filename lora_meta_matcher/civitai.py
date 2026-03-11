import os
import json
import time
import requests
from .db import upsert_lora, get_loras_without_triggers_but_have_hash

CIVITAI_API_URL = "https://civitai.com/api/v1/model-versions/by-hash/"

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

def process_missing_civitai_metadata(token=None, delay=2.0):
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
        filepath = lora["filepath"]
        autov2_hash = lora["autov2_hash"]
        filename = os.path.basename(filepath)
        
        msg_log = f"({count+1}/{total_files}) Fetching data for '{filename}'..."
        msg_sum = f"Processed {count+1} / {total_files} API requests ({int(((count+1)/total_files)*100)}%)"
        
        yield msg_sum, msg_log
        
        data, status_code = fetch_civitai_info(autov2_hash, token)
        
        if status_code == 429:
            yield msg_sum, "CivitAI Rate Limit Exceeded (HTTP 429). Halting API fetches."
            break # Halt immediately
            
        if data:
            # Save the .civitai.info file next to the original file
            base_name = os.path.splitext(filepath)[0]
            info_path = f"{base_name}.civitai.info"
            
            try:
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                yield msg_sum, f"Error saving info file for {filepath}: {e}"
                continue
                
            trigger_words = None
            base_model = None
            civitai_version_id = None
            
            if "trainedWords" in data and isinstance(data["trainedWords"], list):
                trigger_words = ", ".join(data["trainedWords"])
            if "baseModel" in data:
                base_model = data["baseModel"]
            if "id" in data and isinstance(data["id"], int):
                civitai_version_id = data["id"]
                
            # Upsert partial data to populate triggers
            upsert_lora(
                filename=filename,
                filepath=filepath,
                trigger_words=trigger_words,
                base_model=base_model,
                metadata_fetch_attempted=1,
                civitai_version_id=civitai_version_id
            )
            yield msg_sum, f"Successfully updated metadata for '{filename}'."
        elif status_code == 404:
            # Model missing from CivitAI; mark attempted to prevent infinite refetch polling
            upsert_lora(
                filename=filename,
                filepath=filepath,
                metadata_fetch_attempted=1
            )
            yield msg_sum, f"No data found on CivitAI for hash {autov2_hash[:10]}..."
        else:
            yield msg_sum, f"Failed to fetch metadata (API Error {status_code}) for '{filename}'."
            
        time.sleep(delay) # Rate limiting
            
    yield f"API fetching complete. Processed {total_files} files.", f"Finished fetching CivitAI metadata for {total_files} Loras."

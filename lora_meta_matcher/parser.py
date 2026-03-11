import json
import re
import sqlite3
from .db import get_connection

def parse_a1111_metadata(info):
    if 'parameters' not in info:
        return None
    
    params = info['parameters']
    
    positive_prompt = ""
    loras = []
    
    # Split primary prompt from negative prompt and settings
    lines = params.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Negative prompt:") or line.startswith("Steps:"):
            positive_prompt = "\n".join(lines[:i]).strip()
            break
            
    if not positive_prompt:
        positive_prompt = params.strip()
        
    # 1. Standard A1111 prompt Lora tags <lora:name:weight>
    lora_matches = re.findall(r'<lora:([^:]+):([^>]+)>', positive_prompt)
    for name, weight in lora_matches:
        loras.append({"name": name, "weight": weight})
        
    # 2. Civitai Resources JSON extraction
    civitai_match = re.search(r'Civitai resources:\s*(\[.*\])', params, flags=re.DOTALL)
    if civitai_match:
        try:
            resources = json.loads(civitai_match.group(1))
            for res in resources:
                if isinstance(res, dict) and res.get("type") == "lora":
                    name = res.get("modelName") or res.get("modelVersionName")
                    weight = res.get("weight", "1.0")
                    modelVersionId = res.get("modelVersionId")
                    if name:
                        # Ensure we don't duplicate Loras already found in prompt
                        if not any(l["name"] == name for l in loras):
                            loras.append({"name": name, "weight": str(weight), "civitai_version_id": modelVersionId})
        except Exception:
            pass
        
    return {
        "raw_prompt": params, # return the entire raw string
        "loras": loras,
        "positive_prompt": positive_prompt
    }

def parse_comfyui_metadata(info):
    if 'prompt' not in info and 'workflow' not in info:
        return None
        
    prompt_str = info.get('prompt', info.get('workflow', '{}'))
    if not isinstance(prompt_str, str):
        prompt_str = json.dumps(prompt_str)
        
    try:
        prompt_data = json.loads(prompt_str)
    except json.JSONDecodeError:
        return None
        
    positive_prompt = ""
    loras = []
    
    nodes_list = []
    if isinstance(prompt_data, dict):
        if 'nodes' in prompt_data and isinstance(prompt_data['nodes'], list):
            nodes_list = prompt_data['nodes']
        else:
            nodes_list = list(prompt_data.values())
    elif isinstance(prompt_data, list):
        nodes_list = prompt_data

    texts = []
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
            
        inputs = node.get("inputs", node.get("widgets_values", {}))
        
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, str) and (item.endswith('.safetensors') or item.endswith('.pt')):
                    lora_basename = item.replace(".safetensors", "").replace(".pt", "")
                    loras.append({"name": lora_basename, "weight": "1.0"})
                elif isinstance(item, str) and len(item) > 10 and ',' in item:
                    texts.append(item)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, str):
                    k_lower = k.lower()
                    is_lora = False
                    
                    if (v.endswith('.safetensors') or v.endswith('.pt')) and ('lora' in k_lower or 'name' in k_lower):
                        is_lora = True
                    elif 'lora' in k_lower and 'name' in k_lower:
                        is_lora = True
                        
                    if is_lora:
                        lora_basename = v.replace(".safetensors", "").replace(".pt", "")
                        weight = inputs.get("strength_model", "1.0")
                        loras.append({"name": lora_basename, "weight": str(weight)})
                    elif k == "text" or k == "text_positive":
                        texts.append(v)

    if texts:
        positive_prompt = " | ".join([t for t in texts if len(t) > 5])

    return {
        "raw_prompt": prompt_str, # return the entire raw json string
        "loras": loras,
        "positive_prompt": positive_prompt
    }

def decode_user_comment(user_comment):
    if not isinstance(user_comment, bytes):
        return str(user_comment)
        
    header = user_comment[:8]
    data = user_comment[8:]
    
    if header.startswith(b'UNICODE'):
        try:
            text_le = data.decode('utf-16le')
            if text_le.startswith('{') or text_le.startswith('<lora:') or text_le[0].isascii():
                return text_le
        except:
            pass
            
        try:
            text_be = data.decode('utf-16be')
            if text_be.startswith('{') or text_be.startswith('<lora:') or text_be[0].isascii():
                return text_be
        except:
            pass
            
        try:
            return data.decode('utf-8', errors='ignore')
        except:
            pass
            
    elif header.startswith(b'ASCII'):
        return data.decode('ascii', errors='ignore')
    else:
        return user_comment.decode('utf-8', errors='ignore')

def extract_image_metadata(img):
    info = dict(img.info) # Copy to avoid mutating original
    
    # Extract JPEG EXIF UserComment if no standard info found
    if 'parameters' not in info and 'prompt' not in info and 'workflow' not in info:
        exif = img.getexif()
        if exif:
            ifd = exif.get_ifd(0x8769)
            if ifd and 0x9286 in ifd:
                comment = decode_user_comment(ifd[0x9286])
                if comment:
                    # Could be A1111 string or Comfy JSON
                    if comment.startswith('{'):
                        info['prompt'] = comment
                    else:
                        info['parameters'] = comment

    if not info:
        return None
        
    # Attempt parsing as A1111 format
    data = parse_a1111_metadata(info)
    if data:
        return data
        
    # Attempt parsing as ComfyUI format
    data = parse_comfyui_metadata(info)
    if data:
        return data
        
    # If no explicit parser worked but we still found a raw string (e.g. prompt, parameters, workflow, or exif)
    raw_str = info.get('parameters') or info.get('prompt') or info.get('workflow')
    if raw_str:
        return {
            "raw_prompt": raw_str,
            "loras": [],
            "positive_prompt": raw_str
        }
        
    return None

def match_loras_to_db(loras):
    """
    Given a list of loras [{"name": "NAME", "weight": "1.0"}], find their records in DB.
    """
    matched = []
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        for lora in loras:
            name = lora["name"]
            weight = lora["weight"]
            civitai_version_id = lora.get("civitai_version_id")
            
            results = []
            
            # 1. Try to match exact Civitai Version ID
            if civitai_version_id:
                cursor.execute('SELECT * FROM loras WHERE civitai_version_id=?', (civitai_version_id,))
                results = cursor.fetchall()
                
            # 2. Fallback to generic name search
            if not results:
                cursor.execute('SELECT * FROM loras WHERE filename LIKE ? OR filepath LIKE ?', (f"%{name}%.safetensors", f"%{name}%"))
                results = cursor.fetchall()
            
            if results:
                row = dict(results[0])
                matched.append({
                    "original_name": name,
                    "weight": weight,
                    "filename": row["filename"],
                    "filepath": row["filepath"],
                    "autov2_hash": row["autov2_hash"],
                    "base_model": row["base_model"],
                    "civitai_version_id": row.get("civitai_version_id") or civitai_version_id,
                    "trigger_words": row["trigger_words"]
                })
            else:
                matched.append({
                    "original_name": name,
                    "weight": weight,
                    "filename": None,
                    "filepath": None,
                    "autov2_hash": None,
                    "base_model": None,
                    "civitai_version_id": civitai_version_id,
                    "trigger_words": None
                })
                
    return matched

def reconstruct_prompt(parsed_data, matched_loras):
    """
    Returns only the fully qualified lora tags (using filenames)
    and their associated trigger words.
    """
    additions = []
    
    for lora in matched_loras:
        if lora["filename"]:
            basename = lora["filename"].replace(".safetensors", "")
            tag = f"<lora:{basename}:{lora['weight']}>"
            additions.append(tag)
            
            if lora["trigger_words"]:
                additions.append(lora["trigger_words"])
            
    return ", ".join(additions)

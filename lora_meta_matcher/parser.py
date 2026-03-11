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
    
    lines = params.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Negative prompt:") or line.startswith("Steps:"):
            positive_prompt = "\n".join(lines[:i]).strip()
            break
            
    if not positive_prompt:
        positive_prompt = params.strip()
        
    lora_matches = re.findall(r'<lora:([^:]+):([^>]+)>', positive_prompt)
    for name, weight in lora_matches:
        loras.append({"name": name, "weight": weight})
        
    return {
        "raw_prompt": positive_prompt,
        "loras": loras
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
                if isinstance(v, str) and (v.endswith('.safetensors') or v.endswith('.pt')) and ('lora' in k.lower() or 'name' in k.lower()):
                    lora_basename = v.replace(".safetensors", "").replace(".pt", "")
                    weight = inputs.get("strength_model", "1.0")
                    loras.append({"name": lora_basename, "weight": str(weight)})
                elif isinstance(v, str) and (k == "text" or k == "text_positive"):
                    texts.append(v)

    if texts:
        positive_prompt = " | ".join([t for t in texts if len(t) > 5])

    return {
        "raw_prompt": positive_prompt,
        "loras": loras
    }

def extract_image_metadata(img):
    info = img.info
    if not info:
        return None
        
    data = parse_a1111_metadata(info)
    if data and (data["raw_prompt"] or data["loras"]):
        return data
        
    data = parse_comfyui_metadata(info)
    if data and (data["raw_prompt"] or data["loras"]):
        return data
        
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
            
            cursor.execute('SELECT * FROM loras WHERE filename LIKE ? OR filepath LIKE ?', (f"%{name}%.safetensors", f"%{name}%"))
            results = cursor.fetchall()
            
            if results:
                row = dict(results[0])
                matched.append({
                    "original_name": name,
                    "weight": weight,
                    "filename": row["filename"],
                    "trigger_words": row["trigger_words"]
                })
            else:
                matched.append({
                    "original_name": name,
                    "weight": weight,
                    "filename": None,
                    "trigger_words": None
                })
                
    return matched

def reconstruct_prompt(raw_prompt, matched_loras):
    """
    Takes the pure positive prompt and appends the fully qualified lora tags
    and their associated trigger words.
    """
    clean_prompt = re.sub(r'<lora:[^>]+>', '', raw_prompt).strip()
    
    additions = []
    
    for lora in matched_loras:
        if lora["filename"]:
            basename = lora["filename"].replace(".safetensors", "")
            tag = f"<lora:{basename}:{lora['weight']}>"
            additions.append(tag)
            
            if lora["trigger_words"]:
                additions.append(lora["trigger_words"])
        else:
            tag = f"<lora:{lora['original_name']}:{lora['weight']}>"
            additions.append(tag)
            
    if additions:
        return clean_prompt + ", " + ", ".join(additions)
    return clean_prompt

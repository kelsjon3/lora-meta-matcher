import os
import requests
import re
import urllib.parse
import logging
import gradio as gr
from modules import script_callbacks, shared

logger = logging.getLogger(__name__)

from lora_meta_matcher.db import init_db, get_lora_by_hash, upsert_lora
from lora_meta_matcher.scanner import scan_directory
from lora_meta_matcher.hashing import process_missing_hashes, calculate_sha256, get_autov2_hash
from lora_meta_matcher.civitai import process_missing_civitai_metadata, fetch_civitai_version_info, get_api_logs, _extract_api_metadata, _safe_update_civitai_info, search_civitai_model_by_name, fetch_civitai_info
from lora_meta_matcher.parser import extract_image_metadata, match_loras_to_db, reconstruct_prompt

init_db()

import json
STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui_state.json")

def get_include_triggers_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("include_triggers", False)
    except:
        return False

def save_include_triggers_state(value):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"include_triggers": value}, f)
    except:
        pass

def download_missing_lora(vid, subfolder):
    if not vid:
        yield "Error: No version ID provided."
        return

    directory = getattr(shared.opts, "lora_meta_matcher_scan_dir", "")
    if not directory:
        yield "Error: Please set 'Default Lora Directory for Scanning' in Settings first."
        return
        
    if subfolder and str(subfolder).strip() != "":
        clean_subfolder = str(subfolder).replace('..', '').strip('/\\')
        if clean_subfolder:
            directory = os.path.join(directory, clean_subfolder)
            
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            yield f"Error: Could not create directory {directory}: {e}"
            return

    yield "Downloading from CivitAI..."
    url = f"https://civitai.com/api/download/models/{vid}"
    token = getattr(shared.opts, "civitai_api_token", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    try:
        res = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=20)
        res.raise_for_status()
    except Exception as e:
        yield f"Error downloading: {e}"
        return
        
    cd = res.headers.get("content-disposition", "")
    fname = "unknown.safetensors"
    if cd:
        found = re.findall('filename="([^"]+)"', cd)
        if not found:
            found = re.findall("filename\\*=[^']+'[^']*'(.+)", cd)
            if found:
                fname = urllib.parse.unquote(found[0])
            else:
                found = re.findall("filename=([^;]+)", cd)
                if found:
                    fname = found[0]
        else:
            fname = found[0]
    else:
        fname = f"version_{vid}.safetensors"
        
    filepath = os.path.join(directory, fname)
    yield f"Saving to {fname}..."
    try:
        with open(filepath, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1048576):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        yield f"Error saving file: {e}"
        return
        
    yield "Calculating hashes..."
    sha256_hash = calculate_sha256(filepath)
    autov2_hash = get_autov2_hash(filepath)
    
    yield "Fetching metadata..."
    data_res, status = fetch_civitai_version_info(vid, token)
    
    parsed = None
    if data_res and status == 200:
        parsed = _extract_api_metadata(data_res)
        
    upsert_lora(
        filename=fname,
        filepath=filepath,
        autov2_hash=autov2_hash,
        sha256_hash=sha256_hash,
        trigger_words=parsed.get("trigger_words") if parsed else None,
        base_model=parsed.get("base_model") if parsed else None,
        civitai_version_id=vid,
        loraname=parsed.get("loraname") if parsed else None,
        metadata_fetch_attempted=1 if parsed else 0
    )
    
    if parsed:
        _safe_update_civitai_info(filepath, data_res, parsed)
        
    yield f"Done! Saved {fname} to library."

def on_ui_settings():
    section = ('lora_meta_matcher', "Lora Meta Matcher")
    shared.opts.add_option(
        "civitai_api_token",
        shared.OptionInfo(
            "",
            "CivitAI API Token (for downloading trigger words)",
            gr.Textbox,
            {"interactive": True},
            section=section
        )
    )
    shared.opts.add_option(
        "lora_meta_matcher_scan_dir",
        shared.OptionInfo(
            "",
            "Default Lora Directory for Scanning",
            gr.Textbox,
            {"interactive": True},
            section=section
        )
    )
def ui_tab():
    with gr.Blocks(analytics_enabled=False) as interface:
        class UIState:
            halt_hashing = False
            halt_api = False
        st = UIState()
        
        def run_halt():
            st.halt_hashing = True
            return "Halt requested...", "Halting hash calculation gracefully...", get_api_logs()
            
        def run_halt_api():
            st.halt_api = True
            return "Halt requested...", "Halting API fetches gracefully...", get_api_logs()
        
        def run_scan(directory):
            log = ""
            if not directory:
                directory = getattr(shared.opts, "lora_meta_matcher_scan_dir", "")
                
            if not directory:
                msg = "Error: Please enter a directory path here or in Settings."
                shared.log.error(f"[Lora Meta Matcher] {msg}")
                yield "Error", msg, get_api_logs()
                return
                
            shared.log.info(f"[Lora Meta Matcher] Starting directory scan: {directory}")
            for summary, msg in scan_directory(directory):
                shared.log.info(f"[Lora Meta Matcher] {msg}")
                log = log + "\n" + msg if log else msg
                yield summary, log, get_api_logs()
                
        def run_hashing():
            st.halt_hashing = False
            log = ""
            shared.log.info("[Lora Meta Matcher] Starting missing hashes calculation...")
            for summary, msg in process_missing_hashes(halt_check=lambda: st.halt_hashing):
                shared.log.info(f"[Lora Meta Matcher] {msg}")
                log = log + "\n" + msg if log else msg
                yield summary, log, get_api_logs()
                
        def run_api_fetch():
            st.halt_api = False
            log = ""
            token = getattr(shared.opts, "civitai_api_token", "")
            if not token:
                log = "Warning: No CivitAI API token found in Settings. Trying without token...\n"
                shared.log.warning(f"[Lora Meta Matcher] {log.strip()}")
                
            shared.log.info("[Lora Meta Matcher] Starting missing metadata fetch from CivitAI...")
            for summary, msg in process_missing_civitai_metadata(token=token, halt_check=lambda: st.halt_api):
                shared.log.info(f"[Lora Meta Matcher] {msg}")
                log = log + "\n" + msg if log else msg
                yield summary, log, get_api_logs()
                
        def analyze_image(img, include_triggers):
            if img is None:
                return "No image provided.", "", ""
            
            data = extract_image_metadata(img)
            if not data:
                return "Failed to extract metadata from image.", "", ""
                
            raw_prompt = data.get("raw_prompt", "")
            loras = data.get("loras", [])
            
            if not raw_prompt and not loras:
                 return "No prompt or lora data found.", "", ""
            
            matched = match_loras_to_db(loras)
            
            # Fetch missing info for Loras not found in DB or those missing a proper name
            token = getattr(shared.opts, "civitai_api_token", "")
            search_count = 0
            search_limit = 60
            delay_sec = 0.2 if token else 0.5
            
            for m in matched:
                is_unknown = m.get("original_name", "").startswith("UnknownLora_")
                missing_info = not m["filename"] or not m.get("base_model") or is_unknown
                
                if missing_info and m.get("civitai_version_id"):
                    vid = m["civitai_version_id"]
                    try:
                        data_res, status = fetch_civitai_version_info(vid, token, delay=delay_sec)
                        if data_res and status == 200:
                            if "baseModel" in data_res:
                                m["base_model"] = data_res["baseModel"]
                                
                            if is_unknown or m.get("original_name", "").startswith("urn:air:"):
                                m_name = data_res.get("model", {}).get("name")
                                v_name = data_res.get("name")
                                if m_name:
                                    m["loraname"] = f"{m_name} ({v_name})" if v_name else m_name
                                    
                            api_autov2 = None
                            api_autov3 = None
                            api_sha256 = None
                            if "files" in data_res and isinstance(data_res["files"], list):
                                for file_info in data_res["files"]:
                                    if "hashes" in file_info and isinstance(file_info["hashes"], dict):
                                        hashes = file_info["hashes"]
                                        api_autov2 = hashes.get("AutoV2")
                                        api_autov3 = hashes.get("AutoV3")
                                        api_sha256 = hashes.get("SHA256")
                                    if api_autov2 or api_sha256:
                                        break
                                        
                            fetched_hash = api_autov2 or api_autov3 or api_sha256
                            if fetched_hash:
                                m["autov2_hash"] = api_autov2
                                m["autov3_hash"] = api_autov3
                                m["sha256_hash"] = api_sha256
                                            
                            # Use fetched hash to see if the user already scanned the file locally
                            if fetched_hash and not m.get("filename"):
                                loc_matches = get_lora_by_hash(fetched_hash)
                                if loc_matches:
                                    loc = loc_matches[0]
                                    m["filename"] = loc["filename"]
                                    m["filepath"] = loc["filepath"]
                                    m["trigger_words"] = loc["trigger_words"]
                                    # Connect the local file to this Civitai ID so it instant-matches next time
                                    upsert_lora(
                                        filename=loc["filename"],
                                        filepath=loc["filepath"],
                                        civitai_version_id=vid,
                                        loraname=m.get("loraname"),
                                        autov2_hash=api_autov2 if api_autov2 else None,
                                        autov3_hash=api_autov3 if api_autov3 else None,
                                        sha256_hash=api_sha256 if api_sha256 else None
                                    )
                    except Exception as e:
                        print(f"Failed to dynamically fetch ID {vid}: {e}")
                if missing_info and not m.get("civitai_version_id"):
                    if search_count >= search_limit:
                        continue
                    search_count += 1
                    
                    matched_version = None
                    matched_item = None
                    hash_val = m.get("autov2_hash") or m.get("autov3_hash") or m.get("sha256_hash")
                    
                    # Try querying CivitAI by hash first if a hash was parsed
                    if hash_val:
                        try:
                            # Shorten hash to 10 chars if it's AutoV2/sha256 prefix
                            if len(hash_val) > 10:
                                hash_val = hash_val[:10]
                            data_res, status = fetch_civitai_info(hash_val, token, delay=delay_sec)
                            if data_res and status == 200:
                                matched_version = data_res
                                matched_item = data_res.get("model")
                        except Exception as e:
                            print(f"Failed to fetch CivitAI version by hash {hash_val}: {e}")
                            
                    # Fallback to name search if hash lookup didn't succeed
                    if not matched_version and m.get("original_name"):
                        lora_name = m["original_name"]
                        search_name = os.path.basename(lora_name).replace(".safetensors", "").replace(".pt", "")
                        try:
                            data_res, status = search_civitai_model_by_name(search_name, token, delay=delay_sec)
                            if data_res and status == 200:
                                # Search the returned list of items to select the best match
                                matched_item = None
                                # 1. Exact model name match (case-insensitive)
                                for item in data_res:
                                    if item.get("name", "").strip().lower() == search_name.lower():
                                        matched_item = item
                                        break
                                # 2. Try to match by any version filename (case-insensitive)
                                if not matched_item:
                                    for item in data_res:
                                        for v in item.get("modelVersions", []):
                                            for f in v.get("files", []):
                                                fn_no_ext = f.get("name", "").replace(".safetensors", "").replace(".pt", "")
                                                if fn_no_ext.lower() == search_name.lower():
                                                    matched_item = item
                                                    break
                                            if matched_item:
                                                break
                                        if matched_item:
                                            break
                                # 3. Fallback to the first returned item
                                if not matched_item and data_res:
                                    matched_item = data_res[0]
                                    
                                if matched_item:
                                    versions = matched_item.get("modelVersions", [])
                                    if versions:
                                        matched_version = None
                                        for v in versions:
                                            for f in v.get("files", []):
                                                fn_no_ext = f.get("name", "").replace(".safetensors", "").replace(".pt", "")
                                                if fn_no_ext.lower() == search_name.lower():
                                                    matched_version = v
                                                    break
                                            if matched_version:
                                                break
                                        if not matched_version:
                                            matched_version = versions[0]
                        except Exception as e:
                            print(f"Failed to dynamically search CivitAI for '{search_name}': {e}")
                            
                    # Process the matched version (from either hash or name)
                    if matched_version:
                        try:
                            vid = matched_version.get("id")
                            if vid:
                                m["civitai_version_id"] = vid
                                m["base_model"] = matched_version.get("baseModel") or m.get("base_model")
                                
                                m_name = matched_item.get("name") if matched_item else None
                                if not m_name and "model" in matched_version:
                                    m_name = matched_version["model"].get("name")
                                v_name = matched_version.get("name")
                                m["loraname"] = f"{m_name} ({v_name})" if (m_name and v_name) else (m_name or v_name)
                                
                                raw_words = matched_version.get("trainedWords")
                                if isinstance(raw_words, list) and raw_words:
                                    m["trigger_words"] = ", ".join([str(w).strip() for w in raw_words if w])
                                    
                                api_autov2 = None
                                api_autov3 = None
                                api_sha256 = None
                                for file_info in matched_version.get("files", []):
                                    if "hashes" in file_info and isinstance(file_info["hashes"], dict):
                                        hashes = file_info["hashes"]
                                        api_autov2 = hashes.get("AutoV2")
                                        api_autov3 = hashes.get("AutoV3")
                                        api_sha256 = hashes.get("SHA256")
                                    if api_autov2 or api_sha256:
                                        break
                                        
                                fetched_hash = api_autov2 or api_autov3 or api_sha256
                                if fetched_hash:
                                    m["autov2_hash"] = api_autov2
                                    m["autov3_hash"] = api_autov3
                                    m["sha256_hash"] = api_sha256
                                    
                                # Check if the user already has this file locally under a different name
                                if fetched_hash and not m.get("filename"):
                                    loc_matches = get_lora_by_hash(fetched_hash)
                                    if loc_matches:
                                        loc = loc_matches[0]
                                        m["filename"] = loc["filename"]
                                        m["filepath"] = loc["filepath"]
                                        m["trigger_words"] = loc["trigger_words"]
                                        if vid:
                                            upsert_lora(
                                                filename=loc["filename"],
                                                filepath=loc["filepath"],
                                                civitai_version_id=vid,
                                                loraname=m.get("loraname"),
                                                autov2_hash=api_autov2 if api_autov2 else None,
                                                autov3_hash=api_autov3 if api_autov3 else None,
                                                sha256_hash=api_sha256 if api_sha256 else None
                                            )
                        except Exception as e:
                            print(f"Failed to process matched CivitAI info: {e}")
            
            table_html = "<table style='width: 100%; text-align: left; border-collapse: collapse; margin-top: 10px;'>"
            table_html += "<tr><th style='border-bottom: 1px solid #ddd; padding: 8px;'>Saved</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Lora Name</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Lora Filename</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Subfolder</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Base Model</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Hash</th>"
            table_html += "<th style='border-bottom: 1px solid #ddd; padding: 8px;'>Download</th></tr>"
            
            for m in matched:
                saved = "✅" if m["filename"] else "❌"
                name = m.get("loraname") or m.get("original_name", "")
                filename = m.get("filename", "") or ""
                
                # Extract subfolder relative to Lora dir if possible, else just basename dir
                subfolder = ""
                if m.get("filepath"):
                    # Use directory name of the file
                    subfolder = os.path.basename(os.path.dirname(m["filepath"]))
                    
                base_model = m.get("base_model", "") or ""
                hash_val = m.get("autov2_hash") or m.get("autov3_hash") or m.get("sha256_hash") or ""
                
                download_link = ""
                vid = m.get("civitai_version_id")
                if vid and not m["filename"]:
                    subfolder_to_save = base_model.replace("'", "\\'") if base_model else ""
                        
                    onclick_js = (
                        f"const vidInput = document.querySelector('#lora_meta_matcher_download_vid textarea'); "
                        f"const subInput = document.querySelector('#lora_meta_matcher_download_subfolder textarea'); "
                        f"if(vidInput && subInput) {{ "
                        f"vidInput.value = '{vid}'; "
                        f"subInput.value = '{subfolder_to_save}'; "
                        f"vidInput.dispatchEvent(new Event('input', {{ bubbles: true }})); "
                        f"subInput.dispatchEvent(new Event('input', {{ bubbles: true }})); "
                        f"setTimeout(() => {{ const btn = document.querySelector('#lora_meta_matcher_download_btn'); if(btn) btn.click(); }}, 100); "
                        f"}}"
                    )
                    download_link = f"<button onclick=\"{onclick_js}\" style='color: #3b82f6; text-decoration: underline; background: none; border: none; cursor: pointer; padding: 0;'>Download to Library</button>"
                elif vid:
                    download_link = "Saved"
                else:
                    download_link = "Not Found"

                table_html += f"<tr>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{saved}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{name}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{filename}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{subfolder}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{base_model}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{hash_val[:10]}</td>"
                table_html += f"<td style='padding: 8px; border-bottom: 1px solid #eee;'>{download_link}</td>"
                table_html += f"</tr>"
                
            table_html += "</table>"
                    
            new_prompt = reconstruct_prompt(data, matched, include_triggers)
            
            return raw_prompt, new_prompt, table_html

        with gr.Tabs():
            with gr.TabItem("Image Analysis & Lora Matcher"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Analysis & Lora Matcher")
                        
                        with gr.Accordion("Extraction Results", open=True):
                            raw_prompt = gr.Textbox(label="Extracted Raw Metadata", lines=3, interactive=False)
                            
                        with gr.Accordion("Prompt Construction", open=True):
                            include_triggers = gr.Checkbox(label="Include Trigger Words", value=get_include_triggers_state(), elem_id="lora_meta_matcher_include_triggers")
                            include_triggers.do_not_save_to_config = True
                            matched_prompt = gr.Textbox(label="Matched Prompt", lines=4, interactive=False, show_copy_button=True, elem_id="lora_meta_matched_prompt")
                            send_to_txt2img = gr.Button("Send to txt2img", variant="primary")
                            
                    with gr.Column(scale=1):
                        # Use a fixed height for the image to avoid scrolling
                        image_upload = gr.Image(type="pil", label="Upload Image", elem_id="lora_meta_image_upload", height=600)

                with gr.Row():
                    gr.Markdown("### Parsed Loras")
                with gr.Row():
                    lora_table = gr.HTML(value="")
                    
                download_status = gr.Textbox(label="Download Status", lines=1, interactive=False, visible=False)

                with gr.Row(visible=False):
                    download_vid_input = gr.Textbox(elem_id="lora_meta_matcher_download_vid")
                    download_subfolder_input = gr.Textbox(elem_id="lora_meta_matcher_download_subfolder")
                    download_btn = gr.Button(elem_id="lora_meta_matcher_download_btn")

                image_upload.change(fn=analyze_image, inputs=[image_upload, include_triggers], outputs=[raw_prompt, matched_prompt, lora_table])
                include_triggers.change(fn=analyze_image, inputs=[image_upload, include_triggers], outputs=[raw_prompt, matched_prompt, lora_table])
                include_triggers.change(fn=save_include_triggers_state, inputs=[include_triggers], outputs=[])
                
                download_btn.click(
                    fn=lambda: gr.update(visible=True, value="Starting download..."),
                    inputs=[],
                    outputs=[download_status]
                ).then(
                    fn=download_missing_lora,
                    inputs=[download_vid_input, download_subfolder_input],
                    outputs=[download_status]
                ).then(
                    fn=analyze_image,
                    inputs=[image_upload, include_triggers],
                    outputs=[raw_prompt, matched_prompt, lora_table]
                )
                
                send_to_txt2img.click(
                    fn=None,
                    inputs=[matched_prompt],
                    outputs=[],
                    _js="""
                    function(prompt_text) {
                        try {
                            const txt2img_box = document.querySelector('#txt2img_prompt textarea');
                            if (txt2img_box && prompt_text) {
                                let current_val = txt2img_box.value;
                                if (current_val && !current_val.endsWith(' ')) {
                                    current_val += ' ';
                                }
                                if (current_val && !current_val.endsWith(',')) {
                                    current_val += ', ';
                                }
                                txt2img_box.value = current_val + prompt_text;
                                
                                // Dispatch input event to trigger Gradio internal state update
                                const event = new Event('input', { bubbles: true });
                                txt2img_box.dispatchEvent(event);
                            } else {
                                console.warn("Could not find txt2img_prompt textarea or prompt is empty.");
                            }
                        } catch (e) {
                            console.error("Error sending to txt2img:", e);
                        }
                        return [];
                    }
                    """
                )

            with gr.TabItem("Lora Database Manager"):
                with gr.Column():
                    gr.Markdown("### Lora Database Manager")
                    scan_dir_path = gr.Textbox(label="Directory to Scan", value="", placeholder="Leave blank to use Default from Settings, or override here", interactive=True)
                    
                    with gr.Row():
                        scan_btn = gr.Button("1. Scan Directory", variant="primary")
                        hash_btn = gr.Button("2. Calculate Missing Hashes (CPU Intensive)")
                        halt_hash_btn = gr.Button("Halt Hashing", variant="stop")
                    with gr.Row():
                        api_btn = gr.Button("3. Fetch Missing Metadata from CivitAI")
                        halt_api_btn = gr.Button("Halt API Fetches", variant="stop")
                        refresh_api_btn = gr.Button("Refresh API Logs")
                    
                    summary_log = gr.Textbox(label="Progress Summary", lines=1, interactive=False)
                    with gr.Row():
                        output_log = gr.Textbox(label="Detailed Log", lines=10, interactive=False)
                        api_log = gr.Textbox(label="API Transactions", lines=10, interactive=False, value="No API transactions recorded.")

                    scan_btn.click(fn=run_scan, inputs=[scan_dir_path], outputs=[summary_log, output_log, api_log])
                    hash_btn.click(fn=run_hashing, inputs=[], outputs=[summary_log, output_log, api_log])
                    halt_hash_btn.click(fn=run_halt, inputs=[], outputs=[summary_log, output_log, api_log])
                    
                    api_btn.click(fn=run_api_fetch, inputs=[], outputs=[summary_log, output_log, api_log])
                    halt_api_btn.click(fn=run_halt_api, inputs=[], outputs=[summary_log, output_log, api_log])
                    refresh_api_btn.click(fn=get_api_logs, inputs=[], outputs=[api_log])

    return [(interface, "Lora Meta Matcher", "lora_meta_matcher")]

script_callbacks.on_ui_tabs(ui_tab)
script_callbacks.on_ui_settings(on_ui_settings)

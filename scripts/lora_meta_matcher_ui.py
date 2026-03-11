import os
import gradio as gr
from modules import script_callbacks, shared

from lora_meta_matcher.db import init_db
from lora_meta_matcher.scanner import scan_directory
from lora_meta_matcher.hashing import process_missing_hashes
from lora_meta_matcher.civitai import process_missing_civitai_metadata
from lora_meta_matcher.parser import extract_image_metadata, match_loras_to_db, reconstruct_prompt

init_db()

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
        st = UIState()
        
        def run_halt():
            st.halt_hashing = True
            return "Halt requested...", "Halting hash calculation gracefully..."
        
        def run_scan(directory):
            log = ""
            if not directory:
                yield "Error", "Please enter a directory path."
                return
            for summary, msg in scan_directory(directory):
                log = msg + "\n" + log
                yield summary, log
                
        def run_hashing():
            st.halt_hashing = False
            log = ""
            for summary, msg in process_missing_hashes(halt_check=lambda: st.halt_hashing):
                log = msg + "\n" + log
                yield summary, log
                
        def run_api_fetch():
            log = ""
            token = getattr(shared.opts, "civitai_api_token", "")
            if not token:
                log = "Warning: No CivitAI API token found in Settings. Trying without token...\n"
                
            for summary, msg in process_missing_civitai_metadata(token=token):
                log = msg + "\n" + log
                yield summary, log
                
        def analyze_image(img):
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
                name = m.get("original_name", "")
                filename = m.get("filename", "") or ""
                
                # Extract subfolder relative to Lora dir if possible, else just basename dir
                subfolder = ""
                if m.get("filepath"):
                    # Use directory name of the file
                    subfolder = os.path.basename(os.path.dirname(m["filepath"]))
                    
                base_model = m.get("base_model", "") or ""
                hash_val = m.get("autov2_hash", "") or ""
                
                download_link = ""
                vid = m.get("civitai_version_id")
                if vid:
                    download_link = f"<a href='https://civitai.com/api/download/models/{vid}' target='_blank' style='color: #3b82f6; text-decoration: underline;'>Download</a>"
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
                    
            new_prompt = reconstruct_prompt(data, matched)
            
            return raw_prompt, new_prompt, table_html

        with gr.Tabs():
            with gr.TabItem("Image Analysis & Lora Matcher"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Analysis & Lora Matcher")
                        
                        with gr.Accordion("Extraction Results", open=True):
                            raw_prompt = gr.Textbox(label="Extracted Raw Metadata", lines=3, interactive=False)
                            
                        with gr.Accordion("Prompt Construction", open=True):
                            matched_prompt = gr.Textbox(label="Matched Prompt with Trigger Words", lines=4, interactive=False, show_copy_button=True)
                            
                    with gr.Column(scale=1):
                        # Use a fixed height for the image to avoid scrolling
                        image_upload = gr.Image(type="pil", label="Upload Image", elem_id="lora_meta_image_upload", height=600)

                with gr.Row():
                    gr.Markdown("### Parsed Loras")
                with gr.Row():
                    lora_table = gr.HTML(value="")

                image_upload.change(fn=analyze_image, inputs=[image_upload], outputs=[raw_prompt, matched_prompt, lora_table])

            with gr.TabItem("Lora Database Manager"):
                with gr.Column():
                    gr.Markdown("### Lora Database Manager")
                    default_dir = getattr(shared.opts, "lora_meta_matcher_scan_dir", "")
                    scan_dir_path = gr.Textbox(label="Directory to Scan", value=default_dir, placeholder="/path/to/loras", interactive=True)
                    
                    with gr.Row():
                        scan_btn = gr.Button("1. Scan Directory", variant="primary")
                        hash_btn = gr.Button("2. Calculate Missing Hashes (CPU Intensive)")
                        halt_hash_btn = gr.Button("Halt Hashing", variant="stop")
                        api_btn = gr.Button("3. Fetch Missing Metadata from CivitAI (API limit)")
                    
                    summary_log = gr.Textbox(label="Progress Summary", lines=1, interactive=False)
                    output_log = gr.Textbox(label="Detailed Log", lines=10, interactive=False)

                    scan_btn.click(fn=run_scan, inputs=[scan_dir_path], outputs=[summary_log, output_log])
                    hash_btn.click(fn=run_hashing, inputs=[], outputs=[summary_log, output_log])
                    halt_hash_btn.click(fn=run_halt, inputs=[], outputs=[summary_log, output_log])
                    api_btn.click(fn=run_api_fetch, inputs=[], outputs=[summary_log, output_log])

    return [(interface, "Lora Meta Matcher", "lora_meta_matcher")]

script_callbacks.on_ui_tabs(ui_tab)
script_callbacks.on_ui_settings(on_ui_settings)

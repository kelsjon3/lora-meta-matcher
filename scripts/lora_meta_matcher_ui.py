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
                log = msg + "\\n" + log
                yield summary, log
                
        def run_hashing():
            st.halt_hashing = False
            log = ""
            for summary, msg in process_missing_hashes(halt_check=lambda: st.halt_hashing):
                log = msg + "\\n" + log
                yield summary, log
                
        def run_api_fetch():
            log = ""
            token = getattr(shared.opts, "civitai_api_token", "")
            if not token:
                log = "Warning: No CivitAI API token found in Settings. Trying without token...\\n"
                
            for summary, msg in process_missing_civitai_metadata(token=token):
                log = msg + "\\n" + log
                yield summary, log
                
        def analyze_image(img):
            if img is None:
                return "No image provided.", "", "", ""
            
            data = extract_image_metadata(img)
            if not data:
                return "Failed to extract metadata from image.", "", "", ""
                
            raw_prompt = data.get("raw_prompt", "")
            loras = data.get("loras", [])
            
            if not raw_prompt and not loras:
                 return "No prompt or lora data found.", "", "", ""
            
            raw_loras_str = "\\n".join([f"{l['name']} (Weight: {l['weight']})" for l in loras])
            
            matched = match_loras_to_db(loras)
            
            matched_loras_str = ""
            for m in matched:
                if m["filename"]:
                    trigger = m["trigger_words"] if m["trigger_words"] else "None"
                    matched_loras_str += f"[DB MATCH] {m['filename']} (Weight: {m['weight']}) - Triggers: {trigger}\\n"
                else:
                    matched_loras_str += f"[NOT FOUND] {m['original_name']} (Weight: {m['weight']})\\n"
                    
            new_prompt = reconstruct_prompt(data, matched)
            
            return raw_prompt, raw_loras_str.strip(), new_prompt, matched_loras_str.strip()

        with gr.Tabs():
            with gr.TabItem("Image Analysis & Lora Matcher"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Analysis & Lora Matcher")
                        
                        with gr.Accordion("Extraction Results", open=True):
                            raw_prompt = gr.Textbox(label="Extracted Raw Prompt", lines=3, interactive=False)
                            raw_loras = gr.Textbox(label="Detected Loras (Raw)", lines=2, interactive=False)
                            
                        with gr.Accordion("Matched Results", open=True):
                            matched_prompt = gr.Textbox(label="Matched Prompt with Trigger Words", lines=4, interactive=False, show_copy_button=True)
                            matched_loras = gr.Textbox(label="Matched Local Loras", lines=3, interactive=False)
                            
                    with gr.Column(scale=1):
                        # Use a fixed height for the image to avoid scrolling
                        image_upload = gr.Image(type="pil", label="Upload Image", elem_id="lora_meta_image_upload", height=600)

                image_upload.change(fn=analyze_image, inputs=[image_upload], outputs=[raw_prompt, raw_loras, matched_prompt, matched_loras])

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

# Lora Meta Matcher

A [Stable Diffusion Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) extension that extracts LoRA metadata from AI-generated images, matches those LoRAs against a local SQLite database, and fills in missing details from the [CivitAI API](https://github.com/civitai/civitai/wiki/REST-API-Reference).

Drop an image into the UI and the extension tells you which LoRAs were used, whether you already have them on disk, and where to download anything that is missing.

## Features

- **Image metadata extraction** — Reads embedded metadata from PNG/JPEG images produced by Automatic1111/Forge or ComfyUI, including:
  - `<lora:name:weight>` prompt tags
  - `Lora hashes` and `Hashes` fields
  - CivitAI `resources` JSON blocks
  - ComfyUI workflow/prompt JSON (including Power Lora Loader nodes)
  - JPEG EXIF `UserComment` when standard PNG info is absent
  - CivitAI URN identifiers (`urn:air:...civitai:...@versionId`)

- **Local LoRA matching** — Cross-references extracted LoRAs against a SQLite database using, in order:
  1. CivitAI model version ID
  2. AutoV2 / AutoV3 / SHA256 hash
  3. Filename/path fuzzy match

- **Prompt reconstruction** — Builds Forge-ready `<lora:filename:weight>` tags using your local filenames, optionally appending trigger words from the database.

- **LoRA database manager** — Scans your LoRA directory, imports sidecar metadata (`.civitai.info`, `.json`), computes missing file hashes, and fetches trigger words, base model, and CivitAI IDs from the API.

- **CivitAI integration** — Rate-limited, retry-aware API client with hash-based and version-ID lookups. Writes or updates `.civitai.info` sidecar files next to your LoRA files.

- **Download links** — For LoRAs not found locally, provides direct CivitAI download URLs when a version ID is known.

## Installation

1. Clone this repository into your Forge extensions folder:

```bash
cd /path/to/stable-diffusion-webui-forge/extensions
git clone https://github.com/kelsjon3/lora-meta-matcher.git lora-meta-matcher
```

2. Restart Forge (or reload UI).

3. Open the **Lora Meta Matcher** tab in the web UI.

The extension creates `loras.db` in the extension root on first run. This file is local to your machine and is listed in `.gitignore`.

## Configuration

Open **Settings → Lora Meta Matcher**:

| Setting | Description |
|---------|-------------|
| **CivitAI API Token** | Optional Bearer token for CivitAI API requests. Improves rate limits and is recommended for bulk metadata fetching. Create one at [civitai.com/user/account](https://civitai.com/user/account). |
| **Default Lora Directory for Scanning** | Default path used by the database scanner when no override is entered in the UI. Point this at your Forge LoRA folder (for example `models/Lora`). |

## Usage

### Image Analysis & LoRA Matcher

1. Open the **Lora Meta Matcher** tab.
2. Upload a generated image (PNG or JPEG with embedded metadata).
3. Review the results:
   - **Extracted Raw Metadata** — The full metadata string or JSON from the image.
   - **Parsed Loras** — A table showing whether each LoRA is saved locally (✅/❌), its name, filename, subfolder, base model, hash prefix, and a CivitAI download link when available.
   - **Matched Prompt** — Reconstructed `<lora:...>` tags (and optional trigger words) using your local filenames.
4. Toggle **Include Trigger Words** to add or remove trigger words from the matched prompt.
5. Click **Send to txt2img** to append the matched prompt to the txt2img prompt box.

When a LoRA is missing from the database but has a CivitAI version ID, the extension fetches metadata from CivitAI on the fly and attempts to link it to a locally scanned file by hash.

### LoRA Database Manager

Build and maintain the local database in three steps:

1. **Scan Directory** — Recursively finds `.safetensors` files, reads `.civitai.info` / `.civitai_info` / `.json` sidecars, and upserts records into `loras.db`. Skips files that are already fully populated.
2. **Calculate Missing Hashes** — Computes SHA256 / AutoV2 hashes for LoRAs without hash data. CPU-intensive; use **Halt Hashing** to stop gracefully.
3. **Fetch Missing Metadata from CivitAI** — Queries CivitAI by hash for LoRAs missing trigger words, base model, version ID, or display name. Use **Halt API Fetches** to stop gracefully.

Run these steps after adding new LoRAs to your collection, or before analyzing images for best match accuracy.

### Maintenance Script

If duplicate hash entries appear in the database (for example after a bad metadata import), run the cleanup utility from the extension root:

```bash
python scripts/cleanup_corrupted_metadata.py
```

This verifies shared hashes against actual file contents, removes incorrect `.civitai.info` sidecars, and clears corrupted database rows so you can re-scan.

## Project Structure

```
lora-meta-matcher/
├── scripts/
│   ├── lora_meta_matcher_ui.py   # Forge UI entry point (Gradio tab + settings)
│   └── cleanup_corrupted_metadata.py
├── lora_meta_matcher/
│   ├── parser.py                 # Image metadata parsing and DB matching
│   ├── scanner.py                # Directory scan and sidecar import
│   ├── db.py                     # SQLite schema and queries
│   ├── hashing.py                # SHA256 / AutoV2 hash calculation
│   └── civitai.py                # CivitAI API client
├── tests/
│   └── test_civitai_flow.py
├── loras.db                      # Local database (created at runtime)
└── README.md
```

## Database Schema

The `loras` table stores one row per `.safetensors` file:

| Column | Description |
|--------|-------------|
| `filename` | Base filename (e.g. `my_lora.safetensors`) |
| `filepath` | Absolute path (unique key) |
| `autov2_hash` | CivitAI AutoV2 hash (10-character prefix) |
| `autov3_hash` | CivitAI AutoV3 hash |
| `sha256_hash` | Full SHA256 of the file |
| `trigger_words` | Comma-separated trained words |
| `base_model` | e.g. `SD 1.5`, `SDXL 1.0` |
| `loraname` | Human-readable CivitAI model/version name |
| `civitai_version_id` | CivitAI model version ID |
| `metadata_fetch_attempted` | Whether a CivitAI API fetch was attempted |
| `metadata_fetch_hash_signature` | Hash set used for the last API attempt |

## API Reference

The extension can also be used programmatically from Python scripts run inside the Forge environment, or by calling the same CivitAI endpoints directly.

### Python

All modules live under `lora_meta_matcher`. Forge adds the extension root to `sys.path` when the UI script loads.

#### Extract metadata from an image

```python
from PIL import Image
from lora_meta_matcher.parser import extract_image_metadata, match_loras_to_db, reconstruct_prompt

img = Image.open("/path/to/generated.png")
data = extract_image_metadata(img)
# data = {"raw_prompt": "...", "loras": [...], "positive_prompt": "..."}

matched = match_loras_to_db(data["loras"])
prompt = reconstruct_prompt(data, matched, include_triggers=True)
print(prompt)
# <lora:local_filename:0.8>, trigger_word_1, trigger_word_2
```

#### Scan a LoRA directory

```python
from lora_meta_matcher.scanner import scan_directory

for summary, log_line in scan_directory("/path/to/models/Lora"):
    print(summary, log_line)
```

#### Calculate missing hashes

```python
from lora_meta_matcher.hashing import process_missing_hashes

for summary, log_line in process_missing_hashes():
    print(summary, log_line)
```

#### Fetch CivitAI metadata for indexed LoRAs

```python
from lora_meta_matcher.civitai import process_missing_civitai_metadata

token = "your_civitai_api_token"  # optional
for summary, log_line in process_missing_civitai_metadata(token=token):
    print(summary, log_line)
```

#### Look up a LoRA by hash or path

```python
from lora_meta_matcher.db import get_lora_by_hash, get_lora_by_path, get_stats

matches = get_lora_by_hash("ABC123DEF4")
row = get_lora_by_path("/path/to/models/Lora/my_lora.safetensors")
stats = get_stats()  # {"total": N, "hashed": N, "with_triggers": N}
```

#### Fetch a CivitAI model version by ID

```python
from lora_meta_matcher.civitai import fetch_civitai_version_info, fetch_civitai_info

data, status = fetch_civitai_version_info(244808, token="your_token")
data, status = fetch_civitai_info("ABC123DEF456", token="your_token")
```

### JavaScript (Forge UI)

The **Send to txt2img** button uses a small client-side hook to append the matched prompt to the txt2img textarea and dispatch an `input` event so Gradio picks up the change:

```javascript
function(prompt_text) {
    const txt2img_box = document.querySelector('#txt2img_prompt textarea');
    if (txt2img_box && prompt_text) {
        let current_val = txt2img_box.value;
        if (current_val && !current_val.endsWith(' ')) current_val += ' ';
        if (current_val && !current_val.endsWith(',')) current_val += ', ';
        txt2img_box.value = current_val + prompt_text;
        txt2img_box.dispatchEvent(new Event('input', { bubbles: true }));
    }
    return [];
}
```

### CivitAI REST API (curl)

The extension uses these CivitAI endpoints. You can call them directly for debugging or external tooling.

**Look up a model version by file hash:**

```bash
curl -s "https://civitai.com/api/v1/model-versions/by-hash/ABC123DEF456" \
  -H "Authorization: Bearer YOUR_CIVITAI_TOKEN"
```

**Look up a model version by ID:**

```bash
curl -s "https://civitai.com/api/v1/model-versions/244808" \
  -H "Authorization: Bearer YOUR_CIVITAI_TOKEN"
```

**Download a model version** (link shown in the UI when a version ID is known):

```bash
curl -L -o my_lora.safetensors \
  "https://civitai.com/api/download/models/244808" \
  -H "Authorization: Bearer YOUR_CIVITAI_TOKEN"
```

Place downloaded `.safetensors` files in your Forge LoRA directory, then run **Scan Directory** so the extension indexes them.

## Supported Metadata Formats

| Source | Format |
|--------|--------|
| Forge / A1111 | `parameters` PNG text chunk with `<lora:...>` tags, `Lora hashes`, `Hashes`, and `Civitai resources` |
| ComfyUI | `prompt` or `workflow` JSON with LoRA loader nodes |
| JPEG EXIF | `UserComment` (UNICODE/ASCII) containing A1111 or ComfyUI data |
| Sidecar files | `.civitai.info`, `.civitai_info`, `.metadata.json`, `.json` next to LoRA files |

## Testing

```bash
cd /path/to/lora-meta-matcher
python -m unittest tests/test_civitai_flow.py
```

## Requirements

- Stable Diffusion Forge (Gradio UI, `modules.script_callbacks`)
- Python packages available in the Forge environment: `gradio`, `Pillow`, `requests`
- Network access for CivitAI API calls (optional but recommended)

## License

See the repository for license information.

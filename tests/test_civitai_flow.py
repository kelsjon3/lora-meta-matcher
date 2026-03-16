import json
import os
import tempfile
import unittest

from lora_meta_matcher.civitai import (
    _build_hash_candidates,
    _extract_api_metadata,
    _safe_update_civitai_info,
)


class TestCivitaiFlowHelpers(unittest.TestCase):
    def test_build_hash_candidates_prioritized_and_deduped(self):
        candidates = _build_hash_candidates(
            autov2_hash="abcd1234ef56",
            autov3_hash="f" * 64,
            sha256_hash="f" * 64,  # duplicate of autov3 on purpose
        )
        self.assertEqual(
            candidates,
            [
                "F" * 64,
                "ABCD1234EF56",
            ],
        )

    def test_build_hash_candidates_derives_short_from_sha256(self):
        candidates = _build_hash_candidates(
            autov2_hash=None,
            autov3_hash=None,
            sha256_hash="0123456789abcdef" * 4,
        )
        self.assertEqual(candidates[0], ("0123456789ABCDEF" * 4))
        self.assertEqual(candidates[1], "0123456789AB")

    def test_extract_api_metadata_parses_expected_fields(self):
        payload = {
            "id": "12345",
            "baseModel": "SDXL 1.0",
            "trainedWords": ["style_tag", "", "  hero_face  ", 123],
            "name": "v1.2",
            "model": {"name": "My LoRA"},
            "files": [
                {"hashes": {"AutoV2": "abc123def456", "AutoV3": "b" * 64, "SHA256": "a" * 64}}
            ],
        }
        parsed = _extract_api_metadata(payload)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["civitai_version_id"], 12345)
        self.assertEqual(parsed["base_model"], "SDXL 1.0")
        self.assertEqual(parsed["trigger_words"], "style_tag, hero_face")
        self.assertEqual(parsed["loraname"], "My LoRA (v1.2)")
        self.assertEqual(parsed["autov2_hash"], "ABC123DEF456")
        self.assertEqual(parsed["autov3_hash"], "B" * 64)
        self.assertEqual(parsed["sha256_hash"], "A" * 64)

    def test_safe_update_civitai_info_preserves_existing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "example.safetensors")
            info_path = os.path.join(tmpdir, "example.civitai.info")
            with open(model_path, "wb") as f:
                f.write(b"")

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "baseModel": "Existing Base",
                        "trainedWords": ["keep_me"],
                        "files": [{"hashes": {"AutoV2": "OLDHASH"}}],
                    },
                    f,
                    indent=2,
                )

            parsed_metadata = {
                "autov2_hash": "NEWHASH000001",
                "autov3_hash": None,
                "sha256_hash": "A" * 64,
                "trigger_words": "new_word",
                "base_model": "New Base",
                "civitai_version_id": 99,
                "loraname": "Some Name",
            }
            ok = _safe_update_civitai_info(model_path, {"model": {"name": "Model Name"}}, parsed_metadata)
            self.assertTrue(ok)

            with open(info_path, "r", encoding="utf-8") as f:
                merged = json.load(f)

            # Existing values stay, missing values are filled.
            self.assertEqual(merged.get("baseModel"), "Existing Base")
            self.assertEqual(merged.get("trainedWords"), ["keep_me"])
            self.assertEqual(merged.get("id"), 99)
            self.assertEqual(merged.get("name"), "Some Name")
            self.assertEqual(merged.get("sha256"), "A" * 64)
            self.assertEqual(merged["files"][0]["hashes"]["AutoV2"], "OLDHASH")
            self.assertEqual(merged["files"][0]["hashes"]["SHA256"], "A" * 64)


if __name__ == "__main__":
    unittest.main()

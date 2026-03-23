import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_meta_matcher.db import get_connection
from lora_meta_matcher.hashing import get_autov2_hash, get_short_hash

def cleanup_corrupted():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find hashes shared by more than 1 file
        cursor.execute("SELECT autov2_hash, COUNT(*) as c FROM loras GROUP BY autov2_hash HAVING c > 1")
        shared_hashes = [row["autov2_hash"] for row in cursor.fetchall() if row["autov2_hash"]]
        
        print(f"Found {len(shared_hashes)} duplicated hashes to verify. This may take a moment as it requires hashing the files...")
        
        corrupted_count = 0
        
        for h in shared_hashes:
            cursor.execute("SELECT id, filename, filepath, autov2_hash FROM loras WHERE autov2_hash = ?", (h,))
            files = cursor.fetchall()
            
            for f in files:
                filepath = f["filepath"]
                if not os.path.exists(filepath):
                    continue
                    
                print(f"Verifying {f['filename']}...")
                true_sha256 = get_autov2_hash(filepath)
                if not true_sha256:
                    continue
                    
                true_short = get_short_hash(true_sha256)
                
                # If the actual hash doesn't match the database hash, the DB and its .civitai.info are corrupted
                if true_short != h:
                    print(f"  [CORRUPT] Found mismatched hash for {f['filename']} (Real: {true_short}, DB: {h})")
                    
                    # 1. Delete the incorrect .civitai.info file
                    base_name = os.path.splitext(os.path.basename(filepath))[0]
                    dir_name = os.path.dirname(filepath)
                    info_path1 = os.path.join(dir_name, f"{base_name}.civitai.info")
                    info_path2 = os.path.join(dir_name, f"{base_name}.civitai_info")
                    
                    if os.path.exists(info_path1):
                        os.remove(info_path1)
                        print(f"    Deleted {info_path1}")
                    if os.path.exists(info_path2):
                        os.remove(info_path2)
                        print(f"    Deleted {info_path2}")
                        
                    # 2. Clear the corrupted data from the DB so the file can be cleanly re-processed
                    cursor.execute('''
                        UPDATE loras SET 
                        autov2_hash = NULL, 
                        autov3_hash = NULL,
                        sha256_hash = NULL, 
                        base_model = NULL, 
                        civitai_version_id = NULL, 
                        loraname = NULL, 
                        trigger_words = NULL,
                        metadata_fetch_attempted = 0
                        WHERE id = ?
                    ''', (f["id"],))
                    conn.commit()
                    corrupted_count += 1
                    print(f"    Cleared database row for {f['filename']}")
                    
        print(f"\nCleanup complete. Fixed {corrupted_count} corrupted entries.")
        print("Please run 'Scan Directory' and 'Calculate Missing Hashes' in the UI to regenerate the correct metadata.")

if __name__ == "__main__":
    cleanup_corrupted()

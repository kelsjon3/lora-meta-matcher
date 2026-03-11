import hashlib
import os
from .db import get_loras_without_hash, upsert_lora

def calculate_sha256(filepath, halt_check=None):
    """Calculates the full SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in blocks of 1MB for acceptable performance and low memory overhead
        while True:
            if halt_check and halt_check():
                return None
            byte_block = f.read(1048576)
            if not byte_block:
                break
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def get_autov2_hash(filepath, halt_check=None):
    """
    AutoV2 hash for Loras is typically the first 12 characters of the 
    SHA256 of the model file. However, for full compatibility with CivitAI API,
    it's best to use the full 64-character hash. We will return the full hash.
    """
    return calculate_sha256(filepath, halt_check)

def get_short_hash(full_hash):
    """Returns the 12-character AutoV2 hash used by A1111/Forge."""
    if full_hash and len(full_hash) >= 12:
        return full_hash[:12]
    return full_hash

def process_missing_hashes(halt_check=None):
    """
    Generator that calculates missing hashes for loras in the db.
    Yields (summary_text, log_text) tuples.
    """
    filepaths = get_loras_without_hash()
    total_files = len(filepaths)
    
    if not total_files:
        yield "All Loras are already hashed.", "Check complete: No missing hashes found."
        return
        
    yield f"Preparing to hash {total_files} files...", f"Found {total_files} Loras missing an AutoV2 Hash. Starting hash calculation..."
    
    for count, filepath in enumerate(filepaths):
        if halt_check and halt_check():
            yield "Hashing halted.", f"Halted after processing {count} files."
            break
            
        filename = os.path.basename(filepath)
        msg_log = f"({count+1}/{total_files}) Calculating hash for '{filename}'..."
        msg_sum = f"Processed {count+1} / {total_files} files ({int(((count+1)/total_files)*100)}%)"
        
        yield msg_sum, msg_log
        
        autov2_hash = get_autov2_hash(filepath, halt_check)
        if not autov2_hash and halt_check and halt_check():
            yield "Hashing halted.", f"Halted during hashing of '{filename}'."
            break
            
        if autov2_hash:
            upsert_lora(filename=filename, filepath=filepath, autov2_hash=autov2_hash)
            
    if halt_check and halt_check():
        return
        
    yield f"Hash calculation complete. Processed {total_files} files.", f"Finished calculating all {total_files} missing hashes."

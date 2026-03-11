import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "loras.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT UNIQUE NOT NULL,
                autov2_hash TEXT,
                trigger_words TEXT,
                base_model TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add column for tracking API fetches
        try:
            cursor.execute("ALTER TABLE loras ADD COLUMN metadata_fetch_attempted INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filepath ON loras(filepath)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON loras(autov2_hash)')
        conn.commit()

def upsert_lora(filename, filepath, autov2_hash=None, trigger_words=None, base_model=None, metadata_fetch_attempted=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO loras (filename, filepath, autov2_hash, trigger_words, base_model, metadata_fetch_attempted, updated_at)
            VALUES (?, ?, ?, ?, ?, COALESCE(?, 0), CURRENT_TIMESTAMP)
            ON CONFLICT(filepath) DO UPDATE SET
                filename=excluded.filename,
                autov2_hash=COALESCE(excluded.autov2_hash, loras.autov2_hash),
                trigger_words=COALESCE(excluded.trigger_words, loras.trigger_words),
                base_model=COALESCE(excluded.base_model, loras.base_model),
                metadata_fetch_attempted=COALESCE(excluded.metadata_fetch_attempted, loras.metadata_fetch_attempted),
                updated_at=CURRENT_TIMESTAMP
        ''', (filename, filepath, autov2_hash, trigger_words, base_model, metadata_fetch_attempted))
        conn.commit()

def get_lora_by_path(filepath):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM loras WHERE filepath=?', (filepath,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_lora_by_hash(autov2_hash):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM loras WHERE autov2_hash=? OR autov2_hash LIKE ?', (autov2_hash, f"{autov2_hash}%"))
        results = cursor.fetchall()
        return [dict(r) for r in results]
        
def get_loras_without_hash():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT filepath FROM loras WHERE autov2_hash IS NULL OR autov2_hash = ""')
        return [r[0] for r in cursor.fetchall()]

def get_loras_without_triggers_but_have_hash():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT filepath, autov2_hash FROM loras WHERE (trigger_words IS NULL OR trigger_words = "") AND (autov2_hash IS NOT NULL AND autov2_hash != "") AND metadata_fetch_attempted = 0')
        results = cursor.fetchall()
        return [dict(r) for r in results]

def get_stats():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM loras')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM loras WHERE autov2_hash IS NOT NULL AND autov2_hash != ""')
        hashed = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM loras WHERE trigger_words IS NOT NULL AND trigger_words != ""')
        with_triggers = cursor.fetchone()[0]
        return {"total": total, "hashed": hashed, "with_triggers": with_triggers}

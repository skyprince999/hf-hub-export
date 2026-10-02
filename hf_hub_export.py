import os
import json
import sqlite3
import time
from typing import Generator, Any
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Configuration
DB_NAME = "huggingface_hub.db"
JSONL_PREFIX = "hf_export"
BATCH_SIZE = 500  # Commit transactions every N records
MAX_RETRIES = 5

HF_TOKEN = os.getenv("HF_TOKEN", None)
api = HfApi(token=HF_TOKEN)


def init_sqlite_db(db_path: str) -> sqlite3.Connection:
    """Initializes SQLite tables with indexing for efficient querying."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Pragmas for fast bulk ingestion
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA journal_mode = WAL;")

    entities = ["models", "datasets", "spaces"]
    for entity in entities:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {entity} (
                id TEXT PRIMARY KEY,
                author TEXT,
                likes INTEGER,
                downloads INTEGER,
                created_at TEXT,
                last_modified TEXT,
                private INTEGER,
                tags TEXT,
                raw_json TEXT
            );
        """)
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{entity}_likes ON {entity}(likes);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{entity}_author ON {entity}(author);")

    conn.commit()
    return conn


def extract_record(item: Any, entity_type: str) -> tuple[dict, tuple]:
    """Normalizes HfApi Info objects into a structured dict and a DB row."""
    # Split repo id to extract author/org
    repo_id = item.id
    author = repo_id.split("/")[0] if "/" in repo_id else None

    # Handle slight schema variations across entities
    likes = getattr(item, "likes", 0) or 0
    downloads = getattr(item, "downloads", 0) or 0
    created_at = item.created_at.isoformat() if getattr(item, "created_at", None) else None
    last_modified = item.last_modified.isoformat() if getattr(item, "last_modified", None) else None
    private = 1 if getattr(item, "private", False) else 0
    tags = item.tags if hasattr(item, "tags") and item.tags else []

    record_dict = {
        "id": repo_id,
        "entity_type": entity_type,
        "author": author,
        "likes": likes,
        "downloads": downloads,
        "created_at": created_at,
        "last_modified": last_modified,
        "private": bool(private),
        "tags": tags,
    }

    # Add space-specific SDK info if available
    if entity_type == "spaces":
        record_dict["sdk"] = getattr(item, "sdk", None)

    db_row = (
        repo_id,
        author,
        likes,
        downloads,
        created_at,
        last_modified,
        private,
        json.dumps(tags),
        json.dumps(record_dict),
    )

    return record_dict, db_row


def robust_stream(generator: Generator) -> Generator:
    """Wraps the HfApi generator with exponential backoff on network/rate-limit hit."""
    backoff = 2
    for attempt in range(MAX_RETRIES):
        try:
            for item in generator:
                yield item
            return
        except (HfHubHTTPError, Exception) as exc:
            print(f"[Warning] Stream error: {exc}. Retrying in {backoff}s (Attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(backoff)
            backoff *= 2
    print("[Error] Max retries reached on stream.")


def sync_entity(entity_type: str, generator: Generator, conn: sqlite3.Connection):
    """Streams items into both SQLite and JSONL with batching."""
    jsonl_filename = f"{JSONL_PREFIX}_{entity_type}.jsonl"
    cursor = conn.cursor()

    insert_sql = f"""
        INSERT OR REPLACE INTO {entity_type} 
        (id, author, likes, downloads, created_at, last_modified, private, tags, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    print(f"\n--- Starting sync for: {entity_type.upper()} ---")
    print(f"Writing to: {DB_NAME} (Table: {entity_type}) & {jsonl_filename}")

    batch_rows = []
    count = 0
    start_time = time.time()

    with open(jsonl_filename, "a", encoding="utf-8") as jf:
        for item in robust_stream(generator):
            record_dict, db_row = extract_record(item, entity_type)

            # 1. Write to JSONL
            jf.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

            # 2. Add to SQLite batch
            batch_rows.append(db_row)
            count += 1

            if len(batch_rows) >= BATCH_SIZE:
                cursor.executemany(insert_sql, batch_rows)
                conn.commit()
                batch_rows.clear()
                elapsed = time.time() - start_time
                print(f"[{entity_type}] Synced {count:,} items ({count / elapsed:.1f} items/s)...", end="\r")

        # Flush remainder
        if batch_rows:
            cursor.executemany(insert_sql, batch_rows)
            conn.commit()

    total_time = max(time.time() - start_time, 0.1)
    print(f"\nCompleted {entity_type}: {count:,} records processed in {total_time:.2f}s ({count / total_time:.1f} records/s).")


def main():
    conn = init_sqlite_db(DB_NAME)

    try:
        # 1. Models Stream
        # full=False keeps payload lightweight; expand fields if you need specific config info
        sync_entity("models", api.list_models(full=False), conn)

        # 2. Datasets Stream
        sync_entity("datasets", api.list_datasets(full=False), conn)

        # 3. Spaces Stream
        sync_entity("spaces", api.list_spaces(), conn)

    finally:
        conn.close()
        print("\nAll tasks finished. SQLite database and JSONL files are ready.")


if __name__ == "__main__":
    main()

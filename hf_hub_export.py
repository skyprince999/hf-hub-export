"""Export metadata for every public model, dataset and Space on the Hugging Face Hub.

Pages through the Hub API in creation order (oldest first), which gives a stable
cursor: the crawl can be stopped at any time and re-running the script resumes
from the last committed page. Each page and its resume cursor are written to
SQLite in a single transaction, so rows are never lost or duplicated.

Once an entity type is fully synced, its table is exported to
hf_export_<entity>.jsonl (rewritten from SQLite, so re-runs never duplicate lines).
"""

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import requests

API_BASE = "https://huggingface.co/api"
ENTITIES = ["models", "datasets", "spaces"]
PAGE_SIZE = 1000  # Max page size accepted by the Hub API
MAX_RETRIES = 8  # Consecutive failures (network / 5xx) before giving up
REQUEST_TIMEOUT = 60

# Fields requested per entity via expand[]; the API returns only these (plus id/_id)
EXPAND_FIELDS = {
    "models": ["author", "likes", "downloads", "createdAt", "lastModified", "private",
               "gated", "disabled", "tags", "pipeline_tag", "library_name"],
    "datasets": ["author", "likes", "downloads", "createdAt", "lastModified", "private",
                 "gated", "disabled", "tags"],
    "spaces": ["author", "likes", "createdAt", "lastModified", "private", "sdk", "tags"],
}

HF_TOKEN = os.getenv("HF_TOKEN")


def init_sqlite_db(db_path: str) -> sqlite3.Connection:
    """Creates entity tables, indexes and the resume-state table."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA journal_mode = WAL;")

    for entity in ENTITIES:
        conn.execute(f"""
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
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{entity}_likes ON {entity}(likes);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{entity}_author ON {entity}(author);")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_state (
            entity TEXT PRIMARY KEY,
            next_url TEXT,
            done INTEGER NOT NULL DEFAULT 0,
            pages INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        );
    """)
    conn.commit()
    return conn


def first_page_url(entity: str) -> str:
    params = [f"limit={PAGE_SIZE}", "sort=createdAt", "direction=1"]
    params += [f"expand[]={field}" for field in EXPAND_FIELDS[entity]]
    return f"{API_BASE}/{entity}?" + "&".join(params)


def to_row(item: dict, entity: str) -> tuple:
    """Normalizes one API item into a DB row; raw_json keeps the full item."""
    repo_id = item["id"]
    author = item.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None)
    tags = item.get("tags") or []
    record = {k: v for k, v in item.items() if k != "_id"}
    record["entity_type"] = entity
    record["author"] = author
    return (
        repo_id,
        author,
        item.get("likes") or 0,
        item.get("downloads") or 0,
        item.get("createdAt"),
        item.get("lastModified"),
        1 if item.get("private") else 0,
        json.dumps(tags),
        json.dumps(record, ensure_ascii=False),
    )


def rate_limit_wait(response: requests.Response) -> float:
    """Seconds to wait from Retry-After or the RateLimit header ("api";r=..;t=..)."""
    retry_after = response.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return float(retry_after) + 1
    match = re.search(r"t=(\d+)", response.headers.get("RateLimit", ""))
    return float(match.group(1)) + 1 if match else 60.0


def fetch_page(session: requests.Session, url: str) -> requests.Response:
    """GETs one page. Waits out rate limits indefinitely; retries other failures
    with exponential backoff and raises after MAX_RETRIES consecutive failures."""
    failures = 0
    while True:
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            reason = f"network error: {exc}"
        else:
            if response.status_code == 200:
                # Proactively pause when the rate-limit window is nearly spent
                match = re.search(r"r=(\d+)", response.headers.get("RateLimit", ""))
                if match and int(match.group(1)) <= 2:
                    wait = rate_limit_wait(response)
                    print(f"\n[rate limit] quota nearly spent, pausing {wait:.0f}s")
                    time.sleep(wait)
                return response
            if response.status_code == 429:
                wait = rate_limit_wait(response)
                print(f"\n[rate limit] 429 received, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if response.status_code < 500:
                response.raise_for_status()  # 4xx other than 429 is not retryable
            reason = f"HTTP {response.status_code}"

        failures += 1
        if failures > MAX_RETRIES:
            raise RuntimeError(f"Giving up after {MAX_RETRIES} retries ({reason}) on {url}")
        backoff = min(2 ** failures, 300)
        print(f"\n[retry] {reason}; retry {failures}/{MAX_RETRIES} in {backoff}s")
        time.sleep(backoff)


def sync_entity(entity: str, conn: sqlite3.Connection, session: requests.Session):
    """Pages through one entity type, resuming from the saved cursor if any."""
    state = conn.execute(
        "SELECT next_url, done, pages FROM sync_state WHERE entity = ?", (entity,)
    ).fetchone()
    if state and state[1]:
        print(f"[{entity}] already complete, skipping (use --restart to re-crawl)")
        return
    url, pages = (state[0], state[2]) if state else (first_page_url(entity), 0)
    if state:
        print(f"[{entity}] resuming after page {pages:,}")

    insert_sql = f"""
        INSERT OR REPLACE INTO {entity}
        (id, author, likes, downloads, created_at, last_modified, private, tags, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    state_sql = """
        INSERT INTO sync_state (entity, next_url, done, pages, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entity) DO UPDATE SET
            next_url = excluded.next_url, done = excluded.done,
            pages = excluded.pages, updated_at = excluded.updated_at
    """

    start, fetched = time.time(), 0
    while url:
        response = fetch_page(session, url)
        items = response.json()
        next_url = response.links.get("next", {}).get("url")
        pages += 1
        fetched += len(items)

        with conn:  # page rows and resume cursor commit atomically
            conn.executemany(insert_sql, [to_row(item, entity) for item in items])
            conn.execute(state_sql, (entity, next_url, 0 if next_url else 1, pages,
                                     datetime.now(timezone.utc).isoformat()))

        total = conn.execute(f"SELECT COUNT(*) FROM {entity}").fetchone()[0]
        rate = fetched / max(time.time() - start, 0.1)
        print(f"[{entity}] page {pages:,} | {total:,} rows | {rate:,.0f} items/s", end="\r")
        url = next_url

    total = conn.execute(f"SELECT COUNT(*) FROM {entity}").fetchone()[0]
    print(f"\n[{entity}] complete: {total:,} rows in {time.time() - start:,.0f}s")


def export_jsonl(entity: str, conn: sqlite3.Connection, out_dir: str):
    """Rewrites hf_export_<entity>.jsonl from SQLite (atomic replace)."""
    path = os.path.join(out_dir, f"hf_export_{entity}.jsonl")
    tmp_path = path + ".tmp"
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for (raw_json,) in conn.execute(f"SELECT raw_json FROM {entity} ORDER BY created_at, id"):
            f.write(raw_json + "\n")
            count += 1
    os.replace(tmp_path, path)
    print(f"[{entity}] exported {count:,} records to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="huggingface_hub.db", help="SQLite database path")
    parser.add_argument("--out-dir", default=".", help="Directory for JSONL exports")
    parser.add_argument("--entities", nargs="+", choices=ENTITIES, default=ENTITIES)
    parser.add_argument("--restart", action="store_true",
                        help="Discard saved progress and re-crawl the selected entities")
    args = parser.parse_args()

    conn = init_sqlite_db(args.db)
    session = requests.Session()
    if HF_TOKEN:
        session.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    print(f"Authenticated: {'yes' if HF_TOKEN else 'no (set HF_TOKEN for higher rate limits)'}")

    try:
        for entity in args.entities:
            if args.restart:
                with conn:
                    conn.execute("DELETE FROM sync_state WHERE entity = ?", (entity,))
            sync_entity(entity, conn, session)
            export_jsonl(entity, conn, args.out_dir)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run to resume.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

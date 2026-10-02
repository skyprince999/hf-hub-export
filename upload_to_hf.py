"""Convert the SQLite export to Parquet and upload it as a Hugging Face dataset.

Writes one config per entity type (models, datasets, spaces), sharded into
Parquet files of at most --rows-per-shard rows, plus a README dataset card,
then uploads the folder to the dataset repo (created private if missing).

Needs an HF_TOKEN with write access to the target namespace.
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

ENTITIES = ["models", "datasets", "spaces"]
BATCH_ROWS = 50_000

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("author", pa.string()),
    ("likes", pa.int64()),
    ("downloads", pa.int64()),
    ("created_at", pa.timestamp("ms", tz="UTC")),
    ("last_modified", pa.timestamp("ms", tz="UTC")),
    ("private", pa.bool_()),
    ("gated", pa.string()),  # "false", "auto" or "manual"
    ("disabled", pa.bool_()),
    ("tags", pa.list_(pa.string())),
    ("pipeline_tag", pa.string()),  # models only
    ("library_name", pa.string()),  # models only
    ("sdk", pa.string()),  # spaces only
])


def parse_ts(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def to_record(raw_json: str, entity: str) -> dict:
    item = json.loads(raw_json)
    gated = item.get("gated")
    return {
        "id": item["id"],
        "author": item.get("author"),
        "likes": item.get("likes"),
        "downloads": item.get("downloads") if entity != "spaces" else None,
        "created_at": parse_ts(item.get("createdAt")),
        "last_modified": parse_ts(item.get("lastModified")),
        "private": item.get("private"),
        "gated": None if gated is None else str(gated).lower(),
        "disabled": item.get("disabled"),
        "tags": item.get("tags") or [],
        "pipeline_tag": item.get("pipeline_tag"),
        "library_name": item.get("library_name"),
        "sdk": item.get("sdk"),
    }


def write_entity(conn, entity: str, out_dir: str, rows_per_shard: int) -> int:
    """Streams one table into data/<entity>/train-XXXXX-of-YYYYY.parquet."""
    total = conn.execute(f"SELECT COUNT(*) FROM {entity}").fetchone()[0]
    n_shards = max(1, -(-total // rows_per_shard))
    entity_dir = os.path.join(out_dir, "data", entity)
    os.makedirs(entity_dir, exist_ok=True)

    cursor = conn.execute(f"SELECT raw_json FROM {entity} ORDER BY created_at, id")
    written = 0
    for shard in range(n_shards):
        path = os.path.join(entity_dir, f"train-{shard:05d}-of-{n_shards:05d}.parquet")
        with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
            shard_rows = 0
            while shard_rows < rows_per_shard:
                rows = cursor.fetchmany(min(BATCH_ROWS, rows_per_shard - shard_rows))
                if not rows:
                    break
                records = [to_record(r[0], entity) for r in rows]
                writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                shard_rows += len(records)
        written += shard_rows
        print(f"[{entity}] wrote {path} ({shard_rows:,} rows)")
    if written != total:
        raise RuntimeError(f"[{entity}] wrote {written:,} rows but table has {total:,}")
    return written


def dataset_card(counts: dict, snapshot: str) -> str:
    configs = "\n".join(
        f"- config_name: {e}\n  data_files:\n  - split: train\n    path: data/{e}/*.parquet"
        for e in counts
    )
    rows = "\n".join(f"| `{e}` | {n:,} |" for e, n in counts.items())
    return f"""---
license: other
pretty_name: Hugging Face Hub Metadata
tags:
- metadata
- huggingface-hub
configs:
{configs}
---

# Hugging Face Hub Metadata

Metadata for every public model, dataset and Space on the Hugging Face Hub,
crawled from the Hub API on {snapshot} with
[hf-hub-export](https://github.com/skyprince999/hf-hub-export).

| Config | Rows |
|---|---|
{rows}

```python
from datasets import load_dataset
models = load_dataset("REPO_ID", "models", split="train")
```

## Columns

| Column | Type | Notes |
|---|---|---|
| `id` | string | Repo id, e.g. `org/name` |
| `author` | string | Owner (user or org) |
| `likes` | int | |
| `downloads` | int | Last 30 days, as reported by the API; null for Spaces |
| `created_at` | timestamp | |
| `last_modified` | timestamp | |
| `private` | bool | Always false (public repos only) |
| `gated` | string | `false`, `auto` or `manual`; null for Spaces |
| `disabled` | bool | Null for Spaces |
| `tags` | list[string] | |
| `pipeline_tag` | string | Models only |
| `library_name` | string | Models only |
| `sdk` | string | Spaces only |

Rows are ordered by `created_at`. The license field reflects that each listed
repo carries its own license; this dataset contains only public metadata.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="e.g. thinkevolve/hf-hub-metadata")
    parser.add_argument("--db", default="huggingface_hub.db")
    parser.add_argument("--out-dir", default="hf_dataset")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--public", action="store_true", help="Create the repo public")
    parser.add_argument("--no-upload", action="store_true", help="Only write Parquet + card")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    incomplete = [e for e, done in conn.execute("SELECT entity, done FROM sync_state") if not done]
    missing = set(ENTITIES) - {e for (e,) in conn.execute("SELECT entity FROM sync_state")}
    if incomplete or missing:
        raise SystemExit(f"Export not finished for: {sorted(set(incomplete) | missing)}")
    snapshot = conn.execute("SELECT MAX(updated_at) FROM sync_state").fetchone()[0][:10]

    counts = {e: write_entity(conn, e, args.out_dir, args.rows_per_shard) for e in ENTITIES}
    conn.close()
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(dataset_card(counts, snapshot).replace("REPO_ID", args.repo_id))

    if args.no_upload:
        print(f"Wrote {args.out_dir}; skipping upload")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=args.out_dir,
        commit_message=f"Hub metadata snapshot {snapshot}",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

# hf-hub-export

Exports metadata for every public model, dataset and Space on the Hugging Face Hub into a SQLite database (`huggingface_hub.db`) and JSONL files (`hf_export_<type>.jsonl`).

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # optional; raises rate limits
python hf_hub_export.py
```

## Resumable

The crawl pages through the Hub API oldest-first, which gives it a stable cursor. Each page and the cursor for the next one are committed to SQLite together, so you can stop the script at any time (Ctrl-C, crash, lost connection) and re-run it to continue where it left off, without gaps or duplicates.

- Rate limits (HTTP 429) are waited out using the server's `RateLimit` / `Retry-After` headers.
- Network errors and 5xx responses retry with exponential backoff; after repeated failures the script exits, and a re-run resumes.
- JSONL files are rewritten from SQLite after each entity type completes, so they never contain duplicates.

## Options

```
--entities models datasets spaces   # subset to sync (default: all)
--restart                           # discard saved progress and re-crawl
--db PATH                           # SQLite path (default: huggingface_hub.db)
--out-dir DIR                       # JSONL directory (default: .)
```

## Querying

```sql
SELECT id, likes, downloads FROM models ORDER BY likes DESC LIMIT 20;
SELECT entity, pages, done, updated_at FROM sync_state;   -- crawl progress
```

`raw_json` holds the full API record, including `pipeline_tag` / `library_name` (models), `gated` (models, datasets) and `sdk` (Spaces).

## Publishing to the Hugging Face Hub

`upload_to_hf.py` converts the finished export to zstd Parquet (one config each for `models`, `datasets` and `spaces`, sharded at 1M rows), writes a dataset card, and uploads it. The repo is created private unless you pass `--public`. `HF_TOKEN` needs write access to the target namespace.

```bash
python upload_to_hf.py --repo-id thinkevolve/hf-hub-metadata
python upload_to_hf.py --repo-id thinkevolve/hf-hub-metadata --no-upload   # just build hf_dataset/
```

It refuses to run until all three entity types have finished syncing.

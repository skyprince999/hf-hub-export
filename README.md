# hf-hub-export

Streams metadata for every public model, dataset and Space on the Hugging Face Hub into a SQLite database (`huggingface_hub.db`) and JSONL files (`hf_export_<type>.jsonl`).

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # optional; raises rate limits
python hf_hub_export.py
```

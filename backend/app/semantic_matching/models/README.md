# Bundled semantic model (all-MiniLM-L6-v2)

These files are **checked into the repository** so the semantic layer runs with
zero runtime downloads (₹0, offline, CPU-only via ONNX Runtime).

## Files

| File | Purpose | Size |
|------|---------|------|
| `model_qint8_avx512_vnni.onnx` | Quantised ONNX checkpoint of `sentence-transformers/all-MiniLM-L6-v2` (384-dim, mean-pooled + L2-normalized embeddings) | ~23 MB |
| `tokenizer.json` | `tokenizers` fast wordpiece tokenizer for the model's vocab | ~466 KB |
| `vocab.txt` | Original BERT vocab (reference only; `tokenizer.json` embeds it) | ~232 KB |

## Provenance

- Model: `sentence-transformers/all-MiniLM-L6-v2` — the ONNX export is
  produced/verified by the sentence-transformers project and tagged **Apache-2.0**.
- Weights licence: Apache-2.0 (see
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>).
- These bytes were downloaded once from the model's official Hugging Face repo
  (`onnx/model_qint8_avx512_vnni.onnx`, `tokenizer.json`, `vocab.txt`) at
  migration time and pinned in-repo. Nothing is fetched at install or request
  time.

## Runtime behaviour

- Inputs: `input_ids`, `attention_mask`, `token_type_ids` (batch × 128).
- Output: `last_hidden_state`; `app/semantic_matching/model.py` mean-pools over
  the attention mask and L2-normalizes to produce the 384-dim embedding.
- Runs on CPU via `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])`.

Keep these files in sync with the model/version pinned in
`app/semantic_matching/config.py`. Do not add new model files without flagging
their licence and monetary cost in the PR.
# Examples and Data Formats

This folder gives you everything needed to run the production stack on your own data, swap your own evaluation pairs into the benchmark suite, or extend the training data for a follow-up fine-tune.

```
examples/
├── README.md                                    ← this file (field-by-field schema)
├── quickstart_embed.py                          ← 30-line runnable example
├── load_pairs.py                                ← JSONL loaders shared by code/
└── sample_data/
    ├── connected_disconnected_pairs.jsonl       ← B1.1 event-separation pairs
    ├── hard_negative_triplets.jsonl             ← B5 hard-negative triplets
    ├── biosses_style_pairs.jsonl                ← B3 STS pairs with human scores
    ├── domain_sentences.jsonl                   ← B1/B6/B7 per-domain sentences
    └── finetuning_triplets.jsonl                ← Pass-2 BODHI-style training triplets
```

The sample JSONL files are small (5–18 records each) — enough to verify the format end-to-end and run a smoke test of the benchmarks. Drop in your own data with the same fields and everything in `code/` will pick it up via `--data-dir`.

---

## Quick start

```bash
# Activate the project venv (see /docs/SETUP.md if you haven't created it)
source /event_model/.venv/bin/activate

# Embed four built-in biomedical sentences and print the similarity matrix.
python examples/quickstart_embed.py                            # PubMedBERT BODHI, PyTorch BF16
python examples/quickstart_embed.py --backend openvino         # same model, OV BF16
python examples/quickstart_embed.py --model pubmedbert_pass1   # different model
python examples/quickstart_embed.py --sentences \
  "Patient HbA1c 9.2%" "Slept four hours, anxious"             # bring your own sentences
```

Run the full quality benchmark suite against the bundled samples:

```bash
python code/extended_benchmark.py \
  --variant ov-bf16 \
  --data-dir examples/sample_data
```

Run the event-separation benchmark against your own JSONL:

```bash
python code/compare_finetuned.py --model bodhi \
  --data-file /path/to/my_pairs.jsonl
```

When `--data-dir` / `--data-file` is omitted, the scripts fall back to the literals embedded in the source so existing reproduction commands keep working.

---

## File-by-file schema

Every record is one JSON object per line (JSONL / NDJSON). UTF-8. Empty lines and `//`-prefixed lines are ignored by `load_pairs.py`.

### 1. `connected_disconnected_pairs.jsonl` — event separation (B1.1)

Used by `code/compare_finetuned.py` to compute the **discrimination gap** between connected and disconnected event pairs.

| Field | Type | Required | Description |
|---|---|---|---|
| `sentence_a` | string | ✓ | First event description (any domain). |
| `sentence_b` | string | ✓ | Second event description. |
| `label` | `"connected"` \| `"disconnected"` | ✓ | Ground truth. **Connected** = a real-world causal trajectory exists between A and B; **disconnected** = no causal link (even if surface form overlaps). |
| `rationale` | string | optional | One-line explanation of the causal call. Recorded in result JSON; not consumed by training. |

Example:

```json
{"sentence_a": "HbA1c 9.2% sustained hyperglycaemia poor glycaemic control",
 "sentence_b": "Patient describes fatigue, low mood and difficulty concentrating daily",
 "label": "connected",
 "rationale": "metabolic-to-affective causal chain (neuroglycopenia)"}
```

### 2. `hard_negative_triplets.jsonl` — hard-negative ranking (B5)

Used by `code/extended_benchmark.py` for the hard-negative detection benchmark. A triplet **passes** when `cos(anchor, positive) > cos(anchor, hard_negative)`.

| Field | Type | Required | Description |
|---|---|---|---|
| `anchor` | string | ✓ | Query sentence. |
| `positive` | string | ✓ | A semantically equivalent or causally-linked sentence — should rank higher. |
| `hard_negative` | string | ✓ | A sentence that *shares vocabulary or domain* with the anchor but is **not** causally linked — should rank lower. |
| `domain` | string | optional | Source domain tag for reporting. |

### 3. `biosses_style_pairs.jsonl` — semantic textual similarity (B3)

Used by `code/extended_benchmark.py` for Spearman ρ against human-judged similarity (BIOSSES-style).

| Field | Type | Required | Description |
|---|---|---|---|
| `sentence_a` | string | ✓ | First sentence of the STS pair. |
| `sentence_b` | string | ✓ | Second sentence of the STS pair. |
| `human_score` | float ∈ [0, 1] | ✓ | Human-annotated similarity score. (BIOSSES uses 0–4; rescale to [0,1] before storing.) |

### 4. `domain_sentences.jsonl` — within-domain cohesion / domain geometry (B1, B6, B7)

Used for within-domain similarity (B1), inter/intra cluster ratio (B6), and the embedding-space-geometry diagrams (B7).

| Field | Type | Required | Description |
|---|---|---|---|
| `domain` | string | ✓ | Domain label (e.g. `genetics`, `biomarkers`, `psychology`, `journal`, `clinical_notes`, `physiology`). Use the same set across all records. |
| `sentence` | string | ✓ | One sentence from that domain. |

Recommended: ≥ 10 sentences per domain so intra-domain cosine has stable statistics.

### 5. `finetuning_triplets.jsonl` — Pass-2 BODHI-style training data

Used by `code/finetune_pubmed_bodhi.py` to extend the second-pass fine-tune with extra triplets — same schema as the production BODHI export.

| Field | Type | Required | Description |
|---|---|---|---|
| `anchor` | string | ✓ | Training anchor. |
| `positive` | string | ✓ | A node connected to `anchor` along a short causal/ontology path. |
| `negative` | string | ✓ | A node connected by a *longer* path (or no path) — see `level`. |
| `level` | int ∈ {1..5} | optional | Graph-distance bucket (see `docs/FINETUNING.md` §"Distance levels"). Higher = farther apart. |
| `domain_anchor` | string | optional | Source domain of `anchor`. Used for domain-balanced batch sampling. |
| `domain_negative` | string | optional | Source domain of `negative`. |
| `relation` | string | optional | Triplet relation type drawn from the BODHI ontology (e.g. `PRESENT_IN`, `IMPACTS`, `PARAPHRASE`, `INFORMS_TREATMENT`, `EVIDENCED_BY`). |
| `note` | string | optional | Free-form provenance — kept for audit, ignored at training time. |

---

## Bring your own data

Drop in your own JSONL files with the same field names — that's it. Two patterns:

**Override one shape:**
```bash
python code/compare_finetuned.py --model bodhi \
  --data-file /path/to/my_pairs.jsonl
```

**Override the whole sample_data/ directory** (any subset of the five files; missing files fall back to embedded literals):
```bash
python code/extended_benchmark.py --variant ov-bf16 \
  --data-dir /path/to/my_data_dir
```

If a file you point at is malformed, `load_pairs.py` raises with the offending line number — no silent skipping.

See [`docs/FINETUNING.md`](../docs/FINETUNING.md) for the full fine-tuning guide: public dataset registry, BODHI ontology export, synthetic-data generation recipe, and the loss configuration that produced the production weights.

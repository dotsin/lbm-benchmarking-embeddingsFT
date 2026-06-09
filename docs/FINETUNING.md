# Fine-Tuning Guide

This guide is the recipe behind the production weights in `models/pytorch/` and `models/openvino/`. It is written so that anyone outside Dotsin.ai can reproduce the run, extend it on new data, or replace the BERT-base backbone with a different encoder while keeping the **causal-similarity objective** intact.

The training pipeline runs in two passes:

1. **Pass 1 — multi-dataset contrastive fine-tune.** Public biomedical, NLI, clinical, and psychology datasets glued together with `MultipleNegativesRankingLoss` (MNRL) wrapped in `MatryoshkaLoss`. Produces `PubMedBERT Pass1` and `BioBERT Fine-Tuned`.
2. **Pass 2 — BODHI graph-grounded triplets.** Starts from the Pass 1 checkpoint and re-trains on ontology-graph triplets that explicitly encode causal distance. Produces `PubMedBERT BODHI`.

The Pass 2 step is what installs the second axis (causal similarity) on top of the semantic geometry from Pass 1. See [README §"A second axis: causal similarity"](../README.md) for the conceptual framing.

---

## 1. Hardware & environment

The production run used a dual-socket **Intel® Xeon® 6737P (Granite Rapids)** kindly provided to Dotsin.ai by **Intel Corporation** — 2 sockets × 32 cores, 128 logical CPUs, 1 TB DDR5-6400, AMX-BF16. Pass 1 takes ~6.5 h, Pass 2 takes ~14.8 h on this hardware. See [`ACKNOWLEDGEMENTS.md`](../ACKNOWLEDGEMENTS.md) for the full hardware and toolchain credit.

You do **not** need this exact box. The training script auto-scales to whatever cores it sees and switches AMX kernels on/off via `ONEDNN_MAX_CPU_ISA`. On a single-socket 16-core machine expect ~10× longer wall-clock. GPU training is supported by `sentence-transformers` out of the box but the script defaults to CPU + AMX because the production fleet is CPU-only.

```bash
# Reproducible Python environment
python3.11 -m venv /event_model/.venv
source /event_model/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install sentence-transformers==3.3.1 datasets accelerate
```

Verify AMX is on the path:

```bash
python -c "
import torch
print('PT      :', torch.__version__)
print('MKLDNN  :', torch.backends.mkldnn.is_available())
print('AMX BF16:', 'amx_bf16' in open('/proc/cpuinfo').read())
"
```

If `amx_bf16` is `False` you'll still run, just slower — the script will fall back to AVX-512 BF16 automatically.

---

## 2. Public datasets used in Pass 1

All datasets ship via Hugging Face `datasets` and are downloaded on first use into `~/.cache/huggingface/datasets/`. Total download is ~3 GB; budget ~10 min on a fast link.

| Dataset | HF path | Domain | Pairs used | Role in Pass 1 |
|---|---|---|---|---|
| all-nli | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) | General NLI | 570 k entailment + 430 k contradiction | Backbone — hard negatives across general English |
| BIOSSES | [`bigbio/biosses`](https://huggingface.co/datasets/bigbio/biosses) | Biomedical STS gold standard | 100 scored pairs | Calibration — held out for ρ evaluation |
| MedNLI | [`bigbio/mednli`](https://huggingface.co/datasets/bigbio/mednli) | Clinical NLI (MIMIC-III) | 14 049 triples | Clinical entailment / contradiction |
| PubMedQA | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | Biomedical QA | 100 k question + abstract pairs | Within-domain positive anchors |
| medical_q_pairs | [`curaihealth/medical_questions_pairs`](https://huggingface.co/datasets/curaihealth/medical_questions_pairs) | Clinical Q&A | 3 048 expert-paraphrase pairs | Within-domain paraphrase positives |
| emotion | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | Psychology (Twitter, 6 classes) | 20 000 labelled tweets → in-class positives | Within-psych positives |
| go_emotions | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Psychology (Reddit, 27 classes) | 58 000 labelled comments → in-class positives | Fine-grained psych positives |
| mental_health_counseling | [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) | Counselling Q/A | ~1 000 dialogues | Journal-style anchors |
| (synthetic) cross-domain hard negatives | generated inline by `code/finetune_ensemble.py` | All | ~5 000 pairs | Cross-domain distractors that share surface form |

Final training composition (Pass 1 mix, after deduplication and domain balancing): **72 034 pairs** — 28 % PubMedQA + MedQA, 28 % MedNLI + Wikidoc, 26 % mental health (emotion + go_emotions + counselling), 17 % BODHI Pass 2 seed triplets, 1 % synthetic cross-domain hard negatives. Visualised in [`charts/training_config_and_data.png`](../charts/training_config_and_data.png).

### How the public datasets are mapped to (anchor, positive, hard_negative)

`sentence-transformers` accepts datasets as columns `(anchor, positive, negative)` or `(sentence1, sentence2, label)`. The loaders in `code/finetune_ensemble.py` do this remapping per dataset; see the `BUILDERS` dict around line 250 for the exact transformations.

---

## 3. Pass 2 — BODHI graph-grounded triplets

Pass 2 is where causal similarity is installed. The training data is **not** a pre-existing public dataset; it is exported from Dotsin's BODHI biomedical ontology graph. The exported triplets follow the format documented in [`examples/sample_data/finetuning_triplets.jsonl`](../examples/sample_data/finetuning_triplets.jsonl) and described in [`examples/README.md` §5](../examples/README.md#5-finetuning_tripletsjsonl--pass-2-bodhi-style-training-data).

### Where to obtain the BODHI export

The BODHI ontology and its triplet export are part of Dotsin's proprietary biomedical knowledge graph (Dotsin.ai LBM stack). For research collaborations you can request a snapshot via the gated form on the model card — fill in `Intended use` with "BODHI training data". A signed data-use agreement is required because the export embeds curated literature evidence; once signed you receive the BODHI CSV bundle that `code/finetune_pubmed_bodhi.py` reads via `models/BODHI/`.

If you do not have BODHI access, you can still reproduce Pass 2 by **generating your own ontology-grounded triplets** from a public graph (UMLS, SNOMED-CT, MeSH, HPO, MONDO). The schema and the distance-level breakdown below tell you exactly what shape to produce. See §5 "Synthetic-data generation" for the recipe.

### Distance levels written into the geometry

Pass 2 trains the model to put pairs at five graph-distance bands:

| Level | Graph relation in BODHI | Target embedding distance | What it teaches |
|---|---|---|---|
| 1 | Paraphrase / synonym (`SAME_AS`, `LABELED_AS`) | 0.00–0.15 | Surface-form invariance for the same node |
| 2 | Causally linked (`PRESENT_IN`, `IS_INFLUENCED_BY`, `IMPACTS`) | 0.15–0.40 | Cause-effect proximity (the second axis) |
| 3 | Same-domain siblings (`CHILD_OF`, same triage, no direct edge) | 0.40–0.60 | Within-domain coherence without spurious linkage |
| 4 | Cross-domain, no shared path | 0.60–0.80 | Cross-domain separation |
| 5 | Completely unrelated (separate sub-graphs) | 0.80–1.00 | Far-pair regularisation |

The exact mapping from BODHI relation type → level lives at the top of `code/finetune_pubmed_bodhi.py` (`LIK_WEIGHT` and the `level_of()` helper).

---

## 4. Setup — full reproduction

### 4.1 Clone code and pull weights

```bash
# Code (GitHub, public)
git clone git@github.com:dotsin/lbm-benchmarking-embeddingsFT.git
cd lbm-benchmarking-embeddingsFT

# Weights (HF, gated — request access on the model card first)
pip install -U huggingface_hub
huggingface-cli login                  # paste your HF token
huggingface-cli download \
  Dotsin/lbm-benchmarking-embeddingsFT \
  --local-dir . \
  --include "models/**"
```

The `models/` subtree fetched from HF contains the three production checkpoints in both PyTorch SafeTensors (`models/pytorch/`) and OpenVINO BF16 IR (`models/openvino/`) formats.

### 4.2 Install runtime deps

```bash
python3.11 -m venv /event_model/.venv
source /event_model/.venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers==3.3.1 datasets accelerate
```

### 4.3 Verify the install

```bash
python examples/quickstart_embed.py
# → prints a 4×4 cosine similarity matrix for the four built-in sentences
```

If that prints, you can run the full quality suite and the fine-tune.

---

## 5. Synthetic-data generation

You can extend Pass 2 with your own triplets by combining a biomedical ontology with a small set of templates. The recipe below is the one used to grow the BODHI triplet bundle from 17 000 seed triples to 72 034 final training pairs.

### 5.1 Inputs

- An ontology with typed edges. Public options: UMLS, SNOMED-CT, MeSH, HPO, MONDO, the Disease Ontology. BODHI uses a curated biomedical + behavioural superset built on top of these.
- A short list of natural-language templates per relation type (anchor / positive surface-form templates).

### 5.2 Procedure

1. **Sample anchors.** Pull `N_ANCHORS` random nodes from the ontology, stratified across domains so no domain dominates.
2. **Walk for positives.** For each anchor, traverse the graph along one of the allowed Pass 2 relations (`PRESENT_IN`, `IS_INFLUENCED_BY`, `IMPACTS`, `INFORMS_TREATMENT`, `EVIDENCED_BY`, `PARAPHRASE`) up to depth 2 and record the destination node as the positive. Record the relation type and the graph distance (1 hop or 2 hop).
3. **Mine hard negatives.** For each (anchor, positive) pair, draw a negative from one of these three pools, weighted 0.5 / 0.3 / 0.2:
   - **Same-domain sibling** — a node from the same domain that does *not* lie on a short path to the anchor (level 3).
   - **Cross-domain surface-overlap** — a node from a different domain whose label shares 1–2 tokens with the anchor label (level 4). This is the hardest negative type and the one most directly responsible for installing the causal axis.
   - **Far node** — a random node from a disjoint sub-graph (level 5).
4. **Fill the surface form.** Run the (anchor, positive, negative) IDs through your templates to convert each node to a sentence. Vary the template per record so the model sees ≥ 5 surface forms per node — this is what stops the model from over-fitting to the template wording instead of the semantics.
5. **De-duplicate.** Drop triplets where any two of `(anchor, positive, negative)` collapse to the same sentence after templating.
6. **Export as JSONL** matching [`finetuning_triplets.jsonl`](../examples/sample_data/finetuning_triplets.jsonl) — at minimum `anchor`, `positive`, `negative`; ideally also `level`, `domain_anchor`, `domain_negative`, `relation`.

### 5.3 Quality knobs

Two parameters have the largest impact on the final intra/inter ratio:

- **Cross-domain hard-negative fraction.** The production run uses 30 % cross-domain negatives. Below 15 % the model never learns to separate semantically near, causally distant pairs (the `BRCA1 ↔ stock market volatility` failure). Above 50 % the model over-separates and within-domain cohesion drops.
- **Template diversity.** Aim for entropy ≥ 2.0 bits per node label across the templates you sample (i.e. ≥ ~4 effective templates per node). Lower entropy and the model learns the template, not the semantics.

### 5.4 Sanity-checking your triplets

Before launching a 14-h training run, validate the export:

```bash
python - <<'PY'
from examples.load_pairs import load_finetuning_triplets
rows = load_finetuning_triplets("path/to/my_triplets.jsonl")
print("triplets       :", len(rows))
print("levels seen    :", sorted({r.get('level') for r in rows if 'level' in r}))
print("relations seen :", sorted({r.get('relation') for r in rows if 'relation' in r}))
PY
```

A healthy export has all 5 levels populated and at least 6 distinct relations.

---

## 6. Training configuration

These are the exact settings behind the production checkpoints. They sit at the top of `code/finetune_ensemble.py` (Pass 1) and `code/finetune_pubmed_bodhi.py` (Pass 2).

| Knob | Pass 1 (multi-dataset) | Pass 2 (BODHI) |
|---|---|---|
| Starting checkpoint | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (or `dmis-lab/biobert-v1.1`) | `finetuned/pubmedbert/final` (Pass 1 output) |
| Batch size | 128 (MNRL → 127 in-batch negatives per anchor) | 128 |
| Learning rate | 2e-5 with cosine decay, warm-up 6 % | 2e-5, warm-up 3 % |
| Epochs | 3 + 1 hard-negative mining pass | 3 + 1 hard-negative mining pass |
| Loss | `MatryoshkaLoss(MultipleNegativesRankingLoss)` + `TripletLoss` (later epochs) | `MatryoshkaLoss(MultipleNegativesRankingLoss)` |
| Matryoshka dims | [768, 512, 256, 128, 64] | [768, 512, 256, 128, 64] |
| Precision | BF16 (AMX) via `torch.autocast` | BF16 (AMX) |
| Threads | 64 (other 64 reserved for parallel inference benchmarking) | 64 |
| Wall-clock on Xeon 6737P | ~6.5 h | ~14.8 h |

### Matryoshka rationale

Training simultaneously at five embedding widths means downstream consumers can truncate the 768-dim vector to 256 or 64 and lose < 2 % retrieval quality. The smaller widths are what makes the "edge-deployed data hub" line in [`docs/LBM_INTEGRATION.md`](LBM_INTEGRATION.md) §8.3 workable.

### Why two passes instead of one

Mixing BODHI triplets into Pass 1 from the start was tried during development and produced a worse final geometry (intra/inter 1.94× vs the 2.30× the two-pass route hit). The intuition: BODHI triplets are densely causal, and when they share batches with the looser NLI/QA positives, MNRL's in-batch negatives accidentally penalise causally-correct pairs from the BODHI subset. Pass 1 first anchors the semantic geometry; Pass 2 then bends it along causal pathways without that interference.

---

## 7. Running the fine-tune

### Pass 1

```bash
cd code
python finetune_ensemble.py \
  --model pubmedbert \
  --output-dir finetuned/pubmedbert
```

Outputs `finetuned/pubmedbert/final/` in `sentence-transformers` format. Repeat with `--model biobert` for the BioBERT Fine-Tuned checkpoint.

### Pass 2 (BODHI)

```bash
cd code
python finetune_pubmed_bodhi.py \
  --bodhi-dir ../models/BODHI \
  --extra-triplets ../examples/sample_data/finetuning_triplets.jsonl \
  --output-dir finetuned/pubmedbert_bodhi
```

`--extra-triplets` is optional; supply it to concatenate your own JSONL onto the BODHI export. If you have no BODHI access at all, pass only `--extra-triplets` pointing at the JSONL produced by §5 — the script falls back to pure-JSONL mode when `--bodhi-dir` is absent.

### Convert to OpenVINO BF16 IR

```bash
optimum-cli export openvino \
  --model finetuned/pubmedbert_bodhi/final \
  --task feature-extraction \
  --weight-format fp16 \
  ../models/openvino/pubmedbert_bodhi_bf16
```

---

## 8. Evaluating a new fine-tune

Re-run the production B1–B8 suite against your checkpoint:

```bash
python code/extended_benchmark.py --variant ov-bf16 \
  --model-override finetuned/pubmedbert_bodhi/final \
  --data-dir examples/sample_data
```

The targets that mattered for production (see [README §1](../README.md)):

| Metric | Pre-FT | Pass 1 target | Pass 2 (BODHI) target | Production achieved |
|---|---|---|---|---|
| Discrimination gap (B1.1) | 0.051 | > 0.18 | > 0.28 | **0.302** |
| BIOSSES Spearman ρ (B3) | ~0.77 | > 0.85 | > 0.85 | **0.775** |
| Hard-negative accuracy (B5) | ~60–80 % | > 80 % | > 88 % | **100 % (5/5)** |
| Intra/inter ratio (B6) | < 1.0 | > 1.5 | > 2.0 | **2.304** |
| OV-BF16 throughput (bs=256) | — | no regression | no regression | 617.8 sps |

If you hit ratio ≥ 2.0 and discrimination gap ≥ 0.28, your checkpoint is in the production envelope.

---

## 9. Where to push your weights

If you produce a derivative model and want to publish it, the conventional place is your own Hugging Face namespace. We recommend:

- License: keep Apache-2.0 (the LICENSE file in this repo) and credit the BODHI source.
- Gating: if you trained on BODHI data, the data-use agreement requires that you mirror the gated-access form on your model card (anyone downloading must sign in and accept terms). The frontmatter block in [README](../README.md) shows the exact YAML.
- Cite the paper: [arXiv:2606.09672](https://arxiv.org/abs/2606.09672).

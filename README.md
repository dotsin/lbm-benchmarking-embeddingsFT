---
license: apache-2.0
language:
  - en
pretty_name: "Biomedical BERT Embedding Quality & Inference Benchmark — Intel Xeon 6737P"
tags:
  - biomedical
  - bert
  - embeddings
  - benchmark
  - sentence-similarity
  - feature-extraction
  - openvino
  - pytorch
  - safetensors
  - intel-xeon
  - cpu-inference
  - quantization
  - event-separation
  - clinical-nlp
  - arxiv:2606.09672
task_categories:
  - feature-extraction
  - sentence-similarity
multilinguality:
  - monolingual
library_name: transformers
gated: auto
extra_gated_heading: "Access to the Biomedical Causal-Similarity Embedding Layer"
extra_gated_description: |
  These models are released by Dotsin.ai as the open biomedical sentence-embedding
  layer that fronts a secure data hub. To download the weights you must sign in with a
  Hugging Face account and accept the terms below. Access is granted automatically once the
  form is submitted; we keep a record of who has accepted.
extra_gated_prompt: |
  By requesting access you confirm that:
  1. You will not use these weights to identify, profile, or re-identify individuals from
     biomedical or behavioural text.
  2. You will not redistribute the weights outside Hugging Face without preserving this
     gated-access requirement.
  3. You will cite this repository if the weights or benchmark suite contribute to a
     publication.
  4. The weights are provided under Apache-2.0 (see LICENSE) with no warranty of any kind.
extra_gated_fields:
  Full name: text
  Affiliation / Organisation: text
  Country: country
  Intended use: text
  I agree to the terms above: checkbox
extra_gated_button_content: "Request access"
---

# Biomedical BERT Embedding Quality & Inference Benchmark
### PubMedBERT Pass1 · PubMedBERT BODHI · BioBERT Fine-Tuned on Intel Xeon 6737P (Granite Rapids)

[![Hugging Face — Model & Weights](https://img.shields.io/badge/Hugging%20Face-Model%20%26%20Weights-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Dotsin/lbm-benchmarking-embeddingsFT)
[![GitHub — Code](https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/dotsin/lbm-benchmarking-embeddingsFT)
[![arXiv — Paper](https://img.shields.io/badge/arXiv-2606.09672-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.09672)
[![License — Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0A7EBE?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

Three fine-tuned biomedical BERT-base sentence-embedding models, designed around two questions:

1. **Embedding quality** — does the geometry encode *causal* similarity between biomedical events, on top of the semantic similarity inherited from pre-training? Evaluated through event-pair separation, hard-negative ranking, BIOSSES-style correlation, domain geometry, and cosine fidelity under quantization.
2. **Inference efficiency** — PyTorch vs OpenVINO throughput and latency on Intel Granite Rapids with AMX acceleration across six precision variants.

Model weights ship as **PyTorch SafeTensors** and **OpenVINO BF16 IR**, ready to run without re-training or re-export.

---

## Why This Work Exists

These models are the **sentence-embedding layer of a secure biomedical data hub** built by [Dotsin.ai](https://dotsin.ai). The hub stores a user's raw textual life data — clinical notes, journal entries, counselling transcripts, lab text, research context — and uses these 768-dim embeddings to index it. When a downstream task asks the hub for information about a user, the hub assembles an **information stream**: records selected and ordered by embedding proximity, timestamps, and access policy. That stream — not the embeddings — is what crosses the boundary into Dotsin's **Large Behavioral Model (LBM)** service.

The LBM consumes streams and returns a **DAG of behavioural inferences**. The DAG is then traversed over Dotsin's proprietary **LBM graph**, in which millions of behavioural data points are plotted along causal chains induced from **counterfactual analysis** over the user's complete human metadata. The encoders in this repository sit two hops before that reasoning step — they are the retrieval geometry for the data hub, not features that the LBM model consumes. The LBM and the proprietary graph remain closed; this repository is the open layer.

For stream assembly to give the LBM a faithful causal picture, the embedding geometry has to satisfy two requirements:

- A journal entry *"slept 4 hours, felt anxious"* and a biomarker text *"cortisol 28 μg/dL"* should sit close — both belong in the same stream when the requesting task is reasoning about HPA-axis state.
- A genetics record *"BRCA1 pathogenic variant"* and a finance record *"stock market volatility"* should sit far apart — a false neighbour here pulls noise into the stream and the LBM sees a polluted view of the user.

The 5.9× improvement in discrimination gap after fine-tuning (0.051 → 0.302, §1.1) is what makes the hub's stream assembly behave correctly. The inference path (PyTorch → OV-INT8, +33–55% throughput) is what makes the hub keep up with ingest.

### What we're sharing — and why

We publish this repository to make our thought process on biomedical sentence embeddings visible to the wider research and clinical-NLP community. Better encoders, leaner fine-tuning recipes, and sharper benchmarks for **causal-axis** behaviour are the right things for the community to push on — and the only way to get there is to share the full reasoning, not just the numbers. Everything that matters for continuing this line of work is included: the production weights (PubMedBERT BODHI / Pass1 / BioBERT FT), the comparison studies that did not survive (BioM-ELECTRA Large, the three-model averaging configuration), the failure modes of pre-trained baselines, the test cases, and the geometry diagnostics. Whoever wants to build a more efficient, more knowledgeable, more causally-aware embedding model along this direction has what they need here.

### A second axis: causal similarity

Off-the-shelf biomedical encoders — BioBERT, PubMedBERT, BioM-ELECTRA, and general-purpose STS models — are trained for **semantic similarity**: two sentences land close when they share surface form, vocabulary, or topic. For data-hub retrieval that objective is necessary but not sufficient. The hub has to be able to surface records that *cause or follow from* the same underlying event when assembling a stream, not only records that *describe* the same thing.

The fine-tuning regime here targets a second, orthogonal axis: **causal similarity** — events whose real-world consequences converge sit close, even when they share no vocabulary, no register, no surface form; events with overlapping vocabulary but disjoint causal trajectories sit apart. The cause-side and effect-side of the same physiological pathway — the *sin* and *cos* phases of one causal wave — should both land in the same neighbourhood.

**Test-case grounding (from §1.1, fine-tuned model):**

| Pair | Surface / Semantic | Causal trajectory | Required geometry | Achieved cosine |
|---|---|---|---|---|
| `HbA1c 9.2% sustained hyperglycaemia` ↔ `Patient describes fatigue, low mood and difficulty concentrating daily` | Distant — numeric lab marker vs subjective journal-style mood entry; zero token overlap, different registers, different domains | Hyperglycaemia → neuroglycopenic fatigue / mood disturbance | **CLOSE** | **0.69** ✓ |
| `Serum cortisol 32 μg/dL markedly elevated HPA axis dysregulation` ↔ `Patient reports persistent anxiety, sleep disruption and mood instability` | Distant — clinical biomarker vs first-person psychological complaint | HPA-axis activation causes anxiety/insomnia (effect side); chronic stress causes elevated cortisol (cause side) — same wave, both phases | **CLOSE** | **0.69** ✓ |
| `BRCA1 pathogenic variant detected in germline DNA` ↔ `Stock market volatility increased investor anxiety this quarter` | Closer than the rows above by some BERT-base metrics — both invoke "risk", "variant/variability" | No causal pathway between germline oncogenetics and equity markets | **FAR** | **0.42** ✓ |
| `PHQ-9 score 18 severe depression suicidal ideation` ↔ `Bone density scan DEXA normal T-score bilateral hip and spine` | Both are clinical assessments in the same hospital-note register — semantically nearer than the cortisol/anxiety pair above | No shared causal trajectory | **FAR** | **0.36** ✓ |

The first two rows are the thesis. A pure semantic-similarity encoder pushes `HbA1c 9.2%` and `fatigue, low mood` apart (no token overlap) and pulls `BRCA1 variant` and `market volatility` together (shared abstract notion of "risk"). The fine-tuned model inverts both: a lab marker and the mood symptom it causes become neighbours; two technically-worded "risk" statements from incompatible causal worlds become strangers.

The discrimination gap of **0.302** measures exactly this — how much causal structure has been written into the geometry on top of the semantic structure inherited from pre-training. The progression `1.05× → 1.63× → 2.30×` ([embedding_space_geometry_progression](charts/embedding_space_geometry_progression.png)) shows the second axis arriving in two stages: Pass 1 multi-dataset fine-tuning anchors semantic neighbourhoods; BODHI Pass 2 ontology-graph triplets bend those neighbourhoods along causal pathways.

Full architecture context: [`docs/LBM_INTEGRATION.md`](docs/LBM_INTEGRATION.md)
OpenVINO acceleration deep-dive: [`docs/OV_ACCELERATION.md`](docs/OV_ACCELERATION.md)

---

## Quick Links

| | |
|---|---|
| Quality benchmark scripts | `code/extended_benchmark.py`, `code/compare_finetuned.py`, `code/threshold_sweep.py` |
| Throughput benchmark | `code/scenario_bench.py`, `code/load_test.py` |
| Hardware config | `config.yaml` |
| **LBM architecture & motivation** | [`docs/LBM_INTEGRATION.md`](docs/LBM_INTEGRATION.md) |
| **OpenVINO acceleration deep-dive** | [`docs/OV_ACCELERATION.md`](docs/OV_ACCELERATION.md) |
| **Headline results (one page)** | [`RESULTS.md`](RESULTS.md) |
| **Acknowledgements (Intel hardware, datasets, base models)** | [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) |
| **Examples & data formats** | [`examples/README.md`](examples/README.md) |
| **Fine-tuning guide (datasets, BODHI, synthetic data, schemas)** | [`docs/FINETUNING.md`](docs/FINETUNING.md) |
| Runnable quickstart | [`examples/quickstart_embed.py`](examples/quickstart_embed.py) |
| Full throughput + profiling report | [`docs/FINAL_CONSOLIDATED_REPORT.md`](docs/FINAL_CONSOLIDATED_REPORT.md) |
| Decision log (why NUMA, HT, INT8…) | [`docs/REASONING_QNA.md`](docs/REASONING_QNA.md) |
| PyTorch throughput tables | [`docs/PYTORCH_RESULTS.md`](docs/PYTORCH_RESULTS.md) |

---

## Usage

### Load with transformers (PyTorch)

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

model_path = "models/pytorch/pubmedbert_pass1"  # or pubmedbert_bodhi / biobert_finetuned

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval()

def embed(texts, batch_size=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        with torch.inference_mode():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(**enc)
        lhs = out.last_hidden_state.float().numpy()
        mask = enc["attention_mask"].numpy()[..., np.newaxis].astype(np.float32)
        pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        all_vecs.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-8))
    return np.vstack(all_vecs)

texts = [
    "HbA1c 9.2% sustained hyperglycaemia poor glycaemic control",
    "Patient describes fatigue, low mood and difficulty concentrating daily",
    "Stock market volatility increased this quarter",
]
vecs = embed(texts)
print(vecs @ vecs.T)  # cosine similarity matrix
```

### Load with OpenVINO (faster on Intel CPUs)

```python
import openvino as ov
from transformers import AutoTokenizer
import numpy as np

model_path = "models/openvino/pubmedbert_pass1_bf16"

core = ov.Core()
compiled = core.compile_model(
    f"{model_path}/openvino_model.xml", "CPU",
    {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_PRECISION_HINT": "bf16"}
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
valid_inputs = {i.get_any_name() for i in compiled.inputs}

def embed_ov(texts, batch_size=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                        max_length=512, return_tensors="np")
        out = compiled({k: v for k, v in enc.items() if k in valid_inputs})
        lhs = list(out.values())[0]
        mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
        pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        all_vecs.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-8))
    return np.vstack(all_vecs)
```

---

## 1. Embedding Quality — Main Results

![Embedding Space Geometry — Base (inter/intra 1.05×) → Pass 1 multi-dataset FT (1.63×) → BODHI Pass 2 ontology-graph triplets (2.30×). Six biomedical domains shown as colored clusters with confidence ellipses; BODHI training drives domain separation from overlapping cone to cleanly separable regions.](charts/embedding_space_geometry_progression.png)

![Domain cluster quality — intra/inter ratio across all 7 model variants plus per-domain intra-similarity heatmap. BODHI reaches 2.304×, the highest separation achieved.](charts/b7_domain_cluster_quality.png)

### 1.1 Event Separation: Logically Related vs Disconnected

![Pairwise similarity — all models, cross-domain failures highlighted](charts/pairwise_similarity_all_models.png)

The central clinical NLP question: *can the model reliably separate logically connected events from unrelated ones?*

Benchmark: 16 **connected pairs** (lab marker + its clinical consequence, e.g. elevated HbA1c paired with patient fatigue/mood symptoms) vs 13 **disconnected pairs** (events from unrelated clinical domains).

| Model variant | Connected mean cosine | Disconnected mean cosine | **Discrimination gap** |
|---|---|---|---|
| Untuned base (BF16) | 0.846 | 0.795 | **0.051** |
| **Fine-tuned (BF16)** | **0.684** | **0.382** | **0.302 (+5.9×)** |

The base model cannot be threshold-separated — connected and disconnected pairs both fall in the 0.79–0.85 band. After fine-tuning, connected pairs cluster at ~0.68 and disconnected pairs drop to ~0.38, creating a reliable classification margin.

#### Connected pair examples — fine-tuned model cosine scores

| Event A (clinical marker) | Event B (consequence / context) | Cosine |
|---|---|---|
| PHQ-9 score 18 severe depression, suicidal ideation | Patient admitted to psychiatry ward, started on SSRI and CBT | 0.88 |
| APOE e4 allele amyloid accumulation Alzheimer disease risk | Cognitive decline progressive memory loss executive dysfunction in 65yo | 0.84 |
| HbA1c 9.2% sustained hyperglycaemia poor glycaemic control | Patient describes fatigue, low mood and difficulty concentrating daily | 0.69 |
| Serum cortisol 32 μg/dL markedly elevated HPA axis dysregulation | Patient reports persistent anxiety, sleep disruption and mood instability | 0.69 |
| EGFR exon 19 deletion driver mutation lung adenocarcinoma | Patient started on erlotinib targeted therapy for lung cancer | 0.77 |

#### Disconnected pair examples — fine-tuned model correctly scores low

| Event A | Event B (unrelated) | Cosine |
|---|---|---|
| BNP 820 pg/mL congestive heart failure decompensated | Software deployment completed successfully with zero downtime | **0.21** |
| HbA1c 9.2% poor glycaemic control insulin resistance | Stock market volatility increased investor anxiety this quarter | **0.31** |
| PHQ-9 score 18 severe depression suicidal ideation | Bone density scan DEXA normal T-score bilateral hip and spine | **0.36** |
| BRCA1 pathogenic variant detected in germline DNA | Patient reports work-related stress and difficulty sleeping | **0.42** |

---

### 1.2 Hard-Negative Detection

![B5 Hard negative detection accuracy per model](charts/b5_hard_negative_detection.png)

Hard negatives are clinically similar sentences describing different events. The model must rank the semantically correct match higher than the plausible impostor.

Five triplets (anchor · true positive · hard negative), success = `sim(anchor, pos) > sim(anchor, neg)`:

| Anchor | True positive | Hard negative | Pass? |
|---|---|---|---|
| BRCA1 pathogenic variant breast cancer risk | BRCA2 mutation hereditary cancer | BRCA1 protein DNA damage repair pathway | ✓ |
| Patient persistent low mood and hopelessness | Depressive symptoms anhedonia and fatigue | Patient persistently elevated cortisol | ✓ |
| HbA1c 8.5% poor glycaemic control | Glycated haemoglobin above target in diabetic patient | HbA1c 8.5% test performed at 08:00 fasting | ✓ |
| DNA methylation silences tumour suppressor genes | Epigenetic silencing via promoter hypermethylation | DNA methylation patterns change with ageing | ✓ |
| Anxiety disorder GAD-7 score 15 severe | Generalised anxiety PHQ score indicates severe symptoms | Anxiety about upcoming surgery normal response | ✓ |

**5/5 triplets answered correctly** by the fine-tuned OV-BF16 model. This demonstrates the model's ability to distinguish same-entity different-event descriptions — critical for isomorphism detection in clinical event graphs.

---

### 1.3 BIOSSES-Style Semantic Similarity (Spearman ρ)

![B3 BIOSSES-style Spearman correlation](charts/b3_biosses_spearman.png)

![BIOSSES rank correlation and discrimination gap by model — BODHI (2.304×) and PubMedBERT Pass1 (1.632×) anchor the upper bound](charts/b3_biosses_discrimination_gap.png)

Rank correlation between model-predicted cosine similarity and human-annotated semantic similarity on 15 biomedical sentence pairs (10 related, 5 unrelated/cross-domain), modelled after the BIOSSES benchmark.

| Model | Spearman ρ | p-value |
|---|---|---|
| PubMedBERT Pass1 | **0.775** | 0.00069 |
| PubMedBERT BODHI | 0.773 | 0.00072 |
| BioBERT Fine-Tuned | 0.771 | 0.00076 |
| **Three-model Ensemble** | **0.779** | **0.00063** |

All models are statistically significant (p < 0.001). The three-model ensemble row is reported as a comparison point — its +0.006 ρ advantage on this semantic benchmark does not carry over to the causal-similarity metrics that drive the production stack (see §2.1). These correlations are consistent with published BIOSSES results for BERT-base-scale biomedical encoders.

**Sample predictions vs human scores:**

| Sentence pair | Human | Model |
|---|---|---|
| Blood–brain barrier prevents drug penetration / BBB selective barrier limits CNS access | 0.95 | 0.976 |
| Metformin reduces hepatic glucose via AMPK / Metformin activates AMPK to decrease liver output | 0.92 | 0.968 |
| Hippocampus central to new memory formation / Memory consolidation depends on hippocampus | 0.89 | 0.967 |
| BRCA1 mutation breast cancer risk / Stock market high volatility this quarter | 0.02 | 0.794 *(correctly low)* |

---

### 1.4 Domain Geometry — Within-Domain Cohesion

![B1 Within-domain similarity by domain and model](charts/b1_within_domain_similarity.png)

![B6 Domain geometry — intra vs inter distance and isotropy ratio](charts/b6_domain_geometry_isotropy.png)

How tightly the model clusters sentences from the same clinical domain (B1), and whether domain clusters are separable (B6 — inter/intra ratio target ≥ 1.0, all pre-tuned models fail):

| Domain | PubMedBERT Pass1 | PubMedBERT BODHI | BioBERT FT | Ensemble |
|---|---|---|---|---|
| Genetics (mutations, variants) | 0.943 | 0.944 | 0.904 | 0.923 |
| Biomarkers (lab values, assays) | 0.947 | 0.946 | 0.867 | 0.913 |
| Physiology (vital signs, systems) | 0.932 | 0.931 | 0.852 | 0.898 |
| Psychology (mental health, DSM) | 0.948 | 0.947 | 0.857 | 0.896 |
| Patient journal (subjective mood) | 0.973 | 0.972 | 0.916 | 0.941 |
| Clinical notes (SOAP format) | 0.935 | 0.934 | 0.853 | 0.900 |

PubMedBERT variants show very high within-domain cohesion (0.93–0.97). BioBERT Fine-Tuned shows lower cohesion but the largest discrimination gap (§1.1), reflecting its specialisation for separating rather than grouping events.

---

### 1.5 Cosine Fidelity Under Quantization

![B2 Cross-domain discrimination — lower is better, all pre-tuned models fail](charts/b2_cross_domain_discrimination.png)

![Cosine fidelity under quantization variants](charts/chart_accuracy.png)

![Cross-system quantization accuracy — System A (Xeon 6737P Granite Rapids) and System B (Xeon 8375C Ice Lake-SP AWS). OV-BF16 and OV-INT8 maintain ≥ 0.99 cosine across all three models on both systems; PyTorch dynamic INT8 falls to 0.68–0.93.](charts/quantization_accuracy_xeon_systems.png)

Quantization must not degrade semantic quality. Mean cosine similarity of compressed model embeddings vs the PyTorch FP32 reference (30-sentence biomedical corpus).

| Variant | PubMedBERT Pass1 | PubMedBERT BODHI | BioBERT FT | Every sentence > 0.99? |
|---|---|---|---|---|
| PyTorch FP16 | 0.99999 | 0.99999 | 1.00000 | ✓ |
| PyTorch BF16 | 0.99991 | 0.99989 | 0.99998 | ✓ |
| **OV BF16** | **1.00000** | **1.00000** | **1.00000** | **✓ lossless** |
| **OV INT8** | **0.99720** | **0.99742** | **0.99487** | **✓ < 0.3% drift** |
| OV INT4 | 0.904 ⚠️ | 0.917 ⚠️ | 0.985 ⚠️ | ✗ |
| PyTorch INT8 (dynamic) | 0.930 ⚠️ | 0.930 ⚠️ | 0.759 ⚠️ | ✗ broken |

- **OV-BF16**: cosine 1.0000 on all models — zero quality loss.
- **OV-INT8** (NNCF PTQ, 128 calibration samples): < 0.3% cosine drift, 100% of sentences still above 0.99 threshold. Event separation quality is fully preserved.
- **OV-INT4**: PubMedBERT degraded by > 9%. Not suitable for production event classification.
- **PyTorch dynamic INT8**: cosine 0.68–0.94. Broken on both Granite Rapids and Ice Lake-SP — confirmed to be a property of `torch.quantization.quantize_dynamic`, not the hardware.

---

### 1.6 Optimal Classification Threshold

![Threshold sweep — TPR/TNR/F1 across 0.25–0.65 for BODHI plus F1-by-threshold across all models. Optimal classification boundary at t=0.40 (F1=93.0%).](charts/b8_threshold_sweep.png)

From the cosine threshold sweep (0.25–0.65 range) on the fine-tuned models:

| Use-case | Threshold | F1 | TPR | TNR |
|---|---|---|---|---|
| High-recall event inclusion (retrieval) | 0.50 | 0.73 | 1.00 | ~0 |
| Balanced event classification | 0.60–0.62 | peak | ~0.88 | ~0.77 |
| High-precision event deduplication | 0.65 | lower | ~0.70 | ~0.95 |

These thresholds apply to the fine-tuned models only. The base model has no viable threshold — there is no value in [0.25, 0.65] that achieves TNR > 0 while maintaining TPR = 1.0.

---

## 2. Models

All three are BERT-base (12-layer, 768-hidden, 12-head, ~110 M parameters), fine-tuned for biomedical sentence embedding on paired clinical/research text.

| Model | HF Base | Vocab | Avg tokens/sentence | Directory |
|---|---|---|---|---|
| **PubMedBERT Pass1** | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | 30,522 (uncased) | 13.4 | `models/pytorch/pubmedbert_pass1` · `models/openvino/pubmedbert_pass1_bf16` |
| **PubMedBERT BODHI** | PubMedBERT, BODHI fine-tune stage | 30,522 (uncased) | 13.4 | `models/pytorch/pubmedbert_bodhi` · `models/openvino/pubmedbert_bodhi_bf16` |
| **BioBERT Fine-Tuned** | `dmis-lab/biobert-v1.1` | 28,996 (cased) | 21.0 | `models/pytorch/biobert_finetuned` · `models/openvino/biobert_finetuned_bf16` |

**PubMedBERT Pass1** — single-stage fine-tune; strongest within-domain cohesion, highest Spearman ρ. Recommended default for clinical NLP retrieval.

**PubMedBERT BODHI** — BODHI-stage fine-tune emphasising biomarker domain discrimination; marginal gains on lab-value texts over Pass1.

**BioBERT Fine-Tuned** — broader cased vocabulary, longer tokenization. Lower within-domain cohesion but largest hard-negative discrimination margin. Recommended for cross-domain event separation tasks.

![Training data composition and fine-tuning duration per model (BODHI 14.8 h on 64-core AMX server, 72,034 pairs, lr=2e-5, Matryoshka-MNRL)](charts/training_config_and_data.png)

### 2.1 Comparative study — BioM-ELECTRA Large and the three-model ensemble

Earlier iterations of this benchmark carried a fourth encoder (BioM-ELECTRA Large) and an averaging ensemble (BioBERT + PubMedBERT + BioM-ELECTRA). Both were run end-to-end against the full B1–B8 quality suite alongside the production stack. The results are reported here as comparison points; the production deployment uses the three BERT-family models above.

![BioBERT v1.1 + PubMedBERT + BioM-ELECTRA → L2 normalize → average → 768-dim ensemble configuration used in the comparative study](charts/lbm_ensemble_architecture.png)

**BioM-ELECTRA Large**

| Metric | BioM-ELECTRA | PubMedBERT Pass1 | PubMedBERT BODHI | BioBERT FT |
|---|---|---|---|---|
| BIOSSES Spearman ρ | 0.518 | 0.775 | 0.773 | 0.771 |
| Intra/inter cluster ratio | 1.059 → 1.068 (base → FT) | → 1.632 | → **2.304** | → 1.742 |
| Hidden dim | 1024 | 768 | 768 | 768 |
| Pre-training objective | RTD on SQuAD2 | MLM on PubMed abstracts | MLM + BODHI ontology triplets | MLM on PubMed+PMC |
| OV-BF16 peak throughput (bs=256) | 240.4 sps | 617.8 sps | 524 sps | 439.5 sps |
| OV-BF16 single-query p50 | 25.7 ms | 9.1 ms | ~9 ms | 9.4 ms |
| Causal-axis gain from fine-tuning | +0.009 (1.059 → 1.068) | +0.584 | **+1.256** | +0.601 |

Replaced-token-discrimination pre-training gives ELECTRA strong token-level representations but a sentence-pool geometry that is harder to bend along causal pathways. Under the identical fine-tuning regime that takes PubMedBERT from 1.05× to 2.30×, ELECTRA's intra/inter ratio moves by 0.009. Combined with a 1024-dim hidden state (requires a projection head for averaging), 2.8× higher latency and 2.5× lower throughput, BioM-ELECTRA was not carried into the production stack.

**Three-model ensemble (BioBERT FT + PubMedBERT BODHI + BioM-ELECTRA)**

| Metric | Ensemble | BODHI alone | Δ |
|---|---|---|---|
| BIOSSES Spearman ρ | 0.779 | 0.773 | +0.006 (within noise, both p < 0.001) |
| Connected-pair mean cosine | 0.71 | 0.68 | within noise |
| Discrimination gap | 0.28 | **0.302** | −0.022 |
| Cross-domain disconnected mean | 0.43 | 0.382 | +0.048 |
| Single-query p50 (3 sequential forwards) | ~28 ms | 9.1 ms | 3.1× |
| Memory footprint | ~1.88 GB | 627 MB | 3× |
| Quantization | Per-model INT8 calibration + post-average renorm | Single calibration set | — |

The ensemble's +0.006 BIOSSES gain reflects the benchmark's semantic-similarity definition. On the causal-similarity metrics — discrimination gap, cross-domain separation, intra/inter ratio — averaging acts as a smoother: BODHI's installed causal curvature is partly cancelled by ELECTRA's flatter axis and BioBERT's narrower within-domain cohesion. The ensemble therefore trades 0.022 of discrimination gap and 3.1× of latency for 0.006 of semantic ρ.

**Production stack:** PubMedBERT BODHI (primary, intra/inter 2.304×), PubMedBERT Pass1 (within-domain cohesion fallback), BioBERT Fine-Tuned (cross-domain event-separation specialist). All three share 768-dim, AMX-BF16, and a single quantization calibration set.

---

## 3. Inference Efficiency (Secondary)

![Peak TPS comparison](charts/chart_peak_tps.png)

![Cross-platform comparison — Ice Lake FP32 vs Ice Lake VNNI INT8 vs Granite Rapids AMX BF16, bs=256. AMX BF16 delivers 13–27× speedup over the c6i FP32 baseline.](charts/amx_cross_platform_speedup.png)

### 3.1 Peak throughput — OV-INT8, 32 workers + HT, 600 clients, bs=256

| Model | Peak TPS | p50 latency | p99 latency | vs PyTorch-BF16 |
|---|---|---|---|---|
| PubMedBERT Pass1 | **135,668** | 57 ms | 114 ms | +38% |
| PubMedBERT BODHI | **135,721** | 58 ms | — | +33% |
| BioBERT Fine-Tuned | **182,469** | 67 ms | — | +55% |

![Head-to-head PyTorch vs OpenVINO](charts/chart_head_to_head.png)

### 3.2 All variants at 32srv+HT / 600 clients

| Variant | PubMedBERT Pass1 TPS | BioBERT TPS | Embedding quality |
|---|---|---|---|
| PyTorch FP32 | 34,451 | 27,440 | Reference |
| PyTorch FP16 | 92,814 | 114,888 | Lossless |
| PyTorch BF16 | 98,192 | 118,086 | Lossless |
| OV BF16 | 110,215 | 124,160 | Lossless |
| **OV INT8** | **135,668** | **182,469** | **< 0.3% drift** ✓ |
| OV INT4 | 85,877 | 88,150 | 9% degradation ⚠️ |
| PyTorch INT8 | ~627 *(single proc)* | ~997 | Broken ⚠️ |

![TPS vs batch size](charts/chart_tps_vs_bs.png)
![Concurrency scaling](charts/chart_concurrency.png)

![Sentences-per-second heatmap — all three fine-tuned models across batch size (256/512/1024) and worker count (1/2/4/8) on AMX BF16](charts/throughput_heatmap_workers_bs.png)

![Time-to-first-embedding (TTFT) vs concurrency at bs=256 — BODHI achieves 3.6 ms at 4 workers, the lowest TTFT in the sweep](charts/ttft_vs_concurrency.png)

### 3.3 Inference variants

| Variant | Engine | Precision | Hardware path |
|---|---|---|---|
| `pytorch-fp32` | PyTorch 2.11 | FP32 | AVX-512 |
| `pytorch-fp16` | PyTorch 2.11 | FP16 autocast | AVX-512 |
| `pytorch-bf16` | PyTorch 2.11 | BF16 autocast | AMX-BF16 tiles |
| `ov-bf16` | OpenVINO 2026.1 | BF16 IR | AMX-BF16 tiles |
| `ov-int8` | OpenVINO + NNCF PTQ | INT8 | AMX-INT8 tiles |
| `ov-int4` | OpenVINO + NNCF weight compress | INT4 | AMX-INT8 (dequant) |

### 3.4 Why OV-INT8 is the production recommendation

- **Graph fusion**: OV fuses QKV projections, LayerNorm + GeLU, residual adds. PyTorch eager-mode dispatches each op separately.
- **AMX dispatch**: OV NNCF INT8 dispatches directly to AMX INT8 tiles. PyTorch `quantize_dynamic` scales activations per-tensor on every forward pass, blocking the AMX path entirely.
- **NUMA-aware scheduling**: OV `THROUGHPUT` hint picks stream count and threads-per-stream matched to the L3 cache topology per NUMA node.

### 3.5 PyTorch dynamic INT8 — why it is broken

`torch.quantization.quantize_dynamic` on `nn.Linear` applies per-tensor activation scaling at every inference call. This prevents AMX tile dispatch. TPS **decreases** as batch size grows (opposite of all other variants). Mean cosine drops to 0.68–0.94 depending on model. Failure reproduces on both Granite Rapids (Xeon 6737P) and Ice Lake-SP (Xeon Platinum 8375C on AWS).

**Do not use `torch.quantization.quantize_dynamic` for biomedical BERT production on any Intel CPU.**

---

## 4. System Under Test

| Property | Value |
|---|---|
| CPU | 2× Intel Xeon 6737P (Granite Rapids) |
| Sockets / Cores / Threads | 2 sockets · 32 cores/socket · 2 threads/core = **128 logical CPUs** |
| Base / max turbo | 2.90 GHz / 4.00 GHz |
| NUMA | 2 nodes — Node 0: CPUs 0–31, 64–95 · Node 1: CPUs 32–63, 96–127 |
| L1d / L1i | 3 MiB / 4 MiB (64 instances each) |
| L2 | 128 MiB total (2 MiB per core) |
| L3 | **288 MiB total** (144 MiB per NUMA node) |
| RAM | **1,024 GB DDR5 @ 6400 MT/s** — 32 × 64 GB Samsung M321R8GA0BB0-CQKMG |
| NUMA node 0 usable | ~503 GB |
| NUMA node 1 usable | ~500 GB |
| Storage | 2 × Micron 7450 PRO NVMe SSD |
| ISA extensions | AVX-512 VNNI · **AMX-BF16** · **AMX-INT8** |
| OS / kernel | Ubuntu 24.04.4 LTS · 6.8.0-110-generic |
| PyTorch | 2.11.0+cpu |
| OpenVINO | 2026.1.0 |
| NNCF | 3.1.0 |
| transformers | 4.48.0 |

---

## 5. Repository Layout

```
intel_xeon_biobert_bench/
├── README.md                        ← this file (Hugging Face dataset card)
├── config.yaml                      ← hardware & inference config — edit before running
├── requirements.txt
├── .gitattributes                   ← Git LFS tracking for model weights
│
├── code/
│   ├── hardware_config.py           ← reads config.yaml; imported by all scripts
│   ├── config.py                    ← model path config
│   ├── run_all_xeon.sh              ← master driver: all 6 variants with APS/VTune
│   ├── quantize.py                  ← NNCF INT8 PTQ + INT4 weight compression
│   ├── accuracy_eval.py             ← cosine fidelity vs PT-FP32
│   ├── extended_benchmark.py        ← B1–B6 quality suite (BIOSSES, hard-neg, geometry)
│   ├── compare_finetuned.py         ← event separation: connected vs disconnected pairs
│   ├── threshold_sweep.py           ← cosine threshold sweep (TPR / TNR / F1)
│   ├── bench_all.py                 ← single-process throughput sweep
│   ├── scenario_bench.py            ← full grid: batch × workers × variant (288 configs)
│   └── load_test.py                 ← multi-instance dynamic-batching serving test
│
├── models/
│   ├── pytorch/
│   │   ├── pubmedbert_pass1/        ← SafeTensors + tokenizer (~420 MB)
│   │   ├── pubmedbert_bodhi/        ← SafeTensors + tokenizer (~420 MB)
│   │   └── biobert_finetuned/       ← SafeTensors + tokenizer (~415 MB)
│   └── openvino/
│       ├── pubmedbert_pass1_bf16/   ← openvino_model.xml + .bin (~207 MB)
│       ├── pubmedbert_bodhi_bf16/   ← openvino_model.xml + .bin (~207 MB)
│       └── biobert_finetuned_bf16/  ← openvino_model.xml + .bin (~207 MB)
│
├── results/
│   ├── accuracy_all_variants.json   ← cosine fidelity for 5 variants × 3 models
│   ├── pytorch_int8_throughput.json
│   ├── pubmed/                      ← 3 load-test result JSONs for PubMedBERT Pass1
│   ├── bodhi/                       ← 3 load-test result JSONs for PubMedBERT BODHI
│   └── biobert/                     ← 3 load-test result JSONs for BioBERT
│
├── profiling/
│   ├── vtune/                       ← 6 CSV exports: hotspots, µArch, memory (PubMedBERT Pass1)
│   └── aps/                         ← APS reports: GFLOPS, IPC, DRAM BW (3 models)
│
├── charts/                          ← 5 PNG charts
│
└── docs/
    ├── FINAL_CONSOLIDATED_REPORT.md ← full throughput + profiling analysis
    ├── PYTORCH_RESULTS.md           ← complete PyTorch throughput data tables
    └── REASONING_QNA.md             ← decision log (NUMA, HT, INT8, thresholds)
```

---

## 6. Quick Start

### 1. Configure for your hardware

Edit `config.yaml` before running:

```yaml
# Single-socket 16-core machine example:
numa:
  cpunodebind: "0"
  membind: "0"
pytorch:
  omp_num_threads: 16        # physical core count, no HT
openvino:
  inference_precision: "bf16"  # use "f32" for CPUs without AMX/AVX-512 BF16
```

On CPUs without AMX (Ice Lake, Cascade Lake, Skylake): set `inference_precision: "f32"`.

### 2. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Run embedding quality benchmarks

```bash
cd code

# Main quality benchmark: event separation (connected vs disconnected pairs)
python compare_finetuned.py --model pubmed
python compare_finetuned.py --model bodhi
python compare_finetuned.py --model biobert

# Extended quality suite: BIOSSES, hard-negatives, domain geometry (B1–B6)
python extended_benchmark.py --variant ov-bf16

# Cosine fidelity under quantization
python accuracy_eval.py --model all --out ../results/accuracy_all_variants.json

# Classification threshold sweep
python threshold_sweep.py
```

### 4. Quantize to INT8 + INT4 (one-time, ~10 min)

```bash
cd code && python quantize.py --model all --precision all
# outputs: models/openvino_quantized/{pubmed,bodhi,biobert}-{int8,int4}/
```

### 5. Run throughput benchmarks

```bash
cd code
./run_all_xeon.sh               # with Intel APS profiling
./run_all_xeon.sh --no-aps      # plain runs (no oneAPI needed)
./run_all_xeon.sh --vtune       # + VTune hotspot capture
RUN_HEAVY=1 ./run_all_xeon.sh   # + full 288-run scenario sweep + load test (~3–4 hr)
```

All outputs go to `results/xeon_run_<timestamp>/`.

---

## 7. Profiling Data

### VTune CSV exports (`profiling/vtune/`) — PubMedBERT Pass1

| File | Analysis type | What it shows |
|---|---|---|
| `hotspots_pubmed_summary.csv` | Hotspots | Top-level CPU time breakdown |
| `hotspots_pubmed_functions.csv` | Hotspots | Per-function CPU time |
| `uarch_exploration_pubmed_summary.csv` | µArch Exploration | AMX tile utilization, port usage, retiring % |
| `uarch_exploration_pubmed_functions.csv` | µArch Exploration | Per-function pipeline metrics |
| `memory_access_pubmed_summary.csv` | Memory Access | DRAM bandwidth, LLC miss rate |
| `memory_access_pubmed_functions.csv` | Memory Access | Per-function memory metrics |

### APS reports (`profiling/aps/`) — all 3 models at optimal config

Reports captured at 32srv+HT / 600 clients — GFLOPS, IPC, DRAM BW, phys-core utilisation:

- `aps_pubmed.txt` — PubMedBERT Pass1: ~113.5 GB/s DRAM BW sustained
- `aps_bodhi.txt` — PubMedBERT BODHI
- `aps_biobert.txt` — BioBERT Fine-Tuned

DRAM bandwidth at peak: ~113.5 GB/s (32srv+HT / 600c), rising to ~139 GB/s at saturation. This is ~40–50% of the DDR5-6400 practical ceiling — the system is **compute-bound at AMX tile throughput + L3 hit rate**, not memory-bound.

---

## Summary

| Metric | Value |
|---|---|
| Discrimination gap improvement (fine-tuned vs base) | **+5.9×** (0.302 vs 0.051) |
| BIOSSES Spearman ρ | **0.775–0.779** (p < 0.001, all models) |
| Hard-negative accuracy | **5/5 triplets correct** |
| OV-INT8 cosine fidelity | **≥ 99.4%** vs FP32 reference |
| OV-INT4 on PubMedBERT | ⚠️ Not recommended — 9% cosine degradation |
| PyTorch dynamic INT8 | ⚠️ Broken on all tested Intel CPUs |
| Peak throughput (OV-INT8) | **135K TPS** (PubMedBERT) · **182K TPS** (BioBERT) |
| Production p50 latency | **57–67 ms** at 600 concurrent clients |
| Recommended serve config | 32 workers + HT · 600 clients · bs=256 · numactl 0,1 |
| Total repo size | ~1.9 GB (3× PyTorch ~420 MB + 3× OV ~207 MB) |

---

## Acknowledgements

All training, quantization, and benchmark numbers in this repository were produced on an **Intel® Xeon® 6737P (Granite Rapids)** dual-socket system (128 logical CPUs, 1 TB DDR5-6400, AMX-BF16 / AMX-INT8) made available to Dotsin.ai by **Intel Corporation**. The platform delivered the measurable gains that the rest of this document is built on:

- **4.6 – 6.2× throughput** at bs=256 from OpenVINO BF16 vs PyTorch FP32 across all three encoders, with cosine fidelity ≥ 99.4 % under INT8 PTQ.
- **9 – 10 ms p50 single-query latency** — **113 – 145× lower than PyTorch FP32** — enabling embed-on-arrival in the secure data hub.
- **135 K – 182 K TPS** at the production serving preset (32 workers + HT, 600 clients, bs=256); p50 57 – 67 ms, p99 114 ms.
- **17 – 27× speedup** vs Ice Lake-SP (c6i FP32); **3 – 4×** vs c6i VNNI INT8.
- Two-pass BERT-base fine-tune iterated overnight: Pass 1 ~6.5 h, Pass 2 BODHI ~14.8 h.
- **288 MiB L3 (144 MiB per NUMA node)** keeps BERT-base BF16 weights resident in cache per node; **1 TB DDR5-6400** leaves clear headroom for longer context, larger batches, concurrent model instances, and distilled successor models on the same configuration.

This capacity envelope is what carries our **early production embedding traffic** for the secure data hub on this hardware directly — a single 2-socket Granite Rapids node serves the full ingest path at the latency our LBM streams require. **For groups reproducing or extending this work,** a Granite Rapids-class node with AMX-BF16 / AMX-INT8 paired with OpenVINO 2026.1 + NNCF 3.1 + oneDNN is the configuration that reproduces the CPU and DRAM utilization profile end-to-end and gives you the same headroom for production-scale workloads. Full per-metric breakdown, profiling tables, and base-model / dataset credits in [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md).

The weights and benchmark suite released here are the **open public version of one layer of Dotsin.ai's larger system**. Dotsin maintains a deeper-tuned proprietary causal-similarity embedding stack trained against the full BODHI ontology and additional behavioural corpora that we cannot publish for consent, privacy, and clinical-evidence reasons; the LBM service and the proprietary LBM graph remain closed. This repository exists so the open biomedical-NLP community can see the research direction, evaluate the causal-axis framing on their own data, and build something stronger on top of it.

Paper: [arXiv:2606.09672](https://arxiv.org/abs/2606.09672) · Software citation: [`CITATION.cff`](CITATION.cff).

*Intel, the Intel logo, Intel Xeon, AMX, OpenVINO, oneAPI, oneDNN and VTune are trademarks of Intel Corporation or its subsidiaries.*

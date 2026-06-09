# Results at a glance

A one-page summary of what the fine-tuning achieved. Two equivalent ways to state the headline — pick whichever fits the audience.

## The headline claim

> **The discrimination gap between connected and disconnected biomedical event pairs went from 0.051 (base) to 0.302 (fine-tuned) — a 5.9× improvement.** This is the metric the whole repository is built around: connected events (e.g. `HbA1c 9.2%` ↔ `fatigue, low mood`) end up close in embedding space, while disconnected events (e.g. `BRCA1 variant` ↔ `stock market volatility`) end up far apart, even when their surface forms are similar.

If you want a percentage-style framing instead:

> **Cross-domain discrimination accuracy went from 0% (base) to 100% (fine-tuned).** Every pre-trained BERT-family baseline assigned cosine > 0.5 to semantically unrelated cross-domain pairs that should sit near 0; after two-stage fine-tuning, every such pair is correctly pushed apart.

## Headline metrics — base → fine-tuned

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| **Discrimination gap** (connected − disconnected mean cosine) | 0.051 | **0.302** | **+5.9×** |
| **Cross-domain discrimination accuracy** (B2) | 0 % | **100 %** | total failure → resolved |
| **Hard-negative triplet accuracy** (B5) | 60–80 % | **100 %** (5/5) | +20 – 40 pp |
| **Pairwise similarity accuracy** | 80 % | **> 95 %** (production tier) | +15 pp |
| **BIOSSES Spearman ρ** (B3, semantic check) | ~0.77 | **0.775** | unchanged (semantic quality preserved) |

The semantic-similarity score (BIOSSES ρ) does not improve after fine-tuning — and that is the point. The fine-tune installs a second axis (causal similarity) on top of the existing semantic geometry without regressing it.

## Domain geometry — intra/inter cluster ratio

The intra/inter ratio is the cleanest geometric measure of how well-separated the domain clusters are. Target for production: > 1.5.

| Model | Base | Fine-tuned | Δ |
|---|---|---|---|
| PubMedBERT Pass 1 | 1.048 × | 1.632 × | +0.584 |
| **PubMedBERT BODHI** | 1.048 × | **2.304 ×** | **+1.256** (best) |
| BioBERT FT | 1.141 × | 1.742 × | +0.601 |
| BioM-ELECTRA *(retired — see [README §2.1](README.md))* | 1.059 × | 1.068 × | +0.009 (flat) |

BODHI's Pass-2 ontology-triplet training is what closes the gap from ~1.6 × (semantic fine-tune alone) to 2.30 × (semantic + causal).

## Concrete numbers behind the discrimination gap

These four pairs from [`examples/sample_data/connected_disconnected_pairs.jsonl`](examples/sample_data/connected_disconnected_pairs.jsonl) illustrate the geometry change:

| Pair | Surface / Semantic | Causal trajectory | Required geometry | Achieved cosine (fine-tuned) |
|---|---|---|---|---|
| `HbA1c 9.2%` ↔ `fatigue, low mood, difficulty concentrating` | Distant | Hyperglycaemia → neuroglycopenic fatigue / mood | CLOSE | **0.69** ✓ |
| `Cortisol 32 μg/dL HPA axis dysregulation` ↔ `anxiety, sleep disruption, mood instability` | Distant | HPA-axis → anxiety / insomnia | CLOSE | **0.69** ✓ |
| `BRCA1 pathogenic variant` ↔ `stock market volatility raised investor anxiety` | Near (shared "risk/variant") | None | FAR | **0.42** ✓ |
| `PHQ-9 score 18 severe depression` ↔ `DEXA bone density normal` | Near (same clinical register) | None | FAR | **0.36** ✓ |

A pure semantic encoder would invert all four. The fine-tuned model gets all four right.

## Quotable one-liners

> Two-stage fine-tuning raised the connected-vs-disconnected discrimination gap from **0.051 to 0.302 — a 5.9 × improvement** — and lifted cross-domain event-separation accuracy from **0 % to 100 %**, while holding BIOSSES Spearman ρ at **0.775 (p < 0.001)**.

> The BODHI intra/inter cluster ratio rose from **1.05 × to 2.30 ×**, the highest of any model evaluated.

## Underlying data

The full benchmark JSON is in [`results/`](results/) — `accuracy_all_variants.json` for the cosine-fidelity numbers, and the per-model load-test runs in `results/{biobert,pubmed,bodhi}/`. The B1–B8 source tables are in [README §1](README.md) and [docs/LBM_INTEGRATION.md](docs/LBM_INTEGRATION.md) §5–§6.

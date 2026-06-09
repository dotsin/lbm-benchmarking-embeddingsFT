# Acknowledgements

## Hardware and toolchain

The training, quantization, and full benchmark suite in this repository were produced on an **Intel® Xeon® 6737P (Granite Rapids)** dual-socket system (2 sockets × 32 cores, 128 logical CPUs, 1 TB DDR5-6400) made available to Dotsin.ai by **Intel Corporation**. The platform is what turned a CPU-only training and serving stack from a fallback option into a first-class production substrate for our biomedical embedding workloads. Concretely, the hardware delivered:

| Capability | Measurement | What it unlocks |
|---|---|---|
| **AMX-BF16 fine-tuning** | Pass 1 ~6.5 h, Pass 2 ~14.8 h (BODHI, 72 034 pairs, batch 128, MNRL + Matryoshka) | Two-pass BERT-base fine-tune iterated overnight per cycle |
| **AMX-BF16 inference vs PyTorch FP32** | **4.6 – 6.2× throughput** across BioBERT / PubMedBERT / ELECTRA at bs=256 | Single-process serving budget meets real-time data-hub ingest |
| **AMX-BF16 single-query latency** | **9 – 10 ms p50** (down from 1.3 – 2.9 s PyTorch FP32) — **113 – 145× latency reduction** | Real-time embed-on-arrival in the secure data hub without queueing |
| **AMX-INT8 NNCF PTQ (production preset)** | **135 K – 182 K TPS** at 32 workers + HT, 600 clients, bs=256; p50 latency 57 – 67 ms; cosine fidelity ≥ 99.4 % vs FP32 | Early production target met on a single 2-socket node — no GPU fleet needed for first-wave deployment |
| **Cross-platform headroom vs Ice Lake-SP (c6i)** | **17 – 27× speedup** at bs=256 (AMX BF16 vs c6i FP32); **3 – 4× speedup** (c6i VNNI INT8 vs c6i FP32) | Granite Rapids is the inflection where the CPU path matches or beats commodity cloud serving for BERT-class embedding workloads |
| **Memory headroom on top of AMX-saturated compute** | 1 TB DDR5-6400; observed working-set leaves substantial bandwidth and capacity free under peak serving | Room to scale **batch, context length, concurrent model instances, or larger successor encoders** on the same node without re-architecting the serving topology |

The combination of AMX tile throughput and **288 MiB L3 (144 MiB per NUMA node)** is a particularly good fit for BERT-base sentence embedding: full-precision BF16 weights (~208 MB) sit entirely in cache per NUMA node, which is what produces the 4.6 – 6.2× OpenVINO speedup over PyTorch eager mode and the 9 – 10 ms p50 single-query latency. The same shape carries our **early production embedding traffic for the secure data hub** on this hardware directly — a single 2-socket Granite Rapids node serves the full ingest path at the latency our LBM information-stream assembly requires, and the 1 TB DDR5-6400 memory pool leaves clear headroom for memory-heavier follow-ups (longer context, larger batches, multi-instance concurrent serving, distilled successor models) on top of the same configuration.

**For groups reproducing or extending this work:** a Granite Rapids-class node with AMX-BF16 / AMX-INT8, paired with OpenVINO 2026.1 + NNCF 3.1 + oneDNN, is the configuration that lets you get the CPU and DRAM utilization profile reported here end-to-end — fine-tune Pass 1 + Pass 2 overnight, OpenVINO BF16 / INT8 inference dispatched directly to AMX tiles, and weights resident in L3 per NUMA node for the full BERT-base layer. The same hardware envelope makes both the public release and our internal production traffic feasible on a single 2-socket node, and the same envelope is what we recommend if you want to run this stack at production scale yourself.

We additionally use the open Intel software stack end-to-end:

- **OpenVINO™** (2026.1) and **NNCF** (3.1) — model export, INT8 post-training quantization, AMX dispatch
- **oneDNN** — AMX BF16 / AMX INT8 kernel routing under PyTorch and OpenVINO
- **Intel® VTune™ Profiler** and **Intel® Application Performance Snapshot (APS)** — the microarchitectural data behind the profiling tables in [`docs/FINAL_CONSOLIDATED_REPORT.md`](docs/FINAL_CONSOLIDATED_REPORT.md)

All Intel marks are the property of Intel Corporation.

## Relationship to Dotsin's proprietary stack

This repository is the **open public release** of a single layer of Dotsin.ai's larger system. The three fine-tuned BERT-base encoders published here (PubMedBERT BODHI, PubMedBERT Pass 1, BioBERT Fine-Tuned) are the **retrieval geometry of the secure data hub** that fronts Dotsin's Large Behavioral Model (LBM). Dotsin maintains a separate proprietary causal-similarity embedding stack — a deeper-tuned superset of the geometry presented here, trained against the full BODHI ontology and additional behavioural corpora that we cannot publish for consent, privacy, and clinical-evidence licensing reasons. The LBM service itself and the proprietary LBM graph (millions of behavioural data points along counterfactually-derived causal chains) remain closed.

We publish this layer publicly because the open biomedical-NLP and clinical-NLP communities are the right place to push **causal-similarity sentence embeddings** forward. Our goal in releasing the weights, the full benchmark suite, the failure modes, and the comparison studies (BioM-ELECTRA, three-model averaging ensemble) is twofold:

1. **Make our research direction visible** so that other groups can evaluate and challenge the causal-axis framing, not just the numbers.
2. **Give the community a base to build on** — better backbones than BERT-base, leaner fine-tuning recipes, richer ontologies than BODHI, sharper benchmarks for the causal axis, derivative models for adjacent clinical-text tasks. The Apache-2.0 license and the gated registration model are both designed to allow this kind of follow-up while keeping a public record of who is working in the direction.

If you build something on top of this work, we would love to know — drop a note in the HF Discussions tab on the model page, or open an issue on the GitHub repository.

## Open datasets and base models

The Pass 1 multi-dataset fine-tune is built directly on top of open community resources. We thank the maintainers of:

| Resource | Type | Source |
|---|---|---|
| BioBERT v1.1 (`dmis-lab/biobert-v1.1`) | Pre-trained encoder | Lee et al., DMIS Lab, Korea University |
| PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) | Pre-trained encoder | Microsoft Research |
| BioM-ELECTRA Large (`sultan/biom-electra-large`) | Pre-trained encoder (comparison study) | Sultan Alrowili, University of Delaware |
| BIOSSES | Biomedical STS gold standard | Soğancıoğlu et al., 2017 |
| MedNLI | Clinical NLI from MIMIC-III | Romanov & Shivade, 2018 |
| PubMedQA | Biomedical QA | Jin et al., 2019 |
| all-nli | General NLI bundle | Sentence-Transformers project |
| medical_q_pairs | Clinical question pairs | Curai Health |
| emotion / go_emotions | Psychology classification corpora | DAIR.AI / Google Research |
| mental_health_counseling_conversations | Counselling Q/A | Amod (Hugging Face) |

The Pass 2 BODHI training data is Dotsin's own ontology-grounded triplet export and is not redistributed in the open weights; it is referenced here only to make the fine-tuning recipe reproducible. See [`docs/FINETUNING.md`](docs/FINETUNING.md) §3 for access and §5 for a synthetic-data generation recipe that produces equivalent triplets from any public biomedical ontology (UMLS, SNOMED-CT, MeSH, HPO, MONDO).

## Open software stack

| Library | Role |
|---|---|
| PyTorch 2.11 (CPU build) | Training and PT-FP32/FP16/BF16 inference |
| Transformers 4.48 | Model loading, tokenization |
| sentence-transformers 3.3.1 | MNRL + Matryoshka loss, Trainer wrapper |
| `datasets` (Hugging Face) | Streaming the public corpora |
| `huggingface_hub` | Weight publication and gated access |
| OpenVINO 2026.1 + NNCF 3.1 | BF16 IR export, INT8 PTQ |
| Optimum-Intel | `optimum-cli export openvino` conversion path |

## Citation

If you use the weights, the benchmark suite, or the causal-similarity framing in academic work, please cite:

```bibtex
@misc{dotsin2026causal,
  title  = {Biomedical BERT Embedding Quality \& Inference Benchmark — A Causal-Similarity Sentence Embedding Layer for the Large Behavioral Model},
  author = {{Dotsin.ai}},
  year   = {2026},
  eprint = {2606.09672},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  howpublished  = {\url{https://huggingface.co/Dotsin/lbm-benchmarking-embeddingsFT}},
}
```

A machine-readable software citation is provided in [`CITATION.cff`](CITATION.cff).

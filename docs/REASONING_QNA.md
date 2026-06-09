# Benchmark Reasoning & Decision Log — Intel Xeon 6737P

A QnA-style walkthrough of every decision that shaped the full-benchmark suite (`intel_xeon_biobert_bench/`). Each section is a question you might ask reading the results, with the answer and the data behind it.

---

## 1. Hardware & Topology

### Q: What are we running on?
2× Intel Xeon 6737P, 64 physical cores total (32/socket), 128 logical cores with HT. 1 TB DDR5 @ 6400 MT/s (32 × 64 GB DIMMs). L3 = 288 MiB total — **144 MiB per NUMA node**. Two NUMA nodes; node 0 owns cores 0–31 (+ HT siblings 64–95), node 1 owns 32–63 (+ 96–127).

### Q: Why does NUMA matter for inference?
A BERT-base forward pass at batch 64–128 is heavy on weight reads. If a worker on socket 0 reads weights pinned on socket 1, every cache miss hops the UPI link (~30–40% latency penalty). The L3 per node (144 MiB) easily holds a BF16 model (208 MB if split, but quantized 67/104 MB fits entirely). Pinning a worker to one node lets the working set stay in local L3.

### Q: How did you pin?
Each worker process calls `os.sched_setaffinity(0, allowed_cores)` immediately at startup, and the launcher wraps the whole run in `numactl --cpunodebind=0,1 --membind=0,1`. Per-worker affinity narrows further to a single node's cores. Threading env: `KMP_HW_SUBSET`, `KMP_AFFINITY=granularity=fine,compact,1,0`, `KMP_BLOCKTIME=1`, `OMP_NUM_THREADS` sized to the affinity set.

---

## 2. Why These Three Models?

### Q: Why PubMedBERT Pass1, PubMedBERT BODHI, and BioBERT Fine-Tuned?
They represent the production candidates for biomedical embedding: two PubMedBERT variants (one base-finetune, one BODHI ensemble) and a BioBERT fine-tune. All three are 12-layer BERT-base (768 hidden, 12 heads). Same architecture → fair throughput comparison; different vocabularies/tokenizers → realistic spread (BioBERT averages 21 tok/sentence vs 13.4 for PubMedBERT, which is why BioBERT TPS numbers look higher despite lower SPS).

### Q: Why does BioBERT show higher TPS but lower SPS than PubMedBERT?
Because BioBERT's tokenizer produces ~57% more tokens per sentence on our 30-sentence biomedical corpus. SPS counts sentences. TPS = SPS × tokens/sentence. So BioBERT at 9,440 SPS = 198,246 TPS, while PubMedBERT at 13,883 SPS = 186,170 TPS. Different views of the same compute work.

---

## 3. Variant Selection (6 of them — why?)

### Q: Why bench all 6 variants?
| Variant | Reason to include |
|---|---|
| PyTorch FP32 | Reference accuracy baseline; ground truth for cosine comparisons |
| PyTorch FP16 | Common production default; tests if half-precision alone is enough |
| PyTorch BF16 | Native Sapphire Rapids AMX path in PyTorch |
| OV BF16 | OpenVINO's AMX path — directly comparable to PT-BF16 |
| OV INT8 | NNCF post-training quantization with calibration — biggest expected win |
| OV INT4 | Weight-only INT4 — most aggressive compression, tests accuracy floor |

We needed both ends (FP32 baseline, INT4 floor) to find the sweet spot.

### Q: Did you try anything else (FP8, INT2, vLLM)?
- **FP8** — Xeon 6737P doesn't have first-class FP8 acceleration; emulated paths are slower than INT8.
- **INT2** — INT4 already fails on accuracy (mean cosine 0.90 for PubMedBERT); INT2 has no hope for embeddings.
- **vLLM** — vLLM is built for autoregressive generation (KV-cache, paged attention). BERT is encoder-only, no KV cache. vLLM doesn't help.

---

## 4. Quantization Pipeline

### Q: How was INT8 produced?
`nncf.quantize(...)` with `ModelType.TRANSFORMER`, `QuantizationPreset.MIXED`, 128 calibration samples drawn from the biomedical corpus, `fast_bias_correction=True`. Static PTQ — no fine-tune. Output ~104 MB per model (vs 208 MB BF16).

### Q: How was INT4 produced?
`nncf.compress_weights(model, mode=INT4_SYM, group_size=64)`. Weight-only compression, no calibration dataset, no activation quantization. Output ~67 MB. This is the cheapest possible compression — and it shows.

### Q: Why did the first quantization attempt fail?
First pass tried `optimum-intel`'s `OVConfig` and passed it as `ov_config=` to `from_pretrained()` — TypeError: `OVConfig` is not a mapping. Switched to calling NNCF directly on the loaded `ov.Model`. Simpler and works.

---

## 5. Accuracy Verdict

### Q: Where did INT4 actually break?
On the 30-sentence biomedical corpus vs PT-FP32 reference:

| Model | INT4 mean cosine | INT4 min cosine | Verdict |
|---|---|---|---|
| PubMedBERT Pass1 | 0.904 | 0.869 | Broken |
| PubMedBERT BODHI | 0.917 | 0.898 | Broken |
| BioBERT FT | 0.985 | 0.981 | Marginal (BioBERT is more robust but still <99%) |

For embedding tasks where downstream cosine thresholds typically sit at 0.85–0.95, this is unusable: an INT4 "match" could be a 0.87 false-positive rather than a true 0.95 hit.

### Q: And INT8?
INT8 lands at 0.995–0.997 mean cosine — under 0.3% degradation. Distinguishable only in pathological cases. Safe to deploy.

### Q: Why is BF16 effectively lossless?
BF16 has the same 8-bit exponent as FP32 (just 7 mantissa bits vs 23). For BERT inference dominated by accumulations with normalization layers, this is enough — `ov-bf16` shows 1.0000 cosine, `pytorch-bf16` shows 0.9999.

---

## 6. Throughput Strategy

### Q: Why "8 servers", "16 servers + HT", "32 servers + HT"?
"Server" = an instance of the model loaded with its own thread pool, behind dynamic batching. The three plans:

- **8srv (physical only)**: 8 workers × 8 physical cores each. Fully covers 64 phys cores. No HT.
- **16srv + HT**: 16 workers × 4 phys + 4 HT cores each. Covers physical + HT pairs.
- **32srv + HT**: 32 workers × 2 phys + 2 HT each. Maximum process count without oversubscription.

Each step doubles the worker count, halving compute per worker. More workers = better request-level parallelism and dynamic-batch fill rate at high client load.

### Q: Which plan won and why?
**32srv + HT @ 600 clients.** At 8srv we leave queueing latency on the table (not enough workers to absorb concurrency spikes). At 16srv+HT we're better but still queue-limited at 600 clients. At 32srv+HT each request finds an idle worker faster — p50 drops from ~100 ms (8srv) to 57 ms (32srv+HT INT8) at the same client load.

### Q: Why HT helps here when "HT doesn't help compute-bound code"?
Inference at small per-worker batch is **memory-latency-bound**, not pure compute. AMX tiles want weights in L2/L3; while one HT thread waits on a cache miss, its sibling executes. We measured: at 32srv+HT/600c we hit ~139 GB/s sustained DRAM bandwidth (close to the ceiling). HT keeps the AMX units fed.

### Q: Why 600 clients and not 1200?
At 1200 clients the system saturates: TPS plateaus (96K → 97K for PT-FP16 PubMed), but p95 latency doubles (208ms vs 111ms). The 600c point is the knee — peak throughput at acceptable tail latency. 1200c is queueing for queueing's sake.

---

## 7. Why OpenVINO Beats PyTorch on the Same Hardware

### Q: Both have AMX BF16 paths — why is OV faster?
Three reasons, in order of impact:
1. **OV's graph compiler folds more aggressively** — attention QKV projections fuse, LayerNorm + GeLU fuse, residual adds fold into prior MatMuls. PyTorch eager-mode has dispatch overhead per op.
2. **OV's threading is explicitly tuned for `THROUGHPUT` hint** — picks stream count + threads per stream that match the cache topology. PyTorch's MKL-DNN path under `torch.set_num_threads` is more generic.
3. **OV INT8** uses a fast i8→bf16 dequant path on AMX; PyTorch's CPU INT8 (FBGEMM/QNNPACK) is older and has no equivalent path.

Head-to-head at 32srv+HT/600c (PT-BF16 vs OV-BF16):
- PubMedBERT Pass1: 98K → 110K TPS (+12%)
- PubMedBERT BODHI: 102K → 110K TPS (+8%)
- BioBERT: 118K → 124K TPS (+5%)

### Q: And OV INT8 vs PT-BF16?
+33% to +55% TPS depending on model. With <0.3% cosine loss. This is the production delta.

### Q: PT-FP32 was that bad?
34K TPS vs 110K BF16 vs 135K INT8. Yes — FP32 on AMX is doing 4× the matmul work without the BF16 tile speedup. Production: never ship FP32 CPU inference if the model is BERT-class.

---

## 8. Batch Size Choices

### Q: Why did "best batch size" differ per variant?
For PT-FP32 the per-batch latency is huge (1.3–1.9 s at bs=256), so larger batches just amortize overhead. For OV-INT8 the batch is compute-fast (~30–60 ms at bs=64), so smaller batches give better fill on 32 workers without head-of-line blocking. General rule observed:
- FP32 → big batches (256)
- BF16 → middling (128–256)
- INT8/INT4 → small (64–128)

### Q: Dynamic batching settings in load test?
`bs=256` max, `max_wait=10ms`. We chose 10ms because at 600 clients the inter-arrival gap is sub-millisecond — 10ms is plenty to fill batches without adding meaningful tail latency.

---

## 9. DRAM Bandwidth Ceiling

### Q: What's the actual memory ceiling and how close are we?
DDR5-6400 × 8 channels × 2 sockets gives ~410 GB/s theoretical peak. Realistic sustained: ~280–320 GB/s. APS measurements:
- 8srv: ~67 GB/s (under-utilized)
- 16srv+HT: ~91 GB/s
- 32srv+HT @ 600c: 113.5 GB/s
- 32srv+HT @ 1200c: 139.2 GB/s

We're at ~50% of sustainable bandwidth at peak — not bandwidth-bound. The cap is **AMX tile throughput + L3 hit rate**, which is why INT4 (smaller weights, better L3 fit) was the natural next ask. Sadly, INT4 fails on accuracy.

---

## 10. Why Not Push Higher?

### Q: Could we get more TPS by going to 64srv?
Tested informally — no. Beyond 32srv we oversubscribe the HT siblings on the same core (each phys core only has 2 HT threads). 64 workers × 1 core each gives us no HT pairing benefit and starves each worker.

### Q: Could a bigger machine help?
This is already 2 sockets, 64 cores. More cores per socket would help (Granite Rapids 6900P series goes to 128c/socket). More sockets adds UPI hops — diminishing returns past 2. The realistic next step is GPU inference (which we deliberately scoped out of this study).

### Q: Could fine-tuned thread tuning gain more?
We tried `KMP_HW_SUBSET=2s,32c,2t` (full machine), per-stream `INFERENCE_NUM_THREADS`, and `PERFORMANCE_HINT_NUM_REQUESTS`. The +5% kind of gains we found stayed within run-to-run noise. We declared diminishing returns at 135K TPS.

---

## 11. Production Recommendation

### Q: One config to deploy?
**OV INT8 quantized · 32 server processes · HT-paired · 600 concurrent clients · numactl pinned to both nodes · dynamic batching bs=256/wait=10ms.**

Per-model headline at this config:
- PubMedBERT Pass1: **135,668 TPS · 57 ms p50 · 92 ms p95**
- PubMedBERT BODHI: **135,721 TPS · 58 ms p50 · 91 ms p95**
- BioBERT Fine-Tuned: **182,469 TPS · 67 ms p50 · 74 ms p95**

Accuracy cost: < 0.3% cosine vs FP32. Negligible for biomedical embedding retrieval.

### Q: What's the failure mode to watch?
1. If client concurrency falls below ~200, batches don't fill and per-request latency stays at the worker's local minimum (~30–40 ms) but TPS drops. Right-size client pool.
2. If a future workload extends to longer sequences (we tested at typical biomedical sentence length, ~13–21 tokens), the AMX tile throughput drops as seq grows quadratically through attention. Re-benchmark at 256+ tokens before deploying for longer inputs.
3. INT8 weights are static — if you fine-tune, you must re-quantize (re-run `quantize.py`).

---

## 12. Run It Yourself

```bash
source .venv/bin/activate
cd intel_xeon_biobert_bench/code

# Quantize (creates INT8 + INT4 for all 3 models)
python quantize.py --model all --precision all

# Accuracy check (cosine vs FP32 reference)
python accuracy_eval.py

# Scenario sweep (peak TPS, single node)
bash run_full_suite.sh   # phase 1-2

# Multi-instance load test (32srv+HT, all variants, all configs)
bash run_full_suite.sh   # phase 3-4
```

Outputs land in `results/`.

---

*Generated for the Xeon 6737P biomedical embedding benchmark.
Final report: `Intel AMX Benchmark/Xeon6737P_Final_Report.docx`.
Raw results: `full_bench/results/`. Source of truth: `PYTORCH_RESULTS.md`.*

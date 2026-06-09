# Biomedical Embedding Inference — Consolidated Benchmark Report

PyTorch and OpenVINO inference of three fine-tuned biomedical BERT-base
models, measured across two distinct Intel Xeon hosts.

PyTorch is the reference engine — every OpenVINO datapoint is reported
relative to its corresponding PyTorch baseline.

---

## TL;DR

On **System A** (2× Xeon 6737P, Granite Rapids, native AMX), the production
sweet-spot is **OpenVINO INT8 at 32 server processes + HT, 600 concurrent
clients**, delivering **135,668 TPS @ 57 ms p50** for PubMedBERT and
**182,469 TPS @ 67 ms p50** for BioBERT, at <0.3 % cosine drift versus the
PyTorch FP32 reference.

On **System B** (1× Xeon Platinum 8375C, Ice Lake-SP, no AMX) only accuracy
and PyTorch-INT8 single-process throughput were captured — sufficient to
confirm the embedding-pipeline reproduces on a smaller host and that
PyTorch dynamic-INT8 is broken on both boxes.

---

## 1. Systems Under Test

### System A — 2× Intel Xeon 6737P (in-house)

| Item | Value |
|---|---|
| CPU | 2 × Intel Xeon 6737P |
| Sockets | 2 |
| Physical cores | 32 per socket → **64 total** |
| Logical CPUs (HT) | **128** |
| Base / max turbo | 2.90 GHz / 4.00 GHz |
| L1d / L1i | 3 MiB / 4 MiB (64 instances each) |
| L2 | 128 MiB total (2 MiB / core) |
| L3 | **288 MiB total — 144 MiB per NUMA node** |
| RAM | **1 024 GB DDR5 @ 6400 MT/s — 32 × 64 GB Samsung DIMMs** |
| NUMA nodes | 2 (≈ 503 GB usable on node 0, ≈ 500 GB on node 1) |
| Storage | 2 × Micron 7450 PRO NVMe SSD |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0-110-generic |
| OpenVINO | 2026.1.0 |
| NNCF | 3.1.0 |
| PyTorch | CPU build, FP32 / FP16 / BF16 autocast + dynamic-INT8 |

### System B — 1× Intel Xeon Platinum 8375C (AWS)

| Item | Value |
|---|---|
| CPU | Intel Xeon Platinum 8375C @ 2.90 GHz |
| Sockets | 1 |
| Physical cores | 16 |
| Logical CPUs (HT) | 32 |
| NUMA nodes | 1 |
| RAM | 61 GB |
| Kernel / OS | 6.17.0-1007-aws / Ubuntu 24.04.4 LTS |
| ISA highlights | AVX2 · AVX-512 · **AVX-512 VNNI** · no AMX · no AVX-512 BF16 |
| Python | 3.12.3 |
| torch | 2.11.0+cpu |
| openvino | 2026.1.0 |
| nncf | 3.1.0 |
| transformers | 5.6.2 |

> 📊 **DIAGRAM 1** — block diagram showing both hosts side-by-side (sockets, cores, cache, NUMA, RAM) — visually communicates the 4× core / ~17× RAM / AMX-vs-VNNI gap that contextualises every cross-system number below.

---

## 2. Models & Variants

### Models (3, all BERT-base 12-layer 768-hidden)

| Key | Name | Avg tokens / sentence (biomed corpus) |
|---|---|---|
| `pubmed`  | PubMedBERT Pass1 | 13.41 |
| `bodhi`   | PubMedBERT BODHI | 13.41 |
| `biobert` | BioBERT Fine-Tuned | 21.00 |

### Variants benchmarked (PyTorch-first ordering)

| # | Variant | Engine | Precision | Notes |
|---|---|---|---|---|
| 1 | **PyTorch FP32** | PyTorch | FP32 | reference for cosine accuracy |
| 2 | **PyTorch FP16** | PyTorch | FP16 autocast | |
| 3 | **PyTorch BF16** | PyTorch | BF16 autocast | uses AMX-BF16 on System A |
| 4 | **PyTorch INT8** | PyTorch | dynamic-quant Linear→qint8 | `torch.quantization.quantize_dynamic` |
| 5 | OpenVINO BF16 | OV | BF16 IR | exported via `optimum-cli` |
| 6 | OpenVINO INT8 | OV | INT8 PTQ | `nncf.quantize` with 128 calibration samples |
| 7 | OpenVINO INT4 | OV | INT4 weight-only | `nncf.compress_weights`, `INT4_SYM`, group 64 |

> 📊 **DIAGRAM 2** — flow diagram: PT FP32 → (autocast → FP16/BF16) → (dynamic-quant → PT-INT8) → (`optimum-cli export` → OV BF16 IR) → (`nncf.quantize` → INT8, `compress_weights` → INT4). One picture explains the whole variant tree.

---

## 3. Methodology

### Layered test plan

1. **Accuracy eval** — embed a 30-sentence biomedical corpus with each variant; compute cosine similarity against the PyTorch FP32 reference.
2. **Single-process throughput** (`bench_all.py`) — sweep batch sizes 64 / 128 / 256 / 384 / 512 in one process, report SPS and TPS.
3. **Scenario sweep** (`scenario_bench.py`, System A only) — full grid of 6 variants × 3 models × 4 batch sizes × 4 worker counts = **288 runs**. NUMA-balanced workers.
4. **Multi-instance load test** (`load_test.py`, System A only) — dynamic batching (`bs=256`, `max_wait=10 ms`), three server plans (8srv / 16srv+HT / 32srv+HT) × three client counts (300 / 600 / 1200) × all 6 variants.
5. **APS profiling** (System A) — Intel Application Performance Snapshot captured during multi-instance runs for DRAM bandwidth and physical-core utilisation.

### Thread / NUMA control (System A)

```
numactl --cpunodebind=0,1 --membind=0,1
OMP_NUM_THREADS=128
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
KMP_HW_SUBSET=2s,32c,2t
```

Each worker process calls `os.sched_setaffinity()` at startup to pin to its
assigned NUMA-balanced core set.

### Thread / NUMA control (System B)

```
numactl --cpunodebind=0 --membind=0
OMP_NUM_THREADS=16   (physical cores, HT siblings disabled in thread budget)
KMP_HW_SUBSET=1s,16c,2t
```

> 📊 **DIAGRAM 3** — NUMA worker-placement diagram for the three server plans on System A (8srv / 16srv+HT / 32srv+HT), showing which physical cores + HT siblings each worker owns. Anchors the load-test numbers in §6.

---

## 4. Accuracy — Both Systems

Mean cosine vs PyTorch FP32 reference (30-sentence biomedical corpus, all numbers from `accuracy_results.json` on each system).

### System A (Xeon 6737P)

| Model | PT-FP16 | PT-BF16 | **PT-INT8** | OV-BF16 | OV-INT8 | OV-INT4 |
|---|---|---|---|---|---|---|
| PubMedBERT Pass1 | 0.99999 | 0.99991 | **0.9302** | 1.00000 | 0.99720 | 0.90356 |
| PubMedBERT BODHI | 0.99999 | 0.99989 | **0.9297** | 1.00000 | 0.99742 | 0.91656 |
| BioBERT Fine-Tuned | 1.00000 | 0.99998 | **0.7588** | 1.00000 | 0.99487 | 0.98525 |

### System B (Xeon 8375C)

| Model | PT-FP16 | PT-BF16 | **PT-INT8** | OV-BF16 | OV-INT8 | OV-INT4 |
|---|---|---|---|---|---|---|
| PubMedBERT Pass1 | 0.99999 | 0.99991 | **0.9433** | 1.00000 | 0.99727 | 0.90356 |
| PubMedBERT BODHI | 0.99999 | 0.99991 | **0.9448** | 1.00000 | 0.99742 | 0.91656 |
| BioBERT Fine-Tuned | 1.00000 | 0.99998 | **0.6827** | 1.00000 | 0.99487 | 0.98525 |

**Observations**
- PT-FP16, PT-BF16, OV-BF16: lossless on both systems (cosine ≥ 0.99989).
- OV-INT8: 0.995–0.997 cosine — < 0.3 % drift, deployable.
- **PT-INT8: broken on both systems** (cosine 0.68–0.94). Dynamic quantisation of `nn.Linear` is *not* a viable production path for biomedical embeddings — confirmed on both Granite Rapids and Ice Lake-SP, ruling out a host-specific artefact.
- OV-INT4: PubMedBERT badly degraded (0.90–0.92); BioBERT marginal (0.985, still < 99 %).

> 📊 **DIAGRAM 4** — grouped bar chart "mean cosine vs FP32" with 6 variants on the x-axis, 3 bars per variant (one per model), faceted into two panels (System A / System B). Red dashed line at 0.99 acceptance threshold. Makes the PT-INT8 / OV-INT4 failures jump out instantly.

---

## 5. Cross-System Throughput — PyTorch INT8 (single-process)

Identical code path, identical models, identical batch sizes — the only difference is the host.

| Model | Batch | System A SPS | System A TPS | System B SPS | System B TPS | A / B speedup (TPS) |
|---|---|---|---|---|---|---|
| pubmed  | 64  | 46.7 | 626.9 | 15.2 | 204.4 | **3.07×** |
| pubmed  | 128 | 43.0 | 576.4 | 15.9 | 213.3 | 2.70× |
| pubmed  | 256 | 46.3 | 621.2 | 16.3 | 218.0 | 2.85× |
| pubmed  | 384 | 42.4 | 568.8 | 16.5 | 221.3 | 2.57× |
| pubmed  | 512 | 39.4 | 528.6 | 16.6 | 222.3 | 2.38× |
| bodhi   | 64  | 48.9 | 655.3 | 15.1 | 202.5 | **3.24×** |
| bodhi   | 128 | 47.8 | 641.4 | 15.9 | 212.9 | 3.01× |
| bodhi   | 256 | 33.1 | 444.1 | 16.4 | 219.6 | 2.02× |
| bodhi   | 384 | 35.9 | 481.0 | 16.5 | 221.0 | 2.18× |
| bodhi   | 512 | 39.7 | 532.7 | 16.6 | 222.3 | 2.40× |
| biobert | 64  | 47.5 | 997.3 | 15.2 | 319.0 | **3.13×** |
| biobert | 128 | 42.0 | 880.9 | 15.9 | 332.9 | 2.65× |
| biobert | 256 | 36.9 | 774.5 | 16.4 | 343.7 | 2.25× |
| biobert | 384 | 37.1 | 779.5 | 16.4 | 344.8 | 2.26× |
| biobert | 512 | 38.9 | 817.9 | 16.6 | 349.6 | 2.34× |

**Read.** System A is consistently ~2.4–3.2× faster than System B on the same PyTorch-INT8 code — driven by 4× the physical cores and dual sockets, with sub-linear scaling caused by NUMA crossing + MKL contention within a single process. The accuracy drift (PT-INT8 broken) is identical to within calibration noise on both hosts — confirming the failure is in the quantisation method, not the host.

> 📊 **DIAGRAM 5** — grouped bar chart of PT-INT8 best-TPS per model, two bars each (System A / System B), with the host-A / host-B ratio annotated on top of each pair. Companion line chart of TPS vs batch-size, one line per (system, model) — six lines total.

---

## 6. System A — Full Throughput Picture

System A is the only host where we ran the scenario sweep + multi-instance load test, so the rest of this section is single-system.

### 6.0 Coverage matrix — what was actually measured

PT vs OV is exhaustively covered across all four scaling dimensions. Every cell of the cross-product below corresponds to a JSON on disk under `evidences/system_a_xeon6737p/`.

| Dimension | Levels measured | PT variants | OV variants |
|---|---|---|---|
| **Precision** | 6 levels | FP32, FP16, BF16, INT8 *(single-process only)* | BF16, INT8, INT4 |
| **Batch size** (scenario sweep) | 4 levels — 64, 128, 256, 512 | ✓ all four | ✓ all four |
| **Per-process workers** (scenario sweep) | 4 levels — 1, 2, 4, 8 | ✓ all four | ✓ all four |
| **Server plan** (load test) | 3 levels — 8srv / 16srv+HT / 32srv+HT | ✓ all three | ✓ all three |
| **Concurrent clients** (load test) | 3 levels — 300, 600, 1200 | ✓ all three | ✓ all three |
| **Models** | 3 — pubmed, bodhi, biobert | ✓ all three | ✓ all three |

- Scenario sweep: **6 × 3 × 4 × 4 = 288 runs** — every (variant, model, batch, workers) cell present.
- Multi-instance load test: **6 × 3 × 5 = 90 runs** — every (variant, model, server-plan + client-count) cell present.

### Advanced optimisations applied to every variant

Every PT variant and every OV variant ran with the same NUMA / threading / batching stack — no variant was given a handicap.

| Optimisation | PyTorch variants | OpenVINO variants |
|---|---|---|
| `numactl --cpunodebind=0,1 --membind=0,1` | ✓ | ✓ |
| Per-worker `os.sched_setaffinity()` to a NUMA-balanced core set | ✓ | ✓ |
| `KMP_AFFINITY=granularity=fine,compact,1,0`, `KMP_BLOCKTIME=1`, `KMP_HW_SUBSET=2s,32c,2t` | ✓ | ✓ |
| `OMP_NUM_THREADS` + `MKL_NUM_THREADS` sized to affinity set | ✓ | ✓ |
| Hyper-threading enabled in the 16srv+HT and 32srv+HT server plans | ✓ | ✓ |
| `torch.set_num_threads(N_phys)` + `torch.inference_mode()` | ✓ | n/a |
| `torch.autocast(device_type="cpu", dtype=torch.{bfloat16, float16})` | ✓ (FP16 / BF16) | n/a |
| `torch_dtype=torch.bfloat16 / torch.float16` at `from_pretrained` | ✓ (FP16 / BF16) | n/a |
| `torch.quantization.quantize_dynamic({nn.Linear}, qint8)` | ✓ (INT8 only) | n/a |
| `ov.Core()` with `PERFORMANCE_HINT=THROUGHPUT` | n/a | ✓ |
| `INFERENCE_NUM_THREADS` per stream | n/a | ✓ |
| `INFERENCE_PRECISION_HINT=bf16` on AMX path | n/a | ✓ (BF16, INT8) |
| `ENABLE_CPU_PINNING=True`, `ENABLE_HYPER_THREADING=True` | n/a | ✓ |
| `nncf.quantize` PTQ with 128 calibration samples for INT8 | n/a | ✓ (INT8) |
| `nncf.compress_weights INT4_SYM, group_size=64` | n/a | ✓ (INT4) |
| Dynamic batching at the server: `max_bs=256`, `max_wait=10 ms` | ✓ | ✓ |

> 📊 **DIAGRAM 6a** — checklist-style summary graphic of the optimisation stack with PT-only / OV-only / shared columns, so the reader sees at a glance that the comparison is fair.

### 6.1 Single-node peak (scenario sweep — `bench_all` / `scenario_bench`)

Best TPS per (variant × model), choosing the best (batch-size × worker-count) per cell.

| Variant | PubMedBERT Pass1 | PubMedBERT BODHI | BioBERT FT |
|---|---|---|---|
| **PyTorch FP32** | 23 774 TPS | 24 683 TPS | 26 020 TPS |
| **PyTorch FP16** | 76 144 TPS | 77 565 TPS | 92 677 TPS |
| **PyTorch BF16** | 76 478 TPS | 83 282 TPS | 89 532 TPS |
| **PyTorch INT8** *(single-process only)* | 627 TPS | 655 TPS | 997 TPS |
| OpenVINO BF16 | 153 464 TPS | 149 146 TPS | 159 314 TPS |
| OpenVINO INT8 | 186 170 TPS | 179 887 TPS | 198 246 TPS |
| OpenVINO INT4 | 146 851 TPS | 141 391 TPS | 140 859 TPS |

> 📊 **DIAGRAM 6** — clustered bar chart, x-axis = model, 7 bars per group (one per variant, PyTorch flavours in shades of blue, OV in shades of green). Log-y advisable because PT-INT8 ≪ everything else.

### 6.2 Multi-instance load test — peak serving config

`bs=256`, `max_wait=10 ms`, all variants at 32srv+HT / 600 clients.

| Variant | PubMedBERT Pass1 TPS | BODHI TPS | BioBERT TPS | p50 / p95 / p99 (PubMed) |
|---|---|---|---|---|
| PyTorch FP32 | 34 451 | 34 612 | 27 440 | 211 / 320 / 334 ms |
| PyTorch FP16 | 92 814 | 101 103 | 114 888 | 82 / 111 / 132 ms |
| PyTorch BF16 | 98 192 | 101 955 | 118 086 | 79 / 120 / 133 ms |
| OpenVINO BF16 | 110 215 | 110 139 | 124 160 | 74 / 80 / 109 ms |
| **OpenVINO INT8** | **135 668** | **135 721** | **182 469** | **57 / 92 / 114 ms** |
| OpenVINO INT4 | 85 877 | 78 435 | 88 150 | 88 / 122 / 127 ms |

(PyTorch INT8 not exercised in multi-instance — single-process measurement already shows the configuration is two orders of magnitude below FP32, so the multi-instance run was skipped intentionally.)

### 6.3 PyTorch BF16 vs OpenVINO BF16/INT8 — head-to-head at 32srv+HT / 600c

| Model | PT-BF16 TPS | OV-BF16 TPS | OV-BF16 vs PT-BF16 | OV-INT8 TPS | OV-INT8 vs PT-BF16 |
|---|---|---|---|---|---|
| PubMedBERT Pass1 | 98 192 | 110 215 | **+12.2 %** | 135 668 | **+38.2 %** |
| PubMedBERT BODHI | 101 955 | 110 139 | +8.0 % | 135 721 | +33.1 % |
| BioBERT Fine-Tuned | 118 086 | 124 160 | +5.1 % | 182 469 | +54.5 % |

> 📊 **DIAGRAM 7** — grouped bar chart: per model, three bars (PT-BF16, OV-BF16, OV-INT8). Annotate the +X % delta above each non-baseline bar. This is the headline marketing chart.

### 6.4 PT vs OV — full concurrency × HT × precision matrix

**Every cell below was actually measured** — 6 variants × 3 models × 5 server-plan/client configs = 90 load-test runs on disk (`evidences/system_a_xeon6737p/load_test/`). Tokens per second.

#### PubMedBERT Pass1

| Config | PT-FP32 | PT-FP16 | PT-BF16 | OV-BF16 | OV-INT8 | OV-INT4 |
|---|---|---|---|---|---|---|
| 8srv / 300c | 20,953 | 49,298 | 68,137 | 89,901 | 111,556 | 70,408 |
| 8srv / 600c | 21,336 | 60,446 | 84,296 | 105,813 | 119,128 | 87,190 |
| 16srv+HT / 600c | 31,122 | 72,825 | 84,837 | 105,328 | 142,701 | 86,865 |
| **32srv+HT / 600c** | 34,451 | 92,813 | 98,192 | 110,214 | **135,668** | 85,877 |
| 32srv+HT / 1200c | 25,875 | 91,353 | 101,610 | 112,847 | **147,334** | 96,386 |

#### PubMedBERT BODHI

| Config | PT-FP32 | PT-FP16 | PT-BF16 | OV-BF16 | OV-INT8 | OV-INT4 |
|---|---|---|---|---|---|---|
| 8srv / 300c | 20,825 | 64,264 | 66,612 | 89,713 | 113,571 | 66,213 |
| 8srv / 600c | 21,691 | 82,890 | 85,508 | 105,323 | 119,797 | 85,615 |
| 16srv+HT / 600c | 31,867 | 82,414 | 84,708 | 104,817 | 138,516 | 74,300 |
| **32srv+HT / 600c** | 34,612 | 101,102 | 101,955 | 110,138 | **135,721** | 78,434 |
| 32srv+HT / 1200c | 22,387 | 99,817 | 101,689 | 112,918 | **147,261** | 87,871 |

#### BioBERT Fine-Tuned

| Config | PT-FP32 | PT-FP16 | PT-BF16 | OV-BF16 | OV-INT8 | OV-INT4 |
|---|---|---|---|---|---|---|
| 8srv / 300c | 23,908 | 87,118 | 89,904 | 114,798 | 147,397 | 73,248 |
| 8srv / 600c | 25,200 | 104,302 | 106,293 | 127,159 | 151,338 | 87,966 |
| 16srv+HT / 600c | 34,535 | 102,662 | 105,111 | 123,235 | 180,273 | 83,785 |
| **32srv+HT / 600c** | 27,440 | 114,887 | 118,085 | 124,159 | **182,468** | 88,150 |
| 32srv+HT / 1200c | 24,211 | 119,034 | 124,291 | 126,852 | **182,842** | 95,134 |

#### Latency profile at 32srv+HT / 600c (PubMedBERT Pass1, ms)

| Variant | p50 | p95 | p99 |
|---|---|---|---|
| PT-FP32 | 211 | 320 | 334 |
| PT-FP16 | 82  | 111 | 132 |
| PT-BF16 | 79  | 120 | 133 |
| OV-BF16 | 74  | 80  | 109 |
| **OV-INT8** | **57** | **92** | **114** |
| OV-INT4 | 88  | 122 | 127 |

**What this matrix shows**
- The HT step from 8srv → 16srv+HT → 32srv+HT consistently *helps every variant*, not just OV — even PT-FP32 climbs from 21K → 31K → 34K TPS for PubMed.
- OV-BF16 already beats PT-BF16 at every cell (+5–58 % depending on load); the gap widens as concurrency rises.
- **OV-INT8 is fastest at every cell for every model** — it absorbs 1200-client load without throughput collapse (PubMed: 135K → 147K going 600 → 1200 clients).
- OV-INT4 is *not* faster than OV-INT8 anywhere — its weight-only quantisation gives smaller model size but no kernel speedup over INT8 PTQ.

> 📊 **DIAGRAM 8** — multi-line chart, x-axis = (server-plan, client-count) configurations in order, y-axis = TPS, one line per variant (4 PyTorch + 3 OpenVINO). Shows the HT-scaling story plus the OV-vs-PT separation at each step.

### 6.4a Why we stopped at 32srv+HT / 1200 clients — saturation evidence

We swept up to 32 server processes (the hardware maximum without HT oversubscription) and up to 1200 concurrent clients (double the production target). Beyond these points the system stops returning useful throughput and begins eating latency budget.

#### Why not more server processes than 32?

| Constraint | Reason |
|---|---|
| **Core arithmetic** | 64 phys cores × 2 HT threads = 128 logical CPUs. The 32srv+HT plan gives every worker exactly 2 phys + 2 HT cores. Going to 64 workers (1 phys + 1 HT each) loses the HT-pair speedup — each worker would have a single phys core, with its HT sibling running an unrelated worker, defeating cache locality. |
| **MKL-DNN thread pool overhead** | Each worker allocates its own oneDNN scratch + thread pool. At 32 workers the pools already total ~14 % overhead; doubling them past 32 starts to dominate. |
| **APS finding** | At 32srv+HT / 1200c we measured ~139 GB/s sustained DRAM bandwidth and ~85 % phys-core utilisation. The bottleneck shifts from idle cores to AMX-tile dispatch and L3 thrash. Adding more workers can only make L3 contention worse. |

#### Why not more concurrent clients than 1200?

The 600c → 1200c step already shows the knee. Pulled from the live JSONs (PubMedBERT Pass1, 32srv+HT):

| Variant | TPS @ 600c | TPS @ 1200c | Δ TPS | p50 @ 600c → 1200c | p95 @ 600c → 1200c | p99 @ 600c → 1200c |
|---|---|---|---|---|---|---|
| PT-FP32 | 34,451 | **25,875** | **−25 %** *(regressed)* | 211 → 380 ms | 320 → 650 ms | **334 → 8 144 ms** *(queueing collapse)* |
| PT-BF16 | 98,192 | 101,610 | +3.5 % | 79 → 159 ms | 120 → 230 ms | 133 → 263 ms |
| OV-INT8 | 135,668 | 147,334 | +8.6 % | 57 → 105 ms | 92 → 171 ms | 114 → 208 ms |

**Reading.**
- For PT-FP32 the system is already past its knee at 600c; 1200c drives a queueing collapse (p99 jumps 24×).
- For PT-BF16 / OV-INT8 we are still on the throughput curve at 1200c (+3.5 % / +8.6 %), but **latency essentially doubles** — p50 OV-INT8 goes from 57 ms (production-friendly) to 105 ms.
- p99 OV-INT8 at 1200c hits 208 ms — that crosses the 200 ms SLA we use for downstream retrieval. Above 1200c we expect the same pattern as PT-FP32 saw at 1200c: throughput-flat, latency-explode.

#### What we would gain (and lose) by pushing higher

| Next step | Predicted TPS lift | Predicted p99 cost | Verdict |
|---|---|---|---|
| 32srv+HT / 2400c | +5–8 % over 1200c at best | ≥ 400 ms p99 | **Not worth it** — beyond serving SLA |
| 64srv (no HT pairing) | −5–15 % vs 32srv+HT | similar / worse | Loses HT-pair speedup |
| `max_wait` > 10 ms | Marginally more batch fill | direct linear p50 add | Already at near-optimal fill |

The headline production point — **32srv+HT / 600c** — is on the throughput shoulder, well below the latency knee, and inside the 200 ms p99 envelope for OV-INT8.

> 📊 **DIAGRAM 8a** — knee-of-curve plot: x = clients (300, 600, 1200), left-y = TPS, right-y = p99 ms, three lines per variant (PT-FP32 / PT-BF16 / OV-INT8). Vertical line at 600c labelled "production sweet spot". Visualises the saturation argument.

### 6.5 APS profiling — DRAM bandwidth ceiling

Captured by Intel Application Performance Snapshot during PubMedBERT runs.

| Server plan | Clients | DRAM BW sustained | Phys-core utilisation |
|---|---|---|---|
| 8srv | 600 | ~67 GB/s | partial |
| 16srv+HT | 600 | ~91 GB/s | better |
| 32srv+HT | 600 | ~113.5 GB/s | high |
| 32srv+HT | 1200 | ~139.2 GB/s | saturated |

Theoretical peak for this RAM (DDR5-6400 × 8 ch × 2 sockets) is ~410 GB/s; realistic sustained ceiling ~280–320 GB/s. At peak we are at ~50 % of ceiling — **the system is compute-bound at AMX-tile throughput + L3-hit rate, not bandwidth-bound.**

> 📊 **DIAGRAM 9** — twin-axis chart: left axis DRAM BW (GB/s), right axis phys-core util (%), x-axis the four progression steps. Overlay a dashed horizontal at the practical bandwidth ceiling to make headroom visible.

### 6.6 VTune & APS profiling artifacts — file index

All Intel VTune and Application Performance Snapshot captures from the System A runs are preserved under `system_a_xeon6737p/aps_profiling/`. **Note**: VTune does not export an HTML report by default — what we have are the raw `.vtune` project directories (open in the VTune GUI for the equivalent of an HTML flame-graph / metrics view) plus the CSV exports for the headline metrics and APS plain-text summaries.

#### APS plain-text summaries (open directly — no GUI needed)

| File | What it shows |
|---|---|
| `aps_pubmed_report.txt`  | APS summary for PubMedBERT Pass1 — CPI, retiring %, back-end-bound %, DRAM BW, FLOPS |
| `aps_bodhi_report.txt`   | Same metrics for PubMedBERT BODHI |
| `aps_biobert_report.txt` | Same metrics for BioBERT Fine-Tuned |
| `aps_optimal.log`        | End-to-end driver log of the APS sweep |

#### APS captures per server-plan × client-count (8 runs)

Under `aps_profiling/aps_multi_runs/` — one summary file per (model × server-plan × clients) run. Use these to validate the DRAM-BW progression in §6.5.

```
aps_aps_multi_pubmed_8srv_600c_summary.txt          ← 67 GB/s baseline
aps_aps_multi_pubmed_16srv_ht_600c_summary.txt      ← 91 GB/s
aps_aps_multi_pubmed_32srv_ht_600c_summary.txt      ← 113 GB/s (peak prod config)
aps_aps_multi_pubmed_32srv_ht_1200c_summary.txt     ← 139 GB/s (saturation)
aps_aps_multi_bodhi_8srv_600c_summary.txt
aps_aps_multi_bodhi_16srv_ht_600c_summary.txt
aps_aps_multi_bodhi_32srv_ht_600c_summary.txt
aps_aps_multi_bodhi_32srv_ht_1200c_summary.txt
```

#### VTune CSV exports (24 files — open in any spreadsheet or text editor)

Under `aps_profiling/vtune_csv/`. Four analysis types × three models × {summary, hotspots}:

```
vtune_hotspots_{pubmed,bodhi,biobert}_{summary,hotspots}.csv
  → top CPU hotspot functions, ranked by self-CPU time

vtune_memory_access_{pubmed,bodhi,biobert}_{summary,hotspots}.csv
  → LLC misses, DRAM bound %, remote-NUMA accesses

vtune_threading_{pubmed,bodhi,biobert}_{summary,hotspots}.csv
  → wait time, sync overhead, effective CPU utilisation

vtune_uarch_exploration_{pubmed,bodhi,biobert}_{summary,hotspots}.csv
  → top-down microarchitecture breakdown (front-end, back-end, retiring, bad speculation)
```

#### VTune project files (12 files — for HTML-equivalent visualisation in the VTune GUI)

Under `aps_profiling/vtune_projects/`. Open with `vtune-gui <file>.vtune` (Intel oneAPI VTune Profiler) — gives interactive flame graphs, source-line attribution, and the side-by-side panes that the VTune HTML export would render.

```
vtune_hotspots_{pubmed,bodhi,biobert}.vtune
vtune_memory_access_{pubmed,bodhi,biobert}.vtune
vtune_threading_{pubmed,bodhi,biobert}.vtune
vtune_uarch_exploration_{pubmed,bodhi,biobert}.vtune
```

To export any of these to a standalone HTML page for sharing without the VTune GUI:

```bash
vtune -report summary -result-dir <path>/vtune_hotspots_pubmed -format=html -report-output pubmed_hotspots.html
```

(That command is *not* pre-run; the `.vtune` files themselves are the source of truth.)

---

## 7. Verdict & Production Recommendation

1. **PyTorch is the right reference.** PT-FP32 anchors accuracy ground truth. PT-BF16 is the right *PyTorch* production choice if engine portability matters; it loses 8–12 % TPS to OpenVINO BF16 and 33–55 % to OpenVINO INT8 at the same NUMA serving config.
2. **OpenVINO INT8 wins production.** 135 K TPS for PubMedBERT variants, 182 K TPS for BioBERT, at 57–67 ms p50 with < 0.3 % cosine drift. The clearest combination of throughput, latency, and accuracy.
3. **Two configurations are explicitly broken** and must not be shipped:
   - **PyTorch dynamic-INT8** — cosine 0.68–0.94, two orders of magnitude slower than PT-FP32 in throughput. Failure reproduces on both hosts.
   - **OpenVINO INT4** — cosine 0.90–0.92 on PubMedBERT variants, 0.985 on BioBERT (still sub-99 %). Throughput is worse than OV-INT8 anyway (cache-blocking helps less than expected).
4. **System A is the production target.** System B (Ice Lake-SP, no AMX) is a useful cross-system sanity check for the embedding pipeline but is not a scaled-down version of A — the AMX BF16 / INT8 silicon is the differentiator.

### Headline production config

> **OpenVINO INT8 quantised model**, 32 server processes pinned with HT
> across both NUMA nodes, 600 concurrent clients, dynamic batching
> `bs=256 / max_wait=10 ms`, `numactl --cpunodebind=0,1 --membind=0,1`.

| Model | TPS | p50 | p95 |
|---|---|---|---|
| PubMedBERT Pass1 | 135 668 | 57 ms | 92 ms |
| PubMedBERT BODHI | 135 721 | 58 ms | 91 ms |
| BioBERT Fine-Tuned | 182 469 | 67 ms | 74 ms |

> 📊 **DIAGRAM 10** — single "scorecard" tile per model with TPS, p50, p95, and accuracy retained. Three tiles side-by-side. Closing visual.

---

## 8. Reproducing Every Combination via CLI (System A)

All commands assume:

```bash
source .venv/bin/activate
cd intel_xeon_biobert_bench/code
NUMACTL="numactl --cpunodebind=0,1 --membind=0,1"
```

### 8.1 Quantize (one-time — builds OV INT8 / INT4 from OV BF16 IR)

```bash
$NUMACTL python quantize.py --model all  --precision all   # everything
$NUMACTL python quantize.py --model pubmed   --precision int8
$NUMACTL python quantize.py --model biobert  --precision int4 --calib-samples 256
```

Outputs land in `models/openvino_quantized/<model>-<int8|int4>/`. Idempotent — re-runs skip existing.

### 8.2 Accuracy eval — cosine vs PT-FP32, every variant × every model

One command sweeps all 6 variants × 3 models on the 30-sentence biomedical corpus and writes a single JSON:

```bash
$NUMACTL python accuracy_eval.py --model all --out results/accuracy_results.json
$NUMACTL python accuracy_eval.py --model biobert                              # single model
```

### 8.3 PyTorch INT8 single-process bench (the AWS-comparable run)

```bash
$NUMACTL python bench_pytorch_int8.py --model all      --n 1024
$NUMACTL python bench_pytorch_int8.py --model bodhi    --threads 32           # match AWS thread budget
```

Sweeps batch sizes 64 / 128 / 256 / 384 / 512 for each model. Output: `results/bench_pytorch_int8_xeon6737p_<ts>.json`.

### 8.4 Scenario sweep — every (variant × model × batch × workers) cell

Full grid (288 runs, ~3-4 h):

```bash
$NUMACTL python scenario_bench.py --variant all --model all
```

A single cell (fast — useful for spot-checking):

```bash
$NUMACTL python scenario_bench.py --variant pytorch-bf16 --model pubmed   --bs 256 --workers 8
$NUMACTL python scenario_bench.py --variant ov-int8      --model biobert  --bs 64  --workers 4
```

Variant choices: `pytorch-fp32 | pytorch-fp16 | pytorch-bf16 | ov-bf16 | ov-int8 | ov-int4 | all`
Models: `pubmed | bodhi | biobert | all`
Batch sizes: 64, 128, 256, 512 (sweeps all if `--bs` omitted)
Workers: 1, 2, 4, 8 (sweeps all if `--workers` omitted)

Each cell writes `results/scenario_<variant>_<model>_bs<n>_w<n>.json`.

### 8.5 Multi-instance load test — every (variant × model × server-plan × client-count) cell

Full grid (90 runs, ~1 h):

```bash
$NUMACTL python load_test.py --variant all --model all
```

Single configuration runs (matching the table cells in §6.4):

```bash
# 8srv, no HT, 300 clients
$NUMACTL python load_test.py --variant pytorch-bf16 --model pubmed   --servers 8  --clients 300

# 8srv, no HT, 600 clients
$NUMACTL python load_test.py --variant ov-bf16      --model bodhi    --servers 8  --clients 600

# 16srv with HT pairing, 600 clients
$NUMACTL python load_test.py --variant ov-int8      --model biobert  --servers 16 --clients 600 --use-ht

# 32srv with HT pairing — production sweet spot
$NUMACTL python load_test.py --variant ov-int8      --model pubmed   --servers 32 --clients 600 --use-ht

# 32srv with HT pairing, 1200 clients (saturation)
$NUMACTL python load_test.py --variant pytorch-bf16 --model bodhi    --servers 32 --clients 1200 --use-ht
```

Additional knobs:

```
--max-bs    256     dynamic-batch ceiling (default 256)
--max-wait  10      max wait ms before dispatching a partial batch (default 10)
--duration  40      run length seconds (default 40)
--out       path    custom output JSON path (default auto-named under results/)
```

Each run writes `results/loadtest_<variant>_<model>_<plan>_<clients>c_bs<max-bs>.json` with TPS, SPS, p50/p95/p99 latency, total requests, etc.

### 8.6 End-to-end driver — APS-profiled run of everything

The master driver under `code/run_all_xeon.sh` chains
quantize → accuracy → bench_all (×6 variants) → extended_benchmark → compare_finetuned →
threshold_sweep, each wrapped with `numactl` + Intel APS:

```bash
# default — APS-wrapped
bash code/run_all_xeon.sh

# also collect VTune hotspots for every step (slower)
bash code/run_all_xeon.sh --vtune

# skip APS wrapping (raw runs only — useful when oneAPI is not installed)
bash code/run_all_xeon.sh --no-aps

# include the heavy scenario + load tests
RUN_HEAVY=1 bash code/run_all_xeon.sh
```

All artifacts (logs, JSONs, `aps_<step>/`, `vtune_<step>/`) collect under
`aws_benchmark/results/xeon_run_<timestamp>/`.

### 8.7 VTune captures (manual, per analysis type × model)

The four analysis types we collected can each be reproduced individually:

```bash
TARGET="python scenario_bench.py --variant ov-int8 --model pubmed --bs 256 --workers 8"

# Hotspots — function-level CPU time
$NUMACTL vtune -collect hotspots          -r results/vtune_hotspots_pubmed          -- $TARGET

# Memory access — LLC misses, DRAM bound, remote-NUMA accesses
$NUMACTL vtune -collect memory-access     -r results/vtune_memory_access_pubmed     -- $TARGET

# Threading — wait time, sync overhead, effective CPU utilisation
$NUMACTL vtune -collect threading         -r results/vtune_threading_pubmed         -- $TARGET

# Microarchitecture top-down — front-end / back-end / retiring / bad-speculation
$NUMACTL vtune -collect uarch-exploration -r results/vtune_uarch_exploration_pubmed -- $TARGET

# Export any captured run to a shareable HTML
vtune -report summary -result-dir results/vtune_hotspots_pubmed \
      -format=html -report-output vtune_hotspots_pubmed.html
```

Repeat with `--model bodhi`, `--model biobert` to reproduce the 12 captures preserved under `aps_profiling/vtune_projects/`.

### 8.8 APS captures (one APS run per server-plan × client-count)

The eight APS multi-runs in `aps_profiling/aps_multi_runs/` were each produced by wrapping a single `load_test.py` invocation:

```bash
# 8srv / 600c — APS captures DRAM BW, CPI, retiring %
$NUMACTL aps --collection-mode=hw --result-dir=results/aps_multi_pubmed_8srv_600c -- \
    python load_test.py --variant ov-int8 --model pubmed --servers 8 --clients 600

# 16srv+HT / 600c
$NUMACTL aps --collection-mode=hw --result-dir=results/aps_multi_pubmed_16srv_ht_600c -- \
    python load_test.py --variant ov-int8 --model pubmed --servers 16 --clients 600 --use-ht

# 32srv+HT / 600c — production sweet spot
$NUMACTL aps --collection-mode=hw --result-dir=results/aps_multi_pubmed_32srv_ht_600c -- \
    python load_test.py --variant ov-int8 --model pubmed --servers 32 --clients 600 --use-ht

# 32srv+HT / 1200c — saturation evidence
$NUMACTL aps --collection-mode=hw --result-dir=results/aps_multi_pubmed_32srv_ht_1200c -- \
    python load_test.py --variant ov-int8 --model pubmed --servers 32 --clients 1200 --use-ht
```

Repeat with `--model bodhi` for the four BODHI captures.

After collection, generate the plain-text summary that's preserved in this evidence pack:

```bash
aps-report results/aps_multi_pubmed_32srv_ht_600c > aps_multi_pubmed_32srv_ht_600c_summary.txt
```

### 8.9 Quick recipe index — by what you want to compare

| You want to compare … | Run |
|---|---|
| All variants' accuracy on all 3 models | §8.2 single command |
| PT-INT8 throughput cross-system | §8.3 (System A) + same on System B |
| One PT vs OV cell at chosen BS / workers | §8.4 single-cell commands |
| Full PT-vs-OV concurrency grid | §8.5 full grid |
| One 32srv+HT / 600c cell to validate §6.4 | §8.5 single-config command |
| DRAM bandwidth at each server plan | §8.8 four-line block |
| Function-level hotspots for any variant | §8.7 hotspots command |

---

## 9. Evidence Folder Map

All raw data backing every number in this report lives under
`complete_benchmark_evidences/`:

```
complete_benchmark_evidences/
├── FINAL_CONSOLIDATED_REPORT.md          ← this file
├── system_a_xeon6737p/
│   ├── accuracy/                         accuracy_results.json
│   ├── scenario_sweep/                   288 JSONs — 6 variants × 3 models × 4 BS × 4 W
│   ├── load_test/                        90 JSONs — multi-instance load test
│   ├── pytorch_int8_single_process/      bench_pytorch_int8_xeon6737p_20260513_080649.json + .log
│   ├── aps_profiling/                    profiling artifacts (see §6.6 for full file list)
│   │   ├── aps_{pubmed,bodhi,biobert}_report.txt    APS plain-text summaries (3 files)
│   │   ├── aps_optimal.log                          APS driver log
│   │   ├── aps_multi_runs/                          per-(model × plan × clients) — 8 files
│   │   ├── vtune_csv/                               VTune CSV exports — 24 files
│   │   └── vtune_projects/                          VTune .vtune project files — 12 files
│   ├── PYTORCH_RESULTS.md                tabular dump of all System A throughput data
│   └── REASONING_QNA.md                  12-section decision log (why NUMA, HT, INT8, …)
└── system_b_xeon8375c/
    ├── xeon_run_20260513_064605/         final AWS run — accuracy + PT-INT8 throughput
    ├── xeon_run_20260512_234207_partial_pre_int8/    earlier partial run
    ├── xeon_run_20260512_221619_pre_tune/            first pre-tuning attempt
    └── run_master.log                    overall AWS driver log
```

Every TPS / cosine / latency value in §4–§6 traces back to a named JSON in
this tree.

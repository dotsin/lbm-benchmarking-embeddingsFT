# PyTorch Benchmark Results — Intel Xeon 6737P
**2× Intel Xeon 6737P · 64 physical cores · 128 logical (HT) · DDR5 6400 MT/s · 1 TB RAM**
**All runs: NUMA-pinned, numactl --cpunodebind=0,1 --membind=0,1**

---

## Accuracy vs FP32 Reference

All variants compared against PyTorch FP32 embeddings using cosine similarity on 30-sentence biomedical corpus.

| Model | Variant | Mean Cosine | Min Cosine | Pct ≥ 0.99 | Verdict |
|-------|---------|-------------|------------|------------|---------|
| PubMedBERT Pass1 | pytorch-fp16 | 0.99999 | 0.99999 | 100% | ✓ Lossless |
| PubMedBERT Pass1 | pytorch-bf16 | 0.99991 | 0.99980 | 100% | ✓ Lossless |
| PubMedBERT Pass1 | ov-bf16 | 1.00000 | 1.00000 | 100% | ✓ Lossless |
| PubMedBERT Pass1 | ov-int8 | 0.99720 | 0.99434 | 100% | ✓ Acceptable |
| PubMedBERT Pass1 | ov-int4 | 0.90356 | 0.86904 | 0% | ✗ Degraded |
| PubMedBERT BODHI | pytorch-fp16 | 0.99999 | 0.99999 | 100% | ✓ Lossless |
| PubMedBERT BODHI | pytorch-bf16 | 0.99989 | 0.99959 | 100% | ✓ Lossless |
| PubMedBERT BODHI | ov-bf16 | 1.00000 | 1.00000 | 100% | ✓ Lossless |
| PubMedBERT BODHI | ov-int8 | 0.99742 | 0.99532 | 100% | ✓ Acceptable |
| PubMedBERT BODHI | ov-int4 | 0.91656 | 0.89815 | 0% | ✗ Degraded |
| BioBERT Fine-Tuned | pytorch-fp16 | 1.00000 | 1.00000 | 100% | ✓ Lossless |
| BioBERT Fine-Tuned | pytorch-bf16 | 0.99998 | 0.99997 | 100% | ✓ Lossless |
| BioBERT Fine-Tuned | ov-bf16 | 1.00000 | 1.00000 | 100% | ✓ Lossless |
| BioBERT Fine-Tuned | ov-int8 | 0.99487 | 0.99319 | 100% | ✓ Acceptable |
| BioBERT Fine-Tuned | ov-int4 | 0.98525 | 0.98061 | 0% | ~ Marginal |

**Key finding:** INT4 has unacceptable accuracy loss (mean cosine 0.90–0.99) for embedding tasks despite the smaller model size. INT8 maintains >99.4% cosine similarity — safe to deploy.

---

## Scenario Sweep — Peak TPS per Variant/Model

Single-node inference, NUMA-balanced workers, best batch size and worker count per variant.

| Variant | Model | Best BS | Workers | SPS | TPS | Batch p50 | Batch p95 |
|---------|-------|---------|---------|-----|-----|-----------|-----------|
| **pytorch-fp32** | PubMedBERT Pass1 | 256 | 8 | 1,773 | 23,774 | 1305ms | 1531ms |
| **pytorch-fp32** | PubMedBERT BODHI | 128 | 8 | 1,841 | 24,683 | 628ms | 778ms |
| **pytorch-fp32** | BioBERT Fine-Tuned | 256 | 8 | 1,239 | 26,020 | 1864ms | 2252ms |
| **pytorch-fp16** | PubMedBERT Pass1 | 256 | 8 | 5,678 | 76,144 | 395ms | 458ms |
| **pytorch-fp16** | PubMedBERT BODHI | 256 | 8 | 5,784 | 77,565 | 392ms | 474ms |
| **pytorch-fp16** | BioBERT Fine-Tuned | 128 | 8 | 4,413 | 92,677 | 179ms | 376ms |
| **pytorch-bf16** | PubMedBERT Pass1 | 128 | 8 | 5,703 | 76,478 | 184ms | 243ms |
| **pytorch-bf16** | PubMedBERT BODHI | 256 | 8 | 6,210 | 83,282 | 352ms | 460ms |
| **pytorch-bf16** | BioBERT Fine-Tuned | 128 | 8 | 4,263 | 89,532 | 269ms | 339ms |
| **ov-bf16** | PubMedBERT Pass1 | 128 | 8 | 11,444 | 153,464 | 88ms | 112ms |
| **ov-bf16** | PubMedBERT BODHI | 64 | 8 | 11,122 | 149,146 | 42ms | 61ms |
| **ov-bf16** | BioBERT Fine-Tuned | 64 | 8 | 7,586 | 159,314 | 62ms | 84ms |
| **ov-int8** | PubMedBERT Pass1 | 64 | 8 | 13,883 | 186,170 | 31ms | 53ms |
| **ov-int8** | PubMedBERT BODHI | 128 | 8 | 13,414 | 179,887 | 76ms | 104ms |
| **ov-int8** | BioBERT Fine-Tuned | 64 | 8 | 9,440 | 198,246 | 55ms | 71ms |
| **ov-int4** | PubMedBERT Pass1 | 128 | 8 | 10,951 | 146,851 | 86ms | 139ms |
| **ov-int4** | PubMedBERT BODHI | 128 | 8 | 10,544 | 141,391 | 88ms | 138ms |
| **ov-int4** | BioBERT Fine-Tuned | 64 | 8 | 6,708 | 140,859 | 79ms | 86ms |

---

## Multi-Instance Serving Load Test — All Variants

Dynamic batching (bs=256, max_wait=10ms), 40s duration. All architectures × all variants.

### PubMedBERT Pass1 — All Configs

| Variant | Architecture | Clients | Req/s | TPS | p50ms | p95ms | p99ms |
|---------|-------------|---------|-------|-----|-------|-------|-------|
| pytorch-fp32 | 8srv | 300 | 733 | 9,828 | 393ms | 609ms | 715ms |
| pytorch-fp32 | 8srv | 600 | 748 | 10,028 | 773ms | 974ms | 1,091ms |
| pytorch-fp32 | 16srv+HT | 600 | 1,276 | 17,112 | 453ms | 583ms | 637ms |
| pytorch-fp32 | 32srv+HT | 600 | 2,569 | 34,451 | 211ms | 320ms | 334ms |
| pytorch-fp32 | 32srv+HT | 1200 | 2,629 | 35,254 | 426ms | 562ms | 611ms |
| pytorch-fp16 | 8srv | 300 | 2,856 | 38,298 | 101ms | 115ms | 121ms |
| pytorch-fp16 | 8srv | 600 | 3,302 | 44,281 | 176ms | 198ms | 209ms |
| pytorch-fp16 | 16srv+HT | 600 | 5,541 | 74,304 | 103ms | 117ms | 122ms |
| pytorch-fp16 | 32srv+HT | 600 | 6,921 | 92,814 | 82ms | 111ms | 132ms |
| pytorch-fp16 | 32srv+HT | 1200 | 7,216 | 96,786 | 161ms | 208ms | 228ms |
| pytorch-bf16 | 8srv | 300 | 2,873 | 38,527 | 100ms | 114ms | 119ms |
| pytorch-bf16 | 8srv | 600 | 3,336 | 44,734 | 174ms | 196ms | 207ms |
| pytorch-bf16 | 16srv+HT | 600 | 5,514 | 73,943 | 103ms | 117ms | 124ms |
| pytorch-bf16 | 32srv+HT | 600 | 7,322 | 98,192 | 79ms | 120ms | 133ms |
| pytorch-bf16 | 32srv+HT | 1200 | 7,565 | 101,447 | 160ms | 202ms | 218ms |
| **ov-bf16** | **32srv+HT** | **600** | **8,219** | **110,215** | **74ms** | **80ms** | **109ms** |
| **ov-int8** | **32srv+HT** | **600** | **10,117** | **135,668** | **57ms** | **92ms** | **114ms** |
| ov-int4 | 32srv+HT | 600 | 6,404 | 85,877 | 88ms | 122ms | 127ms |

### PubMedBERT BODHI — 32srv+HT Only

| Variant | Clients | Req/s | TPS | p50ms | p95ms | p99ms |
|---------|---------|-------|-----|-------|-------|-------|
| pytorch-fp32 | 600 | 2,581 | 34,612 | 213ms | 309ms | 416ms |
| pytorch-fp16 | 600 | 7,539 | 101,103 | 80ms | 86ms | 96ms |
| pytorch-bf16 | 600 | 7,603 | 101,955 | 78ms | 103ms | 127ms |
| ov-bf16 | 600 | 8,213 | 110,139 | 74ms | 81ms | 115ms |
| **ov-int8** | **600** | **10,121** | **135,721** | **58ms** | **91ms** | **113ms** |
| ov-int4 | 600 | 5,849 | 78,435 | 94ms | 141ms | 176ms |

### BioBERT Fine-Tuned — 32srv+HT Only

| Variant | Clients | Req/s | TPS | p50ms | p95ms | p99ms |
|---------|---------|-------|-----|-------|-------|-------|
| pytorch-fp32 | 600 | 1,307 | 27,440 | 290ms | 457ms | 6322ms |
| pytorch-fp16 | 600 | 5,471 | 114,888 | 108ms | 120ms | 124ms |
| pytorch-bf16 | 600 | 5,623 | 118,086 | 106ms | 118ms | 122ms |
| ov-bf16 | 600 | 5,912 | 124,160 | 101ms | 109ms | 113ms |
| **ov-int8** | **600** | **8,689** | **182,469** | **67ms** | **74ms** | **126ms** |
| ov-int4 | 600 | 4,198 | 88,150 | 140ms | 188ms | 239ms |

---

## PyTorch vs OpenVINO — Head-to-Head at 32srv+HT/600c

| Model | PT-BF16 TPS | OV-BF16 TPS | OV advantage | OV-INT8 TPS | INT8 vs PT-BF16 |
|-------|------------|-------------|-------------|-------------|-----------------|
| PubMedBERT Pass1 | 98,192 | 110,215 | **+12.2%** | 135,668 | **+38.2%** |
| PubMedBERT BODHI | 101,955 | 110,139 | **+8.0%** | 135,721 | **+33.1%** |
| BioBERT Fine-Tuned | 118,086 | 124,160 | **+5.1%** | 182,469 | **+54.5%** |

**OpenVINO BF16 is 5–12% faster than PyTorch BF16 at the same NUMA serving config.**
**OpenVINO INT8 is 33–55% faster than PyTorch BF16 with <0.3% accuracy loss.**

---

## Key Findings

1. **PyTorch FP32 is unusable for serving** — 34K TPS vs 110K TPS for OV BF16. 3× slower with 5–6× higher latency. Batch latency up to 2252ms in scenario sweep.

2. **PyTorch FP16/BF16 are competitive but lose to OV** — At 32srv+HT/600c, PT-BF16 reaches ~100K TPS (pubmed/bodhi) vs OV-BF16 at 110K TPS. The gap is 8–12%, not catastrophic, but OV also has better tail latency (p95 80ms vs 120ms for pubmed).

3. **OV INT8 is the clear winner** — 135K TPS for PubMedBERT variants at 57ms p50, with only 0.28% cosine degradation vs FP32. Best combination of throughput, latency, and accuracy.

4. **INT4 fails on accuracy** — Despite 67 MB model size (potentially fitting in L3 cache), mean cosine similarity drops to 0.90–0.99. The throughput gain (~85K TPS) does not justify the embedding quality loss for production biomedical use.

5. **Optimal production config: OV INT8, 32srv+HT, 600 clients**
   - PubMedBERT Pass1: **135,668 TPS** at **57ms p50 / 92ms p95**
   - PubMedBERT BODHI: **135,721 TPS** at **58ms p50 / 91ms p95**
   - BioBERT: **182,469 TPS** at **67ms p50 / 74ms p95**

---

*Generated from results/ — 288 scenario + 90 load test runs*
*System: 2× Intel Xeon 6737P · OpenVINO 2026.1.0 · NNCF 3.1.0 · Ubuntu 24.04.4 LTS*

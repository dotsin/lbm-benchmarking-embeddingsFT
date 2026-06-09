"""
Scenario sweep benchmark — all variants, all batch sizes, all worker counts.
Tests: pytorch-fp32, pytorch-fp16, pytorch-bf16, ov-bf16, ov-int8, ov-int4

Each worker is NUMA-pinned. Results saved per variant.

Usage:
  python scenario_bench.py --variant ov-bf16 --model pubmed
  python scenario_bench.py --variant all --model all
"""
import argparse, json, os, sys, time, threading
import multiprocessing as mp
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="all",
    choices=["pytorch-fp32","pytorch-fp16","pytorch-bf16",
             "ov-bf16","ov-int8","ov-int4","all"])
ap.add_argument("--model",   default="all", choices=["pubmed","bodhi","biobert","all"])
ap.add_argument("--bs",      type=int, default=None, help="Single batch size (default: all)")
ap.add_argument("--workers", type=int, default=None, help="Single worker count (default: all)")
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from config import (PT_PATHS, OV_PATHS, SENTENCES, MODELS, RESULTS_DIR,
                    BATCH_SIZES, WORKER_COUNTS, WARMUP, BENCH_SECS,
                    TOKENS_PER_SENTENCE)

VARIANTS_TO_RUN = (
    ["pytorch-fp32","pytorch-fp16","pytorch-bf16","ov-bf16","ov-int8","ov-int4"]
    if args.variant == "all" else [args.variant]
)
MODELS_TO_RUN = list(MODELS.keys()) if args.model == "all" else [args.model]
BS_LIST = [args.bs] if args.bs else BATCH_SIZES
W_LIST  = [args.workers] if args.workers else WORKER_COUNTS


def worker_core_plan(n_workers):
    """Return list of core_sets for n_workers, NUMA-balanced."""
    if n_workers == 1:
        return [list(range(0, 64))]
    cores_per_worker = 64 // n_workers
    return [list(range(i * cores_per_worker, (i+1) * cores_per_worker))
            for i in range(n_workers)]


def run_worker(variant, model_key, model_path, core_set, batch_size,
               results_queue, ready_event, stop_event):
    """Single worker process — NUMA-pinned, runs inference in a loop."""
    os.sched_setaffinity(0, set(core_set))
    n_cores = len(core_set)
    os.environ["OMP_NUM_THREADS"]  = str(n_cores)
    os.environ["MKL_NUM_THREADS"]  = str(n_cores)
    os.environ["KMP_AFFINITY"]     = "granularity=fine,compact,1,0"
    os.environ["KMP_HW_SUBSET"]    = f"{n_cores}c,1t"
    os.environ["GOMP_SPINCOUNT"]   = "0"

    import numpy as np
    corpus = (SENTENCES * 100)

    if variant.startswith("pytorch"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        dtype_str = variant.split("-")[1]
        dtype_map  = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[dtype_str]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        model.eval()

        # Warmup
        batch = corpus[:batch_size]
        for _ in range(WARMUP):
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            with torch.no_grad():
                model(**enc)

        ready_event.set()
        latencies = []; total = 0
        while not stop_event.is_set():
            idx   = total % len(corpus)
            batch = (corpus * 2)[idx:idx+batch_size]
            t0    = time.perf_counter()
            enc   = tokenizer(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            with torch.no_grad():
                model(**enc)
            latencies.append((time.perf_counter() - t0) * 1000)
            total += batch_size
        results_queue.put((total, latencies))

    else:  # openvino
        import openvino as ov
        from transformers import AutoTokenizer

        prec_hint = {"ov-bf16": "bf16", "ov-int8": "bf16", "ov-int4": "bf16"}[variant]
        xml = str(Path(model_path) / "openvino_model.xml")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        core = ov.Core()
        compiled = core.compile_model(xml, "CPU", {
            "PERFORMANCE_HINT":         "THROUGHPUT",
            "INFERENCE_NUM_THREADS":    str(n_cores),
            "NUM_STREAMS":              "1",
            "INFERENCE_PRECISION_HINT": prec_hint,
        })
        valid = {i.get_any_name() for i in compiled.inputs}

        batch = corpus[:batch_size]
        for _ in range(WARMUP):
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})

        ready_event.set()
        latencies = []; total = 0
        while not stop_event.is_set():
            idx   = total % len(corpus)
            batch = (corpus * 2)[idx:idx+batch_size]
            t0    = time.perf_counter()
            enc   = tokenizer(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})
            latencies.append((time.perf_counter() - t0) * 1000)
            total += batch_size
        results_queue.put((total, latencies))


if __name__ == "__main__":
    all_results = []

    for variant in VARIANTS_TO_RUN:
        for model_key in MODELS_TO_RUN:
            label = MODELS[model_key]

            # Resolve model path
            if variant.startswith("pytorch"):
                model_path = PT_PATHS[model_key]
            else:
                prec_key   = variant.split("-")[1]
                ov_key     = f"{model_key}_{prec_key}"
                model_path = OV_PATHS[ov_key]
                if not (Path(model_path) / "openvino_model.xml").exists():
                    print(f"[SKIP] {variant}/{model_key} — not quantized yet")
                    continue

            for bs in BS_LIST:
                for n_workers in W_LIST:
                    tag = f"{variant}/{model_key}/bs{bs}/w{n_workers}"
                    out_file = RESULTS_DIR / f"scenario_{variant.replace('-','_')}_{model_key}_bs{bs}_w{n_workers}.json"
                    if out_file.exists():
                        print(f"[skip] {tag}")
                        continue

                    print(f"\n[bench] {tag}", flush=True)
                    core_plan = worker_core_plan(n_workers)

                    rq = [mp.Queue() for _ in range(n_workers)]
                    re = [mp.Event()  for _ in range(n_workers)]
                    se = [mp.Event()  for _ in range(n_workers)]

                    procs = []
                    for i, core_set in enumerate(core_plan):
                        p = mp.Process(
                            target=run_worker,
                            args=(variant, model_key, model_path,
                                  core_set, bs, rq[i], re[i], se[i]),
                            daemon=True,
                        )
                        p.start()
                        procs.append(p)

                    for e in re:
                        e.wait()
                    print(f"  {n_workers} workers ready — running {BENCH_SECS}s...", flush=True)

                    t_start = time.perf_counter()
                    time.sleep(BENCH_SECS)
                    for e in se:
                        e.set()
                    for p in procs:
                        p.join(timeout=15)
                    elapsed = time.perf_counter() - t_start

                    total_sentences = 0
                    all_lats = []
                    for q in rq:
                        try:
                            cnt, lats = q.get(timeout=5)
                            total_sentences += cnt
                            all_lats.extend(lats)
                        except Exception:
                            pass

                    if not all_lats:
                        print(f"  [WARN] no results collected")
                        continue

                    lats = np.array(all_lats)
                    sps  = total_sentences / elapsed
                    tps  = sps * TOKENS_PER_SENTENCE[model_key]

                    r = {
                        "variant":   variant,
                        "model":     model_key,
                        "model_label": label,
                        "batch_size": bs,
                        "n_workers":  n_workers,
                        "cores_per_worker": len(core_plan[0]),
                        "total_sentences": total_sentences,
                        "elapsed_s": round(elapsed, 2),
                        "sentences_per_sec":   round(sps, 1),
                        "tokens_per_sec":      round(tps, 1),
                        "concurrency_x_throughput": round(n_workers * sps, 1),
                        "batch_latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
                        "batch_latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
                        "batch_latency_p99_ms": round(float(np.percentile(lats, 99)), 2),
                        "batch_latency_mean_ms": round(float(lats.mean()), 2),
                    }
                    out_file.write_text(json.dumps(r, indent=2))
                    all_results.append(r)
                    print(f"  SPS={sps:,.0f}  TPS={tps:,.0f}  "
                          f"p50={r['batch_latency_p50_ms']}ms  "
                          f"p95={r['batch_latency_p95_ms']}ms")

    print(f"\n[scenario_bench] done — {len(all_results)} configs completed")

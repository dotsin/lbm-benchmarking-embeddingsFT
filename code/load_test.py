"""
Multi-instance NUMA-pinned load test — all variants.
Runs 8srv/300c, 8srv/600c, 16srv+HT/600c, 32srv+HT/600c, 32srv+HT/1200c for each variant.

Usage:
  python load_test.py --variant ov-int8 --model pubmed
  python load_test.py --variant all --model all
"""
import argparse, json, os, sys, time, threading
import multiprocessing as mp
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="all",
    choices=["pytorch-fp32","pytorch-fp16","pytorch-bf16",
             "ov-bf16","ov-int8","ov-int4","all"])
ap.add_argument("--model",   default="all", choices=["pubmed","bodhi","biobert","all"])
ap.add_argument("--clients", type=int,   default=None)
ap.add_argument("--servers", type=int,   default=None)
ap.add_argument("--use-ht",  action="store_true")
ap.add_argument("--duration",type=int,   default=40)
ap.add_argument("--max-bs",  type=int,   default=256)
ap.add_argument("--max-wait",type=float, default=10)
ap.add_argument("--out",     default=None)
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from config import (PT_PATHS, OV_PATHS, SENTENCES, MODELS, RESULTS_DIR,
                    TOKENS_PER_SENTENCE,
                    SERVER_PLAN_PHYS, SERVER_PLAN_HT_16, SERVER_PLAN_HT_32)

VARIANTS_TO_RUN = (
    ["pytorch-fp32","pytorch-fp16","pytorch-bf16","ov-bf16","ov-int8","ov-int4"]
    if args.variant == "all" else [args.variant]
)
MODELS_TO_RUN = list(MODELS.keys()) if args.model == "all" else [args.model]

# Load test matrix if no specific config requested
LOAD_CONFIGS = []
if args.clients is None:
    LOAD_CONFIGS = [
        # (n_servers, use_ht, clients, plan)
        (8,  False, 300,  SERVER_PLAN_PHYS),
        (8,  False, 600,  SERVER_PLAN_PHYS),
        (16, True,  600,  SERVER_PLAN_HT_16),
        (32, True,  600,  SERVER_PLAN_HT_32),
        (32, True,  1200, SERVER_PLAN_HT_32),
    ]
else:
    plan = (SERVER_PLAN_HT_32 if args.servers == 32 else
            SERVER_PLAN_HT_16 if args.servers == 16 else SERVER_PLAN_PHYS)
    LOAD_CONFIGS = [(args.servers, args.use_ht, args.clients, plan)]


def server_process(server_id, core_set, req_queue, result_queue,
                   variant, model_path, max_bs, max_wait_ms, ready_event, use_ht):
    import os, time, queue as Q
    import numpy as np
    from pathlib import Path

    os.sched_setaffinity(0, set(core_set))
    n_logical = len(core_set)
    n_phys    = n_logical // 2 if use_ht else n_logical

    os.environ["OMP_NUM_THREADS"]  = str(n_logical)
    os.environ["MKL_NUM_THREADS"]  = str(n_logical)
    os.environ["KMP_AFFINITY"]     = "granularity=fine,compact,1,0"
    os.environ["KMP_HW_SUBSET"]    = f"{n_phys}c,2t" if use_ht else f"{n_phys}c,1t"
    os.environ["GOMP_SPINCOUNT"]   = "0"

    corpus = SENTENCES

    if variant.startswith("pytorch"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[variant.split("-")[1]]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        model.eval()

        def infer(texts):
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            with torch.no_grad():
                model(**enc)

    else:
        import openvino as ov
        from transformers import AutoTokenizer
        prec = {"ov-bf16": "bf16", "ov-int8": "bf16", "ov-int4": "bf16"}[variant]
        xml  = str(Path(model_path) / "openvino_model.xml")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        core = ov.Core()
        compiled = core.compile_model(xml, "CPU", {
            "PERFORMANCE_HINT":         "THROUGHPUT",
            "INFERENCE_NUM_THREADS":    str(n_logical),
            "NUM_STREAMS":              "1",
            "INFERENCE_PRECISION_HINT": prec,
        })
        valid = {i.get_any_name() for i in compiled.inputs}

        def infer(texts):
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})

    # Warmup
    for _ in range(3):
        infer(corpus[:8])

    ready_event.set()
    max_wait_s = max_wait_ms / 1000.0

    while True:
        batch_items = []
        deadline = time.perf_counter() + max_wait_s
        while len(batch_items) < max_bs:
            rem = deadline - time.perf_counter()
            if rem <= 0:
                break
            try:
                item = req_queue.get(timeout=min(rem, 0.001))
                if item is None:
                    return
                batch_items.append(item)
            except Q.Empty:
                if time.perf_counter() >= deadline:
                    break
        if not batch_items:
            continue
        infer([it[1] for it in batch_items])
        for (client_id, _) in batch_items:
            result_queue.put(client_id)


def run_load_test(variant, model_key, model_path, plan, use_ht, n_clients, duration, max_bs, max_wait):
    n_servers = len(plan)
    ht_label  = "+HT" if use_ht else ""
    arch      = f"{n_servers}srv{ht_label}"
    tag       = f"{variant}/{model_key}/{arch}/{n_clients}c"
    out_file  = RESULTS_DIR / f"loadtest_{variant.replace('-','_')}_{model_key}_{arch}_{n_clients}c_bs{max_bs}.json"

    if out_file.exists():
        print(f"[skip] {tag}")
        return

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}", flush=True)

    req_queues   = [mp.Queue() for _ in range(n_servers)]
    res_queues   = [mp.Queue() for _ in range(n_servers)]
    ready_events = [mp.Event()  for _ in range(n_servers)]
    client_events = [threading.Event() for _ in range(n_clients)]

    procs = []
    for sid, (_, core_set) in enumerate(plan):
        p = mp.Process(
            target=server_process,
            args=(sid, core_set, req_queues[sid], res_queues[sid],
                  variant, model_path, max_bs, max_wait,
                  ready_events[sid], use_ht),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Dispatcher threads
    dispatch_stop = threading.Event()
    def dispatcher(sid):
        rq = res_queues[sid]
        while not dispatch_stop.is_set():
            try:
                cid = rq.get(timeout=0.01)
                client_events[cid].set()
            except Exception:
                pass

    for sid in range(n_servers):
        threading.Thread(target=dispatcher, args=(sid,), daemon=True).start()

    print("  waiting for servers...", flush=True)
    for e in ready_events:
        e.wait()
    print(f"  all {n_servers} servers ready", flush=True)

    all_rt      = []
    records_lock = threading.Lock()
    barrier      = threading.Barrier(n_clients + 1)
    corpus       = SENTENCES * 20

    def client_fn(cid):
        srv   = cid % n_servers
        rq    = req_queues[srv]
        ev    = client_events[cid]
        idx   = cid
        local = []
        barrier.wait()
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            text = corpus[idx % len(corpus)]; idx += 1
            t0   = time.perf_counter()
            ev.clear()
            rq.put((cid, text))
            ev.wait()
            local.append((time.perf_counter() - t0) * 1000)
        with records_lock:
            all_rt.extend(local)

    threads = [threading.Thread(target=client_fn, args=(i,), daemon=True)
               for i in range(n_clients)]
    for t in threads: t.start()
    print(f"  {n_clients} clients firing...", flush=True)
    wall_start = time.perf_counter()
    barrier.wait()
    for t in threads: t.join()
    elapsed = time.perf_counter() - wall_start

    for q in req_queues: q.put(None)
    for p in procs: p.join(timeout=5)
    dispatch_stop.set()

    rt  = np.array(all_rt)
    rps = len(rt) / elapsed
    tps = rps * TOKENS_PER_SENTENCE[model_key]

    result = {
        "variant":   variant,
        "model":     model_key,
        "model_label": MODELS[model_key],
        "architecture": arch,
        "use_ht":    use_ht,
        "servers":   n_servers,
        "clients":   n_clients,
        "max_bs":    max_bs,
        "duration_s": round(elapsed, 2),
        "total_requests":      len(rt),
        "requests_per_sec":    round(rps, 2),
        "tokens_per_sec":      round(tps, 1),
        "response_time_p50_ms": round(float(np.percentile(rt, 50)), 2),
        "response_time_p95_ms": round(float(np.percentile(rt, 95)), 2),
        "response_time_p99_ms": round(float(np.percentile(rt, 99)), 2),
        "response_time_mean_ms": round(float(rt.mean()), 2),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    else:
        out_file.write_text(json.dumps(result, indent=2))

    print(f"  Req/s={rps:,.1f}  TPS={tps:,.0f}  "
          f"p50={result['response_time_p50_ms']}ms  "
          f"p95={result['response_time_p95_ms']}ms  "
          f"p99={result['response_time_p99_ms']}ms")
    return result


if __name__ == "__main__":
    for variant in VARIANTS_TO_RUN:
        for model_key in MODELS_TO_RUN:
            if variant.startswith("pytorch"):
                model_path = PT_PATHS[model_key]
            else:
                prec_key   = variant.split("-")[1]
                model_path = OV_PATHS[f"{model_key}_{prec_key}"]
                if not (Path(model_path) / "openvino_model.xml").exists():
                    print(f"[SKIP] {variant}/{model_key} — not quantized")
                    continue

            for n_srv, use_ht, n_clients, plan in LOAD_CONFIGS:
                run_load_test(variant, model_key, model_path,
                              plan, use_ht, n_clients,
                              args.duration, args.max_bs, args.max_wait)

    print("\n[load_test] done")

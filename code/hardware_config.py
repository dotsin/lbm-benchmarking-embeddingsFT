"""
Reads config.yaml (repo root) and exposes hardware, NUMA, OV, and
server-plan settings to all benchmark scripts.

Usage:
    from hardware_config import CFG, NUMACTL_ARGS, SERVER_PLAN_PHYS, \
        SERVER_PLAN_HT_16, SERVER_PLAN_HT_32, OV_CONFIG, PT_ENV, \
        BATCH_SIZES, WORKER_COUNTS, WARMUP, BENCH_SECS, VARIANTS
"""
from pathlib import Path
import os, yaml

_ROOT = Path(__file__).resolve().parent.parent
_YAML = Path(os.environ.get("BENCH_CONFIG", _ROOT / "config.yaml"))

with open(_YAML) as _f:
    CFG = yaml.safe_load(_f)

# ── NUMA ──────────────────────────────────────────────────────
NUMACTL_ARGS = [
    "numactl",
    f"--cpunodebind={CFG['numa']['cpunodebind']}",
    f"--membind={CFG['numa']['membind']}",
    "--",
]

# ── Server plans: list of (numa_node, [cpu_ids]) ──────────────
def _parse(plan): return [(w["node"], w["cores"]) for w in plan]
SERVER_PLAN_PHYS  = _parse(CFG["server_plans"]["phys"])
SERVER_PLAN_HT_16 = _parse(CFG["server_plans"]["ht_16"])
SERVER_PLAN_HT_32 = _parse(CFG["server_plans"]["ht_32"])

# ── OpenVINO property dict ────────────────────────────────────
_ov = CFG["openvino"]
OV_CONFIG = {"PERFORMANCE_HINT": _ov["performance_hint"],
             "INFERENCE_PRECISION_HINT": _ov["inference_precision"]}
if _ov["num_streams"]: OV_CONFIG["NUM_STREAMS"] = str(_ov["num_streams"])
if _ov["num_threads"]: OV_CONFIG["INFERENCE_NUM_THREADS"] = str(_ov["num_threads"])

# ── PyTorch environment variables ─────────────────────────────
_pt = CFG["pytorch"]
PT_ENV = {
    "OMP_NUM_THREADS": str(_pt["omp_num_threads"]),
    "MKL_NUM_THREADS": str(_pt["omp_num_threads"]),
    "KMP_BLOCKTIME":   str(_pt["kmp_blocktime"]),
    "KMP_AFFINITY":    _pt["kmp_affinity"],
}

# ── Benchmark parameters ──────────────────────────────────────
BATCH_SIZES   = CFG["bench"]["batch_sizes"]
WORKER_COUNTS = CFG["bench"]["worker_counts"]
WARMUP        = CFG["bench"]["warmup_batches"]
BENCH_SECS    = CFG["bench"]["bench_seconds"]
VARIANTS      = [tuple(v) for v in CFG["variants"]]

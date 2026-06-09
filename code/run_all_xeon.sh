#!/usr/bin/env bash
# run_all_xeon.sh — Master driver for the AWS-benchmark code on Intel Xeon
# 6737P (2× sockets, 64 phys / 128 logical cores, 2 NUMA nodes).
# Self-contained: assumes models live under  aws_benchmark/models/.
#
# Usage:
#   ./run_all_xeon.sh                 # default: APS only
#   ./run_all_xeon.sh --vtune         # also run VTune hotspots
#   ./run_all_xeon.sh --no-aps        # skip APS wrapping
#
# Environment overrides:
#   VENV               path to python venv (default: ../.venv or ~/.venv)
#   AWS_BENCH_MODELS   override models dir (default: $ROOT/models)
#   AWS_BENCH_RESULTS  override results dir (default: $ROOT/results)
set -euo pipefail

USE_VTUNE=0
USE_APS=1
for arg in "$@"; do
  case "$arg" in
    --vtune)   USE_VTUNE=1 ;;
    --no-aps)  USE_APS=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

# ── Paths (ROOT-relative; this script lives in aws_benchmark/code/) ─────────
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$CODE_DIR")"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${AWS_BENCH_RESULTS:-$ROOT/results}/xeon_run_${TS}"
mkdir -p "$RUN_DIR"
export AWS_BENCH_MODELS="${AWS_BENCH_MODELS:-$ROOT/models}"
export AWS_BENCH_RESULTS="$RUN_DIR"

# ── Venv ────────────────────────────────────────────────────────────────────
VENV="${VENV:-}"
if [[ -z "$VENV" ]]; then
  for cand in "$ROOT/.venv" "$(dirname "$ROOT")/.venv" "$HOME/.venv"; do
    [[ -f "$cand/bin/activate" ]] && VENV="$cand" && break
  done
fi
[[ -n "$VENV" && -f "$VENV/bin/activate" ]] || { echo "no venv found — set VENV=…"; exit 3; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── Intel oneAPI tools (vtune, aps) — optional ──────────────────────────────
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -u
fi

# ── Thread env (full machine: 2 sockets × 32 cores × 2 HT) ──────────────────
export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_HW_SUBSET=2s,32c,2t

NUMACTL=(numactl --cpunodebind=0,1 --membind=0,1)

run_wrapped() {
  local tag="$1"; shift
  local logf="$RUN_DIR/${tag}.log"
  echo "[${tag}] $(date +%H:%M:%S)  →  $logf"
  if [[ $USE_VTUNE -eq 1 ]]; then
    "${NUMACTL[@]}" vtune -collect hotspots -r "$RUN_DIR/vtune_${tag}" -- "$@" 2>&1 | tee "$logf"
  elif [[ $USE_APS -eq 1 ]] && command -v aps >/dev/null 2>&1; then
    "${NUMACTL[@]}" aps --collection-mode=hw --result-dir="$RUN_DIR/aps_${tag}" -- "$@" 2>&1 | tee "$logf"
  else
    "${NUMACTL[@]}" "$@" 2>&1 | tee "$logf"
  fi
}

VARIANTS=(pytorch-fp32 pytorch-fp16 pytorch-bf16 ov-bf16 ov-int8 ov-int4)
MODELS_LIST=(pubmed bodhi biobert)

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Xeon 6737P run — ts=$TS  vtune=$USE_VTUNE  aps=$USE_APS"
echo "  ROOT     = $ROOT"
echo "  MODELS   = $AWS_BENCH_MODELS"
echo "  RESULTS  = $RUN_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

# ── 1. Quantize if missing ──────────────────────────────────────────────────
QBASE="$AWS_BENCH_MODELS/openvino_quantized"
need_quant=0
for m in "${MODELS_LIST[@]}"; do
  for p in int8 int4; do
    [[ -f "$QBASE/${m}-${p}/openvino_model.xml" ]] || need_quant=1
  done
done

if [[ $need_quant -eq 1 ]]; then
  echo "[quantize] some INT8/INT4 models missing → running quantize.py"
  "${NUMACTL[@]}" python "$CODE_DIR/quantize.py" --model all --precision all \
    2>&1 | tee "$RUN_DIR/quantize.log"
else
  echo "[quantize] all INT8/INT4 models present — skipping"
fi

# ── 2. Accuracy eval ────────────────────────────────────────────────────────
echo
echo "── Accuracy eval (all variants × all models) ──"
run_wrapped "accuracy_eval" \
  python "$CODE_DIR/accuracy_eval.py" --model all \
    --out "$RUN_DIR/accuracy_results.json"

# ── 3. bench_all for each variant ───────────────────────────────────────────
echo
echo "── bench_all (6 variants × 3 models) ──"
for v in "${VARIANTS[@]}"; do
  run_wrapped "bench_${v//-/_}" \
    python "$CODE_DIR/bench_all.py" --variant "$v" --model all \
      --out-dir "$RUN_DIR"
done

# ── 4. extended_benchmark per variant ───────────────────────────────────────
echo
echo "── extended_benchmark per variant ──"
for v in "${VARIANTS[@]}"; do
  run_wrapped "extended_${v//-/_}" \
    python "$CODE_DIR/extended_benchmark.py" --variant "$v" \
      --out-dir "$RUN_DIR"
done

# ── 5. compare_finetuned per model ──────────────────────────────────────────
echo
echo "── compare_finetuned per model ──"
for m in "${MODELS_LIST[@]}"; do
  run_wrapped "compare_${m}" \
    python "$CODE_DIR/compare_finetuned.py" --model "$m" \
      --out-dir "$RUN_DIR"
done

# ── 6. threshold_sweep (OV only) ────────────────────────────────────────────
echo
echo "── threshold_sweep ──"
( cd "$CODE_DIR" && run_wrapped "threshold_sweep" python "$CODE_DIR/threshold_sweep.py" )

# ── 7. Optional: scenario sweep + multi-instance load test ──────────────────
# These are heavier (hours) — gated on $RUN_HEAVY=1
if [[ "${RUN_HEAVY:-0}" == "1" ]]; then
  echo
  echo "── scenario sweep (heavy) ──"
  run_wrapped "scenario_bench" python "$CODE_DIR/scenario_bench.py"
  echo
  echo "── multi-instance load test (heavy) ──"
  run_wrapped "load_test" python "$CODE_DIR/load_test.py"
fi

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "  DONE  →  $RUN_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

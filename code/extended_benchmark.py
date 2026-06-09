"""
extended_benchmark.py
=====================
Per-model + cross-domain intelligence benchmarks for local finetuned models
across all 6 variants. NUMA-pinned via outer numactl wrapper.

Benchmarks:
  B1 within-domain similarity
  B2 cross-domain discrimination
  B3 STS Spearman (BIOSSES-style)
  B4 throughput (sentences/sec) at varying batch sizes
  B5 hard-negative detection
  B6 domain centroid geometry

Usage:
  python extended_benchmark.py --variant ov-bf16
  python extended_benchmark.py --variant pytorch-bf16 --models biobert pubmed
"""
import argparse, json, os, sys, time, warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="ov-bf16",
    choices=["pytorch-fp32","pytorch-fp16","pytorch-bf16",
             "ov-bf16","ov-int8","ov-int4"])
ap.add_argument("--models", nargs="+", default=["pubmed","bodhi","biobert"],
    choices=["pubmed","bodhi","biobert"])
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

from config import PT_PATHS, OV_PATHS, MODELS as MODEL_LABELS

N_CORES = len(os.sched_getaffinity(0))
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["KMP_BLOCKTIME"]   = "1"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"
os.environ["ONEDNN_VERBOSE"]  = "0"

OUT_DIR = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[platform] variant={args.variant}  cores={N_CORES}")


def resolve(variant, model_key):
    if variant.startswith("pytorch"):
        return PT_PATHS[model_key]
    return OV_PATHS[f"{model_key}_{variant.split('-')[1]}"]


REGISTRY = {}
for mk in args.models:
    path = resolve(args.variant, mk)
    if args.variant.startswith("ov") and not (Path(path) / "openvino_model.xml").exists():
        print(f"  [skip] {mk} — OV model not at {path}")
        continue
    REGISTRY[MODEL_LABELS[mk]] = (mk, path)


# ── Domain corpus ────────────────────────────────────────────────────────────
DOMAIN_CORPUS = {
    "genetics": [
        "BRCA1 pathogenic variant c.5266dupC detected in germline",
        "TP53 missense mutation R175H associated with Li-Fraumeni syndrome",
        "EGFR exon 19 deletion driver mutation in non-small cell lung cancer",
        "KRAS G12D activating mutation found in pancreatic ductal adenocarcinoma",
        "HER2 amplification confirmed by FISH in invasive breast carcinoma",
        "MLH1 promoter hypermethylation causing mismatch repair deficiency",
        "PTEN loss of heterozygosity in endometrial cancer specimen",
        "ALK rearrangement detected via next-generation sequencing panel",
        "POLE exonuclease domain mutation conferring ultramutated phenotype",
        "CDKN2A homozygous deletion identified in melanoma cell line",
    ],
    "biomarkers": [
        "Serum cortisol 28.4 μg/dL, above reference range 6–23 μg/dL",
        "HbA1c 8.5%, fasting plasma glucose 186 mg/dL, insulin resistance suspected",
        "C-reactive protein 18 mg/L elevated, ESR 45 mm/hr indicating inflammation",
        "Troponin I 2.1 ng/mL markedly elevated, consistent with myocardial injury",
        "TSH 0.02 mIU/L suppressed, free T4 2.8 ng/dL elevated, hyperthyroidism",
        "LDL cholesterol 185 mg/dL, HDL 38 mg/dL, triglycerides 290 mg/dL",
        "Ferritin 8 ng/mL low, serum iron 42 μg/dL, TIBC 480 μg/dL, iron deficiency",
        "Creatinine 2.4 mg/dL, eGFR 28 mL/min/1.73m², stage 4 chronic kidney disease",
        "CA-125 156 U/mL elevated in post-menopausal patient with pelvic mass",
        "Procalcitonin 4.2 ng/mL, suggestive of bacterial sepsis",
    ],
    "physiology": [
        "Resting heart rate 82 bpm, blood pressure 148/94 mmHg, hypertensive",
        "VO2 max 38 mL/kg/min, below age-matched average for sedentary adult male",
        "Sleep architecture disrupted, REM latency 95 minutes, slow wave sleep reduced",
        "Sympathetic nervous system hyperactivation detected via heart rate variability",
        "Hypothalamic-pituitary-adrenal axis dysregulation observed under stress",
        "Metabolic rate 1650 kcal/day resting, consistent with hypothyroidism",
        "Circadian rhythm phase delayed by 2 hours, melatonin onset at 11 PM",
        "Gut microbiome diversity index Shannon H=2.8, below healthy range",
        "Vagal tone reduced, low frequency HRV 18 ms², indicating poor recovery",
        "Respiratory rate 18 breaths per minute at rest, within normal limits",
    ],
    "psychology": [
        "Patient meets DSM-5 criteria for major depressive episode, duration 3 months",
        "Generalised anxiety disorder with persistent worry, difficulty concentrating",
        "Cognitive behavioural therapy indicated for maladaptive thought patterns",
        "PHQ-9 score 18 indicating moderately severe depression requiring intervention",
        "GAD-7 score 15 consistent with severe generalised anxiety disorder",
        "Hallucinations and paranoid ideation consistent with first episode psychosis",
        "Trauma history with hypervigilance and intrusive memories, PTSD evaluation",
        "Bipolar II disorder, current hypomanic episode, elevated mood and impulsivity",
        "Attachment insecurity, anxious preoccupied style affecting relationships",
        "Executive function deficits in planning and cognitive flexibility observed",
    ],
    "journal": [
        "Today was exhausting. Mood 3/10. Couldn't focus at work, felt distant from everything.",
        "Woke up at 5am with racing thoughts. Anxiety level 8/10. Hard to get back to sleep.",
        "Had a really good session with my therapist today. Feeling lighter. Mood 7/10.",
        "Slept 9 hours but still tired. Low energy all day. Journaling feels like effort.",
        "Went for a 40-minute run this morning. Mood lifted noticeably. Energy 6/10.",
        "Argument with my partner. Feeling misunderstood and frustrated. Cried twice.",
        "Productive day at work. Completed the project draft. Mood 8/10, feeling accomplished.",
        "Social anxiety spiked at the dinner party. Had to leave early. Felt embarrassed.",
        "Meditation helped this morning. Noticeably calmer. Stress level 3/10 today.",
        "Can't shake this sadness. Nothing specific. Just heavy. Mood 2/10.",
    ],
    "clinical_notes": [
        "Patient presents with 3-week history of progressive fatigue and dyspnoea on exertion",
        "Vital signs: BP 162/98, HR 94, SpO2 94% on room air, temp 37.8°C",
        "Assessment: Decompensated heart failure with bilateral pleural effusions",
        "Plan: IV furosemide 80mg, fluid restriction, daily weight monitoring, cardiology referral",
        "Review of systems positive for orthopnoea, paroxysmal nocturnal dyspnoea, ankle oedema",
        "Past medical history: Type 2 diabetes, hypertension, previous NSTEMI 2019",
        "Social history: 30 pack-year smoking history, current smoker, alcohol 14 units/week",
        "Family history: Father died of MI at 58, mother with breast cancer aged 62",
        "Medications reviewed: metformin 1g BD, ramipril 10mg OD, atorvastatin 40mg nocte",
        "Examination: JVP elevated 5cm, bibasal crackles, pitting oedema to mid-shin bilateral",
    ],
}

BIOSSES_STYLE_PAIRS = [
    ("The blood–brain barrier prevents most drugs from entering the CNS.",
     "The BBB is a selective barrier limiting drug penetration into brain tissue.", 0.95),
    ("BRCA1 mutation carriers have elevated lifetime risk of breast cancer.",
     "Pathogenic BRCA1 variants substantially increase breast and ovarian cancer risk.", 0.90),
    ("Cortisol is the primary glucocorticoid secreted by the adrenal cortex.",
     "The adrenal gland produces cortisol as the main stress hormone.", 0.88),
    ("Metformin reduces hepatic glucose production via AMPK activation.",
     "Metformin works by activating AMPK to decrease liver glucose output.", 0.92),
    ("Sleep deprivation impairs prefrontal cortex function and decision-making.",
     "Lack of sleep reduces executive function controlled by the prefrontal cortex.", 0.87),
    ("Major depressive disorder is characterised by persistent low mood.",
     "Depression involves sustained sadness and loss of interest.", 0.85),
    ("Insulin resistance occurs when cells fail to respond normally to insulin.",
     "Type 2 diabetes develops when tissues become unresponsive to insulin signalling.", 0.78),
    ("C-reactive protein is an acute phase reactant elevated during inflammation.",
     "CRP rises rapidly in response to infection and tissue injury.", 0.83),
    ("The hippocampus plays a central role in the formation of new memories.",
     "Memory consolidation depends critically on hippocampal function.", 0.89),
    ("DNA methylation at CpG islands is associated with gene silencing.",
     "Promoter hypermethylation suppresses gene transcription.", 0.80),
    ("Insulin resistance occurs when cells fail to respond to insulin.",
     "The patient reported feeling tired and emotionally drained.", 0.10),
    ("BRCA1 pathogenic variant detected in germline sample.",
     "The stock market experienced high volatility this quarter.", 0.02),
    ("Cortisol elevation indicates HPA axis dysregulation.",
     "She journaled about her day and had a cup of tea.", 0.05),
    ("Troponin I 2.1 ng/mL consistent with myocardial injury.",
     "Mood 7/10 today, good sleep, productive morning.", 0.04),
    ("Major depressive disorder PHQ-9 score 18.",
     "Genome-wide association study identified 23 novel loci.", 0.08),
]

CROSS_DOMAIN_PAIRS = {
    "genetics×journal":    [("BRCA1 pathogenic variant detected", "Mood 3/10 today, felt tired"),
                             ("TP53 mutation Li-Fraumeni syndrome", "Woke up anxious, journaling helps"),
                             ("EGFR exon 19 deletion lung cancer", "Had a difficult conversation with family")],
    "biomarkers×journal":  [("HbA1c 8.5% fasting glucose elevated", "Feeling very low energy today"),
                             ("Cortisol 28 μg/dL above range", "Stressed at work, poor sleep"),
                             ("CRP 18 mg/L inflammation marker", "Emotional day, cried a lot")],
    "genetics×finance":    [("KRAS G12D pancreatic adenocarcinoma", "Portfolio loss 12% this quarter"),
                             ("BRCA2 germline variant ovarian risk", "Stock market volatility increasing"),
                             ("HER2 amplification breast carcinoma", "Interest rates rising, inflation concern")],
    "biomarkers×finance":  [("Troponin I myocardial injury marker", "Bull market rally tech stocks"),
                             ("TSH suppressed free T4 elevated", "Bond yields inversion recession signal"),
                             ("LDL cholesterol 185 mg/dL", "Revenue growth 24% year over year")],
    "psychology×genetics": [("PHQ-9 score 18 moderately severe depression", "BRCA1 frameshift mutation exon 11"),
                             ("GAD-7 score 15 severe anxiety", "PTEN loss heterozygosity endometrial"),
                             ("Bipolar II hypomanic episode", "CDKN2A deletion melanoma specimen")],
    "journal×clinical":    [("Today was exhausting, mood 2/10", "Patient presents with dyspnoea on exertion"),
                             ("Couldn't focus, felt distant", "Assessment: decompensated heart failure"),
                             ("Went for a run, mood lifted", "Plan: IV furosemide 80mg")],
    "physiology×finance":  [("VO2 max 38 mL/kg/min sedentary", "Hedge fund returns 18% annually"),
                             ("HRV low frequency 18 ms²", "IPO valuation 4.2 billion dollars"),
                             ("Circadian rhythm phase delayed", "Quarterly earnings beat expectations")],
    "journal×genetics":    [("Racing thoughts anxiety 8/10", "ALK rearrangement NGS panel"),
                             ("Social anxiety dinner party", "POLE exonuclease ultramutated"),
                             ("Meditation helped, stress 3/10", "MLH1 methylation mismatch repair")],
}

HARD_NEGATIVES = [
    ("BRCA1 pathogenic variant breast cancer risk",
     "BRCA2 mutation associated with hereditary cancer",
     "BRCA1 protein involved in DNA damage repair pathway"),
    ("Patient reports persistent low mood and hopelessness",
     "Depressive symptoms including anhedonia and fatigue",
     "Patient reports persistently elevated cortisol levels"),
    ("HbA1c 8.5% indicates poor glycaemic control",
     "Glycated haemoglobin above target range in diabetic patient",
     "HbA1c 8.5% test performed at 08:00 fasting"),
    ("DNA methylation silences tumour suppressor genes",
     "Epigenetic silencing via promoter hypermethylation",
     "DNA methylation patterns change with ageing"),
    ("Anxiety disorder GAD-7 score 15 severe",
     "Generalised anxiety PHQ score indicates severe symptoms",
     "Anxiety about upcoming surgery is normal patient response"),
]

# ── Backends ────────────────────────────────────────────────────────────────
_pt_cache, _ov_cache, _tok_cache = {}, {}, {}

def _tok(path):
    if path not in _tok_cache:
        from transformers import AutoTokenizer
        _tok_cache[path] = AutoTokenizer.from_pretrained(path)
    return _tok_cache[path]

def _pt(path, dtype_str):
    key = (path, dtype_str)
    if key in _pt_cache: return _pt_cache[key]
    import torch
    from transformers import AutoModel
    torch.set_num_threads(N_CORES)
    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    print(f"  [pt-load] {Path(path).name}", flush=True)
    _pt_cache[key] = AutoModel.from_pretrained(path, torch_dtype=dmap[dtype_str]).eval()
    return _pt_cache[key]

def _ov(path):
    if path in _ov_cache: return _ov_cache[path]
    import openvino as ov
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "THROUGHPUT",
           "INFERENCE_NUM_THREADS": str(N_CORES),
           "INFERENCE_PRECISION_HINT": "bf16"}
    print(f"  [ov-compile] {Path(path).name}", flush=True)
    _ov_cache[path] = core.compile_model(str(Path(path)/"openvino_model.xml"), "CPU", cfg)
    return _ov_cache[path]


def encode_texts(model_key, texts, batch_size=128):
    variant = args.variant
    path = REGISTRY_PATHS[model_key]
    tok = _tok(path); dtype_str = variant.split("-")[1]
    out = []
    if variant.startswith("pytorch"):
        import torch
        mdl = _pt(path, dtype_str)
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16","bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        o = mdl(**enc)
                else:
                    o = mdl(**enc)
            lhs = o.last_hidden_state.float().numpy()
            mask = enc["attention_mask"].numpy()[..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8, None))
    else:
        compiled = _ov(path)
        valid = {i.get_any_name() for i in compiled.inputs}
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="np")
            o = compiled({k: v for k, v in enc.items() if k in valid})
            lhs = list(o.values())[0]
            mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8, None))
    return np.vstack(out)


REGISTRY_PATHS = {label: path for label, (_, path) in REGISTRY.items()}


def ensemble_avg(texts):
    if len(REGISTRY) == 0:
        raise RuntimeError("No models available")
    vecs = [encode_texts(label, texts) for label in REGISTRY]
    min_d = min(v.shape[1] for v in vecs)
    avg = np.mean(np.stack([v[:,:min_d] for v in vecs]), axis=0)
    return avg / np.linalg.norm(avg, axis=1, keepdims=True).clip(min=1e-9)


def cosine(a, b): return float(np.dot(a, b))


def bench_within_domain():
    print("\n[B1] Within-domain similarity ...", flush=True)
    results = {}
    for label in list(REGISTRY) + ["Ensemble"]:
        scores = {}
        for domain, texts in DOMAIN_CORPUS.items():
            vecs = ensemble_avg(texts) if label == "Ensemble" else encode_texts(label, texts)
            sims = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
            scores[domain] = round(float(np.mean(sims)), 4)
        results[label] = scores
        print(f"  {label:30}  " + "  ".join(f"{d[:6]}={scores[d]:.3f}" for d in DOMAIN_CORPUS), flush=True)
    return results


def bench_cross_domain():
    print("\n[B2] Cross-domain discrimination ...", flush=True)
    results = {}
    for label in list(REGISTRY) + ["Ensemble"]:
        pair_results = {}; correct = total = 0
        for pair_type, pairs in CROSS_DOMAIN_PAIRS.items():
            scores = []
            for t1, t2 in pairs:
                if label == "Ensemble":
                    v1, v2 = ensemble_avg([t1])[0], ensemble_avg([t2])[0]
                else:
                    v1, v2 = encode_texts(label, [t1])[0], encode_texts(label, [t2])[0]
                s = cosine(v1, v2); scores.append(s)
                if s < 0.6: correct += 1
                total += 1
            pair_results[pair_type] = round(float(np.mean(scores)), 4)
        acc = round(correct/total, 4) if total else 0.0
        results[label] = {"cross_domain_accuracy": acc, "pair_scores": pair_results,
                          "avg_cross_domain_sim": round(float(np.mean(list(pair_results.values()))), 4)}
        print(f"  {label:30}  acc={acc:.2%}  avg_sim={results[label]['avg_cross_domain_sim']:.4f}", flush=True)
    return results


def bench_sts_spearman():
    print("\n[B3] STS Spearman ...", flush=True)
    t1s = [p[0] for p in BIOSSES_STYLE_PAIRS]
    t2s = [p[1] for p in BIOSSES_STYLE_PAIRS]
    gold = [p[2] for p in BIOSSES_STYLE_PAIRS]
    results = {}
    for label in list(REGISTRY) + ["Ensemble"]:
        preds = []
        for a, b in zip(t1s, t2s):
            if label == "Ensemble":
                va, vb = ensemble_avg([a])[0], ensemble_avg([b])[0]
            else:
                va, vb = encode_texts(label, [a])[0], encode_texts(label, [b])[0]
            preds.append(cosine(va, vb))
        rho, pval = spearmanr(gold, preds)
        results[label] = {"spearman_rho": round(float(rho),4),
                          "p_value": round(float(pval),6),
                          "predictions": [round(p,4) for p in preds], "gold_scores": gold}
        print(f"  {label:30}  ρ={rho:.4f}  p={pval:.4f}", flush=True)
    return results


def bench_throughput():
    print("\n[B4] Throughput ...", flush=True)
    texts = [t for ts in DOMAIN_CORPUS.values() for t in ts]
    texts = (texts * 10)[:500]
    batch_sizes = [32, 64, 128, 256]
    results = {}
    for label in REGISTRY:
        mr = {}
        for bs in batch_sizes:
            t0 = time.perf_counter()
            _ = encode_texts(label, texts, batch_size=bs)
            elapsed = time.perf_counter() - t0
            sps = round(len(texts)/elapsed, 1)
            mr[f"batch_{bs}"] = {"sentences_per_sec": sps, "total_time_sec": round(elapsed,3)}
            print(f"  {label:30}  bs={bs:3d}  {sps:7.1f} sps", flush=True)
        results[label] = mr
    return results


def bench_hard_negatives():
    print("\n[B5] Hard negatives ...", flush=True)
    results = {}
    for label in list(REGISTRY) + ["Ensemble"]:
        correct = 0; margins = []; details = []
        for anchor, pos, neg in HARD_NEGATIVES:
            if label == "Ensemble":
                va, vp, vn = ensemble_avg([anchor])[0], ensemble_avg([pos])[0], ensemble_avg([neg])[0]
            else:
                va, vp, vn = encode_texts(label, [anchor])[0], encode_texts(label, [pos])[0], encode_texts(label, [neg])[0]
            sp, sn = cosine(va, vp), cosine(va, vn)
            passed = sp > sn
            if passed: correct += 1
            margins.append(sp - sn)
            details.append({"anchor": anchor[:60], "sim_pos": round(sp,4),
                            "sim_neg": round(sn,4), "margin": round(sp-sn,4), "correct": passed})
        acc = round(correct/len(HARD_NEGATIVES), 4)
        results[label] = {"accuracy": acc, "avg_margin": round(float(np.mean(margins)),4), "details": details}
        print(f"  {label:30}  acc={acc:.0%}  margin={np.mean(margins):+.4f}", flush=True)
    return results


def bench_domain_geometry():
    print("\n[B6] Domain geometry ...", flush=True)
    results = {}; domains = list(DOMAIN_CORPUS.keys())
    for label in list(REGISTRY) + ["Ensemble"]:
        centroids = {}; intra = {}
        for domain, texts in DOMAIN_CORPUS.items():
            vecs = ensemble_avg(texts) if label == "Ensemble" else encode_texts(label, texts)
            centroids[domain] = vecs.mean(axis=0)
            d = [1 - cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
            intra[domain] = round(float(np.mean(d)), 4)
        inter = {}
        for d1, d2 in combinations(domains, 2):
            c1 = centroids[d1] / np.linalg.norm(centroids[d1])
            c2 = centroids[d2] / np.linalg.norm(centroids[d2])
            inter[f"{d1}×{d2}"] = round(float(1 - cosine(c1, c2)), 4)
        ai, ae = float(np.mean(list(intra.values()))), float(np.mean(list(inter.values())))
        ratio = round(ae / max(ai, 1e-9), 4)
        results[label] = {"avg_intra_domain_dist": round(ai,4), "avg_inter_domain_dist": round(ae,4),
                          "inter_intra_ratio": ratio, "intra_by_domain": intra, "inter_by_pair": inter}
        print(f"  {label:30}  intra={ai:.4f} inter={ae:.4f} ratio={ratio:.4f}", flush=True)
    return results


def main():
    print("=" * 70)
    print(f"  Extended Benchmark — variant={args.variant}  cores={N_CORES}")
    print("=" * 70)
    if not REGISTRY:
        print("ERROR: no models available for this variant"); sys.exit(1)
    t_start = time.perf_counter()

    b1 = bench_within_domain()
    b2 = bench_cross_domain()
    b3 = bench_sts_spearman()
    b4 = bench_throughput()
    b5 = bench_hard_negatives()
    b6 = bench_domain_geometry()

    total_time = round(time.perf_counter() - t_start, 2)
    print(f"\n  {'Model':<32} {'Within-bio':>10} {'Cross-acc':>10} {'Spearman':>10} {'HardNeg':>10} {'GeoRatio':>10}")
    print(f"  {'-'*82}")
    for label in list(REGISTRY) + ["Ensemble"]:
        wb = np.mean([b1[label][d] for d in ["genetics","biomarkers","physiology"]]) if label in b1 else 0
        ca = b2[label]["cross_domain_accuracy"] if label in b2 else 0
        sp = b3[label]["spearman_rho"] if label in b3 else 0
        hn = b5[label]["accuracy"] if label in b5 else 0
        geo = b6[label]["inter_intra_ratio"] if label in b6 else 0
        print(f"  {label:<32} {wb:>10.4f} {ca:>10.2%} {sp:>10.4f} {hn:>10.2%} {geo:>10.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "run_at": datetime.now().isoformat(),
        "variant": args.variant,
        "platform": {"cpu_cores": N_CORES, "total_time_sec": total_time},
        "models": {label: REGISTRY_PATHS[label] for label in REGISTRY},
        "b1_within_domain_similarity": b1,
        "b2_cross_domain_discrimination": b2,
        "b3_sts_spearman": b3,
        "b4_throughput": b4,
        "b5_hard_negative_detection": b5,
        "b6_domain_geometry": b6,
    }
    out = OUT_DIR / f"extended_benchmark_{args.variant.replace('-','_')}_{ts}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

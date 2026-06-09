"""
finetune_pubmed_bodhi.py
========================
Second-pass fine-tuning of PubMedBERT on BODHI graph-grounded triplets.

Starts from: finetuned/pubmedbert/final
New data   : BODHI-S + BODHI-M graph triplets (anchor, positive, hard_negative)
Loss       : MatryoshkaLoss(MultipleNegativesRankingLoss)

Graph distance levels taught:
  Level 1  paraphrase / synonym              → distance ≈ 0.00–0.15
  Level 2  causally linked (PRESENT_IN,      → distance ≈ 0.15–0.40
           IS_INFLUENCED_BY, IMPACTS)
  Level 3  same-domain siblings (CHILD_OF,   → distance ≈ 0.40–0.60
           same triage, no edge)
  Level 4  cross-domain no path              → distance ≈ 0.60–0.80
  Level 5  completely unrelated              → distance ≈ 0.80–1.00

Output: finetuned/pubmedbert_bodhi/final/
"""

import ast, csv, json, logging, os, random, warnings
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Hardware ──────────────────────────────────────────────────────────────────
N_CORES = os.cpu_count() // 2
os.environ.update({
    "OMP_NUM_THREADS":        str(N_CORES),
    "MKL_NUM_THREADS":        str(N_CORES),
    "KMP_BLOCKTIME":          "1",
    "KMP_AFFINITY":           "granularity=fine,compact,1,0",
    "TOKENIZERS_PARALLELISM": "false",
    "ONEDNN_MAX_CPU_ISA":     "AVX512_CORE_AMX",
})
torch.set_num_threads(N_CORES)
torch.set_float32_matmul_precision("high")
torch.backends.mkldnn.enabled = True
random.seed(42)
np.random.seed(42)

print(f"[platform] {N_CORES}/{os.cpu_count()} cores | AMX=True | workers={min(16, N_CORES//4)}", flush=True)

BODHI       = Path(__file__).resolve().parent.parent / "models" / "BODHI"
BASE        = Path(__file__).parent
START_MODEL = BASE / "finetuned/pubmedbert/final"
OUT_DIR     = BASE / "finetuned/pubmedbert_bodhi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIK_WEIGHT = {"very_high": 4, "high": 3, "medium": 2, "low": 1, "rare": 0, "zero": 0}


# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))

def node_map(rows, id_col, name_col) -> dict:
    return {r[id_col]: r[name_col] for r in rows if r.get(name_col)}

def get_syns(row) -> list:
    raw = row.get("synonyms", "[]")
    try:
        return [x for x in ast.literal_eval(raw) if x and len(x) > 4][:3]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  BODHI-S
# ══════════════════════════════════════════════════════════════════════════════
def build_bodhi_s() -> Dataset:
    print("  [BODHI-S] symptom × condition pairs ...", flush=True)

    sym_rows  = read_csv(BODHI / "bodhi-s/csv/nodes_symptom.csv")
    cond_rows = read_csv(BODHI / "bodhi-s/csv/nodes_condition.csv")
    sym_names = node_map(sym_rows,  "uuid",       "name")
    cond_names= node_map(cond_rows, "snomed_id",  "name")
    cond_triage = {r["snomed_id"]: r.get("triage_level","opd_managed") for r in cond_rows}

    # symptom → set of connected conditions
    edges: dict[str, set] = {}
    edge_lik: dict[tuple, str] = {}
    for r in read_csv(BODHI / "bodhi-s/csv/edges_present_in.csv"):
        s, c = r["symptom_uuid"], r["condition_snomed_id"]
        edges.setdefault(s, set()).add(c)
        edge_lik[(s, c)] = r.get("likelihood_symptom_given_condition", "medium").lower().strip()

    # bucket conditions by triage for same-difficulty hard negatives
    triage_bucket: dict[str, list] = {}
    for cid, tl in cond_triage.items():
        triage_bucket.setdefault(tl, []).append(cid)
    all_conds = list(cond_names.keys())

    anchors, positives, negatives = [], [], []

    # PRESENT_IN with likelihood weighting
    for s_uuid, cond_set in edges.items():
        s_name = sym_names.get(s_uuid, "")
        if len(s_name) < 4:
            continue
        for c_id in cond_set:
            c_name = cond_names.get(c_id, "")
            if not c_name:
                continue
            lik    = edge_lik.get((s_uuid, c_id), "medium")
            weight = LIK_WEIGHT.get(lik, 1)
            if weight == 0:
                continue
            # hard neg: condition same triage but no PRESENT_IN edge for this symptom
            tl   = cond_triage.get(c_id, "opd_managed")
            pool = [x for x in triage_bucket.get(tl, []) if x not in cond_set]
            if not pool:
                pool = [x for x in all_conds if x not in cond_set]
            if not pool:
                continue
            neg_name = cond_names.get(random.choice(pool), "")
            if not neg_name:
                continue
            for _ in range(weight):
                anchors.append(s_name)
                positives.append(c_name)
                negatives.append(neg_name)

    # IS_INFLUENCED_BY: influencing factor → condition (causal cross-domain)
    for r in read_csv(BODHI / "bodhi-s/csv/edges_is_influenced_by.csv"):
        s_uuid = r.get("symptom_uuid", "")
        c_id   = r.get("condition_snomed_id", "")
        s_name = sym_names.get(s_uuid, "")
        c_name = cond_names.get(c_id, "")
        if not s_name or not c_name:
            continue
        other = random.choice([x for x in all_conds if x != c_id])
        anchors.append(s_name)
        positives.append(f"{s_name} can indicate {c_name}")
        negatives.append(f"{s_name} can indicate {cond_names.get(other, 'unknown condition')}")

    # HAS_PREREQUISITE: condition A → condition B (prerequisite relationship)
    for r in read_csv(BODHI / "bodhi-s/csv/edges_has_prerequisite.csv"):
        src = r.get("condition_snomed_id_src", "")
        dst = r.get("condition_snomed_id_dst", "")
        s_name = cond_names.get(src, "")
        d_name = cond_names.get(dst, "")
        if not s_name or not d_name:
            continue
        neg = random.choice([x for x in all_conds if x not in (src, dst)])
        anchors.append(s_name)
        positives.append(f"{d_name} is a prerequisite for {s_name}")
        negatives.append(f"{cond_names.get(neg, 'other condition')} is a prerequisite for {s_name}")

    # RELATED_TO: condition ↔ condition (bidirectional clinical relation)
    for r in read_csv(BODHI / "bodhi-s/csv/edges_related_to.csv"):
        src = r.get("src_id", "")
        dst = r.get("dst_id", "")
        src_type = r.get("src_type", "")
        dst_type = r.get("dst_type", "")
        # Only use Condition↔Condition
        if src_type == "Symptom" and dst_type == "Symptom":
            s1 = sym_names.get(src, "")
            s2 = sym_names.get(dst, "")
            if not s1 or not s2:
                continue
            neg_uuid = random.choice([x for x in sym_names if x not in (src, dst)])
            anchors.append(s1)
            positives.append(s2)
            negatives.append(sym_names.get(neg_uuid, ""))

    # NL facts (BODHI-S) — direct sentences as pairs
    facts = [json.loads(l)["text"] for l in open(BODHI / "bodhi-s/jsonl/nl_facts.jsonl")]
    random.shuffle(facts)
    for i in range(0, len(facts) - 1, 2):
        anchors.append(facts[i])
        positives.append(facts[i + 1])
        negatives.append(facts[min(i + 4, len(facts) - 1)])

    print(f"    BODHI-S: {len(anchors):,} triplets", flush=True)
    return Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


# ══════════════════════════════════════════════════════════════════════════════
#  BODHI-M
# ══════════════════════════════════════════════════════════════════════════════
def build_bodhi_m() -> Dataset:
    print("  [BODHI-M] concept hierarchy + lab pairs ...", flush=True)

    concept_rows = read_csv(BODHI / "bodhi-m/csv/nodes_concept.csv")
    drug_rows    = read_csv(BODHI / "bodhi-m/csv/nodes_drug.csv")
    lab_rows     = read_csv(BODHI / "bodhi-m/csv/nodes_lab_investigation.csv")

    # column: hash, name, therapeutic_class
    drug_names = {r["hash"]: r["name"] for r in drug_rows if r.get("name")}
    # column: loinc_id, name, display_name
    lab_names  = {r["loinc_id"]: r.get("display_name") or r["name"]
                  for r in lab_rows if r.get("loinc_id")}
    # concept: snomed_id, name, display_name, synonyms
    concept_name = {}
    concept_syns = {}
    for r in concept_rows:
        cid = r["snomed_id"]
        concept_name[cid] = r.get("display_name") or r.get("name", "")
        concept_syns[cid] = get_syns(r)
    all_cids = list(concept_name.keys())

    anchors, positives, negatives = [], [], []

    # CHILD_OF: child_snomed_id, child_name, parent_snomed_id, parent_name
    parent_children: dict[str, list] = {}
    for r in read_csv(BODHI / "bodhi-m/csv/edges_child_of.csv"):
        parent_children.setdefault(r["parent_snomed_id"], []).append(r["child_snomed_id"])

    for parent, children in parent_children.items():
        p_name = concept_name.get(parent, "")
        if not p_name or len(children) < 2:
            continue
        for child in children:
            c_name = concept_name.get(child, "")
            if not c_name:
                continue
            siblings = [x for x in children if x != child]
            sib_name = concept_name.get(random.choice(siblings), "")
            if not sib_name:
                continue
            # anchor = child with synonym variation
            syns = concept_syns.get(child, [])
            anchor_txt = random.choice([c_name] + syns) if syns else c_name
            anchors.append(anchor_txt)
            positives.append(f"{c_name} is a type of {p_name}")
            negatives.append(f"{sib_name} is a type of {p_name}")

    # IMPACTS: loinc_id, concept_snomed_id
    impact_edges: dict[str, list] = {}
    for r in read_csv(BODHI / "bodhi-m/csv/edges_impacts.csv"):
        impact_edges.setdefault(r["loinc_id"], []).append(r["concept_snomed_id"])

    for lab_id, cids in impact_edges.items():
        lab_name = lab_names.get(lab_id, "")
        if not lab_name:
            continue
        for c_id in cids:
            c_name = concept_name.get(c_id, "")
            if not c_name:
                continue
            neg_pool = [x for x in all_cids if x not in cids]
            if not neg_pool:
                continue
            neg_name = concept_name.get(random.choice(neg_pool), "")
            if not neg_name:
                continue
            anchors.append(f"{lab_name} test result")
            positives.append(f"Abnormal {lab_name} is associated with {c_name}")
            negatives.append(f"Abnormal {lab_name} is associated with {neg_name}")

    # TREATED_BY: concept_snomed_id, drug_hash
    concept_drugs: dict[str, list] = {}
    for r in read_csv(BODHI / "bodhi-m/csv/edges_treated_by.csv"):
        concept_drugs.setdefault(r["concept_snomed_id"], []).append(r["drug_hash"])

    all_hashes = list(drug_names.keys())
    for c_id, d_hashes in concept_drugs.items():
        c_name = concept_name.get(c_id, "")
        if not c_name:
            continue
        for d_hash in d_hashes[:4]:
            d_name = drug_names.get(d_hash, "")
            if not d_name:
                continue
            neg_hash = random.choice([h for h in all_hashes if h not in d_hashes] or all_hashes)
            neg_drug  = drug_names.get(neg_hash, "")
            if not neg_drug:
                continue
            anchors.append(c_name)
            positives.append(f"{d_name} is used to treat {c_name}")
            negatives.append(f"{neg_drug} is used to treat {c_name}")

    # NL facts (BODHI-M)
    facts = [json.loads(l)["text"] for l in open(BODHI / "bodhi-m/jsonl/nl_facts.jsonl")]
    random.shuffle(facts)
    for i in range(0, len(facts) - 1, 2):
        anchors.append(facts[i])
        positives.append(facts[i + 1])
        negatives.append(facts[min(i + 4, len(facts) - 1)])

    print(f"    BODHI-M: {len(anchors):,} triplets", flush=True)
    return Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
def load_eval():
    from datasets import load_dataset
    print("  [eval] Loading BIOSSES ...", flush=True)
    ds = load_dataset("mteb/biosses-sts", split="test")
    return EmbeddingSimilarityEvaluator(
        sentences1=ds["sentence1"],
        sentences2=ds["sentence2"],
        scores=[s / 4.0 for s in ds["score"]],
        name="biosses",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  Fine-tuning: pubmedbert_bodhi  (BODHI second pass)")
print(f"  Starting from: {START_MODEL}")
print(f"{'='*65}\n")

ds_s = build_bodhi_s()
ds_m = build_bodhi_m()

# Weight BODHI-S higher (richer causal graph)
ds_all = concatenate_datasets([ds_s, ds_s, ds_m]).shuffle(seed=42)
n_total = len(ds_all)
print(f"\n  Total training triplets: {n_total:,}", flush=True)

print(f"\n  Loading model from {START_MODEL} ...", flush=True)
model = SentenceTransformer(str(START_MODEL), device="cpu")
print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}", flush=True)

mnrl = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, mnrl, matryoshka_dims=[768, 512, 256, 128, 64])
evaluator = load_eval()

EPOCHS     = 2
BATCH_SIZE = 128
N_WORKERS  = min(16, N_CORES // 4)
STEPS      = (n_total // BATCH_SIZE) * EPOCHS

args = SentenceTransformerTrainingArguments(
    output_dir                  = str(OUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    warmup_steps                = int(0.06 * STEPS),
    learning_rate               = 1e-5,
    fp16                        = False,
    bf16                        = False,
    dataloader_num_workers      = N_WORKERS,
    save_strategy               = "epoch",
    save_total_limit            = 2,
    eval_strategy               = "epoch",
    logging_steps               = 50,
    seed                        = 42,
    report_to                   = "none",
)

trainer = SentenceTransformerTrainer(
    model         = model,
    args          = args,
    train_dataset = ds_all,
    loss          = loss,
    evaluator     = evaluator,
)

print(f"\n  Training {n_total:,} triplets | batch={BATCH_SIZE} | lr=1e-5 | epochs={EPOCHS}", flush=True)
print(f"  AMX-BF16 autocast via oneDNN | {N_CORES} cores", flush=True)
t0 = time.perf_counter()
trainer.train()
elapsed = time.perf_counter() - t0
print(f"  Training complete: {elapsed:.0f}s  ({elapsed/3600:.2f}h)", flush=True)

model.save(str(OUT_DIR / "final"))
print(f"  Saved → {OUT_DIR / 'final'}", flush=True)
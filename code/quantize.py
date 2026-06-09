"""
Quantize OV BF16 models to INT8 (PTQ) and INT4 (weight compression).
Uses NNCF 3.x API directly with OpenVINO models.

Usage:
  python quantize.py --model pubmed --precision int8
  python quantize.py --model all --precision all
"""
import argparse, os, sys, shutil
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"

ap = argparse.ArgumentParser()
ap.add_argument("--model",         default="all", choices=["pubmed","bodhi","biobert","all"])
ap.add_argument("--precision",     default="all", choices=["int8","int4","all"])
ap.add_argument("--calib-samples", type=int, default=128)
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from config import PT_PATHS, OV_PATHS, SENTENCES, MODELS

MODELS_TO_RUN = list(MODELS.keys()) if args.model == "all" else [args.model]
PRECS_TO_RUN  = ["int8","int4"] if args.precision == "all" else [args.precision]


def make_ov_calib_dataset(tokenizer, n_samples, max_length=128):
    """Build NNCF Dataset from tokenized sentences for OV model."""
    import nncf
    import numpy as np

    corpus = (SENTENCES * ((n_samples // len(SENTENCES)) + 2))[:n_samples]
    def transform(text):
        enc = tokenizer(text, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="np")
        return dict(enc)

    data = [transform(s) for s in corpus]
    return nncf.Dataset(data)


for model_key in MODELS_TO_RUN:
    bf16_path = OV_PATHS[f"{model_key}_bf16"]
    label     = MODELS[model_key]

    for prec in PRECS_TO_RUN:
        out_path = Path(OV_PATHS[f"{model_key}_{prec}"])
        xml_out  = out_path / "openvino_model.xml"

        if xml_out.exists():
            print(f"[skip] {label} {prec.upper()} already at {out_path}")
            continue

        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Quantizing : {label} → {prec.upper()}")
        print(f"  Source     : {bf16_path}")
        print(f"  Output     : {out_path}")
        print(f"{'='*60}")

        import openvino as ov
        import nncf
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(bf16_path)
        core      = ov.Core()
        xml_src   = str(Path(bf16_path) / "openvino_model.xml")
        model     = core.read_model(xml_src)

        if prec == "int8":
            print(f"  [int8] building calibration dataset ({args.calib_samples} samples)...")
            calib_ds = make_ov_calib_dataset(tokenizer, args.calib_samples)

            print(f"  [int8] running PTQ (this takes a few minutes)...")
            quantized = nncf.quantize(
                model,
                calib_ds,
                model_type=nncf.ModelType.TRANSFORMER,
                preset=nncf.QuantizationPreset.MIXED,
                subset_size=args.calib_samples,
                fast_bias_correction=True,
            )
            ov.save_model(quantized, str(xml_out))
            print(f"  [int8] model saved")

        elif prec == "int4":
            print(f"  [int4] weight-only INT4 compression (no calibration)...")
            compressed = nncf.compress_weights(
                model,
                mode=nncf.CompressWeightsMode.INT4_SYM,
                group_size=64,
            )
            ov.save_model(compressed, str(xml_out))
            print(f"  [int4] model saved")

        # Copy tokenizer files from BF16 dir
        for fname in ["tokenizer.json","tokenizer_config.json","vocab.txt",
                      "special_tokens_map.json","config.json"]:
            src = Path(bf16_path) / fname
            if src.exists():
                shutil.copy2(src, out_path / fname)

        # Report size
        bin_file = out_path / "openvino_model.bin"
        if bin_file.exists():
            mb = bin_file.stat().st_size / 1024 / 1024
            print(f"  Model size: {mb:.1f} MB (BF16 was ~208 MB)")
        print(f"  [done] {label} {prec.upper()}")

print("\n[quantize] all done")

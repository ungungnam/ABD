"""Merge the AHA LoRA adapter into vicuna-7b to produce a standalone model.

The AHA checkpoint is a LoRA adapter (+ trained mm-projector) on top of
vicuna-7b. To *further* fine-tune AHA with a fresh LoRA, the trainer needs a
plain, fully-merged base model -- this script produces it.

``robopoint.model.builder.load_pretrained_model`` already applies and merges
the LoRA in memory (the same path the AHA inference server uses); we simply
persist the merged result.

Run with the ``aha`` conda env:

    conda run -n aha python scripts/merge_aha_lora.py \\
        --lora-path /result/AHA/training_output_lora_50ep/checkpoint-1700 \\
        --model-base /ckpt/AHA/vicuna-7b-v1.5 \\
        --out-dir /ckpt/AHA/aha-merged-7b
"""

import argparse
import os
import sys

# Make the bundled RoboPoint package importable (mirrors run_aha_server.py).
ROBOPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "RoboPoint")
sys.path.insert(0, os.path.abspath(ROBOPOINT_DIR))

from robopoint.mm_utils import get_model_name_from_path
from robopoint.model.builder import load_pretrained_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-path",
                    default="/result/AHA/training_output_lora_50ep/checkpoint-1700")
    ap.add_argument("--model-base", default="/ckpt/AHA/vicuna-7b-v1.5")
    ap.add_argument("--out-dir", default="/ckpt/AHA/aha-merged-7b")
    args = ap.parse_args()

    model_name = get_model_name_from_path(args.lora_path)
    print(f"[merge] model_name = {model_name}")
    print(f"[merge] loading LoRA '{args.lora_path}' on base '{args.model_base}'")

    # load_pretrained_model applies the LoRA and calls merge_and_unload().
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.lora_path, args.model_base, model_name
    )
    print(f"[merge] merged model loaded (context_len={context_len})")

    # vicuna's generation_config has do_sample=False with temperature/top_p set,
    # which newer transformers rejects on save. Make it self-consistent;
    # inference passes its own decoding args anyway.
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.do_sample = True

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[merge] saving merged model -> {args.out_dir}")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # Persist the image processor too, so downstream loaders are self-contained.
    try:
        image_processor.save_pretrained(args.out_dir)
    except Exception as exc:  # noqa: BLE001 -- non-fatal; vision tower reloads by name
        print(f"[merge] (image_processor not saved: {exc})")

    print("[merge] done.")


if __name__ == "__main__":
    main()

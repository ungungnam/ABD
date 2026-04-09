"""FastAPI inference server for Qwen3-VL.

Holds Qwen3-VL-8B-Instruct in GPU memory and exposes a single
``POST /inference`` endpoint that accepts ABD's unified
``{images, prompt, task_kind}`` payload (see ``src/vlm_client/base.py``).
The Qwen chat-template wrapping is applied here because it depends on
the loaded ``AutoProcessor``.

Run from ABD root:

    python scripts/run_qwen_server.py \\
        --model-path /ckpt/Qwen3-VL-8B-Instruct \\
        --port 9879
"""

import argparse
import base64
import io
import logging
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor


log = logging.getLogger("qwen_server")
logging.basicConfig(level=logging.INFO)


class InferenceRequest(BaseModel):
    images: Dict[str, str] = {}  # camera name -> base64 JPEG
    prompt: str
    task_kind: Optional[str] = "vqa"


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_app(model_path: str, max_new_tokens: int) -> FastAPI:
    log.info(f"[Qwen] Loading model from {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    log.info("[Qwen] Model loaded")

    app = FastAPI()

    @app.post("/inference")
    def inference(request: InferenceRequest):
        # Build a single user message containing every image followed by
        # the text prompt — this matches the format PaPA's qwen_predict
        # uses and what Qwen3-VL-Instruct expects out of the box.
        content = []
        for cam_name, b64 in request.images.items():
            content.append({"type": "image", "image": decode_image(b64)})
        content.append({"type": "text", "text": request.prompt})

        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {"response": response}

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/ckpt/Qwen3-VL-8B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9879)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args.model_path, args.max_new_tokens)
    uvicorn.run(app, host=args.host, port=args.port)

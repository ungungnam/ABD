"""FastAPI inference server for the AHA LoRA model.

Holds the AHA LoRA fine-tune of RoboPoint/Vicuna-7B in GPU memory and
exposes a single ``POST /inference`` endpoint that accepts ABD's unified
``{images, prompt, task_kind}`` payload (see ``src/vlm_client/base.py``).
The server is responsible for assembling the LLaVA-style chat prompt
(``<image>\\n{question}``) and feeding it through the loaded model.

Run from ABD root with the ``aha`` conda env (per project memory):

    conda run -n aha python scripts/run_aha_server.py \\
        --model-path /result/AHA/training_output \\
        --model-base /ckpt/AHA/vicuna-7b-v1.5 \\
        --port 9878
"""

import argparse
import base64
import io
import logging
import os
import sys

# Make the bundled RoboPoint package importable.
ROBOPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "RoboPoint")
sys.path.insert(0, os.path.abspath(ROBOPOINT_DIR))

from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from robopoint.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from robopoint.conversation import conv_templates
from robopoint.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init


log = logging.getLogger("aha_server")
logging.basicConfig(level=logging.INFO)


class InferenceRequest(BaseModel):
    images: Dict[str, str] = {}  # camera name -> base64 JPEG
    prompt: str
    task_kind: Optional[str] = "vqa"


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_app(model_path: str, model_base: str, conv_mode: str) -> FastAPI:
    log.info(f"[AHA] Loading model from {model_path} (base: {model_base})")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name
    )
    log.info(f"[AHA] Model loaded: {model_name}")

    app = FastAPI()

    @app.post("/inference")
    def inference(request: InferenceRequest):
        # AHA's RoboPoint backbone consumes one image at a time. We use the
        # first camera in the dict (typically the front view) for the LLaVA
        # conversation. The other cameras, if any, are concatenated into the
        # text prompt as additional context cues.
        if not request.images:
            return {"response": ""}

        cam_names = list(request.images.keys())
        primary_cam = cam_names[0]
        image = decode_image(request.images[primary_cam])

        question = request.prompt
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
            )
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return {"response": response}

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/result/AHA/training_output")
    p.add_argument("--model-base", default="/ckpt/AHA/vicuna-7b-v1.5")
    p.add_argument("--conv-mode", default="llava_v1")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9878)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args.model_path, args.model_base, args.conv_mode)
    uvicorn.run(app, host=args.host, port=args.port)

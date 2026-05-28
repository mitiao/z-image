"""Z-Image API Service."""

import os
import sys
import time
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
import torch_br
from torch_br.contrib import transfer_to_supa

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

components = None
_device = None


def _select_device() -> str:
    if torch.cuda.is_available():
        dev = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                dev = "mps"
                print("Chosen device: mps")
            else:
                dev = "cpu"
                print("Chosen device: cpu")
    return dev


@asynccontextmanager
async def lifespan(app: FastAPI):
    global components, _device

    ckpt_dir = os.environ.get("CKPT_DIR", "ckpts/Z-Image-Turbo")
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    compile_model = os.environ.get("COMPILE", "false").lower() == "true"
    dtype = torch.bfloat16

    _device = _select_device()
    model_path = ensure_model_weights(ckpt_dir, verify=False)
    components = load_from_local_dir(model_path, device=_device, dtype=dtype, compile=compile_model)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    print("Warm up ...")
    for _ in range(2):
        generate(
            prompt="warmup",
            **components,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=torch.Generator(_device).manual_seed(42),
        )
    print("Warm up done. API ready.")

    yield

    components = None
    print("Shutdown complete.")


app = FastAPI(title="Z-Image API", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    height: int = Field(1024, ge=512, le=2048)
    width: int = Field(1024, ge=512, le=2048)
    num_inference_steps: int = Field(8, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: int = Field(42)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": components is not None}


@app.post("/generate")
def generate_image(req: GenerateRequest):
    if components is None:
        raise HTTPException(503, "Model not loaded yet")

    gen = torch.Generator(_device).manual_seed(req.seed)
    images = generate(
        prompt=req.prompt,
        **components,
        height=req.height,
        width=req.width,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=gen,
    )

    buf = BytesIO()
    images[0].save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/generate")
def generate_image_get(
    prompt: str = Query(..., description="Text prompt"),
    seed: int = Query(42),
):
    if components is None:
        raise HTTPException(503, "Model not loaded yet")

    gen = torch.Generator(_device).manual_seed(seed)
    images = generate(
        prompt=prompt,
        **components,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=gen,
    )

    buf = BytesIO()
    images[0].save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

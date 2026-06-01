"""Z-Image API Service."""

import os
import sys
import warnings
from collections.abc import Generator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
import torch_br
from torch_br.contrib import transfer_to_supa

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

components: Optional[dict[str, Any]] = None
_device: Optional[str | torch.device] = None


def _select_device() -> str | torch.device:
    if torch.cuda.is_available():
        logger.info("Chosen device: cuda")
        return "cuda"

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
        logger.info("Chosen device: tpu")
        return dev
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            logger.info("Chosen device: mps")
            return "mps"
        logger.info("Chosen device: cpu")
        return "cpu"


def _generate_image(
    prompt: str,
    seed: int,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
) -> Response:
    if components is None:
        raise HTTPException(503, "Model not loaded yet")

    logger.info("Generating: prompt={}", prompt[:80].rstrip())
    gen = torch.Generator(_device).manual_seed(seed)
    try:
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            print_time=True,
        )
    except Exception as e:
        logger.error("Generation failed: {}", e)
        raise HTTPException(500, f"Generation failed: {e}")

    buf = BytesIO()
    images[0].save(buf, format="PNG")
    logger.info("Generated {}x{} image (seed={}) in {:.1f} KB", width, height, seed, buf.tell() / 1024)
    return Response(content=buf.getvalue(), media_type="image/png")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Generator[None]:
    global components, _device

    ckpt_dir = os.environ.get("CKPT_DIR", "/mnt/file/default-gpfsmodels/Z-Image-Turbo")
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    compile_model = os.environ.get("COMPILE", "false").lower() == "true"
    dtype = torch.bfloat16

    _device = _select_device()
    model_path = ensure_model_weights(ckpt_dir, verify=False)
    components = load_from_local_dir(model_path, device=_device, dtype=dtype, compile=compile_model)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    logger.info("Chosen attention backend: {}", attn_backend)

    logger.info("Warm up ...")
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
    logger.info("Warm up done. API ready.")

    yield

    components = None
    logger.info("Shutdown complete.")


app = FastAPI(title="Z-Image API", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    height: int = Field(1024, ge=512, le=2048)
    width: int = Field(1024, ge=512, le=2048)
    num_inference_steps: int = Field(8, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: int = Field(42)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": components is not None}


@app.post("/v1/images/generations")
def create_generation(req: GenerateRequest) -> Response:
    return _generate_image(
        prompt=req.prompt,
        seed=req.seed,
        height=req.height,
        width=req.width,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
    )

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

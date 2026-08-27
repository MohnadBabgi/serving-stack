"""serving-stack: the FastAPI service (week 2, CPU, tiny model).

This is the starter. GET /health is done for you and works as soon as the model
loads: treat it as the worked example. Your job is the two routes marked TODO.
Correctness before speed. The model runs on CPU this week; do not add a GPU.

Run it:
    uvicorn main:app --host 0.0.0.0 --port 8000

Model: Qwen/Qwen2.5-0.5B-Instruct (about 0.5B params; loads on CPU in seconds
once cached). The first ever load downloads weights; the prep-week verify-env
pass pre-seeded the Hugging Face cache, so a cached load is fast.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
API_KEY = os.environ.get("API_KEY", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

if not API_KEY:
    logging.warning("API_KEY is unset — /v1/* is running UNAUTHENTICATED")

app = FastAPI(title="serving-stack", version="wk2")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Reject /v1/* without a matching bearer key. /health stays open."""
    if API_KEY and request.url.path.startswith("/v1/"):
        if request.headers.get("authorization") != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


# Load once at import time. CPU only this week.
print(f"loading {MODEL_ID} on cpu ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.to("cpu")
model.eval()
print("model ready")


# ---------------------------------------------------------------------------
# GET /health  -- DONE. This is the worked example. Copy its shape.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness.

    Contract: returns 200 with {"status": "ok", "model": "<id>"} once the model
    is loaded. Kubernetes probes (week 4) and the agentic client's retry logic
    (weeks 4 to 6) call this. It must be cheap and must not run the model.
    """
    return HealthResponse(status="ok", model=MODEL_ID)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    """List the served model id(s)."""
    card = ModelCard(
        id=MODEL_ID,
        object="model",
        created=int(time.time()),
        owned_by="local",
    )
    return ModelList(object="list", data=[card])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run the model over the messages and return an OpenAI-compatible completion."""
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    input_ids = inputs["input_ids"]


    prompt_tokens = int(input_ids.shape[1])

    
    max_tokens = req.max_tokens if req.max_tokens is not None else 128
    max_tokens = min(max_tokens, MAX_TOKENS)
    do_sample = bool(req.temperature is not None and req.temperature > 0)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }
    if do_sample and req.temperature is not None:
        gen_kwargs["temperature"] = float(req.temperature)

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    new_tokens = out[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    finish_reason = "length" if completion_tokens >= max_tokens else "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

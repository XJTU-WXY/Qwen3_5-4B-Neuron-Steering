"""Web UI for neuron steering on Qwen3.5-4B.

    python serve.py --model .. --port 8420
"""

from __future__ import annotations

import argparse
import json
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import presets as preset_lib
from engine import Steerer

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Qwen3.5-4B Neuron Steering")
engine: Steerer | None = None


class RecipeItem(BaseModel):
    layer: int
    neuron: int
    dose: float


class ChatRequest(BaseModel):
    messages: list[dict]
    recipe: list[list[float]] = []
    max_new_tokens: int = 320
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    thinking: bool = False
    compare: bool = False


class InspectRequest(BaseModel):
    recipe: list[list[float]] = []


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/state")
def state():
    return {"model": engine.describe(), "presets": preset_lib.grouped()}


@app.post("/api/inspect")
def inspect(req: InspectRequest):
    try:
        return {"detail": engine.describe_recipe(req.recipe)}
    except ValueError as exc:
        return {"error": str(exc), "detail": []}


@app.post("/api/stop")
def stop():
    engine.stop()
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    def generator():
        try:
            if req.compare and req.recipe:
                passes = [("base", []), ("steer", req.recipe)]
            else:
                passes = [("single", req.recipe)]
            yield _sse({"type": "begin", "passes": [label for label, _ in passes]})
            for label, recipe in passes:
                for event in engine.stream(
                    req.messages,
                    recipe,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    thinking=req.thinking,
                    label=label,
                ):
                    yield _sse(event)
        except Exception as exc:  # surfaced in the chat pane
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("STEER_MODEL", ".."))
    parser.add_argument("--device", default=os.environ.get("STEER_DEVICE", "cuda:0"))
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()

    global engine
    engine = Steerer(args.model, device=args.device)
    engine.load()
    engine.calibrate(force=args.recalibrate)
    print(f"[steer] http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Local browser demo. Webcam pixels never leave the browser."""
from __future__ import annotations
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .simulation import Tabletop, HORIZON
from .data import save_episode, load_episodes
from .models import LearnedAgent

ROOT = Path.cwd()
UI = Path(__file__).parent / "ui"
app = FastAPI(title="GhostHands", docs_url="/api/docs")
sessions: dict = {}
pool = ThreadPoolExecutor(max_workers=1)
job: dict = {"status": "idle"}
torch.set_num_threads(2)


class NewSession(BaseModel):
    seed: int = Field(default=65001, ge=0, le=2**31-1)


class Control(BaseModel):
    action: list[float] = Field(min_length=5, max_length=5)
    landmarks: list[list[float]] | None = None


class Record(BaseModel):
    source: Literal["webcam", "pointer"] = "pointer"
    split: Literal["train", "val", "test"] = "train"


class Run(BaseModel):
    model: Literal["absolute", "relative", "recurrent", "human"] = "relative"
    seed: int = Field(default=20000, ge=20000, le=2**31-1)


class Perturb(BaseModel):
    kind: Literal["object", "target"]


def session(sid: str) -> dict:
    if sid not in sessions:
        raise HTTPException(404, "Session expired; reload the page")
    return sessions[sid]


def snapshot(s: dict) -> dict:
    return {"human": s["human"].state.tolist(), "robot": s["robot"].state.tolist(),
            "metrics": s["robot"].metrics(), "human_metrics": s["human"].metrics(),
            "running": s["running"], "recording": s["record"] is not None,
            "metadata": s.get("metadata"), "robot_seed": s["robot"].seed}


@app.get("/")
def index():
    return FileResponse(UI / "index.html")


@app.post("/api/session")
def create(req: NewSession):
    if len(sessions) >= 32:
        sessions.pop(next(iter(sessions)))
    sid = secrets.token_hex(16)
    sessions[sid] = {"human": Tabletop(req.seed), "robot": Tabletop(20000),
                     "record": None, "running": False, "agent": None, "seed": req.seed}
    return {"id": sid, **snapshot(sessions[sid])}


@app.post("/api/{sid}/control")
def control(sid: str, req: Control):
    s = session(sid)
    action = np.asarray(req.action, dtype=np.float32)
    if not np.isfinite(action).all() or np.abs(action).max() > 1.001:
        raise HTTPException(422, "Actions must be finite and bounded")
    rec = s["record"]
    if rec is not None:
        if len(rec["actions"]) >= 3000:
            raise HTTPException(400, "Recording limit reached; save this episode")
        rec["observations"].append(s["human"].state.tolist())
        rec["actions"].append(action.tolist())
        if req.landmarks is not None:
            landmarks = np.asarray(req.landmarks)
            if landmarks.shape != (21, 3) or not np.isfinite(landmarks).all():
                rec["observations"].pop(); rec["actions"].pop()
                raise HTTPException(422, "Expected 21 finite xyz landmarks")
        rec["landmarks"].append(req.landmarks)
    s["human"].step(action)
    return snapshot(s)


@app.post("/api/{sid}/record")
def record(sid: str, req: Record):
    s = session(sid)
    if s["record"] is not None:
        raise HTTPException(409, "Save current recording first")
    if s["human"].steps != 0:
        raise HTTPException(409, "Start a new layout before recording")
    s["record"] = {"schema": 1, "source": req.source, "split": req.split,
                    "seed": s["seed"], "dt": 0.05, "observations": [], "actions": [], "landmarks": []}
    return snapshot(s)


@app.post("/api/{sid}/save")
def save(sid: str):
    s = session(sid)
    ep = s["record"]
    if ep is None:
        raise HTTPException(409, "No recording is active")
    ep["final_state"], ep["metrics"] = s["human"].state.tolist(), s["human"].metrics()
    try:
        path = save_episode(ep, ROOT / "data/human")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    s["record"] = None
    return {"saved": path.name, "success": ep["metrics"]["success"]}


@app.post("/api/{sid}/run")
def run(sid: str, req: Run):
    if job["status"] == "training":
        raise HTTPException(409, "Wait for training to finish")
    s = session(sid)
    path = ROOT / ("runs/human/relative.pt" if req.model == "human" else f"runs/default/{req.model}.pt")
    if not path.exists() and req.model != "human":
        path = ROOT / f"assets/{req.model}.pt"
    if not path.exists():
        raise HTTPException(409, "Train this policy first; no checkpoint is available")
    s["agent"] = LearnedAgent(path)
    s["metadata"] = s["agent"].metadata
    s["robot"], s["running"] = Tabletop(req.seed, True), True
    return snapshot(s)


@app.post("/api/{sid}/tick")
def tick(sid: str):
    s = session(sid)
    if s["running"]:
        s["robot"].step(s["agent"].action(s["robot"].state))
        if s["robot"].success or s["robot"].steps >= HORIZON:
            s["running"] = False
    return snapshot(s)


@app.post("/api/{sid}/pause")
def pause(sid: str):
    s = session(sid)
    if s["agent"] and s["robot"].steps < HORIZON and not s["robot"].success:
        s["running"] = not s["running"]
    return snapshot(s)


@app.post("/api/{sid}/perturb")
def perturb(sid: str, req: Perturb):
    s = session(sid)
    env = s["robot"]
    rng = np.random.default_rng(env.seed + env.steps + (70000 if req.kind == "target" else 80000))
    if req.kind == "target":
        env.state[8:10] = rng.uniform(0.1, 0.9, 2)
    else:
        env.state[13] = 0
        env.state[4:6] = rng.uniform(0.1, 0.9, 2)
        env.state[6] = 0.04
    env.stable, env.success = 0, False
    return snapshot(s)


@app.get("/api/status")
def status():
    counts = {split: len(load_episodes(ROOT / "data/human", split, "human")) for split in ("train", "val", "test")}
    return {"job": job, "episodes": counts}


@app.post("/api/train")
def train_human():
    if job["status"] == "training":
        raise HTTPException(409, "Training already running")
    counts = status()["episodes"]
    if counts["train"] < 2 or counts["val"] < 1:
        raise HTTPException(409, "Record at least 2 training episodes and 1 validation episode in different layouts")
    if any(s["running"] for s in sessions.values()):
        raise HTTPException(409, "Pause robot execution before training")
    job.clear(); job.update(status="training")
    def work():
        try:
            from .training import train
            result = train(ROOT / "data/human", ROOT / "runs/human", "relative", 60, 42, "human")
            job.update(status="complete", result=result)
        except Exception as exc:
            job.update(status="failed", error=str(exc))
    pool.submit(work)
    return job


@app.get("/api/results")
def results():
    path = ROOT / "assets/results.json"
    return json.loads(path.read_text()) if path.exists() else {"results": []}


app.mount("/static", StaticFiles(directory=UI), name="static")

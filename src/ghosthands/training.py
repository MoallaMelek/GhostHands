"""Reproducible CPU training; checkpoint selection uses validation only."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_episodes, assert_disjoint
from .models import Policy
from .simulation import features


def windows(episodes: list[dict], relative: bool) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for ep in episodes:
        x = features(np.asarray(ep["observations"]), relative)
        padded = np.concatenate([np.repeat(x[:1], 7, axis=0), x])
        xs.extend(padded[t:t + 8] for t in range(len(x)))
        ys.extend(ep["actions"])
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


def loss_fn(pred: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, :4].tanh(), action[:, :4]) + 0.25 * F.binary_cross_entropy_with_logits(pred[:, 4], (action[:, 4] + 1) / 2)


def train(data: Path, output: Path, kind: str, epochs: int = 60, seed: int = 42,
          source: str = "synthetic_teacher") -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tr, va = load_episodes(data, "train", source), load_episodes(data, "val", source)
    rejected = 0
    if source == "human":
        rejected = sum(not e.get("metrics", {}).get("success", False) for e in tr + va)
        tr = [e for e in tr if e.get("metrics", {}).get("success", False)]
        va = [e for e in va if e.get("metrics", {}).get("success", False)]
    if len(tr) < 2 or not va:
        raise ValueError("Need at least two training episodes and one independent validation episode")
    assert_disjoint(tr, va)
    x, y = windows(tr, kind != "absolute")
    vx, vy = windows(va, kind != "absolute")
    model = Policy(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    best, log = float("inf"), []
    digest = hashlib.sha256(json.dumps(tr + va, sort_keys=True).encode()).hexdigest()
    metadata = {"kind": kind, "seed": seed, "source": source, "train_episodes": len(tr),
                "val_episodes": len(va), "dataset_sha256": digest, "schema": 1,
                "torch_version": str(torch.__version__), "context": 8, "epochs_requested": epochs,
                "excluded_unsuccessful_episodes": rejected}
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for ids in order.split(256):
            loss = loss_fn(model(x[ids]), y[ids])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.inference_mode():
            vl = sum(loss_fn(model(a), b).item() * len(a) for a, b in zip(vx.split(512), vy.split(512))) / len(vx)
        log.append({"epoch": epoch + 1, "train_loss": total / len(x), "val_loss": vl})
        if vl < best:
            best = vl
            torch.save({**metadata, "epoch": epoch + 1, "val_loss": vl, "weights": model.state_dict()}, output / f"{kind}.pt")
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{kind}: epoch {epoch + 1}/{epochs}, train={total/len(x):.4f}, validation={vl:.4f}", flush=True)
    (output / f"{kind}-training.json").write_text(json.dumps({**metadata, "history": log}, indent=2), encoding="utf-8")
    return {**metadata, "best_val_loss": best}

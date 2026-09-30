"""Episode storage and explicitly synthetic expert. Never used at policy inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .simulation import Tabletop, MOVE, TURN, FLOOR, HORIZON, wrap

SCHEMA = 1


def teacher(env: Tabletop) -> np.ndarray:
    """Privileged scripted data generator, isolated from learned inference."""
    s = env.state
    grip = -1.0
    if env.ever_grasped and not s[13] and np.linalg.norm(s[4:7] - s[8:11]) < 0.055:
        goal, yaw = s[:3].copy(), s[3]
    elif s[13]:
        goal, yaw, grip = s[8:11].copy(), s[11], 1.0
        if np.linalg.norm(s[4:6] - s[8:10]) > 0.025:
            goal[2] = 0.27
            if s[6] < 0.23:
                goal[:2] = s[:2]
        elif abs(wrap(s[3] - yaw)) > 0.10:
            goal[2] = max(s[2], 0.20)
        if np.linalg.norm(s[4:7] - s[8:11]) < 0.02 and abs(wrap(s[7] - yaw)) < 0.10:
            grip = -1.0
    else:
        goal, yaw = s[4:7].copy(), s[7]
        if np.linalg.norm(s[:2] - s[4:6]) > 0.025 or abs(wrap(s[3] - yaw)) > 0.12:
            goal[2] = max(0.22, s[2])
        if np.linalg.norm(s[:3] - s[4:7]) < 0.025 and abs(wrap(s[3] - yaw)) < 0.12:
            grip = 1.0
    return np.array([*np.clip((goal - s[:3]) / MOVE, -1, 1),
                     np.clip(wrap(yaw - s[3]) / TURN, -1, 1), grip], dtype=np.float32)


def generate_episode(seed: int, split: str) -> dict:
    env = Tabletop(seed)
    rng = np.random.default_rng(seed + 910000)
    observations, actions = [], []
    for _ in range(HORIZON):
        # Small actuation noise creates corrective examples without using test states.
        a = teacher(env)
        a[:4] = np.clip(a[:4] + rng.normal(0, 0.025, 4), -1, 1)
        observations.append(env.state.tolist())
        actions.append(a.tolist())
        env.step(a)
        if env.success:
            break
    return {"schema": SCHEMA, "source": "synthetic_teacher", "split": split,
            "seed": seed, "dt": 0.05, "observations": observations, "actions": actions,
            "final_state": env.state.tolist(), "metrics": env.metrics()}


def validate_episode(ep: dict) -> None:
    if ep.get("schema") != SCHEMA or ep.get("source") not in {"synthetic_teacher", "webcam", "pointer"}:
        raise ValueError("Unsupported episode schema or provenance")
    if ep.get("split") not in {"train", "val", "test"}:
        raise ValueError("Episode requires an explicit split")
    obs, acts = np.asarray(ep["observations"]), np.asarray(ep["actions"])
    if obs.ndim != 2 or obs.shape[1] != 14 or acts.shape != (len(obs), 5) or not 8 <= len(obs) <= 3000:
        raise ValueError("Expected 8–3000 paired 14D observations and 5D actions")
    if not np.isfinite(obs).all() or not np.isfinite(acts).all() or np.abs(acts).max() > 1.001:
        raise ValueError("Nonfinite data or actions outside [-1, 1]")


def save_episode(ep: dict, root: Path) -> Path:
    validate_episode(ep)
    payload = json.dumps(ep, sort_keys=True)
    identity = hashlib.sha256(payload.encode()).hexdigest()[:16]
    path = root / ep["split"] / f"{ep['source']}-{identity}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def load_episodes(root: Path, split: str, source: str = "all") -> list[dict]:
    episodes = []
    for path in sorted((root / split).glob("*.json")):
        ep = json.loads(path.read_text(encoding="utf-8"))
        validate_episode(ep)
        if ep["split"] != split:
            raise ValueError(f"Split mismatch: {path}")
        if source == "all" or (source == "human" and ep["source"] in {"webcam", "pointer"}) or source == ep["source"]:
            episodes.append(ep)
    return episodes


def assert_disjoint(train: list[dict], val: list[dict]) -> None:
    def key(ep: dict) -> str:
        return json.dumps(ep["observations"][0], sort_keys=True)
    if {key(e) for e in train} & {key(e) for e in val}:
        raise ValueError("Train/validation initial layouts overlap")

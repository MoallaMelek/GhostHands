"""Deterministic kinematic tabletop. Dynamics are not a task controller."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DT = 0.05
MOVE = 0.035
TURN = 0.18
FLOOR = 0.04
HORIZON = 180


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Tabletop:
    seed: int = 0
    ood: bool = False

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        low, high = (0.10, 0.90) if self.ood else (0.25, 0.75)
        obj = rng.uniform(low, high, 2)
        target = rng.uniform(low, high, 2)
        while np.linalg.norm(obj - target) < 0.22:
            target = rng.uniform(low, high, 2)
        self.state = np.array([0.5, 0.5, 0.32, 0, *obj, FLOOR,
                               rng.uniform(-np.pi, np.pi), *target, FLOOR,
                               rng.uniform(-np.pi, np.pi), 0, 0], dtype=np.float32)
        self.steps = 0
        self.stable = 0
        self.success = False
        self.path_length = 0.0
        self.initial = self.state.copy()
        self.ever_grasped = False

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (5,) or not np.isfinite(a).all():
            raise ValueError("Action must contain five finite numbers")
        a = np.clip(a, -1, 1)
        s = self.state
        old = s[:3].copy()
        s[:3] = np.clip(s[:3] + a[:3] * MOVE, [0.05, 0.05, FLOOR], [0.95, 0.95, 0.55])
        s[3] = wrap(s[3] + a[3] * TURN)
        self.path_length += float(np.linalg.norm(s[:3] - old))
        s[12] = float(a[4] > 0)
        if s[13] and not s[12]:
            s[13] = 0
        if not s[13] and s[12] and np.linalg.norm(s[:3] - s[4:7]) < 0.055 and abs(wrap(s[3] - s[7])) < 0.35:
            s[13] = 1
            self.ever_grasped = True
        if s[13]:
            s[4:7] = s[:3]
            s[7] = s[3]
        else:
            s[6] = max(FLOOR, s[6] - 0.045)
        placed = (not s[13] and not s[12] and self.ever_grasped
                  and np.linalg.norm(s[4:7] - s[8:11]) < 0.055
                  and abs(wrap(s[7] - s[11])) < 0.25)
        self.stable = self.stable + 1 if placed else 0
        self.success = self.stable >= 6
        self.steps += 1
        return s.copy()

    def metrics(self) -> dict:
        s = self.state
        lower_bound = float(np.linalg.norm(self.initial[:3] - self.initial[4:7])
                            + np.linalg.norm(self.initial[4:7] - self.initial[8:11]))
        return {"success": bool(self.success), "placement_error": float(np.linalg.norm(s[4:7] - s[8:11])),
                "yaw_error": abs(wrap(s[7] - s[11])), "steps": self.steps,
                "path_length": self.path_length,
                "efficiency": min(1.0, lower_bound / max(self.path_length, 1e-8)) if self.success else None,
                "failure": None if self.success else ("never_grasped" if not self.ever_grasped else "placement_or_release")}


def features(states: np.ndarray, relative: bool = True) -> np.ndarray:
    """No future information; relative xy is invariant under scene translation."""
    s = np.asarray(states, dtype=np.float32)
    if not relative:
        x = s.copy()
        x[..., [3, 7, 11]] /= np.pi
        return x
    a, b = s[..., 7] - s[..., 3], s[..., 11] - s[..., 7]
    return np.concatenate([s[..., 4:7] - s[..., :3], s[..., 8:11] - s[..., 4:7],
                           s[..., 2:3], s[..., 6:7], np.sin(a)[..., None], np.cos(a)[..., None],
                           np.sin(b)[..., None], np.cos(b)[..., None], s[..., 12:14]], axis=-1)

"""Small BC networks and checkpoint-only closed-loop inference."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .simulation import features


class Policy(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in {"absolute", "relative", "recurrent"}:
            raise ValueError("Unknown policy kind")
        self.kind = kind
        if kind == "recurrent":
            self.encoder = nn.GRU(14, 96, batch_first=True)
            self.head = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 5))
        else:
            self.head = nn.Sequential(nn.Linear(14, 128), nn.ReLU(), nn.Linear(128, 128),
                                      nn.ReLU(), nn.Linear(128, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "recurrent":
            x, _ = self.encoder(x)
        return self.head(x[:, -1])


class LearnedAgent:
    def __init__(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = Policy(payload["kind"])
        self.model.load_state_dict(payload["weights"])
        self.model.eval()
        self.metadata = {k: v for k, v in payload.items() if k != "weights"}
        self.history: deque = deque(maxlen=8)

    def reset(self) -> None:
        self.history.clear()

    @torch.inference_mode()
    def action(self, state: np.ndarray) -> np.ndarray:
        x = features(state, self.model.kind != "absolute")
        if not self.history:
            self.history.extend([x.copy() for _ in range(7)])
        self.history.append(x)
        raw = self.model(torch.tensor(np.array(self.history)[None], dtype=torch.float32))[0]
        return np.array([*raw[:4].tanh().numpy(), 1.0 if raw[4].item() > 0 else -1.0], dtype=np.float32)

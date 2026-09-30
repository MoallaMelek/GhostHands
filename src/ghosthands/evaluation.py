"""Paired, seeded closed-loop tests. No expert dependency at inference."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .models import LearnedAgent
from .simulation import Tabletop, HORIZON
from .data import load_episodes


def wilson(k: int, n: int) -> list[float]:
    p, z = k / n, 1.96
    c = (p + z*z / (2*n)) / (1 + z*z/n)
    r = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0, c-r), min(1, c+r)]


def evaluate(data: Path, checkpoints: Path, output: Path, count: int = 100) -> dict:
    if count < 1 or count > 10000:
        raise ValueError("Evaluation count must be 1–10000")
    torch.set_num_threads(2)
    replay = load_episodes(data, "train")[0]["actions"]
    policies = {k: LearnedAgent(checkpoints / f"{k}.pt") for k in ("absolute", "relative", "recurrent")}
    rows, summary = [], []
    for suite, base in (("held_out", 20000), ("spatial_ood", 30000), ("target_shift", 40000), ("object_shift", 50000)):
        for name in ["replay", *policies]:
            batch = []
            for seed in range(base, base + count):
                env = Tabletop(seed, suite == "spatial_ood")
                if name != "replay":
                    policies[name].reset()
                for t in range(HORIZON):
                    if t == 25 and suite in {"target_shift", "object_shift"}:
                        rng = np.random.default_rng(seed + 100000)
                        if suite == "target_shift":
                            env.state[8:10] = rng.uniform(0.15, 0.85, 2)
                        else:
                            env.state[13] = 0
                            env.state[4:6] = rng.uniform(0.15, 0.85, 2)
                            env.state[6] = 0.04
                        env.stable = 0
                    a = np.array(replay[min(t, len(replay)-1)]) if name == "replay" else policies[name].action(env.state)
                    env.step(a)
                    # Perturbation cases must reach the intervention even after early placement.
                    if env.success and (suite not in {"target_shift", "object_shift"} or t >= 25):
                        break
                row = {"suite": suite, "model": name, "seed": seed, **env.metrics()}
                if suite in {"target_shift", "object_shift"}:
                    row["efficiency"] = None  # initial-layout lower bound is invalid after intervention
                rows.append(row)
                batch.append(row)
            wins = sum(r["success"] for r in batch)
            eff = [r["efficiency"] for r in batch if r["efficiency"] is not None]
            summary.append({"suite": suite, "model": name, "n": count, "success_rate": wins/count,
                            "success_ci95": wilson(wins, count),
                            "mean_placement_error": float(np.mean([r["placement_error"] for r in batch])),
                            "mean_yaw_error": float(np.mean([r["yaw_error"] for r in batch])),
                            "mean_success_efficiency": float(np.mean(eff)) if eff else None})
            print(f"{suite:14} {name:10} {wins}/{count}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"training_seeds": 1, "checkpoint_metadata": {k: v.metadata for k, v in policies.items()}, "results": summary}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(summary, output / "results.png")
    return result


def plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, suite in zip(axes, ("held_out", "spatial_ood", "target_shift", "object_shift")):
        rows = [r for r in summary if r["suite"] == suite]
        values = np.array([r["success_rate"] for r in rows])
        intervals = np.array([r["success_ci95"] for r in rows]).T
        ax.bar([r["model"] for r in rows], values, color=["#5b6475", "#ac91ef", "#61e6c3", "#6bb7ff"])
        ax.errorbar(range(4), values, yerr=np.maximum(0, np.array([values-intervals[0], intervals[1]-values])), fmt="none", color="white", capsize=3)
        ax.set_title(suite.replace("_", " "))
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Task success · 95% Wilson interval")
    fig.suptitle("GhostHands / synthetic demonstrations / one training seed", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

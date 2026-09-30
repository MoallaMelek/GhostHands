import numpy as np
from ghosthands.data import generate_episode, save_episode
from ghosthands.training import train
from ghosthands.models import LearnedAgent
from ghosthands.simulation import Tabletop


def test_human_training_checkpoint_roundtrip(tmp_path):
    # Synthetic fixtures are relabeled solely in a temporary test directory.
    for seed, split in [(65000, "train"), (65001, "train"), (66000, "val")]:
        ep = generate_episode(seed, split)
        ep["source"] = "pointer"
        save_episode(ep, tmp_path / "data")
    result = train(tmp_path / "data", tmp_path / "models", "relative", epochs=1, source="human")
    assert result["train_episodes"] == 2 and result["source"] == "human"
    agent = LearnedAgent(tmp_path / "models/relative.pt")
    state = Tabletop(20000).state
    first = agent.action(state)
    assert first.shape == (5,) and np.isfinite(first).all() and np.abs(first).max() <= 1
    agent.reset()
    np.testing.assert_array_equal(first, agent.action(state))

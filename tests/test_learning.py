import numpy as np
import torch
from ghosthands.models import Policy
from ghosthands.training import windows, loss_fn
from ghosthands.data import generate_episode
from ghosthands.evaluation import wilson


def test_windows_never_cross_episode_boundaries():
    a, b = generate_episode(0, "train"), generate_episode(1, "train")
    x, y = windows([a, b], True)
    boundary = len(a["actions"])
    torch.testing.assert_close(x[boundary, 0], x[boundary, -1])
    assert len(x) == len(y) == boundary + len(b["actions"])


def test_all_networks_backpropagate():
    torch.set_num_threads(2)
    for kind in ("absolute", "relative", "recurrent"):
        model = Policy(kind)
        prediction = model(torch.zeros(3, 8, 14))
        assert prediction.shape == (3, 5)
        loss = loss_fn(prediction, torch.zeros(3, 5))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_wilson_extremes():
    assert wilson(0, 100)[0] == 0
    assert wilson(100, 100)[1] >= 0.999999
    assert wilson(50, 100)[0] < 0.5 < wilson(50, 100)[1]

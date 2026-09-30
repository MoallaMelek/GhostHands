import numpy as np
import pytest
from ghosthands.simulation import Tabletop, features, wrap
from ghosthands.data import generate_episode, assert_disjoint, validate_episode


def test_relative_translation_invariance():
    s = Tabletop(3).state
    shifted = s.copy()
    shifted[[0, 4, 8]] += 0.12
    shifted[[1, 5, 9]] -= 0.08
    np.testing.assert_allclose(features(s), features(shifted), atol=1e-7)
    assert not np.allclose(features(s, False), features(shifted, False))


def test_grasp_requires_contact_and_release_is_not_success():
    env = Tabletop(5)
    env.step(np.array([0, 0, 0, 0, 1]))
    assert env.state[13] == 0
    env.state[:4] = env.state[4:8]
    env.step(np.array([0, 0, 0, 0, 1]))
    assert env.state[13] == 1
    env.step(np.array([0, 0, 1, 0, 1]))
    assert env.state[6] > 0.04
    env.step(np.array([0, 0, 0, 0, -1]))
    assert not env.success and env.state[13] == 0


def test_expert_and_reproducibility():
    for seed in range(12):
        ep = generate_episode(seed, "train")
        validate_episode(ep)
        assert ep["metrics"]["success"]
        assert ep == generate_episode(seed, "train")


def test_split_guard_and_invalid_action():
    ep = generate_episode(0, "train")
    with pytest.raises(ValueError):
        assert_disjoint([ep], [ep])
    with pytest.raises(ValueError):
        Tabletop().step(np.full(5, np.nan))
    assert abs(wrap(2 * np.pi + 0.1) - 0.1) < 1e-6

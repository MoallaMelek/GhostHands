import numpy as np
import pytest
from fastapi.testclient import TestClient
from ghosthands import server
from ghosthands.data import teacher, load_episodes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "job", {"status": "idle"})
    server.sessions.clear()
    return TestClient(server.app)


def test_record_save_replay_roundtrip(client):
    sid = client.post("/api/session", json={"seed": 65000}).json()["id"]
    assert client.post(f"/api/{sid}/record", json={"source": "pointer", "split": "train"}).status_code == 200
    env = server.sessions[sid]["human"]
    for _ in range(180):
        response = client.post(f"/api/{sid}/control", json={"action": teacher(env).tolist()})
        assert response.status_code == 200
        if env.success:
            break
    result = client.post(f"/api/{sid}/save").json()
    assert result["success"]
    ep = load_episodes(server.ROOT / "data/human", "train")[0]
    # A test fixture exercises the API; no test demonstration enters the user's dataset.
    from ghosthands.simulation import Tabletop
    replay = Tabletop(ep["seed"])
    for state, action in zip(ep["observations"], ep["actions"]):
        np.testing.assert_allclose(replay.state, state)
        replay.step(np.array(action))
    assert replay.success


def test_missing_checkpoint_never_falls_back_to_teacher(client):
    sid = client.post("/api/session", json={}).json()["id"]
    assert client.post(f"/api/{sid}/run", json={"model": "human"}).status_code == 409
    assert not server.sessions[sid]["running"]


def test_insufficient_human_data_and_bad_control(client):
    assert client.post("/api/train").status_code == 409
    sid = client.post("/api/session", json={}).json()["id"]
    assert client.post(f"/api/{sid}/control", json={"action": [2, 0, 0, 0, 0]}).status_code == 422
    assert server.sessions[sid]["human"].steps == 0


def test_recording_resets_and_rejects_duplicate_start(client):
    sid = client.post("/api/session", json={}).json()["id"]
    client.post(f"/api/{sid}/control", json={"action": [1, 0, 0, 0, -1]})
    assert client.post(f"/api/{sid}/record", json={}).status_code == 200
    assert server.sessions[sid]["human"].steps == 0
    assert client.post(f"/api/{sid}/record", json={}).status_code == 409
    assert client.post(f"/api/{sid}/save").status_code == 422


def test_ragged_landmarks_cannot_partially_mutate_recording(client):
    sid = client.post("/api/session", json={}).json()["id"]
    client.post(f"/api/{sid}/record", json={})
    landmarks = [[0, 0, 0]] * 20 + [[0, 0]]
    result = client.post(f"/api/{sid}/control", json={"action": [1, 0, 0, 0, -1], "landmarks": landmarks})
    assert result.status_code == 422
    state = server.sessions[sid]
    assert state["human"].steps == 0
    assert state["record"]["observations"] == state["record"]["actions"] == state["record"]["landmarks"] == []


def test_new_human_layout_preserves_robot_session(client):
    sid = client.post("/api/session", json={}).json()["id"]
    robot = server.sessions[sid]["robot"]
    result = client.post(f"/api/{sid}/reset", json={"seed": 65002})
    assert result.status_code == 200
    assert len(server.sessions) == 1
    assert server.sessions[sid]["robot"] is robot
    assert server.sessions[sid]["human"].seed == 65002


def test_abandoned_rollout_does_not_block_training(client, monkeypatch):
    from types import SimpleNamespace
    from ghosthands.data import generate_episode, save_episode
    for seed, split in [(65000, "train"), (65001, "train"), (66000, "val")]:
        ep = generate_episode(seed, split)
        ep["source"] = "pointer"  # temporary synthetic test fixtures only
        save_episode(ep, server.ROOT / "data/human")
    abandoned = client.post("/api/session", json={}).json()["id"]
    server.sessions[abandoned]["running"] = True
    client.post("/api/session", json={})
    monkeypatch.setattr(server, "pool", SimpleNamespace(submit=lambda work: None))
    assert client.post("/api/train").status_code == 200
    assert server.job["status"] == "training"
    assert not server.sessions[abandoned]["running"]
    assert client.post(f"/api/{abandoned}/tick").status_code == 409

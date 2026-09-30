# Dataset contract v1

Each UTF-8 JSON file is one complete episode, stored under `data/<source>/<split>/`.
The filename is a digest; repeated saves of identical content are idempotent.
Generated data and user recordings are ignored by Git. Never mix frames from one episode
across splits. Choose validation/test layouts before recording; use **New layout** each time.

| Field | Meaning |
|---|---|
| schema | 1 |
| source | synthetic_teacher, webcam, or pointer |
| split | train, val, or test |
| seed | initial layout identifier |
| dt | 0.05 simulation seconds per action; browser wall time can be slower |
| observations | T × 14, state before each action |
| actions | T × 5, bounded to [-1, 1] |
| landmarks | optional T × 21 × 3 normalized MediaPipe xyz; null for missing sample |
| final_state | state after the final action |
| metrics | success, placement/yaw error, path length, efficiency, failure category |

Observation ordering: `[hand_x, hand_y, hand_z, hand_yaw, object_x, object_y,
object_z, object_yaw, target_x, target_y, target_z, target_yaw, closed, attached]`.
Distances are nominal meters; tabletop is 1 m × 1 m. Yaw is radians in [-π, π].
Actions are `[dx/0.035, dy/0.035, dz/0.035, dyaw/0.18, close]`.
Positive close closes; zero or negative opens. xyz and yaw velocities can be derived
from successive observations and dt; they are not extra model features in this MVP.

Relative features (14): object-minus-hand xyz, target-minus-object xyz, hand_z,
object_z, sin/cos(object_yaw-hand_yaw), sin/cos(target_yaw-object_yaw), closed, attached.
There is no privileged phase label. Attachment is observable simulator contact state.

The server records its own simulated states and actual applied commands. Clients cannot
upload arbitrary state histories through the recording API. Recording starts at reset.
Landmarks are optional auxiliary data, not policy inputs. Webcam frames are never saved.
MediaPipe and font assets require internet on first load; camera processing stays local.

The trainer validates shape, finiteness, source, action bounds, and train/validation
initial-layout disjointness. Recurrent context is left padded with each episode's first
observation and never crosses an episode boundary. Current human-only training excludes
unsuccessful episodes (and records the exclusion count). Failures remain on disk for
inspection; synthetic benchmark episodes are all retained. For mixed training (`--source all`),
curate the dataset deliberately: unsuccessful examples are not automatically filtered.

Minimum accepted episode length is 8 steps; maximum is 3000. A human-trained model needs
at least two successful training episodes and one successful validation episode; this is
an API minimum, not a claim of adequate data. Aim initially for 20–50 varied successes.

Checkpoint metadata includes architecture, context, seed, provenance, episode counts,
dataset SHA-256, PyTorch version, selected epoch and validation loss. Optimizer state is
not saved: checkpoints support inference, not exact interrupted-training resumption.

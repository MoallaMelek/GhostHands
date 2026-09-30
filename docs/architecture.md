# GhostHands: MVP research design

Decision recorded before implementation, 2026-09-30.

## Question and claim boundary
Can state-based imitation transfer a pick-and-place skill to unseen layouts?
This is goal-conditioned behavior cloning, not evidence of human-like intent understanding.
The first benchmark uses clearly labeled synthetic teacher demonstrations. Human webcam
episodes have separate provenance; synthetic success is never presented as human learning.

## Architecture
Webcam → MediaPipe landmarks → calibrated/smoothed Cartesian teleoperation → simulator
→ episode JSON → episode-level split → PyTorch BC → checkpoint → closed-loop simulation.
The simulator owns dynamics and success scoring. Only the network selects learned actions.
No teacher fallback or phase identifier is available to learned policies.

## Environment decision
| Candidate | Strength | Decision |
|---|---|---|
| Custom deterministic tabletop | CPU, inspectable semantics, browser rendering | MVP |
| MuJoCo | contact dynamics, Python bindings | Next validation target |
| PyBullet | accessible rigid-body robotics | Reasonable alternative |
| ManiSkill | established tasks and vectorized training | Defer platform/GPU integration |
| Isaac Sim/Lab | rich rendering and scalable robotics | Excessive MVP hardware/dependencies |

The MVP has xyz translation, yaw, binary gripper, gravity and a proximity/orientation
grasp constraint. It is a kinematic gripper, not a torque-controlled robot arm. No friction,
collision-rich planning, grasp forces or sim-to-real claim. Generalization is conditional
on accurate simulator state. The browser displays an isometric view of that same state.

## Representation and action
14 raw scalars: end effector xyz/yaw, object xyz/yaw, target xyz/yaw, closed, attached.
Actions: normalized delta xyz, delta yaw, close command; fixed step is 1/20 second.
Relative features: object minus hand xyz, target minus object xyz, hand and object height,
sin/cos of object-minus-hand yaw and target-minus-object yaw, closed and attached.
Heights retain gravity reference. Relative xy is translation invariant, not rotation invariant.
No task phase, future state, teacher action, or success flag appears in network input.

## Model decision
1. Fixed action replay from one training episode.
2. Absolute-state MLP BC, 2 × 128 hidden units.
3. Relative-state MLP BC, same capacity.
4. Relative-state GRU BC, 8-frame context and 96 hidden units.

MSE predicts bounded motion and binary cross entropy predicts closure. Receding closed-loop
actions permit response to moved goals; recovery must be measured, not assumed. A finite
history GRU is intentionally simpler than full ACT. ACT is a CVAE/transformer action-chunk
model, not just an MLP with multiple outputs. Diffusion and pixel encoders are deferred
until data volume and multimodality justify them. No fake confidence or attention overlay.

## Data and evaluation commitments
Keep full episodes together. Synthetic train seeds 0–239, validation 10000–10039, test
20000–20099; independently generated spatial OOD seeds 30000–30099. Training layouts
occupy the central region; OOD object and target coordinates extend toward table edges.
Checkpoint selection uses validation action loss only. Test outcomes never choose weights.
Report paired layout seeds for all policies, success with Wilson 95% intervals, final xyz
placement error, yaw error, and successful-path straight-line lower-bound efficiency.
Count every rollout, failures included. Report results for one training seed honestly;
multiple training seeds are a follow-up, not implied by multiple layout seeds.
Success requires release, position tolerance, orientation tolerance, and stable placement.
Record per-rollout data and perturbation suites separately. Do not silently drop failures.

## Webcam scope
MediaPipe 21 landmarks, palm-relative pinch ratio, wrist/palm yaw, smoothed xy, calibrated
palm-scale depth proxy. Monocular depth is approximate. Pointer/keyboard teleoperation is
an explicit fallback. Raw webcam images remain local; only landmarks and trajectories are
saved on explicit recording. Human-only training is available but requires multiple episodes.

## Sources inspected
- [ACT paper](https://arxiv.org/abs/2304.13705)
- [ACT policy source](https://github.com/tonyzhaozh/act/blob/main/policy.py): chunk reconstruction, padding mask, KL objective.
- [robomimic BC source](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/algo/bc.py): MLP and recurrent BC, recurrent resets.
- [MediaPipe web guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js)
- [MuJoCo Python](https://mujoco.readthedocs.io/en/latest/python.html)
- [PyBullet quickstart](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf)
- [ManiSkill installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [Isaac Lab requirements](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)

## Local setup notes
Requested AI_TEAM.md and AGENTS.md templates were absent at the provided template path.
No substitute collaboration protocol was invented. Local GitHub CLI authentication was
invalid at project creation; remote publication remains pending authenticated access.

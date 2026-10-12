# GhostHands validation checklist

Use this checklist before claiming successful real-world imitation transfer.

- [ ] Record human webcam demonstrations with participant consent.
- [ ] Keep training and evaluation scene layouts disjoint.
- [ ] Record the policy checkpoint and random seed for each evaluation.
- [ ] Compare synthetic-teacher and human-trained performance separately.
- [ ] Report success rate, sample count, and failure modes.
- [ ] Verify that no scripted fallback controls the robot during evaluation.

The existing demo is a synthetic-teacher prototype, not proof of human-trained generalization.

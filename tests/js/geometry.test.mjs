// Run with: node --test tests/js
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CRUISE,
  DEPTH_GAIN,
  YAW_GAIN,
  SETTLE,
  project,
  unproject,
  coverPoint,
  handPose,
  HandTeleop,
  OneEuro,
  wrap,
} from "../../src/ghosthands/ui/geometry.js";

const close = (a, b, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tol, `${a} is not within ${tol} of ${b}`);

test("unproject inverts project at any height", () => {
  for (const [w, h] of [[560, 400], [700, 340], [360, 340]])
    for (const p of [[0.2, 0.7, 0], [0.5, 0.5, CRUISE], [0.9, 0.1, 0.5]]) {
      const [px, py] = project(w, h, p);
      const [x, y] = unproject(w, h, px, py, p[2]);
      close(x, p[0]);
      close(y, p[1]);
    }
});

test("gripper at cruise height is drawn exactly under the cursor", () => {
  // Regression: the cursor used to be un-projected onto the floor, ~95 px below the gripper.
  const [w, h, cursor] = [600, 400, [310, 170]];
  const [x, y] = unproject(w, h, ...cursor, CRUISE);
  const drawn = project(w, h, [x, y, CRUISE]);
  close(drawn[0], cursor[0]);
  close(drawn[1], cursor[1]);
  const floor = project(w, h, [...unproject(w, h, ...cursor, 0), CRUISE]);
  assert.ok(cursor[1] - floor[1] > 80, "floor anchoring would put the gripper far above");
});

test("coverPoint mirrors and matches object-fit: cover cropping", () => {
  // 640x480 video in a 560x400 box: scale 0.875 -> 560x420, 10 px cropped top and bottom.
  const [x, y] = coverPoint(0.25, 0.5, 640, 480, 560, 400);
  close(x, 0.75 * 560);
  close(y, 200);
  close(coverPoint(0, 0, 640, 480, 560, 400, false)[1], -10);
});

test("One Euro filter removes jitter but follows a real move", () => {
  const f = new OneEuro(1.5, 6);
  let out = 0;
  for (let i = 0; i < 60; i++) out = f.filter(0.5 + (i % 2 ? 0.004 : -0.004), i / 30);
  assert.ok(Math.abs(out - 0.5) < 0.003);
  for (let i = 60; i < 90; i++) out = f.filter(0.8, i / 30);
  close(out, 0.8, 0.01);
});

// A flat synthetic hand (metres) projected with a known pixels-per-metre scale.
function syntheticHand({ scale = 1500, center = [320, 240], roll = 0, pinch = 0.08 } = {}) {
  const world = Array.from({ length: 21 }, (_, i) => {
    const finger = Math.floor((i - 1) / 4),
      joint = (i - 1) % 4;
    return i === 0
      ? { x: 0, y: 0.04, z: 0 }
      : { x: (finger - 2) * 0.02, y: 0.02 - joint * 0.025, z: 0 };
  });
  world[9] = { x: 0, y: -0.055, z: 0 }; // middle knuckle: wrist -> knuckle is 9.5 cm
  world[4] = { x: -pinch / 2, y: -0.06, z: 0 };
  world[8] = { x: pinch / 2, y: -0.06, z: 0 };
  const [c, s] = [Math.cos(roll), Math.sin(roll)];
  const lm = world.map((p) => ({
    x: (center[0] + scale * (c * p.x - s * p.y)) / 640,
    y: (center[1] + scale * (s * p.x + c * p.y)) / 480,
    z: 0,
  }));
  return { lm, world };
}

test("handPose recovers distance scale, pinch width and roll", () => {
  const { lm, world } = syntheticHand({ scale: 1500, pinch: 0.02 });
  const pose = handPose(lm, world, 640, 480);
  close(pose.scale, 1500, 1e-6);
  close(pose.pinch, 0.02, 1e-9);
  // Tilt is not depth: the fit is rotation invariant.
  const rotated = syntheticHand({ scale: 1500, roll: 0.7, pinch: 0.02 });
  const turned = handPose(rotated.lm, rotated.world, 640, 480);
  close(turned.scale, 1500, 1e-6);
  close(Math.abs(wrap(turned.roll - pose.roll)), 0.7, 1e-9);
  // Without world landmarks the palm length fallback still gives the right order of magnitude.
  close(handPose(lm, null, 640, 480).scale, 1500, 1e-6);
});

function run(teleop, hand, from, to) {
  let out;
  for (let t = from; t <= to + 1e-9; t += 1 / 30)
    out = teleop.update(handPose(hand.lm, hand.world, 640, 480), t, 600, 400, 640, 480);
  return out;
}

test("hand teleop: auto neutral pose, depth, yaw gain, pinch and palm anchoring", () => {
  const teleop = new HandTeleop();
  const neutral = syntheticHand();
  assert.equal(run(teleop, neutral, 0, SETTLE / 2).ready, false);
  let out = run(teleop, neutral, SETTLE / 2, 1.5);
  assert.equal(out.ready, true);
  close(out.desired[2], CRUISE, 1e-6);
  close(out.desired[3], 0, 1e-6);
  const drawn = project(600, 400, [out.desired[0], out.desired[1], CRUISE]);
  close(drawn[0], out.point[0], 1e-6);
  close(drawn[1], out.point[1], 1e-6);
  // Closer to the camera (bigger in the image) lowers the gripper.
  out = run(teleop, syntheticHand({ scale: 1500 * 1.2 }), 1.5, 4);
  close(out.desired[2], CRUISE - DEPTH_GAIN * Math.log(1.2), 1e-3);
  // Wrist roll is amplified so the full circle is reachable.
  out = run(teleop, syntheticHand({ roll: 0.5 }), 4, 7);
  close(Math.abs(out.desired[3]), YAW_GAIN * 0.5, 1e-3);
  // Pinch with hysteresis.
  assert.equal(run(teleop, syntheticHand({ pinch: 0.02 }), 7, 7.1).closed, true);
  assert.equal(run(teleop, syntheticHand({ pinch: 0.04 }), 7.1, 7.2).closed, true);
  assert.equal(run(teleop, syntheticHand({ pinch: 0.07 }), 7.2, 7.3).closed, false);
});

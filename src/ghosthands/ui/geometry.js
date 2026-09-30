// Pure teleoperation geometry. No DOM access, so it runs unchanged under `node --test`.
// Nothing here is a task controller: it only turns a cursor or a tracked hand into the
// commanded gripper pose. Every robot action still comes from a learned checkpoint.

export const CRUISE = 0.32; // reset height; the plane the cursor / palm is anchored to
export const Z_MIN = 0.04;
export const Z_MAX = 0.5;
export const XY_MIN = 0.05;
export const XY_MAX = 0.95;
export const DEPTH_GAIN = 0.9; // metres of gripper height per e-fold of hand distance
export const YAW_GAIN = 2; // +-90 degrees of wrist roll covers the full circle
export const PINCH_CLOSE = 0.03; // metres, thumb tip to index tip
export const PINCH_OPEN = 0.05;
export const SETTLE = 0.6; // seconds of steady tracking before the neutral pose is captured

export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function viewScale(w, h) {
  return Math.min(w * 0.6, h * 0.96);
}

/** Tabletop point [x, y, z] to CSS pixels in a w x h canvas (isometric view). */
export function project(w, h, p) {
  const scale = viewScale(w, h);
  return [
    w / 2 + (p[0] - p[1]) * scale,
    h * 0.25 + (p[0] + p[1]) * 0.34 * scale - p[2] * scale * 0.9,
  ];
}

/** Inverse of `project` on the horizontal plane at height z. Returns unclamped [x, y]. */
export function unproject(w, h, px, py, z) {
  const scale = viewScale(w, h);
  const diff = (px - w / 2) / scale;
  const sum = (py - h * 0.25 + z * scale * 0.9) / (0.34 * scale);
  return [(sum + diff) / 2, (sum - diff) / 2];
}

/**
 * Normalised video coordinates (u, v) to CSS pixels in a w x h box that shows the video
 * with `object-fit: cover`, mirrored like a selfie view by default.
 */
export function coverPoint(u, v, vw, vh, w, h, mirror = true) {
  const s = Math.max(w / vw, h / vh);
  const dw = vw * s;
  const dh = vh * s;
  return [(w - dw) / 2 + (mirror ? 1 - u : u) * dw, (h - dh) / 2 + v * dh];
}

/** One Euro filter (Casiez et al. 2012): steady when still, responsive when moving. */
export class OneEuro {
  constructor(minCutoff = 1.5, beta = 4, dCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }
  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }
  filter(x, t) {
    if (this.x === null) {
      this.x = x;
      this.t = t;
      return x;
    }
    const dt = t - this.t;
    if (dt <= 0) return this.x;
    this.t = t;
    const alpha = (cutoff) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    this.dx += alpha(this.dCutoff) * ((x - this.x) / dt - this.dx);
    this.x += alpha(this.minCutoff + this.beta * Math.abs(this.dx)) * (x - this.x);
    return this.x;
  }
}

const PALM = [0, 5, 9, 13, 17];
const PALM_LENGTH = 0.095; // metres, wrist to middle knuckle; fallback when no world landmarks

/**
 * Reduce 21 MediaPipe landmarks to the four quantities teleoperation needs.
 *   u, v   palm centre in normalised, unmirrored image coordinates
 *   scale  image pixels per metre of hand, proportional to 1 / distance from the camera
 *   roll   image-plane angle of wrist -> middle knuckle in the mirrored view (radians)
 *   pinch  thumb tip to index tip distance in approximate metres
 * `world` are MediaPipe world landmarks (metres). The scale is a rotation-invariant
 * weak-perspective fit of image landmarks to them, so tilting the hand is not read as depth.
 */
export function handPose(lm, world, vw, vh) {
  const px = lm.map((p) => [p.x * vw, p.y * vh]);
  let u = 0,
    v = 0;
  for (const i of PALM) {
    u += lm[i].x / PALM.length;
    v += lm[i].y / PALM.length;
  }
  let scale = 0;
  if (world && world.length === lm.length) {
    const mean = (pts) => [
      pts.reduce((a, p) => a + p[0], 0) / pts.length,
      pts.reduce((a, p) => a + p[1], 0) / pts.length,
    ];
    const spread = (pts) => {
      const [mx, my] = mean(pts);
      return pts.reduce((a, p) => a + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0);
    };
    const metres = spread(world.map((p) => [p.x, p.y]));
    if (metres > 1e-8) scale = Math.sqrt(spread(px) / metres);
  }
  if (!(scale > 0))
    scale = Math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1]) / PALM_LENGTH;
  scale = Math.max(scale, 1e-6);
  return {
    u,
    v,
    scale,
    roll: Math.atan2(px[9][1] - px[0][1], -(px[9][0] - px[0][0])),
    pinch: Math.hypot(px[4][0] - px[8][0], px[4][1] - px[8][1]) / scale,
  };
}

/**
 * Stateful hand -> commanded gripper pose ("claw machine" mapping).
 *   xy   the palm's on-screen point, un-projected onto the cruise-height plane, so at cruise
 *        height the gripper is drawn on the palm
 *   z    hand distance relative to the neutral pose: toward the camera lowers, away lifts
 *   yaw  YAW_GAIN x wrist roll relative to the neutral pose
 *   grip pinch with hysteresis
 * The neutral pose is captured automatically after SETTLE seconds of tracking.
 */
export class HandTeleop {
  constructor() {
    this.fx = new OneEuro(1.5, 6);
    this.fy = new OneEuro(1.5, 6);
    this.fz = new OneEuro(1.0, 3);
    this.fr = new OneEuro(1.0, 2);
    this.reset();
  }
  reset() {
    this.ref = null;
    this.closed = false;
    this.lost();
  }
  /** Tracking dropped: forget filter history so re-acquisition does not blend stale values. */
  lost() {
    for (const f of [this.fx, this.fy, this.fz, this.fr]) f.reset();
    this.samples = [];
  }
  recenter(pose) {
    this.ref = { scale: pose.scale, roll: pose.roll };
    this.fz.reset();
    this.fr.reset();
  }
  /** pose from `handPose`, t in seconds, (w, h) human canvas CSS size, (vw, vh) video size. */
  update(pose, t, w, h, vw, vh) {
    const point = coverPoint(pose.u, pose.v, vw, vh, w, h);
    if (pose.pinch < PINCH_CLOSE) this.closed = true;
    else if (pose.pinch > PINCH_OPEN) this.closed = false;
    if (!this.ref) {
      this.samples.push({ t, scale: pose.scale, roll: pose.roll });
      if (t - this.samples[0].t < SETTLE)
        return { ready: false, point, closed: this.closed, pinch: pose.pinch };
      const scales = this.samples.map((s) => s.scale).sort((a, b) => a - b);
      this.recenter({
        scale: scales[scales.length >> 1],
        roll: Math.atan2(
          this.samples.reduce((a, s) => a + Math.sin(s.roll), 0),
          this.samples.reduce((a, s) => a + Math.cos(s.roll), 0),
        ),
      });
      this.samples = [];
    }
    const [x, y] = unproject(w, h, point[0], point[1], CRUISE);
    const z = CRUISE - DEPTH_GAIN * Math.log(pose.scale / this.ref.scale);
    const roll = wrap(pose.roll - this.ref.roll);
    return {
      ready: true,
      point,
      closed: this.closed,
      pinch: pose.pinch,
      desired: [
        clamp(this.fx.filter(x, t), XY_MIN, XY_MAX),
        clamp(this.fy.filter(y, t), XY_MIN, XY_MAX),
        clamp(this.fz.filter(z, t), Z_MIN, Z_MAX),
        wrap(YAW_GAIN * this.fr.filter(roll, t)),
      ],
    };
  }
}

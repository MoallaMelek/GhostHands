import {
  CRUISE,
  Z_MIN,
  Z_MAX,
  XY_MIN,
  XY_MAX,
  wrap,
  clamp,
  project as projectPoint,
  unproject,
  coverPoint,
  handPose,
  HandTeleop,
} from "./geometry.js";

// A published version: 0.10.22 does not exist on npm, which silently broke the webcam.
const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const $ = (id) => document.getElementById(id);
let sid,
  snap,
  input = "pointer",
  recording = false,
  running = false,
  busy = false,
  close = false;
let desired = [0.5, 0.5, CRUISE, 0],
  tracker = null,
  cameraStream = null,
  hand = null,
  lastVideoTime = -1;
const teleop = new HandTeleop();
let humanTrail = [],
  robotTrail = [],
  nextSeed = 20000,
  demoSeed = 65001,
  controlActive = false;
function message(text) {
  $("message").textContent = text;
}
function stageNote(text) {
  $("stageNote").textContent = text || "";
  $("stageNote").hidden = !text;
}
async function api(path, body) {
  const response = await fetch("/api/" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.detail || "Request failed");
  return data;
}
function bind(id, fn) {
  $(id).addEventListener("click", async () => {
    try {
      await fn();
    } catch (e) {
      message(e.message);
    }
  });
}
function update(data) {
  snap = data;
  running = data.running;
  recording = data.recording;
  $("save").disabled = !recording;
  $("record").disabled = recording;
  $("recordStatus").textContent = recording ? "● RECORDING" : "● NOT RECORDING";
  $("recordStatus").style.color = recording ? "#ff8579" : "#8b9cad";
  $("steps").innerHTML = `${data.metrics.steps} <em>/ 180</em>`;
  $("error").textContent =
    (data.metrics.placement_error * 100).toFixed(1) + " cm";
  $("seedLabel").textContent = "LAYOUT " + data.robot_seed;
  const status = data.metrics.success
    ? "PLACED SUCCESSFULLY"
    : running
      ? "POLICY EXECUTING"
      : data.metrics.steps >= 180
        ? "ROLLOUT FAILED"
        : "READY / PAUSED";
  $("executionStatus").textContent = status;
  $("robotLabel").textContent = data.metrics.success
    ? "TASK COMPLETE"
    : running
      ? "CLOSED-LOOP INFERENCE"
      : "POLICY READY";
  if (data.metadata) {
    $("demoCount").textContent = data.metadata.train_episodes;
    $("provenance").textContent =
      `Training source: ${data.metadata.source} · ${data.metadata.train_episodes} training / ${data.metadata.val_episodes} validation episodes · checkpoint epoch ${data.metadata.epoch}.`;
  }
  $("handStatus").textContent = coach();
}
function setGrip(closed) {
  close = closed;
  $("grip").setAttribute("aria-pressed", String(close));
  $("grip").textContent = close ? "Open gripper" : "Close gripper";
}
async function reset() {
  if (recording)
    throw Error("Save your current recording before starting a new layout.");
  const data = await api(sid ? sid + "/reset" : "session", { seed: demoSeed++ });
  if (data.id) sid = data.id;
  desired = [0.5, 0.5, CRUISE, 0];
  setGrip(false);
  controlActive = false;
  $("height").value = CRUISE;
  $("yaw").value = 0;
  humanTrail = [];
  update(data);
}

// ---------- rendering ----------
function project(canvas, p) {
  return projectPoint(canvas.clientWidth, canvas.clientHeight, p);
}
function line(ctx, canvas, a, b, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(...project(canvas, a));
  ctx.lineTo(...project(canvas, b));
  ctx.stroke();
}
function polygon(ctx, points, fill, stroke, width = 1) {
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}
function cube(ctx, canvas, pos, yaw, color, size = 0.08, glow = false) {
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ].map(([x, y]) => [
    pos[0] + ((x * Math.cos(yaw) - y * Math.sin(yaw)) * size) / 2,
    pos[1] + ((x * Math.sin(yaw) + y * Math.cos(yaw)) * size) / 2,
    pos[2],
  ]);
  const top = corners.map((p) =>
    project(canvas, [p[0], p[1], p[2] + size / 2]),
  );
  const bottom = corners.map((p) =>
    project(canvas, [p[0], p[1], Math.max(0, p[2] - size / 2)]),
  );
  polygon(ctx, [...bottom], "#16272b");
  // Draw only faces turned toward the viewer (+x +y in table coordinates).
  for (let i = 0; i < 4; i++) {
    const a = corners[i],
      b = corners[(i + 1) % 4];
    if ((a[0] + b[0]) / 2 - pos[0] + (a[1] + b[1]) / 2 - pos[1] <= 0) continue;
    polygon(
      ctx,
      [bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]],
      i % 2 ? "#943f47" : "#be5960",
      "#e28b8c",
    );
  }
  polygon(ctx, top, color, glow ? "#8af3ce" : "#ffb2a5", glow ? 3 : 1);
  const center = project(canvas, [pos[0], pos[1], pos[2] + size / 2 + 0.002]);
  const end = project(canvas, [
    pos[0] + Math.cos(yaw) * 0.03,
    pos[1] + Math.sin(yaw) * 0.03,
    pos[2] + size / 2 + 0.002,
  ]);
  ctx.strokeStyle = "#ffe4d7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(...center);
  ctx.lineTo(...end);
  ctx.stroke();
}
// Human-panel teleoperation aid only; never reaches the policy or the dataset.
function alignment(s) {
  const attached = s[13] > 0,
    goal = attached ? [s[8], s[9], s[11]] : [s[4], s[5], s[7]];
  return {
    attached,
    over: Math.hypot(s[0] - goal[0], s[1] - goal[1]) < 0.04,
    turned: Math.abs(wrap(s[3] - goal[2])) < (attached ? 0.2 : 0.3),
  };
}
function draw(canvas, s, trail, isRobot) {
  if (!s) return;
  const dpr = devicePixelRatio || 1;
  const width = canvas.clientWidth,
    height = canvas.clientHeight;
  if (
    canvas.width !== Math.round(width * dpr) ||
    canvas.height !== Math.round(height * dpr)
  ) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, width, height);
  const live = !isRobot && input === "webcam";
  const aligned = isRobot ? null : alignment(s);
  polygon(
    c,
    [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ].map((p) => project(canvas, p)),
    live ? "#15222db0" : "#15222d",
    "#355060",
  );
  for (let k = 0; k <= 10; k++) {
    line(c, canvas, [k / 10, 0, 0], [k / 10, 1, 0], "#263744");
    line(c, canvas, [0, k / 10, 0], [1, k / 10, 0], "#263744");
  }
  const r = 0.065,
    angle = s[11],
    targetGlow = aligned && aligned.attached && aligned.over && aligned.turned,
    target = [
      [-r, -r],
      [r, -r],
      [r, r],
      [-r, r],
    ].map(([x, y]) =>
      project(canvas, [
        s[8] + x * Math.cos(angle) - y * Math.sin(angle),
        s[9] + x * Math.sin(angle) + y * Math.cos(angle),
        0.003,
      ]),
    );
  polygon(
    c,
    target,
    targetGlow ? "#1f6a58" : "#17463e",
    "#8af3ce",
    targetGlow ? 3 : 1,
  );
  line(
    c,
    canvas,
    [s[8], s[9], 0.005],
    [s[8] + Math.cos(angle) * 0.055, s[9] + Math.sin(angle) * 0.055, 0.005],
    "#c5ffe7",
    2,
  );
  const tp = project(canvas, [s[8], s[9], 0]);
  c.fillStyle = "#86ddc1";
  c.font = "9px sans-serif";
  c.fillText("TARGET", tp[0] - 16, tp[1] + 30);
  for (let i = 1; i < trail.length; i++)
    line(
      c,
      canvas,
      trail[i - 1],
      trail[i],
      isRobot ? "#75d7b088" : "#94a8ed77",
      1.5,
    );
  if (isRobot)
    for (let i = 1; i < humanTrail.length; i++)
      line(c, canvas, humanTrail[i - 1], humanTrail[i], "#758eb22a", 1);
  // Shadow on the table, directly below the gripper: this is what you aim with.
  const shadow = project(canvas, [s[0], s[1], 0]);
  c.strokeStyle = "#8fb4cc";
  c.lineWidth = 1.5;
  c.beginPath();
  c.ellipse(shadow[0], shadow[1], 9, 3.5, 0, 0, Math.PI * 2);
  c.stroke();
  line(c, canvas, [s[0], s[1], 0], s.slice(0, 3), "#668598", 1);
  if (!isRobot && s[2] < CRUISE - 0.01)
    // Cable up to the cruise plane, where the cursor / palm anchors the gripper.
    line(c, canvas, [s[0], s[1], s[2] + 0.11], [s[0], s[1], CRUISE], "#b6c6fb55", 1);
  cube(
    c,
    canvas,
    s.slice(4, 7),
    s[7],
    "#ef8a88",
    0.08,
    aligned && !aligned.attached && aligned.over && aligned.turned,
  );
  if (!isRobot && controlActive) {
    const d = project(canvas, desired.slice(0, 3));
    c.strokeStyle = "#b6c6fb66";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(d[0] - 6, d[1]);
    c.lineTo(d[0] + 6, d[1]);
    c.moveTo(d[0], d[1] - 6);
    c.lineTo(d[0], d[1] + 6);
    c.stroke();
  }
  const p = project(canvas, s.slice(0, 3));
  const color = isRobot ? "#9af6d5" : "#b6c6fb";
  const span = s[12] ? 0.03 : 0.08;
  for (const side of [-1, 1]) {
    const x = s[0] + Math.cos(s[3]) * span * side,
      y = s[1] + Math.sin(s[3]) * span * side,
      // The +yaw jaw is tinted so a half-turn misalignment is visible.
      jaw = side > 0 ? "#ffd27a" : color;
    line(c, canvas, [x, y, s[2] + 0.11], [x, y, s[2] + 0.025], jaw, 4);
    line(
      c,
      canvas,
      [x, y, s[2] + 0.025],
      [
        x - Math.cos(s[3]) * 0.015 * side,
        y - Math.sin(s[3]) * 0.015 * side,
        s[2] + 0.025,
      ],
      jaw,
      3,
    );
  }
  line(
    c,
    canvas,
    [s[0] - Math.cos(s[3]) * span, s[1] - Math.sin(s[3]) * span, s[2] + 0.11],
    [s[0] + Math.cos(s[3]) * span, s[1] + Math.sin(s[3]) * span, s[2] + 0.11],
    color,
    6,
  );
  c.strokeStyle = color;
  c.lineWidth = 3;
  c.beginPath();
  c.arc(p[0], p[1], 4, 0, Math.PI * 2);
  c.stroke();
}
const BONES = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [5, 9, 10, 11, 12],
  [9, 13, 14, 15, 16],
  [13, 17, 18, 19, 20],
  [0, 17],
];
function drawHand(lm, out) {
  const canvas = $("handOverlay"),
    video = $("video"),
    dpr = devicePixelRatio || 1,
    w = canvas.clientWidth,
    h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!lm) return;
  const pt = (p) =>
    coverPoint(p.x, p.y, video.videoWidth, video.videoHeight, w, h);
  ctx.strokeStyle = "#8af3ce99";
  ctx.lineWidth = 2;
  for (const finger of BONES) {
    ctx.beginPath();
    finger.forEach((i, j) => (j ? ctx.lineTo(...pt(lm[i])) : ctx.moveTo(...pt(lm[i]))));
    ctx.stroke();
  }
  ctx.fillStyle = "#e5fff5";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(...pt(p), 2, 0, Math.PI * 2);
    ctx.fill();
  }
  const [a, b] = [pt(lm[4]), pt(lm[8])];
  ctx.strokeStyle = out.closed ? "#ffd27a" : "#e5fff566";
  ctx.lineWidth = out.closed ? 3 : 1;
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
  ctx.strokeStyle = "#b6c6fb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(out.point[0], out.point[1], 7, 0, Math.PI * 2);
  ctx.stroke();
}
function render() {
  if (input === "webcam") track();
  if (snap) {
    draw($("human"), snap.human, humanTrail, false);
    draw($("robot"), snap.robot, robotTrail, true);
  }
  requestAnimationFrame(render);
}

// One short instruction for the demonstrator, derived from the human-side state.
function coach() {
  if (!snap) return "";
  const cam = input === "webcam";
  if (cam && !hand) return "Show one hand to the camera";
  if (cam && !hand.out.ready) return "Hold your open hand still…";
  if (snap.human_metrics.success) return "Placed. Save the demonstration";
  const s = snap.human,
    a = alignment(s);
  if (!a.over)
    return a.attached
      ? "Carry the block over the green target"
      : cam
        ? "Move your palm until the shadow is under the red block"
        : "Move until the shadow is under the red block";
  if (!a.turned)
    return cam
      ? "Turn your hand until the yellow jaw matches the white mark"
      : "Q / E until the yellow jaw matches the white mark";
  if (!a.attached && s[2] > 0.07)
    return cam ? "Push your hand toward the camera to lower" : "Scroll down to lower";
  if (a.attached)
    return cam ? "Open your hand to release" : "Space to release";
  return cam ? "Pinch thumb and index to grab" : "Space to grab";
}

// ---------- pointer input ----------
$("human").addEventListener("pointermove", (e) => {
  if (input !== "pointer") return;
  const rect = $("human").getBoundingClientRect();
  // Anchor the cursor on the cruise-height plane so the gripper sits under it.
  const [x, y] = unproject(
    rect.width,
    rect.height,
    e.clientX - rect.left,
    e.clientY - rect.top,
    CRUISE,
  );
  desired[0] = clamp(x, XY_MIN, XY_MAX);
  desired[1] = clamp(y, XY_MIN, XY_MAX);
  controlActive = true;
});
$("human").addEventListener(
  "wheel",
  (e) => {
    if (input !== "pointer") return;
    e.preventDefault();
    desired[2] = clamp(desired[2] - e.deltaY * 0.0006, Z_MIN, Z_MAX);
    $("height").value = desired[2];
    controlActive = true;
  },
  { passive: false },
);
$("height").addEventListener("input", () => {
  desired[2] = +$("height").value;
  controlActive = true;
});
$("yaw").addEventListener("input", () => {
  desired[3] = +$("yaw").value;
  controlActive = true;
});
function grip() {
  if (input === "webcam") return;
  setGrip(!close);
  controlActive = true;
}
bind("grip", grip);
document.addEventListener("keydown", (e) => {
  if (input === "webcam") return;
  if (["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName))
    return;
  if (e.code === "Space") {
    e.preventDefault();
    grip();
  }
  if (e.key === "q" || e.key === "e") {
    desired[3] = wrap(desired[3] + (e.key === "q" ? -0.12 : 0.12));
    $("yaw").value = desired[3];
  }
});

// ---------- demo / recording / rollout ----------
bind("newDemo", reset);
bind("record", async () => {
  if (input === "webcam" && !hand?.out.ready)
    throw Error("Show your hand to the camera and wait for tracking before recording.");
  update(
    await api(sid + "/record", { source: input, split: $("split").value }),
  );
  controlActive = true;
  message(
    "Recording started. Pick up, align, place and release the red block.",
  );
});
bind("save", async () => {
  const result = await api(sid + "/save", {});
  recording = false;
  $("save").disabled = true;
  $("record").disabled = false;
  $("recordStatus").textContent = "● SAVED";
  message(
    result.success
      ? "Successful demonstration saved."
      : "Episode saved with failure label. Check the trajectory before including it in training.",
  );
  await refresh();
});
bind("run", async () => {
  robotTrail = [];
  update(
    await api(sid + "/run", { model: $("model").value, seed: nextSeed++ }),
  );
  message("The selected neural checkpoint is choosing every robot action.");
});
bind("pause", async () => update(await api(sid + "/pause", {})));
for (const [id, kind] of [
  ["moveObject", "object"],
  ["moveTarget", "target"],
])
  bind(id, async () => {
    update(await api(sid + "/perturb", { kind }));
    message(
      `Moved ${kind}. Recovery is an experiment, not a guarantee. Resume if paused.`,
    );
  });
bind("train", async () => {
  await api("train", {});
  message("Training exclusively on your recorded human episodes.");
  refresh();
});
async function tick() {
  if (busy || !sid) return;
  busy = true;
  try {
    const handReady = input === "webcam" && hand?.out.ready;
    if (controlActive && (input === "pointer" || handReady)) {
      const s = snap.human,
        action = [
          ...desired
            .slice(0, 3)
            .map((v, i) => clamp((v - s[i]) / 0.035, -1, 1)),
          clamp(wrap(desired[3] - s[3]) / 0.18, -1, 1),
          close ? 1 : -1,
        ];
      update(
        await api(sid + "/control", {
          action,
          landmarks: handReady ? hand.landmarks : null,
        }),
      );
      humanTrail.push(snap.human.slice(0, 3));
      if (humanTrail.length > 1000) humanTrail.shift();
    }
    if (running) {
      update(await api(sid + "/tick", {}));
      robotTrail.push(snap.robot.slice(0, 3));
    }
  } catch (e) {
    controlActive = false;
    running = false;
    message(e.message);
  } finally {
    busy = false;
  }
}

// ---------- webcam input ----------
function cameraError(e) {
  switch (e?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access was blocked. Allow the camera for this page (camera icon in the address bar), then press Enable webcam again. Embedded previews often cannot use cameras: open http://127.0.0.1:8765 in Chrome or Edge.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect or enable a webcam, then try again.";
    case "NotReadableError":
    case "AbortError":
      return "The camera is in use or failed to start. Close other apps using it (Teams, Zoom, Camera, another browser tab) and try again.";
    default:
      return "Could not start the camera: " + (e?.message || e);
  }
}
async function loadTracker() {
  if (tracker) return tracker;
  const { HandLandmarker, FilesetResolver } = await import(
    VISION + "/vision_bundle.mjs"
  );
  const vision = await FilesetResolver.forVisionTasks(VISION + "/wasm");
  // GPU is ~3x faster per frame but compiles shaders on first use (seconds), so warm it up
  // here, behind the loading note, and fall back to CPU if WebGL is unavailable.
  const warmup = document.createElement("canvas");
  warmup.width = 64;
  warmup.height = 48;
  for (const delegate of ["GPU", "CPU"]) {
    try {
      const candidate = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
      candidate.detectForVideo(warmup, performance.now());
      tracker = candidate;
      return tracker;
    } catch (e) {
      if (delegate === "CPU") throw e;
    }
  }
}
function stopCamera() {
  cameraStream?.getTracks().forEach((t) => t.stop());
  cameraStream = null;
  $("video").srcObject = null;
  $("video").hidden = true;
  input = "pointer";
  hand = null;
  teleop.reset();
  lastVideoTime = -1;
  controlActive = false;
  drawHand(null);
  stageNote("");
  for (const id of ["height", "yaw", "grip"]) $(id).disabled = false;
  $("calibrate").disabled = true;
  $("camera").textContent = "Enable webcam";
  $("inputBadge").textContent = "POINTER INPUT";
  $("sceneLabel").textContent = "TEACH THE MOVEMENT";
  if (snap) $("handStatus").textContent = coach();
}
bind("camera", async () => {
  if (recording)
    throw Error("Save your recording before changing input source.");
  if (cameraStream) return stopCamera();
  if (!navigator.mediaDevices?.getUserMedia)
    throw Error(
      "This page cannot reach a camera. Open http://127.0.0.1:8765 (not a LAN address or file) in Chrome or Edge.",
    );
  $("camera").disabled = true;
  try {
    stageNote("Waiting for camera permission…");
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      stageNote("");
      throw Error(cameraError(e));
    }
    const video = $("video");
    video.srcObject = cameraStream;
    video.hidden = false;
    await video.play();
    stageNote("Camera on. Loading hand tracker…");
    try {
      await loadTracker();
    } catch (e) {
      stopCamera();
      throw Error(
        "The camera works, but the hand tracker could not be downloaded (internet is needed on first use): " +
          e.message,
      );
    }
    input = "webcam";
    teleop.reset();
    controlActive = false;
    for (const id of ["height", "yaw", "grip"]) $(id).disabled = true;
    $("calibrate").disabled = false;
    $("camera").textContent = "Use pointer";
    $("inputBadge").textContent = "WEBCAM INPUT";
    stageNote("");
    message(
      "Webcam on. Your palm carries the gripper: push toward the camera to lower, pull back to lift, turn your hand to rotate, pinch to grab.",
    );
  } finally {
    $("camera").disabled = false;
  }
});
bind("calibrate", () => {
  if (!hand) throw Error("Show your hand to the camera first.");
  teleop.reset();
  message("Hold your open hand still for a moment to set the new neutral pose.");
});
function track() {
  const video = $("video");
  if (!tracker || video.readyState < 2 || video.currentTime === lastVideoTime)
    return;
  lastVideoTime = video.currentTime;
  let result;
  try {
    result = tracker.detectForVideo(video, performance.now());
  } catch (e) {
    message("Hand tracking error: " + e.message);
    return;
  }
  const lm = result.landmarks?.[0];
  if (!lm) {
    if (hand) teleop.lost();
    hand = null;
    drawHand(null);
    $("sceneLabel").textContent = "NO HAND · CONTROL PAUSED";
    if (snap) $("handStatus").textContent = coach();
    return;
  }
  const canvas = $("human"),
    pose = handPose(
      lm,
      result.worldLandmarks?.[0],
      video.videoWidth,
      video.videoHeight,
    ),
    out = teleop.update(
      pose,
      performance.now() / 1000,
      canvas.clientWidth,
      canvas.clientHeight,
      video.videoWidth,
      video.videoHeight,
    );
  hand = { landmarks: lm.map((p) => [p.x, p.y, p.z]), out };
  if (out.ready) {
    desired = out.desired;
    setGrip(out.closed);
    controlActive = true;
    $("height").value = desired[2];
    $("yaw").value = desired[3];
  }
  $("sceneLabel").textContent = out.ready
    ? `HAND TRACKED · HEIGHT ${Math.round(desired[2] * 100)} CM · ${out.closed ? "PINCHED" : "OPEN"}`
    : "HAND FOUND · HOLD STILL";
  drawHand(lm, out);
  if (snap) $("handStatus").textContent = coach();
}
async function refresh() {
  try {
    const status = await api("status");
    $("counts").textContent =
      `${status.episodes.train} training · ${status.episodes.val} validation · ${status.episodes.test} test episodes`;
    $("train").disabled = status.job.status === "training";
    if (status.job.status !== "idle")
      $("job").textContent =
        status.job.status === "failed"
          ? status.job.error
          : status.job.status === "complete"
            ? "Training complete. Select “My demonstrations” to evaluate on new layouts."
            : "Training your policy on CPU…";
  } catch (e) {
    message(e.message);
  }
}
try {
  await reset();
  const result = await api("results");
  if (result.results.length) {
    $("results").innerHTML = "";
    for (const model of ["replay", "absolute", "relative", "recurrent"]) {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = {
        replay: "Exact replay",
        absolute: "Absolute BC",
        relative: "Relative BC",
        recurrent: "Recurrent BC",
      }[model];
      row.append(name);
      for (const suite of ["held_out", "spatial_ood", "target_shift"]) {
        const r = result.results.find(
            (x) => x.model === model && x.suite === suite,
          ),
          cell = document.createElement("td");
        cell.textContent = r ? Math.round(r.success_rate * 100) + "%" : "—";
        if (r)
          cell.title = `95% CI: ${r.success_ci95.map((v) => (v * 100).toFixed(1)).join("–")}%`;
        row.append(cell);
      }
      $("results").append(row);
    }
  }
  await refresh();
  render();
  setInterval(tick, 50);
  setInterval(refresh, 3000);
} catch (e) {
  message(e.message);
}
window.addEventListener("beforeunload", () =>
  cameraStream?.getTracks().forEach((t) => t.stop()),
);

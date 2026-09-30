const $ = (id) => document.getElementById(id);
let sid,
  snap,
  input = "pointer",
  recording = false,
  running = false,
  busy = false,
  close = false;
let desired = [0.5, 0.5, 0.32, 0],
  landmarks = null,
  tracker = null,
  cameraStream = null,
  calibrated = false,
  palmReference = 0.2;
let humanTrail = [],
  robotTrail = [],
  nextSeed = 20000,
  demoSeed = 65001,
  controlActive = false;
function message(text) {
  $("message").textContent = text;
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
}
async function reset() {
  if (recording)
    throw Error("Save your current recording before starting a new layout.");
  const data = await api(sid ? sid + "/reset" : "session", { seed: demoSeed++ });
  if (data.id) sid = data.id;
  desired = [0.5, 0.5, 0.32, 0];
  close = false;
  controlActive = false;
  $("height").value = 0.32;
  $("yaw").value = 0;
  humanTrail = [];
  $("grip").setAttribute("aria-pressed", "false");
  $("grip").textContent = "Close gripper";
  update(data);
}
function project(canvas, p) {
  const w = canvas.clientWidth,
    h = canvas.clientHeight,
    scale = Math.min(w * 0.6, h * 0.96);
  return [
    w / 2 + (p[0] - p[1]) * scale,
    h * 0.25 + (p[0] + p[1]) * 0.34 * scale - p[2] * scale * 0.9,
  ];
}
function line(ctx, canvas, a, b, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(...project(canvas, a));
  ctx.lineTo(...project(canvas, b));
  ctx.stroke();
}
function polygon(ctx, points, fill, stroke) {
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}
function cube(ctx, canvas, pos, yaw, color, size = 0.08) {
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
  for (let i = 0; i < 4; i++)
    polygon(
      ctx,
      [bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]],
      i % 2 ? "#943f47" : "#be5960",
      "#e28b8c",
    );
  polygon(ctx, top, color, "#ffb2a5");
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
  polygon(
    c,
    [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ].map((p) => project(canvas, p)),
    "#15222d",
    "#355060",
  );
  for (let k = 0; k <= 10; k++) {
    line(c, canvas, [k / 10, 0, 0], [k / 10, 1, 0], "#263744");
    line(c, canvas, [0, k / 10, 0], [1, k / 10, 0], "#263744");
  }
  const r = 0.065,
    angle = s[11],
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
  polygon(c, target, "#17463e", "#8af3ce");
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
  line(c, canvas, [s[0], s[1], 0], s.slice(0, 3), "#668598", 1);
  cube(c, canvas, s.slice(4, 7), s[7], "#ef8a88");
  const p = project(canvas, s.slice(0, 3));
  c.strokeStyle = isRobot ? "#9af6d5" : "#b6c6fb";
  c.lineWidth = 3;
  const span = s[12] ? 0.03 : 0.08;
  for (const side of [-1, 1]) {
    const x = s[0] + Math.cos(s[3]) * span * side,
      y = s[1] + Math.sin(s[3]) * span * side;
    line(
      c,
      canvas,
      [x, y, s[2] + 0.11],
      [x, y, s[2] + 0.025],
      c.strokeStyle,
      4,
    );
    line(
      c,
      canvas,
      [x, y, s[2] + 0.025],
      [
        x - Math.cos(s[3]) * 0.015 * side,
        y - Math.sin(s[3]) * 0.015 * side,
        s[2] + 0.025,
      ],
      c.strokeStyle,
      3,
    );
  }
  line(
    c,
    canvas,
    [s[0] - Math.cos(s[3]) * span, s[1] - Math.sin(s[3]) * span, s[2] + 0.11],
    [s[0] + Math.cos(s[3]) * span, s[1] + Math.sin(s[3]) * span, s[2] + 0.11],
    c.strokeStyle,
    6,
  );
  c.beginPath();
  c.arc(p[0], p[1], 4, 0, Math.PI * 2);
  c.stroke();
}
function render() {
  if (snap) {
    draw($("human"), snap.human, humanTrail, false);
    draw($("robot"), snap.robot, robotTrail, true);
  }
  requestAnimationFrame(render);
}
$("human").addEventListener("pointermove", (e) => {
  if (input !== "pointer") return;
  const rect = $("human").getBoundingClientRect(),
    w = rect.width,
    h = rect.height,
    scale = Math.min(w * 0.6, h * 0.96);
  const diff = (e.clientX - rect.left - w / 2) / scale,
    sum = (e.clientY - rect.top - h * 0.25) / (0.34 * scale);
  desired[0] = Math.max(0.05, Math.min(0.95, (sum + diff) / 2));
  desired[1] = Math.max(0.05, Math.min(0.95, (sum - diff) / 2));
  controlActive = true;
});
$("human").addEventListener(
  "wheel",
  (e) => {
    if (input !== "pointer") return;
    e.preventDefault();
    desired[2] = Math.max(0.04, Math.min(0.5, desired[2] - e.deltaY * 0.0006));
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
  close = !close;
  $("grip").setAttribute("aria-pressed", String(close));
  $("grip").textContent = close ? "Open gripper" : "Close gripper";
  controlActive = true;
}
bind("grip", grip);
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName))
    return;
  if (e.code === "Space") {
    e.preventDefault();
    grip();
  }
  if (e.key === "q" || e.key === "e") {
    desired[3] += e.key === "q" ? -0.12 : 0.12;
    desired[3] = Math.atan2(Math.sin(desired[3]), Math.cos(desired[3]));
    $("yaw").value = desired[3];
  }
});
bind("newDemo", reset);
bind("record", async () => {
  if (input === "webcam" && !calibrated)
    throw Error("Calibrate your open palm before recording.");
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
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (x) => Math.max(-1, Math.min(1, x));
async function tick() {
  if (busy || !sid) return;
  busy = true;
  try {
    if (input === "webcam" && tracker && $("video").readyState >= 2) track();
    if (controlActive && (input === "pointer" || landmarks)) {
      const s = snap.human,
        action = [
          ...desired.slice(0, 3).map((v, i) => clamp((v - s[i]) / 0.035)),
          clamp(wrap(desired[3] - s[3]) / 0.18),
          close ? 1 : -1,
        ];
      update(
        await api(sid + "/control", {
          action,
          landmarks: input === "webcam" ? landmarks : null,
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
let lastVideo = -1,
  lastPalm = null;
bind("camera", async () => {
  if (cameraStream) {
    if (recording)
      throw Error("Save your recording before changing input source.");
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
    input = "pointer";
    landmarks = null;
    controlActive = false;
    $("videoWrap").hidden = true;
    $("camera").textContent = "Enable webcam";
    $("inputBadge").textContent = "POINTER INPUT";
    return;
  }
  if (recording)
    throw Error("Save your recording before changing input source.");
  message(
    "Loading hand tracking. Your browser will request webcam permission.",
  );
  const { HandLandmarker, FilesetResolver } =
    await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm",
  );
  tracker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  $("video").srcObject = cameraStream;
  await $("video").play();
  $("videoWrap").hidden = false;
  input = "webcam";
  calibrated = false;
  controlActive = false;
  $("camera").textContent = "Use pointer";
  $("calibrate").disabled = false;
  $("inputBadge").textContent = "WEBCAM INPUT";
  message("Hold your open palm at comfortable distance, then calibrate depth.");
});
bind("calibrate", () => {
  if (!lastPalm) throw Error("Show your hand to the camera first.");
  palmReference = lastPalm;
  calibrated = true;
  controlActive = true;
  message(
    "Depth calibrated. Bring your palm closer to lift; move it away to lower.",
  );
});
function track() {
  const video = $("video");
  if (video.currentTime === lastVideo) return;
  lastVideo = video.currentTime;
  const result = tracker.detectForVideo(video, performance.now());
  if (!result.landmarks.length) {
    landmarks = null;
    lastPalm = null;
    $("handStatus").textContent = "Hand lost · control paused";
    return;
  }
  const lm = result.landmarks[0];
  landmarks = lm.map((p) => [p.x, p.y, p.z]);
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  lastPalm = dist(lm[0], lm[9]);
  const ratio = dist(lm[4], lm[8]) / Math.max(0.01, dist(lm[5], lm[17]));
  if (ratio < 0.35) close = true;
  else if (ratio > 0.55) close = false;
  const target = [
    0.05 + 0.9 * (1 - lm[9].x),
    0.05 + 0.9 * lm[9].y,
    Math.max(0.04, Math.min(0.5, 0.22 + (lastPalm / palmReference - 1) * 0.5)),
    Math.atan2(lm[5].y - lm[17].y, -(lm[5].x - lm[17].x)),
  ];
  if (calibrated) {
    for (let i = 0; i < 3; i++) desired[i] += 0.25 * (target[i] - desired[i]);
    desired[3] = wrap(desired[3] + 0.25 * wrap(target[3] - desired[3]));
  }
  const c = $("landmarks"),
    ctx = c.getContext("2d");
  c.width = 290;
  c.height = 216;
  ctx.clearRect(0, 0, 290, 216);
  ctx.strokeStyle = "#8af3ce";
  ctx.lineWidth = 2;
  for (const finger of [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [5, 9, 10, 11, 12],
    [9, 13, 14, 15, 16],
    [13, 17, 18, 19, 20],
    [0, 17],
  ]) {
    ctx.beginPath();
    finger.forEach((i, j) =>
      j
        ? ctx.lineTo(lm[i].x * 290, lm[i].y * 216)
        : ctx.moveTo(lm[i].x * 290, lm[i].y * 216),
    );
    ctx.stroke();
  }
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * 290, p.y * 216, 2, 0, Math.PI * 2);
    ctx.fillStyle = "#e5fff5";
    ctx.fill();
  }
  $("handStatus").textContent = calibrated
    ? close
      ? "PINCH / CLOSED"
      : "OPEN HAND"
    : "Open palm → calibrate depth";
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

"use strict";

const sceneCanvas = document.getElementById("sceneCanvas");
const sceneCtx = sceneCanvas.getContext("2d");
const iqCanvas = document.getElementById("iqCanvas");
const iqCtx = iqCanvas?.getContext("2d") || null;

const radarTabs = document.getElementById("radarTabs");
const targetTabs = document.getElementById("targetTabs");
const playPauseBtn = document.getElementById("playPauseBtn");
const playPauseIcon = document.getElementById("playPauseIcon");
const playPauseText = document.getElementById("playPauseText");
const resetBtn = document.getElementById("resetBtn");
const speedRange = document.getElementById("speedRange");
const clutterRange = document.getElementById("clutterRange");
const uncertaintyRange = document.getElementById("uncertaintyRange");
const showBeamsInput = document.getElementById("showBeams");
const showTracksInput = document.getElementById("showTracks");
const showGatesInput = document.getElementById("showGates");

const ui = {
  runStatus: document.getElementById("runStatus"),
  backendStatusValue: document.getElementById("backendStatusValue"),
  llmStatusValue: document.getElementById("llmStatusValue"),
  cycleValue: document.getElementById("cycleValue"),
  timeValue: document.getElementById("timeValue"),
  loopRateValue: document.getElementById("loopRateValue"),
  speedValue: document.getElementById("speedValue"),
  clutterValue: document.getElementById("clutterValue"),
  uncertaintyValue: document.getElementById("uncertaintyValue"),
  mapFocus: document.getElementById("mapFocus"),
  backendOverlay: document.getElementById("backendOverlay"),
  selectedRadarName: document.getElementById("selectedRadarName"),
  selectedRadarRole: document.getElementById("selectedRadarRole"),
  bearingMetric: document.getElementById("bearingMetric"),
  rangeMetric: document.getElementById("rangeMetric"),
  confidenceMetric: document.getElementById("confidenceMetric"),
  snrMetric: document.getElementById("snrMetric"),
  beamWidthMetric: document.getElementById("beamWidthMetric"),
  dwellMetric: document.getElementById("dwellMetric"),
  iqSource: document.getElementById("iqSource"),
  estimatedBearing: document.getElementById("estimatedBearing"),
  estimatedRange: document.getElementById("estimatedRange"),
  dopplerMetric: document.getElementById("dopplerMetric"),
  fusionMetric: document.getElementById("fusionMetric"),
  intentMetric: document.getElementById("intentMetric"),
  threatMetric: document.getElementById("threatMetric"),
  llmNarrative: document.getElementById("llmNarrative"),
  controlMetric: document.getElementById("controlMetric"),
  controllerImpactList: document.getElementById("controllerImpactList"),
  loopList: document.getElementById("loopList"),
  eventLog: document.getElementById("eventLog"),
  cardShoreTitle: document.getElementById("cardShoreTitle"),
  cardShoreBeam: document.getElementById("cardShoreBeam"),
  cardShoreConfidence: document.getElementById("cardShoreConfidence"),
  cardShoreBar: document.getElementById("cardShoreBar"),
  cardV01Title: document.getElementById("cardV01Title"),
  cardV01Beam: document.getElementById("cardV01Beam"),
  cardV01Confidence: document.getElementById("cardV01Confidence"),
  cardV01Bar: document.getElementById("cardV01Bar"),
  cardV02Title: document.getElementById("cardV02Title"),
  cardV02Beam: document.getElementById("cardV02Beam"),
  cardV02Confidence: document.getElementById("cardV02Confidence"),
  cardV02Bar: document.getElementById("cardV02Bar"),
};

const world = {
  minX: -72,
  maxX: 92,
  minY: -58,
  maxY: 66,
};

const radarMeta = {
  shore: {
    id: "shore",
    name: "岸基雷达",
    role: "广域发现",
    short: "SHORE",
    color: "#f5b84b",
    beamAlpha: "rgba(245, 184, 75, 0.20)",
    lineAlpha: "rgba(245, 184, 75, 0.82)",
    maxRange: 185,
    baseDwell: 104,
  },
  v01: {
    id: "v01",
    name: "舰 01 雷达",
    role: "近距精跟",
    short: "V-01",
    color: "#4cd7a3",
    beamAlpha: "rgba(76, 215, 163, 0.20)",
    lineAlpha: "rgba(76, 215, 163, 0.84)",
    maxRange: 128,
    baseDwell: 86,
  },
  v02: {
    id: "v02",
    name: "舰 02 雷达",
    role: "协同确认",
    short: "V-02",
    color: "#57c7d4",
    beamAlpha: "rgba(87, 199, 212, 0.18)",
    lineAlpha: "rgba(87, 199, 212, 0.84)",
    maxRange: 132,
    baseDwell: 90,
  },
};

const targetMeta = {
  "T-01": { id: "T-01", label: "T-01", color: "#ff5a68", kind: "red" },
  "T-02": { id: "T-02", label: "T-02", color: "#5c8dff", kind: "blue" },
};

const routes = {
  ships: {
    v01: {
      period: 92,
      points: [
        { x: -28, y: -10 },
        { x: -12, y: -2 },
        { x: 8, y: 5 },
        { x: 30, y: 10 },
        { x: 44, y: 22 },
        { x: 22, y: 12 },
        { x: -4, y: 0 },
      ],
    },
    v02: {
      period: 96,
      points: [
        { x: 26, y: -26 },
        { x: 36, y: -12 },
        { x: 48, y: 6 },
        { x: 62, y: 24 },
        { x: 54, y: 38 },
        { x: 34, y: 18 },
        { x: 20, y: -8 },
      ],
    },
  },
  targets: {
    "T-01": {
      period: 110,
      points: [
        { x: 76, y: 45 },
        { x: 62, y: 38 },
        { x: 42, y: 26 },
        { x: 18, y: 9 },
        { x: -8, y: -8 },
        { x: 12, y: 4 },
        { x: 45, y: 24 },
        { x: 70, y: 40 },
      ],
    },
    "T-02": {
      period: 120,
      points: [
        { x: 54, y: 47 },
        { x: 30, y: 43 },
        { x: 6, y: 38 },
        { x: -22, y: 30 },
        { x: -42, y: 20 },
        { x: -18, y: 24 },
        { x: 18, y: 36 },
      ],
    },
  },
};

const cardBindings = {
  shore: {
    title: ui.cardShoreTitle,
    beam: ui.cardShoreBeam,
    confidence: ui.cardShoreConfidence,
    bar: ui.cardShoreBar,
  },
  v01: {
    title: ui.cardV01Title,
    beam: ui.cardV01Beam,
    confidence: ui.cardV01Confidence,
    bar: ui.cardV01Bar,
  },
  v02: {
    title: ui.cardV02Title,
    beam: ui.cardV02Beam,
    confidence: ui.cardV02Confidence,
    bar: ui.cardV02Bar,
  },
};

const radarOrder = ["shore", "v01", "v02"];

const radarLabels = {
  shore: "岸基",
  v01: "舰01",
  v02: "舰02",
};

const state = {
  running: true,
  selectedRadarId: "shore",
  targetId: "T-01",
  simTime: 0,
  cycle: 0,
  loopAccumulator: 0,
  loopStep: 0,
  lastFrameTime: 0,
  speed: 1,
  clutter: 0.26,
  uncertainty: 0.22,
  showBeams: true,
  showTracks: true,
  showGates: true,
  resizeToken: 0,
  logs: [],
  previousFusionScore: 0,
  backendRequired: true,
  backendConnected: false,
  backendSocket: null,
  backendReconnectTimer: 0,
  backendLastMessageAt: 0,
  backendEverConnected: false,
  backendTracks: {},
  backendModel: null,
  backendLlm: null,
};

const ships = {
  v01: {
    id: "V-01",
    label: "V-01",
    color: radarMeta.v01.color,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    heading: 0,
    trail: [],
  },
  v02: {
    id: "V-02",
    label: "V-02",
    color: radarMeta.v02.color,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    heading: 0,
    trail: [],
  },
};

const targets = {
  "T-01": {
    id: "T-01",
    label: "T-01",
    color: targetMeta["T-01"].color,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    heading: 0,
    trail: [],
  },
  "T-02": {
    id: "T-02",
    label: "T-02",
    color: targetMeta["T-02"].color,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    heading: 0,
    trail: [],
  },
};

const radars = {
  shore: createRadar("shore", { x: -58, y: -35 }),
  v01: createRadar("v01", { x: 0, y: 0 }),
  v02: createRadar("v02", { x: 0, y: 0 }),
};

function createRadar(id, origin) {
  const meta = radarMeta[id];
  return {
    id,
    name: meta.name,
    role: meta.role,
    origin: { x: origin.x, y: origin.y },
    beamAzimuth: id === "shore" ? 34 : id === "v01" ? 54 : 120,
    beamWidth: id === "shore" ? 42 : 30,
    dwell: meta.baseDwell,
    rangeGate: meta.maxRange * 0.62,
    iq: makeMatrix(48, 64, 0),
    snr: 0,
    perception: {
      bearing: 0,
      range: 0,
      confidence: 0,
      doppler: 0,
      angularError: 0,
      estimate: { x: origin.x, y: origin.y },
    },
    reasoning: {
      intent: "search",
      threat: "low",
      action: "search",
      narrative: "初始化闭环状态。",
    },
    controller: {
      desiredAzimuth: 0,
      desiredWidth: 30,
      desiredDwell: meta.baseDwell,
      constraint: "nominal",
    },
    handoffUntil: 0,
  };
}

function makeMatrix(rows, cols, value) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => value));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function numberOr(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function wrap360(angle) {
  return ((angle % 360) + 360) % 360;
}

function angleDiff(a, b) {
  let diff = ((b - a + 540) % 360) - 180;
  return diff;
}

function smoothAngle(current, target, factor, maxStep) {
  const delta = clamp(angleDiff(current, target), -maxStep, maxStep);
  return wrap360(current + delta * factor);
}

function blendAngle(from, to, factor) {
  return wrap360(from + angleDiff(from, to) * factor);
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function trackingIntensityFor(radarId, beamWidth, dwell, confidence = 0) {
  const minWidth = radarId === "shore" ? 12 : 8;
  const maxWidth = radarId === "shore" ? 62 : 48;
  const widthFocus = 1 - clamp((beamWidth - minWidth) / (maxWidth - minWidth), 0, 1);
  const dwellFocus = clamp((dwell - 45) / 195, 0, 1);
  return clamp(0.38 * widthFocus + 0.42 * dwellFocus + 0.2 * confidence, 0, 1);
}

function bearingDeg(a, b) {
  return wrap360((Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI);
}

function radialVelocity(observer, target) {
  const dx = target.x - observer.x;
  const dy = target.y - observer.y;
  const range = Math.hypot(dx, dy) || 1;
  const tx = target.vx || 0;
  const ty = target.vy || 0;
  const ox = observer.vx || 0;
  const oy = observer.vy || 0;
  return ((tx - ox) * dx + (ty - oy) * dy) / range;
}

function noise(seed, amplitude) {
  const v =
    Math.sin(seed * 12.9898 + state.simTime * 7.233) * 43758.5453 +
    Math.sin(seed * 78.233 + state.cycle * 0.73) * 21413.159;
  return (v - Math.floor(v) - 0.5) * 2 * amplitude;
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function followRoute(route, t) {
  const count = route.points.length;
  const wrapped = ((t % route.period) + route.period) % route.period;
  const scaled = (wrapped / route.period) * count;
  const index = Math.floor(scaled) % count;
  const nextIndex = (index + 1) % count;
  const local = smoothstep(scaled - index);
  return {
    x: lerp(route.points[index].x, route.points[nextIndex].x, local),
    y: lerp(route.points[index].y, route.points[nextIndex].y, local),
  };
}

function targetPosition(id, t) {
  return followRoute(routes.targets[id], t);
}

function shipPosition(id, t) {
  return followRoute(routes.ships[id], t);
}

function pointFromPolar(origin, bearing, range) {
  const radians = (bearing * Math.PI) / 180;
  return {
    x: origin.x + Math.cos(radians) * range,
    y: origin.y + Math.sin(radians) * range,
  };
}

function pushTrail(entity, maxLength) {
  const latest = entity.trail[entity.trail.length - 1];
  if (!latest || Math.hypot(latest.x - entity.x, latest.y - entity.y) > 0.18) {
    entity.trail.push({ x: entity.x, y: entity.y });
  }
  while (entity.trail.length > maxLength) {
    entity.trail.shift();
  }
}

function updateKinematics(dt) {
  for (const ship of Object.values(ships)) {
    const next = shipPosition(ship.id === "V-01" ? "v01" : "v02", state.simTime);
    const hasPrevious = ship.trail.length > 0;
    ship.vx = hasPrevious && dt > 0 ? (next.x - ship.x) / dt : 0;
    ship.vy = hasPrevious && dt > 0 ? (next.y - ship.y) / dt : 0;
    ship.heading = wrap360((Math.atan2(ship.vy, ship.vx) * 180) / Math.PI);
    ship.x = next.x;
    ship.y = next.y;
    pushTrail(ship, 180);
  }

  for (const target of Object.values(targets)) {
    const next = targetPosition(target.id, state.simTime);
    const hasPrevious = target.trail.length > 0;
    target.vx = hasPrevious && dt > 0 ? (next.x - target.x) / dt : 0;
    target.vy = hasPrevious && dt > 0 ? (next.y - target.y) / dt : 0;
    target.heading = wrap360((Math.atan2(target.vy, target.vx) * 180) / Math.PI);
    target.x = next.x;
    target.y = next.y;
    pushTrail(target, 220);
  }

  radars.v01.origin.x = ships.v01.x;
  radars.v01.origin.y = ships.v01.y;
  radars.v02.origin.x = ships.v02.x;
  radars.v02.origin.y = ships.v02.y;
}

function generateIq(radar, target) {
  const rows = 48;
  const cols = 64;
  const matrix = makeMatrix(rows, cols, 0);
  const observer = radar.id === "v01" ? ships.v01 : radar.id === "v02" ? ships.v02 : { ...radar.origin, vx: 0, vy: 0 };
  const range = distance(radar.origin, target);
  const rv = radialVelocity(observer, target);
  const rangeIndex = clamp(Math.round((range / radarMeta[radar.id].maxRange) * (rows - 8)) + 3, 2, rows - 3);
  const dopplerIndex = clamp(Math.round(cols / 2 + rv * 5.2), 3, cols - 4);
  const angularError = Math.abs(angleDiff(radar.beamAzimuth, bearingDeg(radar.origin, target)));
  const designationAssist = radar.id === "shore" ? 0.16 : 0.26;
  const inBeam = clamp(
    designationAssist + (1 - designationAssist) * (1 - angularError / Math.max(8, radar.beamWidth * 0.72)),
    0,
    1,
  );
  const rangeLoss = clamp(1 - range / (radarMeta[radar.id].maxRange * 1.08), 0.04, 1);
  const focusGain = 0.42 + 0.86 * inBeam * rangeLoss;
  const clutter = state.clutter;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const wave = Math.sin(r * 0.45 + state.simTime * 1.8) * Math.cos(c * 0.31 - state.simTime * 1.1);
      const ridge = Math.exp(-Math.pow((r - rows * 0.58) / 8, 2)) * (0.05 + 0.08 * Math.sin(c * 0.4 + state.simTime));
      const background = clutter * (0.32 + 0.26 * wave + 0.22 * Math.sin((r + c) * 0.17 + state.simTime * 0.9));
      matrix[r][c] = clamp(background + ridge + 0.025 * noise(r * 97 + c * 13 + radar.id.length, 1), 0, 1);
    }
  }

  const paintPeak = (row, col, amp, sigmaR, sigmaC) => {
    for (let r = Math.max(0, row - 7); r <= Math.min(rows - 1, row + 7); r += 1) {
      for (let c = Math.max(0, col - 9); c <= Math.min(cols - 1, col + 9); c += 1) {
        const value = amp * Math.exp(-Math.pow((r - row) / sigmaR, 2) - Math.pow((c - col) / sigmaC, 2));
        matrix[r][c] = clamp(matrix[r][c] + value, 0, 1.25);
      }
    }
  };

  paintPeak(rangeIndex, dopplerIndex, focusGain, 2.8, 3.6);

  const secondaryTarget = targets[state.targetId === "T-01" ? "T-02" : "T-01"];
  const secondaryRange = distance(radar.origin, secondaryTarget);
  const secondaryBearing = bearingDeg(radar.origin, secondaryTarget);
  const secondaryRv = radialVelocity(observer, secondaryTarget);
  const secondaryAngle = Math.abs(angleDiff(radar.beamAzimuth, secondaryBearing));
  const secondaryVisible = clamp(1 - secondaryAngle / Math.max(14, radar.beamWidth * 1.4), 0, 1);
  const sr = clamp(Math.round((secondaryRange / radarMeta[radar.id].maxRange) * (rows - 8)) + 3, 2, rows - 3);
  const sd = clamp(Math.round(cols / 2 + secondaryRv * 5.2), 3, cols - 4);
  paintPeak(sr, sd, 0.25 * secondaryVisible, 3.6, 4.8);

  radar.iq = matrix;
  radar.snr = 7 + focusGain * 18 - clutter * 7 + noise(range * 1.7 + radar.id.length, 0.7);
  return { range, rv, angularError, peakRow: rangeIndex, peakCol: dopplerIndex };
}

function perceiveWithTransformer(radar, target, iqInfo) {
  const trueBearing = bearingDeg(radar.origin, target);
  const trueRange = iqInfo.range;
  const angularError = Math.abs(angleDiff(radar.beamAzimuth, trueBearing));
  const beamScore = clamp(1 - angularError / Math.max(7, radar.beamWidth), 0, 1);
  const rangeScore = clamp(1 - trueRange / radarMeta[radar.id].maxRange, 0, 1);
  const snrScore = clamp((radar.snr - 4) / 22, 0, 1);
  const cueScore = radar.id === "shore" ? 0.12 : 0.2;
  const confidence = clamp(
    0.22 + cueScore + 0.34 * beamScore + 0.24 * snrScore + 0.16 * rangeScore - state.uncertainty * 0.11,
    0.08,
    0.99,
  );
  const jitter = state.uncertainty * (1.35 - confidence);
  const bearing = wrap360(trueBearing + noise(trueRange + radar.beamWidth, 4.5 * jitter));
  const range = Math.max(0, trueRange + noise(trueBearing + radar.snr, 1.7 * jitter));

  radar.perception = {
    bearing,
    range,
    confidence,
    doppler: iqInfo.rv * 1000,
    angularError,
    estimate: pointFromPolar(radar.origin, bearing, range),
  };
}

function inferSituation(radar, target, fusionScore) {
  const shore = radars.shore.origin;
  const distToShore = distance(shore, target);
  const closingShore = -radialVelocity({ ...shore, vx: 0, vy: 0 }, target);
  const speed = Math.hypot(target.vx, target.vy);
  const confidence = radar.perception.confidence;
  let intent = "monitor";
  let threat = "low";
  let action = "maintain_track";

  if (target.id === "T-01") {
    if (distToShore < 55 && closingShore > 0.08) {
      intent = "intercept";
      threat = "high";
      action = "narrow_beam_high_dwell";
    } else if (closingShore > 0.02) {
      intent = "approach";
      threat = "elevated";
      action = "increase_tracking_rate";
    } else {
      intent = "loiter_patrol";
      threat = "guarded";
      action = "classify_and_shadow";
    }
  } else if (Math.abs(closingShore) < 0.025 && speed > 0.16) {
    intent = "benign_transit";
    threat = "low";
    action = "monitor";
  } else {
    intent = closingShore > 0 ? "approach" : "retreat";
    threat = closingShore > 0 ? "guarded" : "low";
    action = closingShore > 0 ? "increase_tracking_rate" : "monitor";
  }

  if (confidence < 0.42) {
    action = "expand_beam_reacquire";
  }

  const sourcePhrase =
    radar.id === "shore" ? "岸基广域链路" : radar.id === "v01" ? "舰 01 近距链路" : "舰 02 侧向链路";
  const targetPhrase = target.id === "T-01" ? "红色目标" : "蓝色目标";
  const threatPhrase = threat === "high" ? "高威胁" : threat === "elevated" ? "威胁升高" : threat === "guarded" ? "警戒" : "低威胁";

  radar.reasoning = {
    intent,
    threat,
    action,
    narrative: `${sourcePhrase}对${targetPhrase}保持闭环跟踪；三源融合一致性 ${fusionScore.toFixed(
      2,
    )}，当前判定为${threatPhrase}，执行 ${action}。`,
  };
}

function controlBeam(radar, dt) {
  const confidence = radar.perception.confidence;
  const threat = radar.reasoning.threat;
  const target = targets[state.targetId];
  const designationBearing = bearingDeg(radar.origin, target);
  const desiredAzimuth = blendAngle(radar.perception.bearing, designationBearing, radar.id === "shore" ? 0.16 : 0.24);
  const handoffActive = state.simTime < radar.handoffUntil;
  let desiredWidth = 38 - confidence * 22;
  let desiredDwell = radarMeta[radar.id].baseDwell + (1 - confidence) * 55;

  if (threat === "high") {
    desiredWidth -= 5;
    desiredDwell += 52;
  } else if (threat === "elevated") {
    desiredWidth -= 2;
    desiredDwell += 28;
  }

  if (radar.reasoning.action === "expand_beam_reacquire") {
    desiredWidth = 52;
    desiredDwell = 150;
  }

  if (handoffActive) {
    desiredWidth = Math.max(desiredWidth, radar.id === "shore" ? 46 : 38);
    desiredDwell += radar.id === "shore" ? 24 : 34;
  }

  const constrainedWidth = clamp(desiredWidth, radar.id === "shore" ? 12 : 8, radar.id === "shore" ? 62 : 48);
  const constrainedDwell = clamp(desiredDwell, 45, 240);
  const slewRate = handoffActive ? (radar.id === "shore" ? 34 : 46) : radar.id === "shore" ? 24 : 34;
  const maxStep = clamp(slewRate * Math.max(dt, 0.016), 0.18, 7.5);
  const desiredRangeGate = clamp(radar.perception.range + 18 + (1 - confidence) * 16, 18, radarMeta[radar.id].maxRange);
  const trackingIntensity = trackingIntensityFor(radar.id, constrainedWidth, constrainedDwell, confidence);

  radar.controller = {
    desiredAzimuth,
    desiredWidth: constrainedWidth,
    desiredDwell: constrainedDwell,
    desiredRangeGate,
    slewRate,
    maxSlewStep: maxStep,
    trackingIntensity,
    constraint:
      handoffActive
        ? "handoff_slew"
        : constrainedWidth !== desiredWidth || constrainedDwell !== desiredDwell
        ? "clamped"
        : confidence < 0.42
          ? "reacquire"
          : "nominal",
  };

  radar.beamAzimuth = smoothAngle(radar.beamAzimuth, desiredAzimuth, 1, maxStep);
  radar.beamWidth = lerp(radar.beamWidth, constrainedWidth, handoffActive ? 0.08 : 0.14);
  radar.dwell = lerp(radar.dwell, constrainedDwell, handoffActive ? 0.1 : 0.18);
  radar.rangeGate = lerp(radar.rangeGate, desiredRangeGate, 0.15);
}

function updateClosedLoop(dt) {
  updateKinematics(dt);
  const target = targets[state.targetId];
  const estimates = [];

  for (const radar of Object.values(radars)) {
    const iqInfo = generateIq(radar, target);
    perceiveWithTransformer(radar, target, iqInfo);
    estimates.push(radar.perception.estimate);
  }

  const centroid = estimates.reduce(
    (acc, estimate) => ({ x: acc.x + estimate.x / estimates.length, y: acc.y + estimate.y / estimates.length }),
    { x: 0, y: 0 },
  );
  const estimateError = estimates.reduce((acc, estimate) => acc + distance(estimate, target), 0) / estimates.length;
  const estimateSpread = estimates.reduce((acc, estimate) => acc + distance(estimate, centroid), 0) / estimates.length;
  const fusionScore = clamp(1 - (estimateError * 0.58 + estimateSpread * 0.42) / 18, 0, 1);
  state.previousFusionScore = fusionScore;

  for (const radar of Object.values(radars)) {
    inferSituation(radar, target, fusionScore);
    controlBeam(radar, dt);
  }
}

function resizeCanvases() {
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const sceneRect = sceneCanvas.getBoundingClientRect();
  sceneCanvas.width = Math.round(sceneRect.width * dpr);
  sceneCanvas.height = Math.round(sceneRect.height * dpr);
  sceneCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!iqCanvas || !iqCtx) {
    return;
  }
  const iqRect = iqCanvas.getBoundingClientRect();
  iqCanvas.width = Math.round(iqRect.width * dpr);
  iqCanvas.height = Math.round(iqRect.height * dpr);
  iqCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function mapPoint(point) {
  const rect = sceneCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const x = ((point.x - world.minX) / (world.maxX - world.minX)) * width;
  const y = height - ((point.y - world.minY) / (world.maxY - world.minY)) * height;
  return { x, y };
}

function worldLengthToPx(length) {
  const rect = sceneCanvas.getBoundingClientRect();
  return (length / (world.maxX - world.minX)) * rect.width;
}

function drawScene() {
  const rect = sceneCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  sceneCtx.clearRect(0, 0, width, height);

  drawSea(width, height);
  drawGrid(width, height);
  drawOperationalZones();

  if (state.showTracks) {
    for (const target of Object.values(targets)) {
      drawTrail(target.trail, target.color, target.id === state.targetId ? 0.88 : 0.44, 2);
    }
    drawTrail(ships.v01.trail, ships.v01.color, 0.62, 1.6);
    drawTrail(ships.v02.trail, ships.v02.color, 0.62, 1.6);
  }

  drawBackendPrediction();

  if (state.showGates) {
    for (const radar of Object.values(radars)) {
      drawRangeGate(radar);
    }
  }

  if (state.showBeams) {
    for (const radar of Object.values(radars)) {
      drawBeam(radar);
    }
  }

  drawPerceptionMarks();
  drawTrackLines();
  drawShoreRadar();
  drawShip(ships.v01);
  drawShip(ships.v02);
  drawTarget(targets["T-01"]);
  drawTarget(targets["T-02"]);
}

function drawSea(width, height) {
  const gradient = sceneCtx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#0b3335");
  gradient.addColorStop(0.48, "#092327");
  gradient.addColorStop(1, "#111820");
  sceneCtx.fillStyle = gradient;
  sceneCtx.fillRect(0, 0, width, height);

  sceneCtx.save();
  sceneCtx.globalAlpha = 0.12;
  sceneCtx.strokeStyle = "#78c7bc";
  sceneCtx.lineWidth = 1;
  for (let y = -20; y < height + 80; y += 34) {
    sceneCtx.beginPath();
    for (let x = -30; x < width + 30; x += 18) {
      const waveY = y + Math.sin(x * 0.02 + state.simTime * 0.8) * 3;
      if (x === -30) {
        sceneCtx.moveTo(x, waveY);
      } else {
        sceneCtx.lineTo(x, waveY);
      }
    }
    sceneCtx.stroke();
  }
  sceneCtx.restore();

  const shoreTop = mapPoint({ x: world.minX, y: world.maxY }).y;
  const shoreBottom = mapPoint({ x: -62, y: world.minY }).y;
  const shoreRight = mapPoint({ x: -62, y: 0 }).x;
  const landGradient = sceneCtx.createLinearGradient(0, 0, shoreRight, 0);
  landGradient.addColorStop(0, "#313024");
  landGradient.addColorStop(1, "rgba(49, 48, 36, 0.08)");
  sceneCtx.fillStyle = landGradient;
  sceneCtx.fillRect(0, shoreTop, shoreRight, shoreBottom);
}

function drawGrid(width, height) {
  sceneCtx.save();
  sceneCtx.strokeStyle = "rgba(187, 218, 210, 0.13)";
  sceneCtx.lineWidth = 1;
  sceneCtx.font = "11px Segoe UI, sans-serif";
  sceneCtx.fillStyle = "rgba(198, 215, 209, 0.55)";
  for (let x = -60; x <= 90; x += 20) {
    const a = mapPoint({ x, y: world.minY });
    sceneCtx.beginPath();
    sceneCtx.moveTo(a.x, 0);
    sceneCtx.lineTo(a.x, height);
    sceneCtx.stroke();
    sceneCtx.fillText(`${x}`, a.x + 4, height - 8);
  }
  for (let y = -40; y <= 60; y += 20) {
    const a = mapPoint({ x: world.minX, y });
    sceneCtx.beginPath();
    sceneCtx.moveTo(0, a.y);
    sceneCtx.lineTo(width, a.y);
    sceneCtx.stroke();
    sceneCtx.fillText(`${y}`, 8, a.y - 4);
  }
  sceneCtx.restore();
}

function drawOperationalZones() {
  const shore = radars.shore.origin;
  const center = mapPoint(shore);
  const rings = [28, 52, 78];
  sceneCtx.save();
  for (const [index, radius] of rings.entries()) {
    sceneCtx.beginPath();
    sceneCtx.arc(center.x, center.y, worldLengthToPx(radius), 0, Math.PI * 2);
    sceneCtx.strokeStyle = index === 0 ? "rgba(255, 90, 104, 0.28)" : "rgba(245, 184, 75, 0.18)";
    sceneCtx.setLineDash(index === 0 ? [] : [7, 8]);
    sceneCtx.lineWidth = index === 0 ? 1.5 : 1;
    sceneCtx.stroke();
  }
  sceneCtx.restore();
}

function drawTrail(points, color, alpha, width) {
  if (points.length < 2) {
    return;
  }
  sceneCtx.save();
  sceneCtx.strokeStyle = color;
  sceneCtx.globalAlpha = alpha;
  sceneCtx.lineWidth = width;
  sceneCtx.beginPath();
  points.forEach((point, index) => {
    const p = mapPoint(point);
    if (index === 0) {
      sceneCtx.moveTo(p.x, p.y);
    } else {
      sceneCtx.lineTo(p.x, p.y);
    }
  });
  sceneCtx.stroke();
  sceneCtx.restore();
}

function drawBackendPrediction() {
  if (!state.backendConnected || !state.backendTracks[state.targetId]) {
    return;
  }
  const track = state.backendTracks[state.targetId];
  const prediction = track.prediction || [];
  if (prediction.length < 2) {
    return;
  }
  const color = targetMeta[state.targetId]?.color || "#edf7f2";
  sceneCtx.save();
  sceneCtx.strokeStyle = color;
  sceneCtx.fillStyle = color;
  sceneCtx.globalAlpha = 0.86;
  sceneCtx.lineWidth = 2.4;
  sceneCtx.setLineDash([10, 7]);
  sceneCtx.beginPath();
  prediction.forEach((stateRow, index) => {
    const p = mapPoint({ x: stateRow[0], y: stateRow[1] });
    if (index === 0) {
      sceneCtx.moveTo(p.x, p.y);
    } else {
      sceneCtx.lineTo(p.x, p.y);
    }
  });
  sceneCtx.stroke();
  sceneCtx.setLineDash([]);
  prediction.forEach((stateRow, index) => {
    if (index % 2 !== 0 && index !== prediction.length - 1) {
      return;
    }
    const p = mapPoint({ x: stateRow[0], y: stateRow[1] });
    sceneCtx.beginPath();
    sceneCtx.arc(p.x, p.y, index === prediction.length - 1 ? 5 : 3, 0, Math.PI * 2);
    sceneCtx.fill();
  });
  sceneCtx.restore();
}

function drawRangeGate(radar) {
  const center = mapPoint(radar.origin);
  sceneCtx.save();
  sceneCtx.beginPath();
  sceneCtx.arc(center.x, center.y, worldLengthToPx(radar.rangeGate), 0, Math.PI * 2);
  sceneCtx.setLineDash([4, 8]);
  sceneCtx.strokeStyle = radarMeta[radar.id].lineAlpha;
  sceneCtx.globalAlpha = 0.28;
  sceneCtx.lineWidth = 1;
  sceneCtx.stroke();
  sceneCtx.restore();
}

function drawBeam(radar) {
  const origin = mapPoint(radar.origin);
  const radius = worldLengthToPx(radar.rangeGate);
  const start = ((radar.beamAzimuth - radar.beamWidth / 2) * Math.PI) / 180;
  const end = ((radar.beamAzimuth + radar.beamWidth / 2) * Math.PI) / 180;

  sceneCtx.save();
  sceneCtx.beginPath();
  sceneCtx.moveTo(origin.x, origin.y);
  sceneCtx.arc(origin.x, origin.y, radius, -end, -start, false);
  sceneCtx.closePath();
  sceneCtx.fillStyle = radarMeta[radar.id].beamAlpha;
  sceneCtx.fill();
  sceneCtx.strokeStyle = radarMeta[radar.id].lineAlpha;
  sceneCtx.globalAlpha = 0.82;
  sceneCtx.lineWidth = radar.id === state.selectedRadarId ? 2.2 : 1.35;
  sceneCtx.stroke();

  const centerAngle = (-radar.beamAzimuth * Math.PI) / 180;
  sceneCtx.beginPath();
  sceneCtx.moveTo(origin.x, origin.y);
  sceneCtx.lineTo(origin.x + Math.cos(centerAngle) * radius, origin.y + Math.sin(centerAngle) * radius);
  sceneCtx.setLineDash([8, 7]);
  sceneCtx.stroke();
  sceneCtx.restore();
}

function drawTrackLines() {
  const target = targets[state.targetId];
  const targetPoint = mapPoint(target);
  sceneCtx.save();
  sceneCtx.setLineDash([5, 6]);
  for (const radar of Object.values(radars)) {
    const origin = mapPoint(radar.origin);
    sceneCtx.beginPath();
    sceneCtx.moveTo(origin.x, origin.y);
    sceneCtx.lineTo(targetPoint.x, targetPoint.y);
    sceneCtx.strokeStyle = radarMeta[radar.id].lineAlpha;
    sceneCtx.globalAlpha = 0.44 + radar.perception.confidence * 0.35;
    sceneCtx.lineWidth = radar.id === state.selectedRadarId ? 1.8 : 1;
    sceneCtx.stroke();
  }
  sceneCtx.restore();
}

function drawPerceptionMarks() {
  sceneCtx.save();
  for (const radar of Object.values(radars)) {
    const estimate = radar.perception.estimate;
    if (!estimate) {
      continue;
    }
    const p = mapPoint(estimate);
    const size = 5 + radar.perception.confidence * 5;
    sceneCtx.strokeStyle = radarMeta[radar.id].lineAlpha;
    sceneCtx.fillStyle = "rgba(7, 19, 19, 0.72)";
    sceneCtx.lineWidth = radar.id === state.selectedRadarId ? 2 : 1.3;
    sceneCtx.beginPath();
    sceneCtx.arc(p.x, p.y, size, 0, Math.PI * 2);
    sceneCtx.fill();
    sceneCtx.stroke();
    sceneCtx.beginPath();
    sceneCtx.moveTo(p.x - size - 3, p.y);
    sceneCtx.lineTo(p.x + size + 3, p.y);
    sceneCtx.moveTo(p.x, p.y - size - 3);
    sceneCtx.lineTo(p.x, p.y + size + 3);
    sceneCtx.globalAlpha = 0.74;
    sceneCtx.stroke();
    sceneCtx.globalAlpha = 1;
  }
  sceneCtx.restore();
}

function drawShoreRadar() {
  const radar = radars.shore;
  const p = mapPoint(radar.origin);
  sceneCtx.save();
  sceneCtx.translate(p.x, p.y);
  sceneCtx.fillStyle = radarMeta.shore.color;
  sceneCtx.strokeStyle = "#fff2cf";
  sceneCtx.lineWidth = 1.4;
  sceneCtx.beginPath();
  sceneCtx.moveTo(0, -15);
  sceneCtx.lineTo(13, 12);
  sceneCtx.lineTo(-13, 12);
  sceneCtx.closePath();
  sceneCtx.fill();
  sceneCtx.stroke();
  sceneCtx.fillStyle = "#071313";
  sceneCtx.fillRect(-5, 2, 10, 10);
  sceneCtx.restore();
  drawLabel(p, "岸基", radarMeta.shore.color, -18);
}

function drawShip(ship) {
  const p = mapPoint(ship);
  sceneCtx.save();
  sceneCtx.translate(p.x, p.y);
  sceneCtx.rotate((-ship.heading * Math.PI) / 180);
  sceneCtx.fillStyle = ship.color;
  sceneCtx.strokeStyle = "rgba(237, 247, 242, 0.9)";
  sceneCtx.lineWidth = 1.2;
  sceneCtx.beginPath();
  sceneCtx.moveTo(15, 0);
  sceneCtx.lineTo(5, 8);
  sceneCtx.lineTo(-13, 6);
  sceneCtx.lineTo(-13, -6);
  sceneCtx.lineTo(5, -8);
  sceneCtx.closePath();
  sceneCtx.fill();
  sceneCtx.stroke();
  sceneCtx.fillStyle = "rgba(7, 19, 19, 0.62)";
  sceneCtx.fillRect(-3, -4, 8, 8);
  sceneCtx.restore();
  drawLabel(p, ship.label, ship.color, -18);
}

function drawTarget(target) {
  const p = mapPoint(target);
  const selected = target.id === state.targetId;
  sceneCtx.save();
  sceneCtx.translate(p.x, p.y);
  sceneCtx.rotate((-target.heading * Math.PI) / 180);
  sceneCtx.fillStyle = target.color;
  sceneCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  sceneCtx.lineWidth = selected ? 2 : 1.2;
  sceneCtx.beginPath();
  sceneCtx.moveTo(13, 0);
  sceneCtx.lineTo(-8, 8);
  sceneCtx.lineTo(-4, 0);
  sceneCtx.lineTo(-8, -8);
  sceneCtx.closePath();
  sceneCtx.fill();
  sceneCtx.stroke();
  sceneCtx.restore();

  if (selected) {
    sceneCtx.save();
    sceneCtx.strokeStyle = target.color;
    sceneCtx.lineWidth = 1.4;
    sceneCtx.globalAlpha = 0.8;
    sceneCtx.beginPath();
    sceneCtx.arc(p.x, p.y, 16 + 3 * Math.sin(state.simTime * 3), 0, Math.PI * 2);
    sceneCtx.stroke();
    sceneCtx.restore();
  }
  drawLabel(p, target.label, target.color, 19);
}

function drawLabel(point, text, color, offsetY) {
  sceneCtx.save();
  sceneCtx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
  sceneCtx.textAlign = "center";
  sceneCtx.textBaseline = "middle";
  const metrics = sceneCtx.measureText(text);
  const width = Math.max(38, metrics.width + 12);
  const x = point.x;
  const y = point.y + offsetY;
  sceneCtx.fillStyle = "rgba(7, 19, 19, 0.78)";
  sceneCtx.strokeStyle = color;
  sceneCtx.lineWidth = 1;
  roundRect(sceneCtx, x - width / 2, y - 10, width, 20, 6);
  sceneCtx.fill();
  sceneCtx.stroke();
  sceneCtx.fillStyle = "#edf7f2";
  sceneCtx.fillText(text, x, y + 1);
  sceneCtx.restore();
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawIq() {
  if (!iqCanvas || !iqCtx) {
    return;
  }
  const radar = radars[state.selectedRadarId];
  const rect = iqCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  iqCtx.clearRect(0, 0, width, height);
  iqCtx.fillStyle = "#050b0d";
  iqCtx.fillRect(0, 0, width, height);

  const rows = radar.iq.length;
  const cols = radar.iq[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const v = clamp(radar.iq[r][c], 0, 1);
      iqCtx.fillStyle = heatColor(v);
      iqCtx.fillRect(c * cellW, height - (r + 1) * cellH, Math.ceil(cellW) + 0.2, Math.ceil(cellH) + 0.2);
    }
  }

  iqCtx.save();
  iqCtx.strokeStyle = "rgba(237, 247, 242, 0.32)";
  iqCtx.lineWidth = 1;
  iqCtx.beginPath();
  iqCtx.moveTo(width / 2, 0);
  iqCtx.lineTo(width / 2, height);
  iqCtx.moveTo(0, height * 0.5);
  iqCtx.lineTo(width, height * 0.5);
  iqCtx.stroke();
  iqCtx.restore();
}

function heatColor(value) {
  const v = clamp(value, 0, 1);
  const stops = [
    [0.0, [5, 13, 14]],
    [0.24, [21, 55, 58]],
    [0.5, [42, 128, 119]],
    [0.74, [245, 184, 75]],
    [1.0, [255, 90, 104]],
  ];
  for (let i = 1; i < stops.length; i += 1) {
    if (v <= stops[i][0]) {
      const [p0, c0] = stops[i - 1];
      const [p1, c1] = stops[i];
      const t = (v - p0) / (p1 - p0);
      const r = Math.round(lerp(c0[0], c1[0], t));
      const g = Math.round(lerp(c0[1], c1[1], t));
      const b = Math.round(lerp(c0[2], c1[2], t));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return "rgb(255, 90, 104)";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function controllerValueHtml(label, value) {
  return `<span><small>${label}</small><strong>${value}</strong></span>`;
}

function renderControllerImpact() {
  if (!ui.controllerImpactList) {
    return;
  }

  ui.controllerImpactList.innerHTML = radarOrder
    .map((id) => {
      const radar = radars[id];
      const controller = radar.controller || {};
      const confidence = numberOr(radar.perception?.confidence, 0);
      const desiredAzimuth = numberOr(controller.desiredAzimuth, radar.beamAzimuth);
      const desiredWidth = numberOr(controller.desiredWidth, radar.beamWidth);
      const desiredDwell = numberOr(controller.desiredDwell, radar.dwell);
      const desiredRangeGate = numberOr(
        controller.desiredRangeGate ?? controller.targetRangeGate ?? controller.rangeGate,
        radar.rangeGate,
      );
      const trackingIntensity = numberOr(
        controller.trackingIntensity,
        trackingIntensityFor(id, desiredWidth, desiredDwell, confidence),
      );
      const slewRate = numberOr(controller.slewRate, id === "shore" ? 24 : 34);
      const mode = escapeHtml(controller.mode || radar.reasoning?.action || "monitor");
      const constraint = escapeHtml(controller.constraint || "nominal");

      return `<article class="controller-impact-row" data-radar="${id}">
        <div class="controller-impact-head">
          <span class="controller-impact-name"><i></i>${radarLabels[id]}</span>
          <span class="controller-impact-mode">${mode} / ${constraint}</span>
        </div>
        <div class="controller-impact-values">
          ${controllerValueHtml("波束", `${radar.beamAzimuth.toFixed(1)}&deg; / ${radar.beamWidth.toFixed(1)}&deg;`)}
          ${controllerValueHtml("目标", `${desiredAzimuth.toFixed(1)}&deg; / ${desiredWidth.toFixed(1)}&deg;`)}
          ${controllerValueHtml("强度", `${Math.round(trackingIntensity * 100)}%`)}
          ${controllerValueHtml("速度", `${slewRate.toFixed(1)}&deg;/s`)}
          ${controllerValueHtml("驻留", `${Math.round(radar.dwell)}&rarr;${Math.round(desiredDwell)} ms`)}
          ${controllerValueHtml("距离门", `${desiredRangeGate.toFixed(1)} km`)}
        </div>
      </article>`;
    })
    .join("");
}

function updateUi() {
  const radar = radars[state.selectedRadarId];
  const handoffActive = Object.values(radars).some((item) => state.simTime < item.handoffUntil);
  ui.cycleValue.textContent = String(state.cycle).padStart(4, "0");
  ui.timeValue.textContent = `${state.simTime.toFixed(1)}s`;
  ui.loopRateValue.textContent = `${(4 * state.speed).toFixed(1)}Hz`;
  ui.speedValue.textContent = `${state.speed.toFixed(2)}x`;
  ui.clutterValue.textContent = state.clutter.toFixed(2);
  ui.uncertaintyValue.textContent = state.uncertainty.toFixed(2);
  ui.mapFocus.textContent = `${state.backendConnected ? "BACKEND" : state.backendRequired ? "DISCONNECTED" : handoffActive ? "HANDOFF" : "TRACK"} ${state.targetId}`;

  ui.runStatus.textContent = state.backendConnected ? "BACKEND" : state.backendRequired ? "DISCONNECTED" : state.running ? "RUNNING" : "PAUSED";
  ui.runStatus.classList.toggle("paused", !state.running || (state.backendRequired && !state.backendConnected));
  ui.backendStatusValue.textContent = state.backendConnected
    ? state.backendModel?.modelType || "CONNECTED"
    : "LOCAL";
  ui.llmStatusValue.textContent = state.backendConnected
    ? state.backendLlm?.configured
      ? state.backendLlm.model || state.backendLlm.provider
      : "rule"
    : "fallback";
  ui.backendOverlay.classList.toggle("is-hidden", state.backendConnected);
  playPauseIcon.textContent = state.running ? "||" : "▶";
  playPauseText.textContent = state.running ? "暂停" : "运行";
  playPauseBtn.title = state.running ? "暂停" : "运行";

  ui.selectedRadarName.textContent = radar.name;
  ui.selectedRadarRole.textContent = radar.role;
  ui.bearingMetric.textContent = `${radar.beamAzimuth.toFixed(1)}°`;
  ui.rangeMetric.textContent = `${radar.perception.range.toFixed(1)} km`;
  ui.confidenceMetric.textContent = radar.perception.confidence.toFixed(2);
  ui.snrMetric.textContent = `${radar.snr.toFixed(1)} dB`;
  ui.beamWidthMetric.textContent = `${radar.beamWidth.toFixed(1)}°`;
  ui.dwellMetric.textContent = `${Math.round(radar.dwell)} ms`;
  if (ui.iqSource) {
    ui.iqSource.textContent = radarMeta[radar.id].short;
  }
  ui.estimatedBearing.textContent = `${radar.perception.bearing.toFixed(1)}°`;
  ui.estimatedRange.textContent = `${radar.perception.range.toFixed(1)} km`;
  ui.dopplerMetric.textContent = `${radar.perception.doppler.toFixed(1)} m/s`;
  ui.fusionMetric.textContent = state.previousFusionScore.toFixed(2);
  ui.intentMetric.textContent = radar.reasoning.intent;
  ui.threatMetric.textContent = radar.reasoning.threat;
  ui.llmNarrative.textContent = radar.reasoning.narrative;
  ui.controlMetric.textContent = `${radar.reasoning.action} / ${radar.controller.constraint}`;

  const loopItems = Array.from(ui.loopList.querySelectorAll("li"));
  loopItems.forEach((item, index) => {
    item.classList.toggle("is-active", index === state.loopStep);
  });

  for (const [id, bindings] of Object.entries(cardBindings)) {
    const item = radars[id];
    bindings.title.textContent = `目标 ${state.targetId}`;
    bindings.beam.textContent = `${Math.round(item.beamAzimuth).toString().padStart(3, "0")}° / ${Math.round(
      item.beamWidth,
    )
      .toString()
      .padStart(2, "0")}°`;
    bindings.confidence.textContent = item.perception.confidence.toFixed(2);
    bindings.bar.style.width = `${Math.round(item.perception.confidence * 100)}%`;
  }

  renderControllerImpact();
  ui.eventLog.innerHTML = state.logs.map((line) => `<div>${line}</div>`).join("");
}

function logEvent(message) {
  const text = `[${state.simTime.toFixed(1)}s] ${message}`;
  if (state.logs[0] !== text) {
    state.logs.unshift(text);
    state.logs = state.logs.slice(0, 3);
  }
}

function applyBackendSnapshot(snapshot) {
  state.backendConnected = true;
  state.backendEverConnected = true;
  state.backendLastMessageAt = performance.now();
  state.backendTracks = snapshot.tracks || {};
  state.backendModel = snapshot.model || null;
  state.backendLlm = snapshot.llm || null;
  state.simTime = Number(snapshot.simTime || 0);
  state.cycle = Number(snapshot.cycle || 0);
  state.speed = Number(snapshot.speed ?? state.speed);
  state.clutter = Number(snapshot.clutter ?? state.clutter);
  state.uncertainty = Number(snapshot.uncertainty ?? state.uncertainty);
  state.previousFusionScore = Number(snapshot.fusionScore ?? state.previousFusionScore);
  state.targetId = snapshot.selectedTargetId || state.targetId;
  state.logs = Array.isArray(snapshot.events) ? snapshot.events.slice(-4).reverse() : state.logs;

  syncTargetButtons();

  const backendShips = snapshot.entities?.ships || {};
  applyEntitySnapshot(ships.v01, backendShips.v01);
  applyEntitySnapshot(ships.v02, backendShips.v02);

  const backendTargets = snapshot.entities?.targets || {};
  applyEntitySnapshot(targets["T-01"], backendTargets["T-01"]);
  applyEntitySnapshot(targets["T-02"], backendTargets["T-02"]);

  const backendRadars = snapshot.radars || {};
  for (const id of Object.keys(radars)) {
    const incoming = backendRadars[id];
    if (!incoming) {
      continue;
    }
    const radar = radars[id];
    radar.origin.x = Number(incoming.origin?.x ?? radar.origin.x);
    radar.origin.y = Number(incoming.origin?.y ?? radar.origin.y);
    radar.beamAzimuth = Number(incoming.beamAzimuth ?? radar.beamAzimuth);
    radar.beamWidth = Number(incoming.beamWidth ?? radar.beamWidth);
    radar.dwell = Number(incoming.dwell ?? radar.dwell);
    radar.rangeGate = Number(incoming.rangeGate ?? radar.rangeGate);
    radar.snr = Number(incoming.snr ?? radar.snr);
    radar.perception = {
      bearing: Number(incoming.perception?.bearing ?? 0),
      range: Number(incoming.perception?.range ?? 0),
      confidence: Number(incoming.perception?.confidence ?? 0),
      doppler: Number(incoming.perception?.doppler ?? 0),
      angularError: radar.perception.angularError || 0,
      estimate: {
        x: Number(incoming.perception?.estimate?.x ?? radar.origin.x),
        y: Number(incoming.perception?.estimate?.y ?? radar.origin.y),
      },
    };
    radar.reasoning = incoming.reasoning || radar.reasoning;
    radar.controller = incoming.controller || radar.controller;
  }
}

function applyEntitySnapshot(entity, incoming) {
  if (!incoming) {
    return;
  }
  entity.x = Number(incoming.x ?? entity.x);
  entity.y = Number(incoming.y ?? entity.y);
  entity.vx = Number(incoming.vx ?? entity.vx);
  entity.vy = Number(incoming.vy ?? entity.vy);
  entity.heading = Number(incoming.heading ?? entity.heading);
  if (Array.isArray(incoming.trail)) {
    entity.trail = incoming.trail.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  }
}

function syncTargetButtons() {
  Array.from(targetTabs.querySelectorAll("button")).forEach((button) => {
    button.classList.toggle("is-active", button.dataset.target === state.targetId);
  });
}

function sendBackendCommand(command) {
  if (!state.backendSocket || state.backendSocket.readyState !== WebSocket.OPEN) {
    return;
  }
  state.backendSocket.send(JSON.stringify(command));
}

function connectBackend() {
  if (window.location.protocol === "file:" || state.backendSocket) {
    return;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
  state.backendSocket = socket;

  socket.addEventListener("open", () => {
    state.backendConnected = true;
    logEvent("backend closed-loop connected");
  });

  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "snapshot") {
        applyBackendSnapshot(message);
      }
    } catch (error) {
      console.warn("Bad backend message", error);
    }
  });

  socket.addEventListener("close", () => {
    state.backendConnected = false;
    state.backendSocket = null;
    window.clearTimeout(state.backendReconnectTimer);
    state.backendReconnectTimer = window.setTimeout(connectBackend, 1800);
  });

  socket.addEventListener("error", () => {
    socket.close();
  });
}

function alignRadarsToTarget() {
  const target = targets[state.targetId];
  for (const radar of Object.values(radars)) {
    const bearing = bearingDeg(radar.origin, target);
    const range = distance(radar.origin, target);
    radar.beamAzimuth = bearing;
    radar.beamWidth = radar.id === "shore" ? 34 : 26;
    radar.rangeGate = clamp(range + 22, 18, radarMeta[radar.id].maxRange);
    radar.controller.desiredAzimuth = bearing;
    radar.controller.constraint = "target_designated";
  }
}

function simulationTick(dt) {
  if (state.backendRequired && !state.backendConnected) {
    return;
  }
  if (state.backendConnected) {
    if (performance.now() - state.backendLastMessageAt > 15000) {
      state.backendConnected = false;
      logEvent("backend timeout, waiting for reconnect");
    }
    return;
  }
  if (state.backendEverConnected) {
    return;
  }
  if (!state.running) {
    return;
  }
  const scaledDt = dt * state.speed;
  state.simTime += scaledDt;
  state.loopAccumulator += scaledDt;

  updateClosedLoop(Math.max(scaledDt, 0.016));

  while (state.loopAccumulator >= 0.25) {
    state.loopAccumulator -= 0.25;
    state.cycle += 1;
    state.loopStep = (state.loopStep + 1) % 7;
    const selectedRadar = radars[state.selectedRadarId];
    if (state.cycle % 4 === 0) {
      logEvent(
        `${selectedRadar.name} ${state.targetId} conf=${selectedRadar.perception.confidence.toFixed(2)} beam=${Math.round(
          selectedRadar.beamAzimuth,
        )}°`,
      );
    }
  }
}

function animationFrame(timestamp) {
  if (!state.lastFrameTime) {
    state.lastFrameTime = timestamp;
  }
  const dt = clamp((timestamp - state.lastFrameTime) / 1000, 0, 0.06);
  state.lastFrameTime = timestamp;

  simulationTick(dt);
  drawScene();
  drawIq();
  updateUi();
  requestAnimationFrame(animationFrame);
}

function setSelectedRadar(id) {
  state.selectedRadarId = id;
  Array.from(radarTabs.querySelectorAll("button")).forEach((button) => {
    button.classList.toggle("is-active", button.dataset.radar === id);
  });
  logEvent(`${radarMeta[id].name} 数据视图`);
}

function setTarget(id) {
  if (state.targetId === id) {
    return;
  }
  state.targetId = id;
  Array.from(targetTabs.querySelectorAll("button")).forEach((button) => {
    button.classList.toggle("is-active", button.dataset.target === id);
  });
  if (state.backendConnected) {
    sendBackendCommand({ type: "set_target", targetId: id });
    logEvent(`backend target handoff ${id}`);
    return;
  }
  if (state.backendRequired) {
    logEvent("backend disconnected, target command pending");
    connectBackend();
    return;
  }
  for (const radar of Object.values(radars)) {
    radar.handoffUntil = state.simTime + (radar.id === "shore" ? 2.4 : 1.9);
    radar.controller.constraint = "handoff_slew";
  }
  logEvent(`跟踪目标切换为 ${id}`);
}

function resetSimulation() {
  state.simTime = 0;
  state.cycle = 0;
  state.loopAccumulator = 0;
  state.loopStep = 0;
  state.logs = [];
  for (const ship of Object.values(ships)) {
    ship.trail = [];
    ship.x = 0;
    ship.y = 0;
  }
  for (const target of Object.values(targets)) {
    target.trail = [];
    target.x = 0;
    target.y = 0;
  }
  radars.shore = Object.assign(radars.shore, createRadar("shore", { x: -58, y: -35 }));
  radars.v01 = Object.assign(radars.v01, createRadar("v01", { x: 0, y: 0 }));
  radars.v02 = Object.assign(radars.v02, createRadar("v02", { x: 0, y: 0 }));
  updateKinematics(0);
  alignRadarsToTarget();
  updateClosedLoop(0.1);
  logEvent("闭环仿真复位");
}

radarTabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-radar]");
  if (button) {
    setSelectedRadar(button.dataset.radar);
  }
});

targetTabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-target]");
  if (button) {
    setTarget(button.dataset.target);
  }
});

playPauseBtn.addEventListener("click", () => {
  state.running = !state.running;
  logEvent(state.running ? "闭环继续运行" : "闭环暂停");
});

resetBtn.addEventListener("click", () => {
  if (state.backendConnected) {
    sendBackendCommand({ type: "reset" });
    return;
  }
  if (state.backendRequired) {
    logEvent("backend disconnected, waiting for reconnect");
    connectBackend();
    return;
  }
  resetSimulation();
});

speedRange.addEventListener("input", () => {
  state.speed = Number(speedRange.value);
  sendBackendCommand({ type: "set_speed", speed: state.speed });
});

clutterRange.addEventListener("input", () => {
  state.clutter = Number(clutterRange.value);
  sendBackendCommand({ type: "set_clutter", clutter: state.clutter });
});

uncertaintyRange.addEventListener("input", () => {
  state.uncertainty = Number(uncertaintyRange.value);
  sendBackendCommand({ type: "set_uncertainty", uncertainty: state.uncertainty });
});

showBeamsInput.addEventListener("change", () => {
  state.showBeams = showBeamsInput.checked;
});

showTracksInput.addEventListener("change", () => {
  state.showTracks = showTracksInput.checked;
});

showGatesInput.addEventListener("change", () => {
  state.showGates = showGatesInput.checked;
});

window.addEventListener("resize", () => {
  window.clearTimeout(state.resizeToken);
  state.resizeToken = window.setTimeout(resizeCanvases, 80);
});

resizeCanvases();
resetSimulation();
connectBackend();
requestAnimationFrame(animationFrame);

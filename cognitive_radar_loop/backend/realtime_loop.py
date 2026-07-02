from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "cognitive_radar_loop"
CHECKPOINT = ROOT / "detect_intention" / "train_intention" / "runs" / "run_20260604_005306" / "best_by_intent.pt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_IDS = ("T-01", "T-02")
RADAR_IDS = ("shore", "v01", "v02")

INTENT_NAMES = {
    1: "benign_transit",
    2: "approach",
    3: "retreat",
    4: "loiter_patrol",
    5: "intercept",
}

THREAT_NAMES = {
    1: "low",
    2: "guarded",
    3: "elevated",
    4: "high",
}

ACTION_BY_INTENT = {
    "benign_transit": "monitor",
    "approach": "increase_tracking_rate",
    "retreat": "monitor",
    "loiter_patrol": "classify_and_shadow",
    "intercept": "alert_and_allocate_tracker",
}


ROUTES = {
    "ships": {
        "v01": {
            "period": 92.0,
            "points": [
                (-28.0, -10.0),
                (-12.0, -2.0),
                (8.0, 5.0),
                (30.0, 10.0),
                (44.0, 22.0),
                (22.0, 12.0),
                (-4.0, 0.0),
            ],
        },
        "v02": {
            "period": 96.0,
            "points": [
                (26.0, -26.0),
                (36.0, -12.0),
                (48.0, 6.0),
                (62.0, 24.0),
                (54.0, 38.0),
                (34.0, 18.0),
                (20.0, -8.0),
            ],
        },
    },
    "targets": {
        "T-01": {
            "period": 110.0,
            "points": [
                (76.0, 45.0),
                (62.0, 38.0),
                (42.0, 26.0),
                (18.0, 9.0),
                (-8.0, -8.0),
                (12.0, 4.0),
                (45.0, 24.0),
                (70.0, 40.0),
            ],
        },
        "T-02": {
            "period": 120.0,
            "points": [
                (54.0, 47.0),
                (30.0, 43.0),
                (6.0, 38.0),
                (-22.0, 30.0),
                (-42.0, 20.0),
                (-18.0, 24.0),
                (18.0, 36.0),
            ],
        },
    },
}


RADAR_META = {
    "shore": {"name": "shore", "max_range": 185.0, "base_dwell": 104.0, "slew_rate": 24.0},
    "v01": {"name": "V-01", "max_range": 128.0, "base_dwell": 86.0, "slew_rate": 34.0},
    "v02": {"name": "V-02", "max_range": 132.0, "base_dwell": 90.0, "slew_rate": 34.0},
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def tracking_intensity(radar_id: str, beam_width: float, dwell_ms: float, confidence: float = 0.0) -> float:
    min_width, max_width = (12.0, 62.0) if radar_id == "shore" else (8.0, 48.0)
    width_focus = 1.0 - clamp((beam_width - min_width) / (max_width - min_width), 0.0, 1.0)
    dwell_focus = clamp((dwell_ms - 45.0) / 195.0, 0.0, 1.0)
    return clamp(0.38 * width_focus + 0.42 * dwell_focus + 0.2 * confidence, 0.0, 1.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def wrap360(angle: float) -> float:
    return angle % 360.0


def angle_diff(a: float, b: float) -> float:
    return ((b - a + 540.0) % 360.0) - 180.0


def slew_angle(current: float, target: float, max_step: float) -> float:
    return wrap360(current + clamp(angle_diff(current, target), -max_step, max_step))


def bearing_deg(ax: float, ay: float, bx: float, by: float) -> float:
    return wrap360(math.degrees(math.atan2(by - ay, bx - ax)))


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def route_position(route: dict[str, Any], t: float) -> tuple[float, float]:
    points = route["points"]
    count = len(points)
    wrapped = t % float(route["period"])
    scaled = wrapped / float(route["period"]) * count
    index = int(math.floor(scaled)) % count
    nxt = (index + 1) % count
    local = smoothstep(scaled - index)
    return (
        lerp(points[index][0], points[nxt][0], local),
        lerp(points[index][1], points[nxt][1], local),
    )


def noise(seed: float, t: float, amplitude: float) -> float:
    value = math.sin(seed * 12.9898 + t * 7.233) * 43758.5453 + math.sin(seed * 78.233 + t * 0.73) * 21413.159
    return (value - math.floor(value) - 0.5) * 2.0 * amplitude


@dataclass
class Entity:
    id: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    trail: list[dict[str, float]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "heading": self.heading,
            "trail": self.trail[-180:],
        }


@dataclass
class Measurement:
    radar_id: str
    target_id: str
    range_m: float
    bearing_local_deg: float
    bearing_world_deg: float
    range_rate: float
    confidence: float
    x: float
    y: float
    doppler: float


@dataclass
class Track:
    target_id: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    initialized: bool = False
    history: list[list[float]] = field(default_factory=list)
    prediction: list[list[float]] = field(default_factory=list)
    intent: str = "monitor"
    threat: str = "low"
    intent_confidence: float = 0.0
    threat_confidence: float = 0.0
    objectness: float = 0.0
    llm_decision: dict[str, Any] = field(default_factory=dict)

    def append_history(self, max_len: int = 48) -> None:
        self.history.append([self.x, self.y, self.vx, self.vy])
        if len(self.history) > max_len:
            del self.history[: len(self.history) - max_len]

    def snapshot(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "state": {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy},
            "history": self.history[-32:],
            "prediction": self.prediction,
            "intent": self.intent,
            "threat": self.threat,
            "intent_confidence": self.intent_confidence,
            "threat_confidence": self.threat_confidence,
            "objectness": self.objectness,
            "llm_decision": self.llm_decision,
        }


@dataclass
class Radar:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    beam_azimuth: float = 0.0
    beam_width: float = 34.0
    dwell_ms: float = 90.0
    range_gate: float = 80.0
    snr: float = 18.0
    perception: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": {"x": self.x, "y": self.y},
            "heading": self.heading,
            "beamAzimuth": self.beam_azimuth,
            "beamWidth": self.beam_width,
            "dwell": self.dwell_ms,
            "rangeGate": self.range_gate,
            "snr": self.snr,
            "perception": self.perception,
            "reasoning": self.reasoning,
            "controller": self.controller,
        }


class TrackModelService:
    def __init__(self, checkpoint: Path = CHECKPOINT) -> None:
        self.checkpoint = checkpoint
        self.available = False
        self.error = ""
        self.tin = 24
        self.tout = 8
        self.max_targets = 4
        self.model_type = "constant_velocity"
        self.input_mode = "track"
        self.model: Any = None
        self.torch: Any = None
        self.state_mean: Any = None
        self.state_std: Any = None
        self._load()

    def _load(self) -> None:
        try:
            import torch

            from detect_intention.train_intention.config import STATE_MEAN, STATE_STD
            from detect_intention.train_intention.model import TrackIntentTransformer

            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            args = checkpoint.get("args", {})
            if checkpoint.get("input_mode") != "track":
                raise ValueError(f"checkpoint input_mode is {checkpoint.get('input_mode')!r}, expected 'track'")

            self.tin = int(args.get("tin", self.tin))
            self.tout = int(args.get("tout", self.tout))
            self.max_targets = int(args.get("max_targets", self.max_targets))
            model = TrackIntentTransformer(
                max_targets=self.max_targets,
                tin=self.tin,
                tout=self.tout,
                embed_dim=int(args.get("embed_dim", 128)),
                num_heads=int(args.get("num_heads", 4)),
                temporal_layers=int(args.get("temporal_layers", 2)),
                mlp_ratio=float(args.get("mlp_ratio", 2.0)),
                dropout=float(args.get("dropout", 0.1)),
                num_target_classes=4,
                num_intents=5,
                num_threats=4,
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            self.model = model
            self.torch = torch
            self.state_mean = torch.as_tensor(STATE_MEAN, dtype=torch.float32).view(1, 1, 1, 4)
            self.state_std = torch.as_tensor(STATE_STD, dtype=torch.float32).view(1, 1, 1, 4)
            self.model_type = checkpoint.get("model_type", "track_intent_transformer")
            self.input_mode = checkpoint.get("input_mode", "track")
            self.available = True
        except Exception as exc:
            self.error = repr(exc)
            self.available = False

    def infer(self, tracks: dict[str, Track]) -> None:
        if not self.available:
            self._constant_velocity(tracks)
            return

        if not all(len(tracks[target_id].history) >= self.tin for target_id in TARGET_IDS):
            self._constant_velocity(tracks)
            return

        torch = self.torch
        assert torch is not None

        tensor = torch.zeros((1, self.max_targets, self.tin, 4), dtype=torch.float32)
        for slot, target_id in enumerate(TARGET_IDS):
            window = tracks[target_id].history[-self.tin :]
            tensor[0, slot] = torch.as_tensor(window, dtype=torch.float32)

        normalized = (tensor - self.state_mean) / self.state_std
        with torch.no_grad():
            outputs = self.model(normalized)
            state_pred = outputs["state_pred"] * self.state_std + self.state_mean
            intent_prob = torch.softmax(outputs["intent_logits"], dim=-1)
            threat_prob = torch.softmax(outputs["threat_logits"], dim=-1)
            objectness = torch.sigmoid(outputs["objectness_logits"])

        state_np = state_pred[0].cpu().numpy()
        intent_np = intent_prob[0].cpu().numpy()
        threat_np = threat_prob[0].cpu().numpy()
        obj_np = objectness[0].cpu().numpy()

        for slot, target_id in enumerate(TARGET_IDS):
            track = tracks[target_id]
            track.prediction = [[float(v) for v in row] for row in state_np[slot].tolist()]
            intent_idx = int(intent_np[slot].argmax()) + 1
            threat_idx = int(threat_np[slot].argmax()) + 1
            track.intent = INTENT_NAMES.get(intent_idx, "monitor")
            track.threat = THREAT_NAMES.get(threat_idx, "low")
            track.intent_confidence = float(intent_np[slot, intent_idx - 1])
            track.threat_confidence = float(threat_np[slot, threat_idx - 1])
            track.objectness = float(obj_np[slot])

    def _constant_velocity(self, tracks: dict[str, Track]) -> None:
        for track in tracks.values():
            track.prediction = [
                [track.x + track.vx * i * 0.4, track.y + track.vy * i * 0.4, track.vx, track.vy]
                for i in range(1, self.tout + 1)
            ]
            speed = math.hypot(track.vx, track.vy)
            rng = math.hypot(track.x + 58.0, track.y + 35.0)
            future_rng = math.hypot(track.prediction[-1][0] + 58.0, track.prediction[-1][1] + 35.0)
            if future_rng < rng - 2.0:
                track.intent = "approach"
                track.threat = "elevated" if speed > 0.5 else "guarded"
            elif future_rng > rng + 2.0:
                track.intent = "retreat"
                track.threat = "low"
            else:
                track.intent = "loiter_patrol"
                track.threat = "guarded"
            track.intent_confidence = 0.55
            track.threat_confidence = 0.55
            track.objectness = 0.5


class OnlineLLMDecisionService:
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "rule").strip() or "rule"
        self.api_url = os.getenv("LLM_API_URL", "").strip()
        self.api_key = (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.last_call: dict[str, float] = {}
        self.cache: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")

    def decide(self, target_id: str, payload: dict[str, Any], now: float) -> dict[str, Any]:
        future = self.pending.get(target_id)
        if future is not None and future.done():
            try:
                self.cache[target_id] = future.result()
            except Exception as exc:
                print(f"[LLM] fallback for {target_id}: {exc.__class__.__name__}: {exc}", flush=True)
                self.cache[target_id] = self._rule_decision(payload, note=f"llm_fallback:{exc.__class__.__name__}")
            self.pending.pop(target_id, None)

        if future is not None and not future.done():
            return self.cache.get(target_id) or self._rule_decision(payload, note="llm_pending")

        if now - self.last_call.get(target_id, -999.0) < 6.0 and target_id in self.cache:
            return self.cache[target_id]

        self.last_call[target_id] = now
        if self.provider == "openai_compatible" and self.api_url and self.model:
            print(f"[LLM] queued {self.model} for {target_id}", flush=True)
            self.pending[target_id] = self.executor.submit(self._call_openai_compatible, payload)
            decision = self.cache.get(target_id) or self._rule_decision(payload, note="llm_pending")
        else:
            decision = self._rule_decision(payload, note="rule_provider")
        self.cache[target_id] = decision
        return decision

    def _call_openai_compatible(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are an online radar mission advisor. Return strict JSON only. "
            "Give task-level guidance for shore, v01, and v02 radars. "
            "Allowed action values: monitor, increase_tracking_rate, classify_and_shadow, alert_and_allocate_tracker. "
            "Required top-level keys: target_id, final_intent, priority, action, reason, radar_guidance, platform_guidance. "
            "Do not output raw beam angles; the constraint controller will convert guidance to executable parameters.\n\n"
            f"Structured state:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return compact JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
            "enable_thinking": False,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.api_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        decision = self._parse_decision_content(content)
        decision = self._normalize_decision(payload, decision, note="llm")
        print(
            f"[LLM] {payload.get('target_id')} action={decision.get('action', 'unknown')} "
            f"intent={decision.get('final_intent', payload.get('intent', 'unknown'))}",
            flush=True,
        )
        return decision

    def _parse_decision_content(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        return json.loads(text)

    def _normalize_decision(self, payload: dict[str, Any], decision: dict[str, Any], note: str) -> dict[str, Any]:
        allowed_actions = {
            "monitor",
            "increase_tracking_rate",
            "classify_and_shadow",
            "alert_and_allocate_tracker",
        }
        target_id = decision.get("target_id", payload.get("target_id"))
        intent = decision.get("final_intent") or decision.get("intent") or payload.get("intent", "monitor")
        threat = payload.get("threat", "low")
        action = decision.get("action") or ACTION_BY_INTENT.get(intent, "monitor")
        if action not in allowed_actions:
            action = ACTION_BY_INTENT.get(intent, "monitor")

        try:
            priority = int(decision.get("priority", 0))
        except (TypeError, ValueError):
            priority = 0
        if priority <= 0:
            priority = 4 if action == "alert_and_allocate_tracker" else 3 if action == "increase_tracking_rate" else 2 if action == "classify_and_shadow" else 1

        reason = decision.get("reason") or (
            f"LLM normalized decision: intent={intent}, threat={threat}, action={action}."
        )
        radar_guidance = decision.get("radar_guidance") or {
            "shore": "continuous_designation" if action != "monitor" else "wide_area_track",
            "v01": action,
            "v02": "keep_cross_bearing_track",
        }
        platform_guidance = decision.get("platform_guidance") or {
            "v01": "keep standoff and maintain precision track",
            "v02": "hold side-looking geometry for confirmation",
        }

        return {
            **decision,
            "target_id": target_id,
            "final_intent": intent,
            "priority": priority,
            "action": action,
            "radar_guidance": radar_guidance,
            "platform_guidance": platform_guidance,
            "reason": reason,
            "provider": self.provider,
            "note": note,
        }

    def _rule_decision(self, payload: dict[str, Any], note: str) -> dict[str, Any]:
        intent = payload.get("intent", "monitor")
        threat = payload.get("threat", "low")
        predicted_range_delta = float(payload.get("predicted_range_delta", 0.0))
        confidence = float(payload.get("intent_confidence", 0.0))

        if threat == "high" or intent == "intercept":
            action = "alert_and_allocate_tracker"
            priority = 4
            guidance = "allocate dedicated precision tracking and keep cross bearing"
        elif intent == "approach" or predicted_range_delta < -4.0:
            action = "increase_tracking_rate"
            priority = 3
            guidance = "increase tracking rate and narrow beam after lock"
        elif intent == "loiter_patrol":
            action = "classify_and_shadow"
            priority = 2
            guidance = "shadow target with staggered dwell"
        else:
            action = "monitor"
            priority = 1
            guidance = "continue routine monitoring"

        if confidence < 0.55:
            guidance = "widen beam until confidence recovers; " + guidance

        return {
            "target_id": payload.get("target_id"),
            "final_intent": intent,
            "priority": priority,
            "action": action,
            "radar_guidance": {
                "shore": "wide_area_track" if action == "monitor" else "continuous_designation",
                "v01": guidance,
                "v02": "keep_cross_bearing_track",
            },
            "platform_guidance": {
                "v01": "keep standoff and maintain precision track",
                "v02": "hold side-looking geometry for confirmation",
            },
            "reason": f"intent={intent}, threat={threat}, predicted_range_delta={predicted_range_delta:.2f}",
            "provider": self.provider,
            "note": note,
        }


class ConstraintController:
    def update(self, radar: Radar, track: Track, selected: bool, dt: float) -> None:
        prediction = track.prediction[2] if len(track.prediction) >= 3 else [track.x, track.y, track.vx, track.vy]
        px, py = float(prediction[0]), float(prediction[1])
        desired = bearing_deg(radar.x, radar.y, px, py)
        target_range = distance(radar.x, radar.y, px, py)
        decision = track.llm_decision or {}
        action = decision.get("action", ACTION_BY_INTENT.get(track.intent, "monitor"))
        priority = int(decision.get("priority", 1) or 1)
        confidence = clamp(track.intent_confidence, 0.0, 1.0)

        if action == "alert_and_allocate_tracker":
            desired_width = 10.0
            desired_dwell = 210.0
        elif action == "increase_tracking_rate":
            desired_width = 14.0
            desired_dwell = 165.0
        elif action == "classify_and_shadow":
            desired_width = 20.0
            desired_dwell = 135.0
        else:
            desired_width = 30.0
            desired_dwell = RADAR_META[radar.id]["base_dwell"]

        if not selected:
            desired_width += 12.0
            desired_dwell *= 0.78
        if confidence < 0.55:
            desired_width += 14.0
            desired_dwell += 24.0

        if radar.id == "shore":
            width_bounds = (12.0, 62.0)
            slew_rate = 28.0 + 4.0 * priority
        else:
            width_bounds = (8.0, 48.0)
            slew_rate = 38.0 + 5.0 * priority

        desired_width = clamp(desired_width, *width_bounds)
        desired_dwell = clamp(desired_dwell, 45.0, 240.0)
        max_step = slew_rate * max(dt, 0.016)
        desired_range_gate = clamp(target_range + 18.0, 18.0, RADAR_META[radar.id]["max_range"])
        intensity = tracking_intensity(radar.id, desired_width, desired_dwell, confidence)

        radar.beam_azimuth = slew_angle(radar.beam_azimuth, desired, max_step)
        radar.beam_width = lerp(radar.beam_width, desired_width, 0.18)
        radar.dwell_ms = lerp(radar.dwell_ms, desired_dwell, 0.18)
        radar.range_gate = lerp(radar.range_gate, desired_range_gate, 0.2)
        radar.controller = {
            "desiredAzimuth": desired,
            "desiredWidth": desired_width,
            "desiredDwell": desired_dwell,
            "desiredRangeGate": desired_range_gate,
            "slewRate": slew_rate,
            "maxSlewStep": max_step,
            "trackingIntensity": intensity,
            "mode": action,
            "constraint": "bounded_slew",
            "rangeGate": radar.range_gate,
            "platformGuidance": decision.get("platform_guidance", {}),
        }


class ClosedLoopEngine:
    def __init__(self, checkpoint: Path = CHECKPOINT) -> None:
        self.sim_time = 0.0
        self.cycle = 0
        self.speed = 1.0
        self.clutter = 0.26
        self.uncertainty = 0.22
        self.selected_target = "T-01"
        self.events: list[str] = []
        self.ships = {"v01": Entity("V-01"), "v02": Entity("V-02")}
        self.targets = {"T-01": Entity("T-01"), "T-02": Entity("T-02")}
        self.radars = {
            "shore": Radar("shore", -58.0, -35.0, beam_azimuth=32.0, beam_width=36.0, range_gate=120.0),
            "v01": Radar("v01", 0.0, 0.0, beam_azimuth=28.0, beam_width=26.0, range_gate=90.0),
            "v02": Radar("v02", 0.0, 0.0, beam_azimuth=55.0, beam_width=26.0, range_gate=90.0),
        }
        self.tracks = {target_id: Track(target_id) for target_id in TARGET_IDS}
        self.model = TrackModelService(checkpoint)
        self.llm = OnlineLLMDecisionService()
        self.controller = ConstraintController()
        self.last_infer = -999.0
        self.last_llm = -999.0
        self._update_kinematics(0.1)
        self._align_initial_beams()

    def command(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "set_target":
            target_id = message.get("targetId") or message.get("target_id")
            if target_id in TARGET_IDS:
                self.selected_target = target_id
                self._log(f"target handoff -> {target_id}")
        elif msg_type == "set_speed":
            self.speed = clamp(float(message.get("speed", self.speed)), 0.25, 3.0)
        elif msg_type == "set_clutter":
            self.clutter = clamp(float(message.get("clutter", self.clutter)), 0.02, 0.8)
        elif msg_type == "set_uncertainty":
            self.uncertainty = clamp(float(message.get("uncertainty", self.uncertainty)), 0.0, 0.8)
        elif msg_type == "reset":
            self.__init__(self.model.checkpoint)
            self._log("backend loop reset")

    def step(self, dt: float) -> None:
        scaled = dt * self.speed
        self.sim_time += scaled
        self.cycle += 1
        self._update_kinematics(scaled)
        all_measurements = self._measure_all_targets()
        self._fuse_tracks(all_measurements, scaled)

        if self.sim_time - self.last_infer >= 0.35:
            self.model.infer(self.tracks)
            self.last_infer = self.sim_time

        if self.sim_time - self.last_llm >= 6.0:
            for target_id, track in self.tracks.items():
                payload = self._decision_payload(target_id, track)
                track.llm_decision = self.llm.decide(target_id, payload, self.sim_time)
            self.last_llm = self.sim_time

        selected_track = self.tracks[self.selected_target]
        for radar in self.radars.values():
            measurement = self._best_measurement_for_radar(all_measurements, radar.id, self.selected_target)
            if measurement:
                radar.perception = {
                    "bearing": measurement.bearing_world_deg,
                    "range": measurement.range_m,
                    "confidence": measurement.confidence,
                    "doppler": measurement.doppler,
                    "estimate": {"x": measurement.x, "y": measurement.y},
                    "localBearing": measurement.bearing_local_deg,
                }
                radar.snr = 8.0 + measurement.confidence * 18.0 - self.clutter * 5.0
            radar.reasoning = {
                "intent": selected_track.intent,
                "threat": selected_track.threat,
                "action": selected_track.llm_decision.get("action", ACTION_BY_INTENT.get(selected_track.intent, "monitor")),
                "narrative": selected_track.llm_decision.get("reason", "waiting for decision"),
            }
            self.controller.update(radar, selected_track, True, scaled)

    def snapshot(self) -> dict[str, Any]:
        selected = self.tracks[self.selected_target]
        return {
            "type": "snapshot",
            "backend": True,
            "simTime": self.sim_time,
            "cycle": self.cycle,
            "speed": self.speed,
            "clutter": self.clutter,
            "uncertainty": self.uncertainty,
            "selectedTargetId": self.selected_target,
            "model": {
                "available": self.model.available,
                "error": self.model.error,
                "modelType": self.model.model_type,
                "inputMode": self.model.input_mode,
                "tin": self.model.tin,
                "tout": self.model.tout,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model or "rule",
                "apiUrl": self.llm.api_url,
                "configured": bool(self.llm.api_key and self.llm.api_url and self.llm.model),
            },
            "fusionScore": clamp((selected.intent_confidence + selected.threat_confidence) * 0.5, 0.0, 1.0),
            "entities": {
                "ships": {key: value.snapshot() for key, value in self.ships.items()},
                "targets": {key: value.snapshot() for key, value in self.targets.items()},
            },
            "radars": {key: value.snapshot() for key, value in self.radars.items()},
            "tracks": {key: value.snapshot() for key, value in self.tracks.items()},
            "events": self.events[-4:],
        }

    def _update_kinematics(self, dt: float) -> None:
        for key, ship in self.ships.items():
            px, py = ship.x, ship.y
            ship.x, ship.y = route_position(ROUTES["ships"][key], self.sim_time)
            self._update_velocity(ship, px, py, dt)
            self._push_trail(ship, 180)

        for key, target in self.targets.items():
            px, py = target.x, target.y
            target.x, target.y = route_position(ROUTES["targets"][key], self.sim_time)
            self._update_velocity(target, px, py, dt)
            self._push_trail(target, 220)

        self.radars["v01"].x = self.ships["v01"].x
        self.radars["v01"].y = self.ships["v01"].y
        self.radars["v01"].vx = self.ships["v01"].vx
        self.radars["v01"].vy = self.ships["v01"].vy
        self.radars["v01"].heading = self.ships["v01"].heading
        self.radars["v02"].x = self.ships["v02"].x
        self.radars["v02"].y = self.ships["v02"].y
        self.radars["v02"].vx = self.ships["v02"].vx
        self.radars["v02"].vy = self.ships["v02"].vy
        self.radars["v02"].heading = self.ships["v02"].heading

    def _update_velocity(self, entity: Entity, px: float, py: float, dt: float) -> None:
        if len(entity.trail) > 0 and dt > 1e-6:
            entity.vx = (entity.x - px) / dt
            entity.vy = (entity.y - py) / dt
        else:
            entity.vx = 0.0
            entity.vy = 0.0
        if abs(entity.vx) + abs(entity.vy) > 1e-6:
            entity.heading = wrap360(math.degrees(math.atan2(entity.vy, entity.vx)))

    def _push_trail(self, entity: Entity, max_len: int) -> None:
        if not entity.trail or distance(entity.trail[-1]["x"], entity.trail[-1]["y"], entity.x, entity.y) > 0.18:
            entity.trail.append({"x": entity.x, "y": entity.y})
        if len(entity.trail) > max_len:
            del entity.trail[: len(entity.trail) - max_len]

    def _measure_all_targets(self) -> list[Measurement]:
        measurements = []
        for target_id, target in self.targets.items():
            for radar in self.radars.values():
                measurements.append(self._measure(radar, target, target_id))
        return measurements

    def _measure(self, radar: Radar, target: Entity, target_id: str) -> Measurement:
        true_range = distance(radar.x, radar.y, target.x, target.y)
        true_bearing = bearing_deg(radar.x, radar.y, target.x, target.y)
        local_bearing = angle_diff(radar.heading, true_bearing)
        dx = target.x - radar.x
        dy = target.y - radar.y
        rng = math.hypot(dx, dy) or 1.0
        range_rate = ((target.vx - radar.vx) * dx + (target.vy - radar.vy) * dy) / rng
        beam_error = abs(angle_diff(radar.beam_azimuth, true_bearing))
        beam_score = clamp(1.0 - beam_error / max(radar.beam_width, 8.0), 0.0, 1.0)
        range_score = clamp(1.0 - true_range / (RADAR_META[radar.id]["max_range"] * 1.1), 0.0, 1.0)
        selected_bonus = 0.18 if target_id == self.selected_target else 0.04
        confidence = clamp(0.18 + selected_bonus + 0.38 * beam_score + 0.28 * range_score - self.uncertainty * 0.12, 0.05, 0.99)
        jitter = self.uncertainty * (1.3 - confidence)
        measured_range = max(0.0, true_range + noise(true_range + len(radar.id), self.sim_time, 1.7 * jitter))
        measured_bearing = wrap360(true_bearing + noise(true_bearing + true_range, self.sim_time, 3.8 * jitter))
        mx = radar.x + math.cos(math.radians(measured_bearing)) * measured_range
        my = radar.y + math.sin(math.radians(measured_bearing)) * measured_range
        return Measurement(
            radar_id=radar.id,
            target_id=target_id,
            range_m=measured_range,
            bearing_local_deg=angle_diff(radar.heading, measured_bearing),
            bearing_world_deg=measured_bearing,
            range_rate=range_rate,
            confidence=confidence,
            x=mx,
            y=my,
            doppler=range_rate * 1000.0,
        )

    def _fuse_tracks(self, measurements: list[Measurement], dt: float) -> None:
        for target_id in TARGET_IDS:
            target_measurements = [m for m in measurements if m.target_id == target_id]
            total = sum(max(0.01, m.confidence) for m in target_measurements) or 1.0
            mx = sum(m.x * max(0.01, m.confidence) for m in target_measurements) / total
            my = sum(m.y * max(0.01, m.confidence) for m in target_measurements) / total
            track = self.tracks[target_id]
            if not track.initialized:
                track.x = mx
                track.y = my
                track.vx = 0.0
                track.vy = 0.0
                track.initialized = True
            else:
                pred_x = track.x + track.vx * dt
                pred_y = track.y + track.vy * dt
                rx = mx - pred_x
                ry = my - pred_y
                alpha = 0.42
                beta = 0.12
                track.x = pred_x + alpha * rx
                track.y = pred_y + alpha * ry
                if dt > 1e-6:
                    track.vx = track.vx + beta * rx / dt
                    track.vy = track.vy + beta * ry / dt
            track.append_history()

    def _best_measurement_for_radar(self, measurements: list[Measurement], radar_id: str, target_id: str) -> Measurement | None:
        matches = [m for m in measurements if m.radar_id == radar_id and m.target_id == target_id]
        return max(matches, key=lambda item: item.confidence) if matches else None

    def _decision_payload(self, target_id: str, track: Track) -> dict[str, Any]:
        rng_now = distance(-58.0, -35.0, track.x, track.y)
        if track.prediction:
            px, py = track.prediction[-1][0], track.prediction[-1][1]
            rng_future = distance(-58.0, -35.0, px, py)
        else:
            rng_future = rng_now
        return {
            "target_id": target_id,
            "state": {"x": track.x, "y": track.y, "vx": track.vx, "vy": track.vy},
            "intent": track.intent,
            "threat": track.threat,
            "intent_confidence": track.intent_confidence,
            "threat_confidence": track.threat_confidence,
            "predicted_range_delta": rng_future - rng_now,
            "predicted_track": track.prediction,
            "selected_target": self.selected_target,
        }

    def _align_initial_beams(self) -> None:
        target = self.targets[self.selected_target]
        for radar in self.radars.values():
            radar.beam_azimuth = bearing_deg(radar.x, radar.y, target.x, target.y)
            radar.range_gate = clamp(distance(radar.x, radar.y, target.x, target.y) + 18.0, 18.0, RADAR_META[radar.id]["max_range"])

    def _log(self, message: str) -> None:
        self.events.append(f"[{self.sim_time:.1f}s] {message}")
        if len(self.events) > 8:
            del self.events[: len(self.events) - 8]


class StaticAndWebSocketServer:
    def __init__(self, host: str, port: int, engine: ClosedLoopEngine) -> None:
        self.host = host
        self.port = port
        self.engine = engine

    async def serve(self) -> None:
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"cognitive radar loop backend listening on {sockets}", flush=True)
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
        except Exception:
            writer.close()
            await writer.wait_closed()
            return

        header_text = request.decode("iso-8859-1", errors="replace")
        request_line, headers = self._parse_headers(header_text)
        if not request_line:
            writer.close()
            await writer.wait_closed()
            return

        method, path, _ = request_line.split(" ", 2)
        if headers.get("upgrade", "").lower() == "websocket":
            await self._websocket(reader, writer, headers)
            return

        if method != "GET":
            await self._send_response(writer, 405, b"Method Not Allowed", "text/plain")
            return
        await self._serve_static(writer, path)

    def _parse_headers(self, text: str) -> tuple[str, dict[str, str]]:
        lines = text.split("\r\n")
        request_line = lines[0]
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return request_line, headers

    async def _websocket(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        writer.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode("ascii")
        )
        await writer.drain()

        last = time.perf_counter()
        try:
            while not writer.is_closing():
                now = time.perf_counter()
                dt = min(max(now - last, 0.02), 0.12)
                last = now
                self.engine.step(dt)
                writer.write(self._encode_ws(json.dumps(self.engine.snapshot(), ensure_ascii=False).encode("utf-8")))
                await writer.drain()
                await self._read_commands(reader)
                await asyncio.sleep(0.1)
        except Exception:
            writer.close()
            await writer.wait_closed()

    async def _read_commands(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                frame = await asyncio.wait_for(reader.readexactly(2), timeout=0.001)
            except asyncio.TimeoutError:
                return
            except Exception:
                return
            fin_opcode = frame[0]
            length = frame[1] & 0x7F
            masked = frame[1] & 0x80
            if length == 126:
                length = struct.unpack("!H", await reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", await reader.readexactly(8))[0]
            mask = await reader.readexactly(4) if masked else b""
            payload = await reader.readexactly(length)
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            opcode = fin_opcode & 0x0F
            if opcode == 8:
                raise ConnectionError("client closed websocket")
            if opcode == 1:
                try:
                    self.engine.command(json.loads(payload.decode("utf-8")))
                except Exception:
                    pass

    def _encode_ws(self, payload: bytes) -> bytes:
        length = len(payload)
        if length < 126:
            return bytes([0x81, length]) + payload
        if length < 65536:
            return bytes([0x81, 126]) + struct.pack("!H", length) + payload
        return bytes([0x81, 127]) + struct.pack("!Q", length) + payload

    async def _serve_static(self, writer: asyncio.StreamWriter, path: str) -> None:
        clean = path.split("?", 1)[0].strip("/")
        if clean in ("", "cognitive_radar_loop", "cognitive_radar_loop/"):
            file_path = APP_DIR / "index.html"
        elif clean.startswith("cognitive_radar_loop/"):
            file_path = APP_DIR / clean.removeprefix("cognitive_radar_loop/")
        else:
            file_path = APP_DIR / clean

        try:
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(APP_DIR.resolve())) or not resolved.is_file():
                await self._send_response(writer, 404, b"Not Found", "text/plain")
                return
            content = resolved.read_bytes()
            await self._send_response(writer, 200, content, self._content_type(resolved))
        except Exception as exc:
            await self._send_response(writer, 500, repr(exc).encode("utf-8"), "text/plain")

    async def _send_response(self, writer: asyncio.StreamWriter, status: int, body: bytes, content_type: str) -> None:
        reason = {200: "OK", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"}.get(status, "OK")
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: {content_type}; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def _content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
        }.get(suffix, "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online cognitive radar closed-loop backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5177)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    engine = ClosedLoopEngine(args.checkpoint)
    if engine.model.available:
        print(
            f"loaded {engine.model.model_type} input_mode={engine.model.input_mode} "
            f"Tin={engine.model.tin} Tout={engine.model.tout}",
            flush=True,
        )
    else:
        print(f"model unavailable, using constant-velocity fallback: {engine.model.error}", flush=True)
    asyncio.run(StaticAndWebSocketServer(args.host, args.port, engine).serve())


if __name__ == "__main__":
    main()

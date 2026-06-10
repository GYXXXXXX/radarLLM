#!/usr/bin/env python3
"""Generate target-response explanations from structured intention predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import ACTION_BY_INTENT
except ImportError:  # pragma: no cover
    from config import ACTION_BY_INTENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LLM-style decision explanations.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--provider", choices=("rule", "openai_compatible", "qwen"), default="rule")
    parser.add_argument(
        "--llm-task",
        choices=("explain_prediction", "infer_intent"),
        default="infer_intent",
        help=(
            "explain_prediction gives the LLM the Transformer's predicted intent; "
            "infer_intent hides it and asks the LLM to infer intent from trends."
        ),
    )
    parser.add_argument("--api-url", type=str, default=os.getenv("LLM_API_URL", ""))
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Append output and skip records already in output.")
    parser.add_argument("--print-config", action="store_true", help="Print LLM request configuration and exit.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(record.get("scene_file", "")),
        int(record.get("start_frame", 0) or 0),
        int(record.get("slot", 0) or 0),
    )


def decision_key(item: dict[str, Any]) -> tuple[str, int, int]:
    prediction = item.get("prediction", {})
    if isinstance(prediction, dict):
        return record_key(prediction)
    return (
        str(item.get("scene_file", "")),
        int(item.get("start_frame", 0) or 0),
        int(item.get("slot", 0) or 0),
    )


def load_done_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = decision_key(item)
            if key[0] and key[2] > 0:
                done.add(key)
    return done


def build_compact_prediction(record: dict[str, Any], include_model_intent: bool) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "scene_file": record.get("scene_file"),
        "target_slot": record.get("slot"),
        "observed_range_trend": record.get("observed_range_trend"),
        "observed_range_delta_m": record.get("observed_range_delta_m"),
        "observed_speed_mean_mps": record.get("observed_speed_mean_mps"),
        "observed_protected_distance_trend": record.get("observed_protected_distance_trend"),
        "observed_protected_distance_delta_m": record.get("observed_protected_distance_delta_m"),
        "predicted_range_trend": record.get("predicted_range_trend"),
        "predicted_range_delta_m": record.get("predicted_range_delta_m"),
        "predicted_protected_distance_trend": record.get("predicted_protected_distance_trend"),
        "predicted_protected_distance_delta_m": record.get("predicted_protected_distance_delta_m"),
        "predicted_speed_mean_mps": record.get("predicted_speed_mean_mps"),
        "recommended_action_rule": record.get("recommended_action_rule"),
    }
    if include_model_intent:
        compact.update(
            {
                "predicted_target_class": record.get("pred_target_class_name"),
                "predicted_intent": record.get("pred_intent_name"),
                "intent_confidence": record.get("pred_intent_confidence"),
                "predicted_threat_level": record.get("pred_threat_level"),
                "predicted_threat_name": record.get("pred_threat_name"),
                "threat_confidence": record.get("pred_threat_confidence"),
            }
        )
    return compact


def build_prompt(record: dict[str, Any], llm_task: str = "infer_intent") -> str:
    include_model_intent = llm_task == "explain_prediction"
    compact = build_compact_prediction(record, include_model_intent=include_model_intent)
    if include_model_intent:
        instruction = (
            "The upstream Transformer prediction is included. Verify whether the action is reasonable, "
            "then return compact JSON with keys target_id, final_intent, priority, action, reason, and follow_up."
        )
    else:
        instruction = (
            "Infer the target intention from the trajectory trends without using the upstream intent label. "
            "Choose final_intent from exactly one of: benign_transit, approach, retreat, loiter_patrol, intercept. "
            "Choose action from exactly one of: monitor, increase_tracking_rate, classify_and_shadow, "
            "alert_and_allocate_tracker. Return compact JSON with keys target_id, final_intent, priority, "
            "action, reason, and follow_up."
        )
    return (
        "You are a radar target-intention decision assistant. "
        f"{instruction}\n\n"
        f"Structured prediction:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def priority_from_record(record: dict[str, Any]) -> str:
    intent = record.get("pred_intent_name", "")
    threat = int(record.get("pred_threat_level", 1))
    protected_trend = record.get("predicted_protected_distance_trend", "")
    confidence = float(record.get("pred_intent_confidence", 0.0))

    if intent == "intercept" or threat >= 4:
        return "high"
    if intent == "approach" or (threat >= 3 and protected_trend == "decreasing"):
        return "medium_high" if confidence >= 0.55 else "medium"
    if intent == "loiter_patrol" or threat == 2:
        return "medium"
    return "low"


def rule_decision(record: dict[str, Any]) -> dict[str, Any]:
    intent = record.get("pred_intent_name", "unknown")
    action = ACTION_BY_INTENT.get(intent, record.get("recommended_action_rule", "monitor"))
    priority = priority_from_record(record)

    reasons = [
        f"model intent={intent}",
        f"confidence={float(record.get('pred_intent_confidence', 0.0)):.2f}",
        f"threat_level={record.get('pred_threat_level')}",
        f"observed_range_trend={record.get('observed_range_trend')}",
        f"predicted_protected_distance_trend={record.get('predicted_protected_distance_trend')}",
    ]
    if intent == "intercept":
        follow_up = "allocate a dedicated tracker and alert the operator"
    elif intent == "approach":
        follow_up = "increase update rate and confirm identity"
    elif intent == "loiter_patrol":
        follow_up = "keep classification active and shadow the target"
    else:
        follow_up = "continue normal monitoring"

    return {
        "target_id": record.get("slot"),
        "final_intent": intent,
        "priority": priority,
        "action": action,
        "reason": "; ".join(reasons),
        "follow_up": follow_up,
        "provider": "rule",
    }


def normalize_api_url(api_url: str) -> str:
    api_url = api_url.rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url
    return api_url + "/chat/completions"


def apply_provider_defaults(args: argparse.Namespace) -> None:
    if args.provider == "qwen":
        if not args.api_url:
            args.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not args.api_key_env:
            args.api_key_env = "DASHSCOPE_API_KEY"
        if not args.model:
            args.model = "qwen-plus"
    else:
        if not args.api_key_env:
            args.api_key_env = "LLM_API_KEY"


def call_openai_compatible(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.api_url:
        raise ValueError("--api-url or LLM_API_URL is required for openai_compatible provider.")
    if not args.model:
        raise ValueError("--model or LLM_MODEL is required for openai_compatible provider.")

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(
            f"Environment variable {args.api_key_env} is not set. "
            "For Qwen, run: $env:DASHSCOPE_API_KEY=\"your-api-key\""
        )
    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You produce concise JSON decisions for radar target-intention analysis. "
                    "Do not include markdown."
                ),
            },
            {"role": "user", "content": build_prompt(record, args.llm_task)},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        normalize_api_url(args.api_url),
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "LLM request failed: "
            f"HTTP {exc.code} {exc.reason}. "
            f"Provider={args.provider}, model={args.model}, url={normalize_api_url(args.api_url)}, "
            f"api_key_env={args.api_key_env}, response_body={body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    content = payload["choices"][0]["message"]["content"]
    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        decision = {"raw_response": content}
    decision["provider"] = "openai_compatible"
    return decision


def make_decision(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if record.get("target_present_label") is False:
        return {
            "scene_file": record.get("scene_file"),
            "slot": record.get("slot"),
            "prediction": record,
            "decision": {
                "target_id": record.get("slot"),
                "final_intent": "empty_slot",
                "priority": "none",
                "action": "ignore",
                "reason": "This slot is labeled as empty, so no target-intention decision is generated.",
                "follow_up": "skip empty slot",
                "provider": args.provider,
            },
        }

    if args.provider == "rule":
        decision = rule_decision(record)
    else:
        decision = call_openai_compatible(record, args)

    return {
        "scene_file": record.get("scene_file"),
        "slot": record.get("slot"),
        "prompt": build_prompt(record, args.llm_task),
        "prediction": record,
        "decision": decision,
    }


def main() -> None:
    args = parse_args()
    apply_provider_defaults(args)
    if args.print_config:
        api_key = os.getenv(args.api_key_env, "")
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "llm_task": args.llm_task,
                    "api_url": normalize_api_url(args.api_url) if args.api_url else "",
                    "api_key_env": args.api_key_env,
                    "api_key_present": bool(api_key),
                    "api_key_prefix": api_key[:6] + "..." if api_key else "",
                    "model": args.model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    records = load_jsonl(args.predictions)

    output = Path(args.output) if args.output else Path(args.predictions).with_name("llm_decisions.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(output) if args.resume else set()
    if done_keys:
        before = len(records)
        records = [record for record in records if record_key(record) not in done_keys]
        print(f"Resume enabled: skipped {before - len(records)} completed records from {output}")
    if args.max_records:
        records = records[: args.max_records]

    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for record in records:
            item = make_decision(record, args)
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    print(f"Saved decisions: {output}")


if __name__ == "__main__":
    main()

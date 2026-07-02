from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DashScope/OpenAI-compatible LLM connectivity.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-url", default=os.getenv("LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3.6-flash"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("No API key found. Set DASHSCOPE_API_KEY or LLM_API_KEY first.")

    endpoint = args.api_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true, \"action\": \"monitor\"}."},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    print(f"endpoint = {endpoint}")
    print(f"model    = {args.model}")
    print(f"timeout  = {args.timeout}s")
    print(f"key      = {'*' * max(8, min(16, len(args.api_key)))}")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            print(f"HTTP {response.status} in {elapsed:.2f}s")
            print(body[:2000])
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTPError {exc.code} in {elapsed:.2f}s")
        print(body[:2000])
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - started
        print(f"URLError in {elapsed:.2f}s: {exc}")
        raise SystemExit(1)
    except TimeoutError as exc:
        elapsed = time.perf_counter() - started
        print(f"TimeoutError in {elapsed:.2f}s: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

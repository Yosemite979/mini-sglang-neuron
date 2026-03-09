import argparse
import json
import os
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HTTP chat call.")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "1919")))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("MODEL", ""))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    chat_url = base_url.removesuffix("/v1") + "/v1/chat/completions"

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "stream": True,
        }
    ).encode("utf-8")

    req = Request(
        chat_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue

            piece = line[6:]
            if piece == "[DONE]":
                break

            try:
                obj = json.loads(piece)
                text_piece = obj["choices"][0]["delta"].get("content")
            except Exception:
                continue

            if text_piece:
                print(text_piece, end="", flush=True)

    print()


if __name__ == "__main__":
    main()

"""Minimal tool call parser for Qwen3 models."""
import json
import re
import uuid
from typing import List, Optional, Tuple

BOT = "<tool_call>\n"
EOT = "\n</tool_call>"
PATTERN = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> Tuple[Optional[str], Optional[List[dict]]]:
    if BOT not in text:
        return None, None
    matches = PATTERN.findall(text)
    if not matches:
        return None, None
    idx = text.find(BOT)
    normal = text[:idx].strip() or None
    calls = []
    for i, m in enumerate(matches):
        try:
            parsed = json.loads(m.strip())
            calls.append({
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed.get("arguments", {})),
                },
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return (normal, calls) if calls else (None, None)

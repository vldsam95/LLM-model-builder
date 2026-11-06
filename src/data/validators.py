from __future__ import annotations

import json
from pathlib import Path


REQUIRED_DIALOGUE_KEYS = {"user", "assistant"}


def validate_text_file(path: str | Path) -> list[str]:
    path = Path(path)
    issues: list[str] = []
    if not path.exists():
        issues.append(f"Missing file: {path}")
        return issues
    if path.stat().st_size == 0:
        issues.append(f"Empty file: {path}")
    return issues


def validate_jsonl_dialogues(path: str | Path) -> list[str]:
    path = Path(path)
    issues: list[str] = []
    if not path.exists():
        return [f"Missing file: {path}"]
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(f"{path}:{line_no} invalid JSON: {exc}")
                continue
            missing = REQUIRED_DIALOGUE_KEYS - obj.keys()
            if missing:
                issues.append(f"{path}:{line_no} missing keys: {sorted(missing)}")
            for key in ("system", "user", "assistant"):
                if key in obj and obj[key] is not None and not isinstance(obj[key], str):
                    issues.append(f"{path}:{line_no} field '{key}' must be a string or null")
    return issues

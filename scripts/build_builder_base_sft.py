#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import normalize_text

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "sft_builder_base.jsonl"
SEED = 42

EN_ALLOWED_TITLES = {
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Chatbot",
    "Automation",
    "Computer",
    "Website",
    "Translation",
    "JSON",
    "User interface",
    "Data",
}

HE_ALLOWED_TITLES = {
    "בינה מלאכותית",
    "למידת מכונה",
    "עיבוד שפה טבעית",
    "צ'טבוט",
    "אוטומציה",
    "מחשב",
    "אתר אינטרנט",
    "תרגום",
    "JSON",
    "ממשק משתמש",
    "נתונים",
}


def first_sentence(text: str) -> str:
    text = normalize_text(text)
    for sep in [". ", "? ", "! ", ":", " - "]:
        if sep in text:
            return text.split(sep)[0].strip().rstrip(".") + "."
    return text[:180].rstrip(".") + "."


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def load_squad_rows(path: Path, seed: int, limit: int = 80) -> list[dict[str, str]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue
                question = normalize_text(qa.get("question", ""))
                answer = normalize_text(answers[0].get("text", ""))
                if question and 3 <= len(answer) <= 160:
                    rows.append({"user": question, "assistant": answer})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def add(rows: list[dict[str, str]], system: str, user: str, assistant: str) -> None:
    rows.append(
        {
            "system": normalize_text(system),
            "user": normalize_text(user),
            "assistant": normalize_text(assistant),
        }
    )


def generic_manual_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    greetings = [
        ("Answer briefly and clearly.", "Hello", "Hello. I am a tiny neutral assistant for short answers, JSON extraction, and lightweight classification."),
        ("Answer briefly and clearly.", "What can you do?", "I can answer short questions, summarize notes, extract compact JSON, and classify short text."),
        ("Respond in compact JSON only.", "What can you do in JSON?", json.dumps({"task": "capabilities", "summary": "short answers, JSON extraction, and classification"}, ensure_ascii=False)),
        ("ענה בקצרה ובעברית.", "שלום", "שלום. אני עוזר tiny ניטרלי לתשובות קצרות, חילוץ JSON וסיווג בסיסי."),
        ("ענה בקצרה ובעברית.", "מה אתה יודע לעשות?", "אני יכול לענות בקצרה, לסכם, לחלץ JSON ולסווג טקסטים קצרים."),
        ("ענה ב-JSON בלבד.", "מה אתה יודע לעשות ב-JSON?", json.dumps({"task": "capabilities", "summary": "תשובות קצרות, חילוץ JSON וסיווג"}, ensure_ascii=False)),
    ]
    for system, user, assistant in greetings:
        add(rows, system, user, assistant)

    translations = [
        ("house", "בית"),
        ("computer", "מחשב"),
        ("invoice", "חשבונית"),
        ("priority", "עדיפות"),
        ("policy", "מדיניות"),
        ("ticket", "כרטיס"),
    ]
    for english, hebrew in translations:
        add(rows, "Act as a translation assistant.", f"Translate '{english}' to Hebrew.", hebrew)
        add(rows, "פעל כעוזר תרגום קצר.", f"תתרגם לאנגלית: {hebrew}", english)

    extraction_examples = [
        (
            "Extract structured fields and return compact JSON only.",
            "Extract the fields customer_name, product, renewal_date from this text: Dana Levi renewed the Analytics Pro plan on 2026-04-01.",
            {"customer_name": "Dana Levi", "product": "Analytics Pro", "renewal_date": "2026-04-01"},
        ),
        (
            "Extract structured fields and return compact JSON only.",
            "Extract the fields company, contact_email, product from this text: Nova Labs asked about the Insight Suite. Contact: ops@example.com.",
            {"company": "Nova Labs", "contact_email": "ops@example.com", "product": "Insight Suite"},
        ),
        (
            "חלץ מידע מובנה והחזר JSON קומפקטי בלבד.",
            "חלץ את השדות customer_name, amount, due_date מהטקסט: ללקוח רון כהן הופקה חשבונית על סך 1,250 ש\"ח, לתשלום עד 2026-05-10.",
            {"customer_name": "רון כהן", "amount": "1,250 ש\"ח", "due_date": "2026-05-10"},
        ),
    ]
    for system, user, payload in extraction_examples:
        add(rows, system, user, json.dumps(payload, ensure_ascii=False))

    knowledge_examples = [
        (
            "Answer only from the provided internal notes.",
            "Notes: Employees can work remotely up to 3 days a week. Question: How many remote days are allowed?",
            "Employees can work remotely up to 3 days per week.",
        ),
        (
            "Answer only from the provided internal notes.",
            "Notes: API keys rotate every 90 days. Question: How often should API keys rotate?",
            "API keys should rotate every 90 days.",
        ),
        (
            "ענה רק על בסיס החומר שסופק.",
            "הערות: לקוחות enterprise מקבלים SLA של שעתיים. שאלה: מה זמן התגובה ללקוחות enterprise?",
            "זמן התגובה ללקוחות enterprise הוא שעתיים.",
        ),
    ]
    for system, user, assistant in knowledge_examples:
        add(rows, system, user, assistant)

    classification_examples = [
        (
            "Classify the text and return compact JSON only.",
            "Labels: hot_lead, warm_lead, low_priority. Text: I need a proposal today and want to sign this week.",
            {"label": "hot_lead", "score": 5, "reason": "urgent buying intent"},
        ),
        (
            "Classify the text and return compact JSON only.",
            "Labels: bug, feature_request, question. Text: The export button crashes every time I click it.",
            {"label": "bug", "score": 5, "reason": "clear product failure"},
        ),
        (
            "סווג את הטקסט והחזר רק JSON קומפקטי.",
            "קטגוריות: חיובי, נייטרלי, שלילי. טקסט: השירות היה מהיר ומדויק ואני מרוצה מאוד.",
            {"label": "חיובי", "score": 5, "reason": "שביעות רצון גבוהה"},
        ),
    ]
    for system, user, payload in classification_examples:
        add(rows, system, user, json.dumps(payload, ensure_ascii=False))

    return rows


def wiki_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in load_jsonl(RAW_DIR / "wiki_pages_en.jsonl"):
        if row.get("title") not in EN_ALLOWED_TITLES:
            continue
        summary = first_sentence(row.get("text", ""))
        title = row["title"]
        add(rows, "Answer briefly in English.", f"What is {title}?", summary)
        add(rows, "Answer briefly in English.", f"Explain {title} in one sentence.", summary)
        add(rows, "Respond in compact JSON only.", f"Explain {title} in JSON.", json.dumps({"topic": title, "summary": summary}, ensure_ascii=False))
    for row in load_jsonl(RAW_DIR / "wiki_pages_he.jsonl"):
        if row.get("title") not in HE_ALLOWED_TITLES:
            continue
        summary = first_sentence(row.get("text", ""))
        title = row["title"]
        add(rows, "ענה בקצרה ובעברית.", f"מה זה {title}?", summary)
        add(rows, "ענה בעברית פשוטה.", f"תסביר בקצרה מה זה {title}.", summary)
        add(rows, "ענה ב-JSON בלבד.", f"תסביר את {title} ב-JSON.", json.dumps({"topic": title, "summary": summary}, ensure_ascii=False))
    return rows


def squad_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in load_squad_rows(RAW_DIR / "squad_train_v2.json", seed=SEED, limit=80):
        add(rows, "Answer briefly and factually.", row["user"], row["assistant"])
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["system"], row["user"], row["assistant"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    rng = random.Random(SEED)
    rows = dedupe(generic_manual_rows() + wiki_rows() + squad_rows())
    rng.shuffle(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

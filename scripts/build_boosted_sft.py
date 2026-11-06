#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import build_sft_tensors
from tokenizer.tokenizer_utils import SentencePieceTokenizer


TOPICS = [
    {
        "topic_en": "SEO",
        "topic_he": "SEO",
        "summary_en": "SEO is the practice of improving a website so it can appear more clearly in search engine results.",
        "summary_he": "SEO הוא תהליך של שיפור נראות אתר במנועי חיפוש.",
    },
    {
        "topic_en": "Artificial intelligence",
        "topic_he": "בינה מלאכותית",
        "summary_en": "Artificial intelligence is the field of building systems that perform tasks associated with human intelligence.",
        "summary_he": "בינה מלאכותית היא תחום שמפתח מערכות שמבצעות משימות הקשורות לאינטליגנציה אנושית.",
    },
    {
        "topic_en": "Machine learning",
        "topic_he": "למידת מכונה",
        "summary_en": "Machine learning is a branch of AI that learns patterns from data in order to make predictions or decisions.",
        "summary_he": "למידת מכונה היא תחום בבינה מלאכותית שבו מודלים לומדים דפוסים מנתונים כדי לבצע חיזוי או קבלת החלטות.",
    },
    {
        "topic_en": "Natural language processing",
        "topic_he": "עיבוד שפה טבעית",
        "summary_en": "Natural language processing is the field of helping computers understand and generate human language.",
        "summary_he": "עיבוד שפה טבעית הוא תחום שמאפשר למחשבים להבין ולהפיק שפה אנושית.",
    },
    {
        "topic_en": "Google",
        "topic_he": "גוגל",
        "summary_en": "Google is a technology company best known for its search engine, ads platform, Android, YouTube, and cloud products.",
        "summary_he": "גוגל היא חברת טכנולוגיה שמוכרת בעיקר בזכות מנוע החיפוש שלה, הפרסום הדיגיטלי, אנדרואיד, יוטיוב ושירותי הענן שלה.",
    },
    {
        "topic_en": "Search engine",
        "topic_he": "מנוע חיפוש",
        "summary_en": "A search engine is a system that indexes information and returns relevant results for a query.",
        "summary_he": "מנוע חיפוש הוא מערכת שמאנדקסת מידע ומחזירה תוצאות רלוונטיות לשאילתה.",
    },
    {
        "topic_en": "Digital marketing",
        "topic_he": "שיווק דיגיטלי",
        "summary_en": "Digital marketing is the use of online channels to promote products, services, or brands.",
        "summary_he": "שיווק דיגיטלי הוא שימוש בערוצים מקוונים כדי לקדם מוצרים, שירותים או מותגים.",
    },
    {
        "topic_en": "Content marketing",
        "topic_he": "שיווק באמצעות תוכן",
        "summary_en": "Content marketing focuses on creating useful content that attracts and educates a target audience.",
        "summary_he": "שיווק באמצעות תוכן מתמקד ביצירת תוכן מועיל שמושך קהל יעד ומסייע לו.",
    },
    {
        "topic_en": "Automation",
        "topic_he": "אוטומציה",
        "summary_en": "Automation is the use of systems or software to perform tasks with minimal manual work.",
        "summary_he": "אוטומציה היא שימוש במערכות או תוכנה כדי לבצע משימות עם מינימום עבודה ידנית.",
    },
    {
        "topic_en": "JSON",
        "topic_he": "JSON",
        "summary_en": "JSON is a lightweight text format used to store and exchange structured data.",
        "summary_he": "JSON הוא פורמט טקסט קל המשמש לייצוג ולהעברת מידע מובנה.",
    },
    {
        "topic_en": "Website",
        "topic_he": "אתר אינטרנט",
        "summary_en": "A website is a collection of web pages published under a shared domain.",
        "summary_he": "אתר אינטרנט הוא אוסף של דפי רשת שמתפרסמים תחת דומיין משותף.",
    },
    {
        "topic_en": "User interface",
        "topic_he": "ממשק משתמש",
        "summary_en": "A user interface is the part of a system through which people interact with software or devices.",
        "summary_he": "ממשק משתמש הוא החלק במערכת שבאמצעותו אנשים מתקשרים עם תוכנה או מכשיר.",
    },
]

TRANSLATION_PAIRS = [
    ("house", "בית"),
    ("computer", "מחשב"),
    ("website", "אתר"),
    ("search engine", "מנוע חיפוש"),
    ("marketing", "שיווק"),
    ("content", "תוכן"),
    ("automation", "אוטומציה"),
    ("assistant", "עוזר"),
    ("translation", "תרגום"),
    ("title", "כותרת"),
    ("description", "תיאור"),
    ("data", "נתונים"),
    ("article", "מאמר"),
    ("ai", "בינה מלאכותית"),
    ("seo", "קידום אתרים"),
    ("prompt", "פרומפט"),
    ("hello", "שלום"),
    ("thank you", "תודה"),
]

MARKETING_TOPICS = [
    ("AI article", "מאמר על בינה מלאכותית"),
    ("SEO guide", "מדריך SEO"),
    ("marketing automation tool", "כלי אוטומציה שיווקית"),
    ("small business website", "אתר לעסק קטן"),
    ("machine learning blog", "בלוג על למידת מכונה"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a boosted SFT dataset and tensors for the demo checkpoint.")
    parser.add_argument("--base", default="data/raw/sft_dialogues.jsonl")
    parser.add_argument("--output-raw", default="data/raw/sft_dialogues_boosted.jsonl")
    parser.add_argument("--tokenizer", default="tokenizer/tiny_bilingual.model")
    parser.add_argument("--train-out", default="data/processed/sft_train_boosted_128.pt")
    parser.add_argument("--val-out", default="data/processed/sft_val_boosted_128.pt")
    parser.add_argument("--stats-out", default="data/processed/sft_boosted_stats.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def add_row(rows: list[dict[str, str]], system: str, user: str, assistant: str) -> None:
    rows.append({"system": system, "user": user, "assistant": assistant})


def generate_topic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in TOPICS:
        topic_en = item["topic_en"]
        topic_he = item["topic_he"]
        summary_en = item["summary_en"]
        summary_he = item["summary_he"]

        for user in (
            f"What is {topic_en}?",
            f"Explain {topic_en} briefly.",
            f"Give a short definition of {topic_en}.",
        ):
            add_row(rows, "Answer briefly in English.", user, summary_en)

        for user in (
            f"מה זה {topic_he}?",
            f"תסביר בקצרה מה זה {topic_he}.",
            f"תן הגדרה קצרה ל{topic_he}.",
        ):
            add_row(rows, "ענה בקצרה ובעברית.", user, summary_he)

        add_row(rows, "Respond in compact JSON only.", f"Explain {topic_en} in JSON.", json.dumps({"topic": topic_en, "summary": summary_en}, ensure_ascii=False))
        add_row(rows, "Respond in compact JSON only.", f"Give JSON for {topic_en}.", json.dumps({"topic": topic_en, "summary": summary_en}, ensure_ascii=False))
        add_row(rows, "ענה ב-JSON קומפקטי.", f"תסביר את {topic_he} ב-JSON.", json.dumps({"topic": topic_he, "summary": summary_he}, ensure_ascii=False))
        add_row(rows, "ענה ב-JSON קומפקטי.", f"ענה לי ב-JSON מה זה {topic_he}", json.dumps({"topic": topic_he, "summary": summary_he}, ensure_ascii=False))

    add_row(rows, "Answer briefly and factually.", "Who are Google?", "Google is a technology company best known for search, ads, Android, YouTube, and cloud products.")
    add_row(rows, "Answer briefly and factually.", "Who is the greatest search engine?", "Google is generally the most widely used search engine.")
    add_row(rows, "Answer briefly and factually.", "Who is the best search engine?", "Google is generally the most widely used search engine.")
    add_row(rows, "ענה בקצרה ובעברית.", "מי זאת גוגל?", "גוגל היא חברת טכנולוגיה שמוכרת בעיקר בזכות מנוע החיפוש שלה, הפרסום הדיגיטלי, אנדרואיד, יוטיוב ושירותי ענן.")
    add_row(rows, "ענה בקצרה ובעברית.", "מה מנוע החיפוש הנפוץ ביותר?", "גוגל הוא מנוע החיפוש הנפוץ ביותר בעולם.")
    return rows


def generate_translation_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for english, hebrew in TRANSLATION_PAIRS:
        add_row(rows, "Act as a translation assistant.", f"Translate '{english}' to Hebrew.", hebrew)
        add_row(rows, "Act as a translation assistant.", f"Translate {english} to Hebrew.", hebrew)
        add_row(rows, "פעל כעוזר תרגום קצר.", f"תתרגם לאנגלית: {hebrew}", english)
        add_row(rows, "פעל כעוזר תרגום קצר.", f"תרגם לאנגלית: {hebrew}", english)
        add_row(
            rows,
            "Respond in compact JSON only.",
            f"Translate '{english}' to Hebrew in JSON.",
            json.dumps({"task": "translation", "source_text": english, "target_language": "hebrew", "translation": hebrew}, ensure_ascii=False),
        )
        add_row(
            rows,
            "ענה ב-JSON קומפקטי.",
            f"תתרגם לאנגלית ב-JSON: {hebrew}",
            json.dumps({"task": "translation", "source_text": hebrew, "target_language": "english", "translation": english}, ensure_ascii=False),
        )
    return rows


def generate_marketing_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for topic_en, topic_he in MARKETING_TOPICS:
        title_en = f"Clear {topic_en.title()} Guide"
        meta_en = f"Short clear overview of {topic_en}, highlighting the value, key use cases, and why it matters."
        title_he = f"מדריך קצר וברור ל{topic_he}"
        meta_he = f"סקירה קצרה וברורה על {topic_he}, עם הדגשת הערך, השימושים והיתרונות לקורא."

        add_row(rows, "Act as a tiny marketing assistant.", f"Write a short title for {topic_en}.", title_en)
        add_row(rows, "Act as a tiny SEO assistant.", f"Write a short meta description for {topic_en}.", meta_en)
        add_row(rows, "ענה בקצרה ובעברית.", f"כתוב לי כותרת קצרה ל{topic_he}.", title_he)
        add_row(rows, "ענה בקצרה ובעברית.", f"כתוב לי תיאור מטא קצר ל{topic_he}.", meta_he)
        add_row(
            rows,
            "Respond in compact JSON only.",
            f"Write a short meta description for {topic_en} in JSON.",
            json.dumps({"task": "marketing_meta_description", "topic": topic_en, "meta_description": meta_en}, ensure_ascii=False),
        )
        add_row(
            rows,
            "ענה ב-JSON קומפקטי.",
            f"כתוב לי תיאור מטא קצר ב-JSON ל{topic_he}.",
            json.dumps({"task": "marketing_meta_description", "topic": topic_he, "meta_description": meta_he}, ensure_ascii=False),
        )
    return rows


def generate_chat_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    english_rows = [
        ("Hello", "Hello. I am a tiny bilingual demo assistant."),
        ("Hi", "Hi. Ask me for a short explanation, translation, or SEO help."),
        ("How are you?", "I am fine. I am a tiny demo assistant, so I keep answers short."),
        ("What can you do?", "I can answer short questions, translate simple words, explain basic concepts, and draft short marketing text."),
        ("Help", "Try short prompts about SEO, AI, translation, titles, meta descriptions, or JSON output."),
    ]
    hebrew_rows = [
        ("היי", "שלום. אני עוזר דמו קטן לשאלות קצרות, תרגום ו-SEO."),
        ("שלום", "שלום. אפשר לשאול אותי על SEO, AI, תרגום, כותרות ותיאורי מטא."),
        ("מה שלומך?", "אני בסדר. אני עוזר דמו קטן ולכן אני עונה בקצרה."),
        ("מה אתה יודע לעשות?", "אני יכול לענות בקצרה, לתרגם מילים פשוטות, להסביר מושגים בסיסיים ולנסח טקסט שיווקי קצר."),
        ("עזרה", "נסה שאלות קצרות על SEO, AI, תרגום, כותרות, תיאורי מטא או JSON."),
    ]
    for user, assistant in english_rows:
        add_row(rows, "Answer briefly in English.", user, assistant)
    for user, assistant in hebrew_rows:
        add_row(rows, "ענה בקצרה ובעברית.", user, assistant)
    add_row(rows, "Respond in compact JSON only.", "What can you do in JSON?", json.dumps({"task": "capabilities", "summary": "short answers, translation, SEO, AI basics, and short marketing text"}, ensure_ascii=False))
    add_row(rows, "ענה ב-JSON קומפקטי.", "מה אתה יודע לעשות ב-JSON?", json.dumps({"task": "capabilities", "summary": "תשובות קצרות, תרגום, SEO, מושגי AI וטקסט שיווקי קצר"}, ensure_ascii=False))
    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("system", "") or "", row["user"], row["assistant"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base)
    generated_rows = []
    generated_rows.extend(generate_topic_rows())
    generated_rows.extend(generate_translation_rows())
    generated_rows.extend(generate_marketing_rows())
    generated_rows.extend(generate_chat_rows())

    final_rows = dedupe_rows(base_rows + generated_rows)

    output_raw = Path(args.output_raw)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    with output_raw.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    tensor_stats = build_sft_tensors(
        tokenizer=tokenizer,
        input_path=output_raw,
        train_out=args.train_out,
        val_out=args.val_out,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    stats = {
        "base_rows": len(base_rows),
        "generated_rows": len(generated_rows),
        "final_rows": len(final_rows),
        "train_examples": tensor_stats["train_examples"],
        "val_examples": tensor_stats["val_examples"],
        "output_raw": str(output_raw),
        "train_out": str(args.train_out),
        "val_out": str(args.val_out),
    }
    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

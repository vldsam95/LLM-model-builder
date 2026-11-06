from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

from src.data.preprocess import normalize_text

USER_AGENT = "tiny-bilingual-assistant/0.1 (educational project)"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

EN_TITLES = [
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Large language model",
    "Chatbot",
    "Automation",
    "Computer",
    "Website",
    "Translation",
    "JSON",
    "Marketing",
    "Digital marketing",
    "Search engine optimization",
    "Search engine",
    "Content marketing",
    "User interface",
    "Advertising",
    "Data",
]

HE_TITLES = [
    "בינה מלאכותית",
    "למידת מכונה",
    "עיבוד שפה טבעית",
    "מודל שפה גדול",
    "צ'טבוט",
    "אוטומציה",
    "מחשב",
    "אתר אינטרנט",
    "תרגום",
    "JSON",
    "שיווק",
    "שיווק דיגיטלי",
    "אופטימיזציה למנועי חיפוש",
    "מנוע חיפוש",
    "שיווק באמצעות תוכן",
    "ממשק משתמש",
    "פרסום",
    "נתונים",
]

DOMAIN_TITLE_KEYWORDS = {
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Large language model",
    "Chatbot",
    "Marketing",
    "Digital marketing",
    "Search engine optimization",
    "Search engine",
    "Content marketing",
    "Automation",
    "בינה מלאכותית",
    "למידת מכונה",
    "עיבוד שפה טבעית",
    "מודל שפה גדול",
    "צ'טבוט",
    "שיווק",
    "שיווק דיגיטלי",
    "אופטימיזציה למנועי חיפוש",
    "מנוע חיפוש",
    "שיווק באמצעות תוכן",
    "אוטומציה",
}

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
    ("AI", "בינה מלאכותית"),
    ("SEO", "קידום אתרים"),
    ("prompt", "פרומפט"),
    ("hello", "שלום"),
    ("good morning", "בוקר טוב"),
    ("good evening", "ערב טוב"),
    ("thank you", "תודה"),
]

MARKETING_TOPICS = [
    "AI article",
    "SEO guide",
    "marketing automation tool",
    "small business website",
    "blog about machine learning",
]


def _wiki_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def fetch_wikipedia_extract(session: requests.Session, lang: str, title: str) -> dict[str, str] | None:
    response = session.get(
        _wiki_url(lang),
        params={
            "action": "query",
            "prop": "extracts|info|revisions",
            "inprop": "url",
            "rvprop": "ids",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
            "titles": title,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    extract = normalize_text(page.get("extract", ""))
    if not extract:
        return None
    revisions = page.get("revisions", [])
    revision_id = str(revisions[0]["revid"]) if revisions else ""
    source_url = page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(page.get('title', title).replace(' ', '_'))}"
    return {
        "lang": lang,
        "title": page.get("title", title),
        "text": extract,
        "source_url": source_url,
        "revision_id": revision_id,
        "source_revision_url": f"{source_url}?oldid={revision_id}" if revision_id else source_url,
        "license": "CC-BY-SA-4.0",
        "attribution": f"{lang} Wikipedia contributors (see article history)",
    }


def save_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_wikipedia_corpus(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    en_rows = [row for title in EN_TITLES if (row := fetch_wikipedia_extract(session, "en", title))]
    he_rows = [row for title in HE_TITLES if (row := fetch_wikipedia_extract(session, "he", title))]

    save_jsonl(en_rows, output_dir / "wiki_pages_en.jsonl")
    save_jsonl(he_rows, output_dir / "wiki_pages_he.jsonl")

    en_text = "\n\n".join(row["text"] for row in en_rows)
    he_text = "\n\n".join(row["text"] for row in he_rows)
    domain_text = "\n\n".join(
        row["text"]
        for row in en_rows + he_rows
        if row["title"] in DOMAIN_TITLE_KEYWORDS
    )

    (output_dir / "pretrain_en.txt").write_text(en_text, encoding="utf-8")
    (output_dir / "pretrain_he.txt").write_text(he_text, encoding="utf-8")
    (output_dir / "domain_notes.txt").write_text(domain_text, encoding="utf-8")

    return {
        "pretrain_en": str(output_dir / "pretrain_en.txt"),
        "pretrain_he": str(output_dir / "pretrain_he.txt"),
        "domain_notes": str(output_dir / "domain_notes.txt"),
        "wiki_en_jsonl": str(output_dir / "wiki_pages_en.jsonl"),
        "wiki_he_jsonl": str(output_dir / "wiki_pages_he.jsonl"),
    }


def download_squad(output_path: str | Path) -> str:
    response = requests.get(SQUAD_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    return str(output_path)


def _first_sentence(text: str) -> str:
    for sep in [". ", "? ", "! ", "। ", ":", " - "]:
        if sep in text:
            return text.split(sep)[0].strip() + ("" if sep == ":" else ".")
    return text[:160].strip()


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _sample_squad_examples(path: str | Path, seed: int, limit: int = 120) -> list[dict[str, str]]:
    rng = random.Random(seed)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[dict[str, str]] = []
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer = normalize_text(answers[0].get("text", ""))
                question = normalize_text(qa.get("question", ""))
                if 3 <= len(answer) <= 160 and question:
                    samples.append({"user": question, "assistant": answer})
    rng.shuffle(samples)
    return samples[:limit]


def build_sft_dataset(raw_dir: str | Path, output_path: str | Path, seed: int = 42) -> str:
    raw_dir = Path(raw_dir)
    rng = random.Random(seed)

    en_rows = _load_jsonl(raw_dir / "wiki_pages_en.jsonl")
    he_rows = _load_jsonl(raw_dir / "wiki_pages_he.jsonl")
    squad_rows = _sample_squad_examples(raw_dir / "squad_train_v2.json", seed=seed)

    dialogues: list[dict[str, str | None]] = []

    greetings = [
        ("Hello", "Hello. I am a tiny bilingual assistant."),
        ("Hi", "Hi. Ask me for a short explanation, translation, or SEO help."),
        ("שלום", "שלום. אני עוזר דו-לשוני קטן להדגמה לימודית."),
        ("מה אתה יודע לעשות?", "אני יכול לענות בקצרה, לתרגם בין אנגלית לעברית, ולהסביר מושגים בסיסיים."),
        ("What is SEO?", "SEO is the practice of improving a website so it can appear more clearly in search results."),
        ("What is AI?", "AI is the field of building systems that perform tasks associated with human intelligence."),
        ("מה זה SEO?", "SEO הוא תהליך של שיפור נראות אתר במנועי חיפוש."),
        ("מה זה בינה מלאכותית?", "בינה מלאכותית היא תחום שבונה מערכות שמבצעות משימות הקשורות לאינטליגנציה אנושית."),
    ]
    for user, assistant in greetings:
        dialogues.append({"system": "Answer briefly.", "user": user, "assistant": assistant})

    for row in en_rows:
        title = row["title"]
        answer = _first_sentence(row["text"])
        dialogues.extend(
            [
                {"system": "Answer briefly in English.", "user": f"What is {title}?", "assistant": answer},
                {"system": "Answer briefly in English.", "user": f"Explain {title} in one sentence.", "assistant": answer},
                {
                    "system": "Respond in compact JSON.",
                    "user": f"Explain {title} in JSON.",
                    "assistant": json.dumps({"topic": title, "summary": answer}, ensure_ascii=False),
                },
            ]
        )

    for row in he_rows:
        title = row["title"]
        answer = _first_sentence(row["text"])
        dialogues.extend(
            [
                {"system": "ענה בקצרה ובעברית.", "user": f"מה זה {title}?", "assistant": answer},
                {"system": "ענה בעברית פשוטה.", "user": f"תסביר בקצרה מה זה {title}.", "assistant": answer},
                {
                    "system": "ענה ב-JSON קומפקטי.",
                    "user": f"תסביר את {title} ב-JSON.",
                    "assistant": json.dumps({"topic": title, "summary": answer}, ensure_ascii=False),
                },
            ]
        )

    for english, hebrew in TRANSLATION_PAIRS:
        dialogues.extend(
            [
                {
                    "system": "Act as a translation assistant.",
                    "user": f"Translate '{english}' to Hebrew.",
                    "assistant": hebrew,
                },
                {
                    "system": "פעל כעוזר תרגום קצר.",
                    "user": f"תתרגם לאנגלית: {hebrew}",
                    "assistant": english,
                },
            ]
        )

    for topic in MARKETING_TOPICS:
        dialogues.extend(
            [
                {
                    "system": "Act as a tiny marketing assistant.",
                    "user": f"Write a short title for a {topic}.",
                    "assistant": f"Clear {topic.title()} Guide",
                },
                {
                    "system": "Act as a tiny SEO assistant.",
                    "user": f"Write a short meta description for a {topic}.",
                    "assistant": f"Short practical overview of a {topic} for readers who want a quick introduction.",
                },
            ]
        )

    for row in squad_rows:
        dialogues.append({"system": "Answer briefly and factually.", "user": row["user"], "assistant": row["assistant"]})

    extra_json = [
        ("Explain SEO in JSON", {"topic": "SEO", "summary": "SEO is the practice of improving website visibility in search engines."}),
        ("Explain AI in JSON", {"topic": "AI", "summary": "AI is the field of building systems that perform tasks associated with human intelligence."}),
        ("ענה לי ב-JSON מה זה בינה מלאכותית", {"topic": "בינה מלאכותית", "summary": "תחום במדעי המחשב שמפתח מערכות המבצעות משימות הדורשות אינטליגנציה אנושית."}),
    ]
    for user, payload in extra_json:
        dialogues.append({"system": "Respond in compact JSON only.", "user": user, "assistant": json.dumps(payload, ensure_ascii=False)})

    dialogues.extend(
        [
            {
                "system": "Act as a translation assistant.",
                "user": "Translate 'house' to Hebrew",
                "assistant": "בית",
            },
            {
                "system": "Act as a tiny SEO assistant.",
                "user": "Write a short meta description for an AI article",
                "assistant": "Short clear meta description about AI basics, use cases, and why the topic matters.",
            },
        ]
    )

    rng.shuffle(dialogues)
    save_jsonl(dialogues, output_path)
    return str(output_path)


def write_source_manifest(output_path: str | Path) -> str:
    manifest = [
        {
            "name": "English and Hebrew Wikipedia extracts",
            "url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "English and Hebrew Wikipedia contributors; see linked article histories",
            "license_note": "CC BY-SA 4.0. Extracted, normalized, concatenated and converted into educational Q&A. See data/ATTRIBUTION.md.",
        },
        {
            "name": "SQuAD v2.0",
            "url": "https://rajpurkar.github.io/SQuAD-explorer/",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "Pranav Rajpurkar, Robin Jia, Percy Liang and the SQuAD contributors",
            "license_note": "CC BY-SA 4.0. Answerable Q&A sampled, filtered and reformatted into dialogues. Underlying passages originate from Wikipedia.",
        },
    ]
    output_path = Path(output_path)
    for lang, titles in (("en", EN_TITLES), ("he", HE_TITLES)):
        saved_path = output_path.parent / f"wiki_pages_{lang}.jsonl"
        saved = _load_jsonl(saved_path) if saved_path.exists() else []
        articles = saved or [{"title": title} for title in titles]
        for article in articles:
            title = article["title"]
            url = article.get("source_url") or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            manifest.append({
                "name": f"Wikipedia ({lang}): {title}",
                "url": url,
                "license": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": f"{lang} Wikipedia contributors (see linked article history)",
                "revision_id": article.get("revision_id", ""),
                "source_revision_url": article.get("source_revision_url", url),
                "provenance_status": "downloaded_article" if saved else "requested_inventory_historical_revision_unknown",
                "license_note": "Plain-text extract normalized and used in consolidated corpora and educational Q&A; see data/ATTRIBUTION.md.",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def download_all_data(raw_dir: str | Path, seed: int = 42) -> dict[str, str]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = download_wikipedia_corpus(raw_dir)
    paths["squad_json"] = download_squad(raw_dir / "squad_train_v2.json")
    paths["sft_dialogues"] = build_sft_dataset(raw_dir, raw_dir / "sft_dialogues.jsonl", seed=seed)
    paths["source_manifest"] = write_source_manifest(raw_dir / "source_manifest.json")
    return paths

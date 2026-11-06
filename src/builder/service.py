from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = PROJECT_ROOT / "builder_projects"
DEFAULT_BASE_CHECKPOINT = "checkpoints/tiny_cpu_builder_base/best.pt"
FALLBACK_BASE_CHECKPOINT = "checkpoints/tiny_cpu_pretrain/best.pt"
DEFAULT_TOKENIZER = "tokenizer/tiny_bilingual.model"
NEUTRAL_FALLBACK_DIALOGUES = "data/raw/sft_builder_base.jsonl"
USER_AGENT = "tiny-model-builder/0.1"
PROJECT_FAMILY_LABELS = {
    "language_model": "Language Model",
    "predictive_ai": "Predictive AI",
}
PREDICTIVE_TASK_LABELS = {
    "classification": "Classification",
    "regression": "Regression",
    "forecasting": "Forecasting",
}

LEGACY_PROFILE_MAP = {
    "assistant_qa": "general_assistant",
    "faq": "internal_knowledge",
    "marketing_helper": "general_assistant",
}

PROFILE_LABELS = {
    "general_assistant": "General Assistant",
    "extraction_json": "Extraction / JSON",
    "internal_knowledge": "Internal Knowledge",
    "classification_scoring": "Classification / Scoring",
}

FOUNDATION_PACKS = [
    {
        "value": "none",
        "label": "Project Only",
        "description": "Use only the text, files, and examples that were added to this project.",
    },
    {
        "value": "general_open_pack",
        "label": "General Open Pack",
        "description": "Blend the project data with the local open bilingual foundation pack: Wikipedia extracts, SQuAD-derived QA, and neutral builder SFT seeds.",
    },
]

MODEL_PRESETS = {
    "safe_6m": {
        "label": "~5.6M Safe CPU",
        "description": "Best default for responsive CPU-only demos and short replies.",
        "sequence_length": 128,
        "vocab_size": 3200,
        "config": {
            "d_model": 256,
            "n_heads": 8,
            "n_kv_heads": 8,
            "n_layers": 6,
            "ffn_mult": 4.0,
            "dropout": 0.05,
            "bias": True,
            "norm_type": "layernorm",
            "ffn_type": "gelu",
            "tie_embeddings": True,
        },
    },
    "balanced_11m": {
        "label": "~11.2M Balanced CPU",
        "description": "The practical upper sweet spot for this VPS if you still want reasonable response latency.",
        "sequence_length": 192,
        "vocab_size": 4000,
        "config": {
            "d_model": 288,
            "n_heads": 8,
            "n_kv_heads": 4,
            "n_layers": 8,
            "ffn_mult": 4.0,
            "dropout": 0.1,
            "bias": True,
            "norm_type": "rmsnorm",
            "ffn_type": "swiglu",
            "tie_embeddings": True,
        },
    },
    "stretch_16m": {
        "label": "~15.2M Stretch CPU",
        "description": "Possible on this VPS for experimentation, but training is slower and live replies are noticeably less snappy.",
        "sequence_length": 192,
        "vocab_size": 4000,
        "config": {
            "d_model": 320,
            "n_heads": 8,
            "n_kv_heads": 4,
            "n_layers": 9,
            "ffn_mult": 4.0,
            "dropout": 0.1,
            "bias": True,
            "norm_type": "rmsnorm",
            "ffn_type": "swiglu",
            "tie_embeddings": True,
        },
    },
}

PROFILE_EXAMPLES = [
    {
        "profile": "extraction_json",
        "title": "Extraction / JSON Schema Builder",
        "summary": "Turn messy text into compact JSON with a stable field set.",
        "sample_input": "Schema fields: customer_name, product, renewal_date\nText: Dana Levi renewed the Analytics Pro plan on 2026-04-01.",
        "sample_output": '{"customer_name":"Dana Levi","product":"Analytics Pro","renewal_date":"2026-04-01"}',
    },
    {
        "profile": "internal_knowledge",
        "title": "FAQ / Internal Knowledge Assistant",
        "summary": "Use uploaded notes, URLs, or policy text to answer short internal questions.",
        "sample_input": "Notes: Employees can work from home up to 3 days a week.\nQuestion: How many remote days are allowed?",
        "sample_output": "Employees can work from home up to 3 days per week.",
    },
    {
        "profile": "classification_scoring",
        "title": "Classification / Scoring Mini-Models",
        "summary": "Label or score short text such as leads, tickets, intents, or risk levels.",
        "sample_input": "Labels: hot_lead, warm_lead, low_priority\nText: I need a proposal today and want to sign this week.",
        "sample_output": '{"label":"hot_lead","score":5,"reason":"urgent buying intent"}',
    },
    {
        "profile": "general_assistant",
        "title": "General Tiny Assistant",
        "summary": "A neutral base for short answers, translation, summaries, and JSON responses.",
        "sample_input": "Topic: mortgage\nQuestion: What is a mortgage?",
        "sample_output": "A mortgage is a loan used to buy property and is usually repaid over many years.",
    },
    {
        "profile": "predictive_tabular",
        "title": "Predictive AI / Forecasting",
        "summary": "Train a CPU-friendly model on CSV/XLSX data for classification, regression, or simple forecasting.",
        "sample_input": "Columns: carrier, origin, dep_hour, weather_score, previous_delay, delay_minutes\nTarget: delay_minutes",
        "sample_output": '{"prediction": 27.4, "summary": "Higher weather impact and previous delay increase expected delay."}',
    },
]


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


def project_dir(slug: str) -> Path:
    return PROJECTS_ROOT / slug


def ensure_project_dirs(slug: str) -> dict[str, Path]:
    root = project_dir(slug)
    paths = {
        "root": root,
        "raw": root / "raw",
        "processed": root / "processed",
        "configs": root / "configs",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "tokenizer": root / "tokenizer",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def manifest_path(slug: str) -> Path:
    return project_dir(slug) / "manifest.json"


def runtime_path(slug: str) -> Path:
    return project_dir(slug) / "runtime.json"


def predictive_model_path(slug: str) -> Path:
    return project_dir(slug) / "artifacts" / "predictive_model.joblib"


def predictive_evaluation_path(slug: str) -> Path:
    return project_dir(slug) / "artifacts" / "evaluation.json"


def predictive_review_path(slug: str) -> Path:
    return project_dir(slug) / "artifacts" / "dataset_review.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u200f", " ").replace("\u200e", " ")).strip()


def detect_language(text: str) -> str:
    if re.search(r"[\u0590-\u05FF]", text):
        return "he"
    return "en"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return [part.strip() for part in parts if part.strip()]


def first_sentences(text: str, count: int = 2, max_chars: int = 260) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_space(text)[:max_chars]
    joined = " ".join(sentences[:count]).strip()
    return joined[:max_chars].strip()


def fetch_url_text(url: str) -> str:
    response = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/plain" in content_type:
        return normalize_space(response.text)
    parser = _TextHTMLParser()
    parser.feed(response.text)
    return parser.get_text()


def parse_topics(raw: str) -> list[str]:
    topics = [normalize_space(item) for item in re.split(r"[,\n]", raw) if normalize_space(item)]
    return topics


def parse_list_field(raw: str) -> list[str]:
    return parse_topics(raw)


def parse_urls(raw: str) -> list[str]:
    urls = [line.strip() for line in raw.splitlines() if line.strip()]
    return urls


def normalize_profile(profile: str) -> str:
    profile = normalize_space(profile or "general_assistant")
    return LEGACY_PROFILE_MAP.get(profile, profile or "general_assistant")


def normalize_project_family(project_family: str) -> str:
    value = normalize_space(project_family or "language_model").lower()
    return value if value in PROJECT_FAMILY_LABELS else "language_model"


def normalize_predictive_task(task: str) -> str:
    value = normalize_space(task or "regression").lower()
    return value if value in PREDICTIVE_TASK_LABELS else "regression"


def effective_default_base_checkpoint() -> str:
    for candidate in (DEFAULT_BASE_CHECKPOINT, FALLBACK_BASE_CHECKPOINT):
        if (PROJECT_ROOT / candidate).exists():
            return candidate
    return FALLBACK_BASE_CHECKPOINT


def checkpoint_tokenizer_rel(checkpoint_rel: str) -> str:
    checkpoint_path = PROJECT_ROOT / checkpoint_rel
    if not checkpoint_path.exists():
        return DEFAULT_TOKENIZER
    parts = checkpoint_path.relative_to(PROJECT_ROOT).parts
    if len(parts) >= 3 and parts[0] == "builder_projects":
        return resolve_project_tokenizer(parts[1])
    return DEFAULT_TOKENIZER


def split_text_by_language(raw_text: str) -> tuple[str, str]:
    english_parts: list[str] = []
    hebrew_parts: list[str] = []
    paragraphs = [normalize_space(chunk) for chunk in re.split(r"\n{2,}", raw_text) if normalize_space(chunk)]
    for paragraph in paragraphs:
        if detect_language(paragraph) == "he":
            hebrew_parts.append(paragraph)
        else:
            english_parts.append(paragraph)
    return "\n\n".join(english_parts), "\n\n".join(hebrew_parts)


def parse_scoring_range(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", raw or "")
    if not match:
        return 1, 5
    low = int(match.group(1))
    high = int(match.group(2))
    if low == high:
        return low, high + 1
    return (low, high) if low < high else (high, low)


def midpoint(low: int, high: int) -> int:
    return int(round((low + high) / 2))


def parse_custom_dialogues(raw_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        obj = json.loads(candidate)
        if not obj.get("user") or not obj.get("assistant"):
            continue
        rows.append(
            {
                "system": normalize_space(obj.get("system", "")),
                "user": normalize_space(obj["user"]),
                "assistant": normalize_space(obj["assistant"]),
            }
        )
    return rows


def read_uploaded_texts(files: list[Any]) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    text_chunks: list[str] = []
    custom_dialogues: list[dict[str, str]] = []
    uploaded_meta: list[dict[str, str]] = []
    for storage in files:
        if not storage or not storage.filename:
            continue
        filename = storage.filename
        suffix = Path(filename).suffix.lower()
        content = storage.read()
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("utf-8", errors="ignore")
        uploaded_meta.append({"filename": filename, "suffix": suffix})
        if suffix == ".jsonl":
            custom_dialogues.extend(parse_custom_dialogues(decoded))
        else:
            text_chunks.append(decoded)
    return text_chunks, custom_dialogues, uploaded_meta


def save_dataset_file(storage: Any, output_dir: Path, existing_rel: str | None = None) -> tuple[str | None, dict[str, str] | None]:
    if storage and getattr(storage, "filename", ""):
        filename = Path(storage.filename).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xlsm"}:
            raise ValueError("Dataset file must be CSV or XLSX.")
        destination = output_dir / f"dataset{suffix}"
        storage.stream.seek(0)
        destination.write_bytes(storage.read())
        return str(destination.relative_to(PROJECT_ROOT)), {"filename": filename, "suffix": suffix}

    if existing_rel:
        existing_path = PROJECT_ROOT / existing_rel
        if existing_path.exists():
            return existing_rel, {"filename": existing_path.name, "suffix": existing_path.suffix.lower()}
    return None, None


def summarize_topic(topic: str, raw_text: str, specialization: str) -> str:
    paragraphs = [normalize_space(chunk) for chunk in re.split(r"\n{2,}", raw_text) if normalize_space(chunk)]
    topic_norm = topic.lower()
    for paragraph in paragraphs:
        paragraph_sentences = split_sentences(paragraph)
        for sentence in paragraph_sentences:
            if topic_norm in sentence.lower():
                return normalize_space(sentence)
        if topic_norm in paragraph.lower():
            return first_sentences(paragraph)
    if paragraphs:
        return first_sentences(paragraphs[0])
    if detect_language(topic) == "he":
        return f"{topic} הוא חלק מההתמחות {specialization}."
    return f"{topic} is part of the specialization {specialization}."


def add_row(rows: list[dict[str, str]], system: str, user: str, assistant: str) -> None:
    rows.append(
        {
            "system": normalize_space(system),
            "user": normalize_space(user),
            "assistant": normalize_space(assistant),
        }
    )


def extraction_payload(schema_fields: list[str], topic: str, summary: str, snippet: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    keywords = [word for word in re.findall(r"[\w\u0590-\u05FF]+", topic) if len(word) > 2][:3]
    for index, field in enumerate(schema_fields or ["topic", "summary"]):
        key = normalize_space(field).replace(" ", "_")
        key_lower = key.lower()
        if key_lower in {"topic", "subject", "name", "title"}:
            values[key] = topic
        elif key_lower in {"summary", "description", "answer", "details", "overview"}:
            values[key] = summary
        elif key_lower in {"language", "lang"}:
            values[key] = detect_language(summary)
        elif key_lower in {"keywords", "tags"}:
            values[key] = keywords or [topic]
        elif key_lower in {"source_text", "text"}:
            values[key] = snippet[:180]
        elif key_lower in {"confidence"}:
            values[key] = 0.92
        elif key_lower in {"score"}:
            values[key] = 5
        else:
            values[key] = topic if index == 0 else summary
    return values


def generate_seed_sft_rows(manifest: dict[str, Any], raw_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specialization = manifest.get("specialization") or manifest["name"]
    language_mode = manifest.get("language_mode", "bilingual")
    profile = normalize_profile(manifest.get("profile", "general_assistant"))
    system_prompt = manifest.get("system_prompt") or (
        "Answer briefly and clearly."
        if language_mode != "hebrew"
        else "ענה בקצרה ובעברית ברורה."
    )
    topics = manifest.get("topics") or [specialization]
    schema_fields = manifest.get("schema_fields") or ["topic", "summary"]
    label_options = manifest.get("label_options") or ["high_priority", "medium_priority", "low_priority"]
    score_low, score_high = parse_scoring_range(manifest.get("scoring_scale", "1-5"))
    score_mid = midpoint(score_low, score_high)
    positive_label = label_options[0]
    neutral_label = label_options[1] if len(label_options) > 1 else label_options[0]
    low_label = label_options[-1]

    for topic in topics:
        summary = summarize_topic(topic, raw_text, specialization)
        snippet = first_sentences(summary, count=2, max_chars=220)
        topic_language = detect_language(topic if len(topic) >= len(summary) else summary)
        if language_mode == "hebrew":
            topic_language = "he"
        elif language_mode == "english":
            topic_language = "en"

        if profile in {"general_assistant", "internal_knowledge"} and topic_language == "he":
            add_row(rows, "ענה בקצרה ובעברית.", f"מה זה {topic}?", summary)
            add_row(rows, "ענה בקצרה ובעברית.", f"תסביר בקצרה מה זה {topic}.", summary)
            add_row(rows, "ענה ב-JSON קומפקטי.", f"תסביר את {topic} ב-JSON.", json.dumps({"topic": topic, "summary": summary}, ensure_ascii=False))
        elif profile in {"general_assistant", "internal_knowledge"}:
            add_row(rows, "Answer briefly in English.", f"What is {topic}?", summary)
            add_row(rows, "Answer briefly in English.", f"Explain {topic} briefly.", summary)
            add_row(rows, "Respond in compact JSON only.", f"Explain {topic} in JSON.", json.dumps({"topic": topic, "summary": summary}, ensure_ascii=False))

        if profile == "internal_knowledge":
            if topic_language == "he":
                add_row(rows, "ענה רק על בסיס הידע שסופק לך. אם המידע חסר, אמור זאת.", f"מה אומר החומר על {topic}?", summary)
                add_row(rows, "ענה ב-JSON קומפקטי.", f"סכם את הידע הפנימי על {topic} ב-JSON.", json.dumps({"topic": topic, "answer": summary, "source": "project_notes"}, ensure_ascii=False))
            else:
                add_row(rows, "Answer only from the provided project notes. If the information is missing, say that the notes do not contain it.", f"What do the internal notes say about {topic}?", summary)
                add_row(rows, "Respond in compact JSON only.", f"Summarize the internal knowledge about {topic} in JSON.", json.dumps({"topic": topic, "answer": summary, "source": "project_notes"}, ensure_ascii=False))

        if profile == "extraction_json":
            payload = extraction_payload(schema_fields, topic, summary, snippet)
            if topic_language == "he":
                add_row(
                    rows,
                    "חלץ מידע מובנה והחזר JSON קומפקטי בלבד.",
                    f"חלץ את השדות {', '.join(schema_fields)} מהטקסט הבא: {snippet}",
                    json.dumps(payload, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "ענה ב-JSON בלבד.",
                    f"צור אובייקט JSON עם השדות {', '.join(schema_fields)} עבור הטקסט: {summary}",
                    json.dumps(payload, ensure_ascii=False),
                )
            else:
                add_row(
                    rows,
                    "Extract structured fields and return compact JSON only.",
                    f"Extract the fields {', '.join(schema_fields)} from this text: {snippet}",
                    json.dumps(payload, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "Respond with JSON only.",
                    f"Build a JSON object with the fields {', '.join(schema_fields)} for this text: {summary}",
                    json.dumps(payload, ensure_ascii=False),
                )

        if profile == "classification_scoring":
            labels_text = ", ".join(label_options)
            if topic_language == "he":
                add_row(
                    rows,
                    "סווג את הטקסט והחזר רק JSON קומפקטי.",
                    f"סווג את הטקסט הבא לאחת מהקטגוריות: {labels_text}. הטקסט: דחוף, צריך תשובה היום בנושא {topic}.",
                    json.dumps({"label": positive_label, "score": score_high, "reason": f"פנייה דחופה ורלוונטית ל-{topic}"}, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "סווג את הטקסט והחזר רק JSON קומפקטי.",
                    f"סווג את הטקסט הבא לאחת מהקטגוריות: {labels_text}. הטקסט: אזכור כללי של {topic} לבדיקה עתידית.",
                    json.dumps({"label": neutral_label, "score": score_mid, "reason": f"אזכור כללי של {topic}"}, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "סווג את הטקסט והחזר רק JSON קומפקטי.",
                    f"סווג את הטקסט הבא לאחת מהקטגוריות: {labels_text}. הטקסט: תוכן לא קשור לתחום המבוקש.",
                    json.dumps({"label": low_label, "score": score_low, "reason": "לא רלוונטי לנושא המרכזי"}, ensure_ascii=False),
                )
            else:
                add_row(
                    rows,
                    "Classify the text and return compact JSON only.",
                    f"Classify this text into one of: {labels_text}. Text: Urgent request, I need a reply today about {topic}.",
                    json.dumps({"label": positive_label, "score": score_high, "reason": f"urgent and clearly about {topic}"}, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "Classify the text and return compact JSON only.",
                    f"Classify this text into one of: {labels_text}. Text: General note mentioning {topic} for later review.",
                    json.dumps({"label": neutral_label, "score": score_mid, "reason": f"mentions {topic} without urgency"}, ensure_ascii=False),
                )
                add_row(
                    rows,
                    "Classify the text and return compact JSON only.",
                    f"Classify this text into one of: {labels_text}. Text: Unrelated low-signal content.",
                    json.dumps({"label": low_label, "score": score_low, "reason": "weak relevance"}, ensure_ascii=False),
                )

    if profile in {"general_assistant", "internal_knowledge", "extraction_json", "classification_scoring"}:
        if language_mode == "hebrew":
            capability = {
                "general_assistant": "אני יכול לענות בקצרה, לסכם, לתרגם ולתת JSON פשוט.",
                "internal_knowledge": "אני יכול לענות על בסיס החומר שהוזן לפרויקט ולסכם אותו בקצרה.",
                "extraction_json": "אני יכול לחלץ שדות מוגדרים מראש ולהחזיר JSON קומפקטי.",
                "classification_scoring": "אני יכול לסווג טקסטים קצרים ולהחזיר תווית או ציון ב-JSON.",
            }[profile]
            add_row(rows, "ענה בקצרה ובעברית.", "מה אתה יודע לעשות?", capability)
        else:
            capability = {
                "general_assistant": "I can answer short questions, summarize content, translate simple phrases, and return compact JSON.",
                "internal_knowledge": "I can answer short questions from the uploaded project notes and summarize them.",
                "extraction_json": "I can extract predefined fields from text and return compact JSON.",
                "classification_scoring": "I can classify short text and return labels or scores in JSON.",
            }[profile]
            add_row(rows, "Answer briefly and clearly.", "What can you do?", capability)
            add_row(rows, "Respond in compact JSON only.", "What can you do in JSON?", json.dumps({"task": profile, "summary": capability}, ensure_ascii=False))

    if system_prompt:
        if language_mode == "hebrew":
            add_row(rows, system_prompt, "היי", "שלום. אני מודל tiny מותאם לפרויקט הזה.")
        else:
            add_row(rows, system_prompt, "Hello", "Hello. I am a tiny model adapted for this project.")

    return dedupe_rows(rows)


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


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_or_update_project(form: dict[str, str], files: list[Any], dataset_file: Any | None = None) -> dict[str, Any]:
    name = normalize_space(form.get("name", ""))
    if not name:
        raise ValueError("Project name is required.")

    slug = slugify(form.get("slug") or name)
    paths = ensure_project_dirs(slug)
    existing_manifest = read_json(manifest_path(slug), {})
    project_family = normalize_project_family(form.get("project_family", existing_manifest.get("project_family", "language_model")))

    uploaded_texts, uploaded_dialogues, uploaded_meta = read_uploaded_texts(files)
    raw_dir = paths["raw"]
    manifest_base = {
        "name": name,
        "slug": slug,
        "description": normalize_space(form.get("description", "")),
        "specialization": normalize_space(form.get("specialization", "")) or name,
        "project_family": project_family,
        "created_at": existing_manifest.get("created_at") or utc_now(),
        "updated_at": utc_now(),
        "uploaded_files": uploaded_meta,
    }

    if project_family == "predictive_ai":
        dataset_path_rel, dataset_meta = save_dataset_file(dataset_file, raw_dir, existing_manifest.get("dataset_path"))
        if dataset_meta:
            uploaded_meta.append(dataset_meta)
        if not dataset_path_rel:
            raise ValueError("Predictive AI projects require a CSV or XLSX dataset file.")

        raw_notes = "\n\n".join(chunk for chunk in [form.get("predictive_notes", "")] + uploaded_texts if normalize_space(chunk))
        (raw_dir / "notes.txt").write_text(raw_notes, encoding="utf-8")

        dataset_overview: dict[str, Any] = {}
        try:
            from src.predictive.tabular import inspect_tabular_dataset

            overview = inspect_tabular_dataset(PROJECT_ROOT / dataset_path_rel)
            dataset_overview = {
                "rows": overview.rows,
                "columns": overview.columns,
                "column_names": overview.column_names,
                "preview": overview.preview,
                "dtypes": overview.dtypes,
            }
        except Exception as exc:
            dataset_overview = {"error": str(exc)}

        manifest = {
            **manifest_base,
            "profile": "predictive_tabular",
            "build_mode": "predictive_tabular",
            "predictive_task": normalize_predictive_task(form.get("predictive_task", existing_manifest.get("predictive_task", "regression"))),
            "target_column": normalize_space(form.get("target_column", existing_manifest.get("target_column", ""))),
            "time_column": normalize_space(form.get("time_column", existing_manifest.get("time_column", ""))),
            "feature_columns": parse_list_field(form.get("feature_columns", "")) or existing_manifest.get("feature_columns", []),
            "forecast_horizon": int(form.get("forecast_horizon", existing_manifest.get("forecast_horizon", 3)) or 3),
            "dataset_path": dataset_path_rel,
            "dataset_overview": dataset_overview,
            "system_prompt": normalize_space(form.get("system_prompt", "")),
            "topics": parse_topics(form.get("topics", "")),
            "base_checkpoint": None,
            "tokenizer_model": None,
            "language_mode": "structured",
            "foundation_pack": "none",
            "model_size": "tabular_cpu",
            "schema_fields": [],
            "label_options": [],
            "scoring_scale": "",
            "url_records": [],
        }
        manifest["stats"] = {
            "raw_notes_chars": len(raw_notes),
            "dataset_rows": dataset_overview.get("rows", 0),
            "dataset_columns": dataset_overview.get("columns", 0),
        }
        write_json(manifest_path(slug), manifest)

        runtime = read_json(runtime_path(slug), {})
        runtime.setdefault("status", "idle")
        runtime.setdefault("active_model", str(predictive_model_path(slug).relative_to(PROJECT_ROOT)) if predictive_model_path(slug).exists() else None)
        runtime.setdefault("evaluation_path", str(predictive_evaluation_path(slug).relative_to(PROJECT_ROOT)) if predictive_evaluation_path(slug).exists() else None)
        runtime["active_checkpoint"] = None
        runtime["active_tokenizer"] = None
        runtime.setdefault("log_path", str(paths["logs"] / "train.log"))
        write_json(runtime_path(slug), runtime)
        return project_summary(slug)

    url_records: list[dict[str, str]] = []
    url_texts: list[str] = []
    for url in parse_urls(form.get("urls", "")):
        try:
            text = fetch_url_text(url)
            url_texts.append(text)
            url_records.append({"url": url, "status": "ok", "chars": str(len(text))})
        except Exception as exc:  # pragma: no cover - best effort fetch
            url_records.append({"url": url, "status": "error", "error": str(exc)})

    raw_text_chunks = [form.get("raw_text", "")] + uploaded_texts + url_texts
    raw_text = "\n\n".join(chunk for chunk in raw_text_chunks if normalize_space(chunk))
    custom_dialogues = parse_custom_dialogues(form.get("custom_dialogues", "")) + uploaded_dialogues
    profile = normalize_profile(form.get("profile", "general_assistant"))
    foundation_pack = form.get("foundation_pack", "none") or "none"
    model_size = form.get("model_size", "safe_6m") or "safe_6m"
    schema_fields = parse_list_field(form.get("schema_fields", ""))
    label_options = parse_list_field(form.get("label_options", ""))
    scoring_scale = normalize_space(form.get("scoring_scale", "1-5")) or "1-5"
    default_base = effective_default_base_checkpoint()
    base_checkpoint = form.get("base_checkpoint") or default_base

    manifest = {
        **manifest_base,
        "profile": profile,
        "build_mode": form.get("build_mode", "fast_adapter"),
        "language_mode": form.get("language_mode", "bilingual"),
        "foundation_pack": foundation_pack,
        "model_size": model_size,
        "topics": parse_topics(form.get("topics", "")),
        "schema_fields": schema_fields,
        "label_options": label_options,
        "scoring_scale": scoring_scale,
        "system_prompt": normalize_space(form.get("system_prompt", "")),
        "base_checkpoint": base_checkpoint,
        "tokenizer_model": checkpoint_tokenizer_rel(base_checkpoint),
        "sequence_length": int(form.get("sequence_length", "128") or 128),
        "vocab_size": int(form.get("vocab_size", "3200") or 3200),
        "url_records": url_records,
    }

    seed_rows = generate_seed_sft_rows(manifest, raw_text)
    final_rows = dedupe_rows(seed_rows + custom_dialogues)

    pretrain_en, pretrain_he = split_text_by_language(raw_text)
    if manifest["language_mode"] == "hebrew" and raw_text and not pretrain_he:
        pretrain_he = raw_text
    if manifest["language_mode"] == "english" and raw_text and not pretrain_en:
        pretrain_en = raw_text
    if raw_text and not pretrain_en and not pretrain_he:
        pretrain_en = raw_text
    (raw_dir / "source_text.txt").write_text(raw_text, encoding="utf-8")
    (raw_dir / "pretrain_en.txt").write_text(pretrain_en, encoding="utf-8")
    (raw_dir / "pretrain_he.txt").write_text(pretrain_he, encoding="utf-8")
    (raw_dir / "domain_notes.txt").write_text(raw_text, encoding="utf-8")
    write_jsonl(raw_dir / "sft_dialogues.jsonl", final_rows)

    manifest["stats"] = {
        "raw_text_chars": len(raw_text),
        "pretrain_en_chars": len(pretrain_en),
        "pretrain_he_chars": len(pretrain_he),
        "seed_dialogues": len(seed_rows),
        "custom_dialogues": len(custom_dialogues),
        "final_dialogues": len(final_rows),
    }
    write_json(manifest_path(slug), manifest)

    runtime = read_json(runtime_path(slug), {})
    runtime.setdefault("status", "idle")
    runtime.setdefault("active_checkpoint", manifest["base_checkpoint"])
    runtime.setdefault("active_tokenizer", manifest["tokenizer_model"])
    runtime["active_model"] = None
    runtime["evaluation_path"] = None
    runtime.setdefault("log_path", str(paths["logs"] / "train.log"))
    write_json(runtime_path(slug), runtime)
    return project_summary(slug)


def project_summary(slug: str) -> dict[str, Any]:
    manifest = read_json(manifest_path(slug), {})
    runtime = read_json(runtime_path(slug), {})
    evaluation = read_json(predictive_evaluation_path(slug), {}) if predictive_evaluation_path(slug).exists() else {}
    project_family = manifest.get("project_family", "language_model")
    return {
        "slug": slug,
        "name": manifest.get("name", slug),
        "description": manifest.get("description", ""),
        "specialization": manifest.get("specialization", ""),
        "project_family": project_family,
        "profile": manifest.get("profile", ""),
        "build_mode": manifest.get("build_mode", ""),
        "language_mode": manifest.get("language_mode", ""),
        "foundation_pack": manifest.get("foundation_pack", "none"),
        "model_size": manifest.get("model_size", "safe_6m"),
        "topics": manifest.get("topics", []),
        "schema_fields": manifest.get("schema_fields", []),
        "label_options": manifest.get("label_options", []),
        "scoring_scale": manifest.get("scoring_scale", "1-5"),
        "predictive_task": manifest.get("predictive_task", ""),
        "target_column": manifest.get("target_column", ""),
        "time_column": manifest.get("time_column", ""),
        "feature_columns": manifest.get("feature_columns", []),
        "forecast_horizon": manifest.get("forecast_horizon", 0),
        "dataset_path": manifest.get("dataset_path"),
        "dataset_overview": manifest.get("dataset_overview", {}),
        "updated_at": manifest.get("updated_at"),
        "stats": manifest.get("stats", {}),
        "status": runtime.get("status", "idle"),
        "active_checkpoint": runtime.get("active_checkpoint", manifest.get("base_checkpoint", effective_default_base_checkpoint())) if project_family != "predictive_ai" else None,
        "active_tokenizer": runtime.get("active_tokenizer", manifest.get("tokenizer_model", DEFAULT_TOKENIZER)) if project_family != "predictive_ai" else None,
        "active_model": runtime.get("active_model"),
        "evaluation_path": runtime.get("evaluation_path"),
        "evaluation_summary": evaluation.get("summary_text", ""),
        "evaluation_metrics": evaluation.get("metrics", {}),
        "input_schema": evaluation.get("input_schema", []),
        "log_path": runtime.get("log_path", str(project_dir(slug) / "logs" / "train.log")),
        "pid": runtime.get("pid"),
        "error": runtime.get("error"),
    }


def list_projects() -> list[dict[str, Any]]:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = [project_summary(path.name) for path in PROJECTS_ROOT.iterdir() if path.is_dir()]
    return sorted(projects, key=lambda item: item.get("updated_at") or "", reverse=True)


def model_catalog() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in sorted((PROJECT_ROOT / "checkpoints").rglob("best.pt")):
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel in seen:
            continue
        seen.add(rel)
        entries.append(
            {
                "path": rel,
                "origin": "core",
                "project": None,
                "profile": "base",
                "build_mode": "built_in",
                "recommended_for": "global base / evaluation",
            }
        )

    for path in sorted(PROJECTS_ROOT.rglob("best.pt")):
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel in seen:
            continue
        slug = path.relative_to(PROJECTS_ROOT).parts[0]
        manifest = read_json(manifest_path(slug), {})
        seen.add(rel)
        entries.append(
            {
                "path": rel,
                "origin": "project",
                "project": manifest.get("name", slug),
                "profile": normalize_profile(manifest.get("profile", "")),
                "build_mode": manifest.get("build_mode", ""),
                "recommended_for": manifest.get("specialization", "") or manifest.get("description", "") or slug,
            }
        )

    for path in sorted(PROJECTS_ROOT.rglob("predictive_model.joblib")):
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel in seen:
            continue
        slug = path.relative_to(PROJECTS_ROOT).parts[0]
        manifest = read_json(manifest_path(slug), {})
        seen.add(rel)
        entries.append(
            {
                "path": rel,
                "origin": "project",
                "project": manifest.get("name", slug),
                "profile": "predictive_tabular",
                "build_mode": manifest.get("build_mode", "predictive_tabular"),
                "recommended_for": manifest.get("specialization", "") or manifest.get("description", "") or slug,
            }
        )

    default_path = effective_default_base_checkpoint()
    for entry in entries:
        entry["is_default_builder_base"] = entry["path"] == default_path
    return entries


def checkpoint_catalog_paths() -> list[str]:
    paths = [entry["path"] for entry in model_catalog() if entry["path"].endswith(".pt")]
    default_path = effective_default_base_checkpoint()
    if default_path in paths:
        paths.remove(default_path)
        paths.insert(0, default_path)
    return paths


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def refresh_runtime(slug: str) -> dict[str, Any]:
    runtime = read_json(runtime_path(slug), {"status": "idle"})
    if runtime.get("status") == "running" and not is_pid_running(runtime.get("pid")):
        manifest = load_project_manifest(slug)
        if manifest.get("project_family") == "predictive_ai":
            model_rel = resolve_project_predictive_model(slug)
            evaluation_rel = resolve_project_predictive_evaluation(slug)
            runtime["status"] = "completed" if model_rel else "failed"
            runtime["completed_at"] = utc_now()
            if model_rel:
                runtime["active_model"] = model_rel
                runtime["evaluation_path"] = evaluation_rel
                runtime["active_checkpoint"] = None
                runtime["active_tokenizer"] = None
            write_json(runtime_path(slug), runtime)
            return runtime
        checkpoint = resolve_project_checkpoint(slug)
        runtime["status"] = "completed" if checkpoint else "failed"
        runtime["completed_at"] = utc_now()
        if checkpoint:
            runtime["active_checkpoint"] = checkpoint
            runtime["active_tokenizer"] = resolve_project_tokenizer(slug)
        write_json(runtime_path(slug), runtime)
    return runtime


def resolve_project_checkpoint(slug: str) -> str | None:
    root = project_dir(slug)
    candidates = [
        root / "checkpoints" / "fast_adapter" / "best.pt",
        root / "checkpoints" / "sft" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path.relative_to(PROJECT_ROOT))
    return None


def resolve_project_predictive_model(slug: str) -> str | None:
    path = predictive_model_path(slug)
    if path.exists():
        return str(path.relative_to(PROJECT_ROOT))
    return None


def resolve_project_predictive_evaluation(slug: str) -> str | None:
    path = predictive_evaluation_path(slug)
    if path.exists():
        return str(path.relative_to(PROJECT_ROOT))
    return None


def resolve_project_tokenizer(slug: str) -> str:
    custom_tokenizer = project_dir(slug) / "tokenizer" / "project.model"
    if custom_tokenizer.exists():
        return str(custom_tokenizer.relative_to(PROJECT_ROOT))
    return DEFAULT_TOKENIZER


def start_training_job(slug: str) -> dict[str, Any]:
    ensure_project_dirs(slug)
    manifest = load_project_manifest(slug)
    runtime = refresh_runtime(slug)
    if runtime.get("status") == "running" and is_pid_running(runtime.get("pid")):
        raise ValueError("Training job is already running for this project.")

    log_path = project_dir(slug) / "logs" / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "scripts/run_builder_predictive_project.py" if manifest.get("project_family") == "predictive_ai" else "scripts/run_builder_project.py",
        "--project-slug",
        slug,
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime = {
        "status": "running",
        "pid": process.pid,
        "started_at": utc_now(),
        "completed_at": None,
        "error": None,
        "log_path": str(log_path),
        "active_checkpoint": manifest.get("base_checkpoint", effective_default_base_checkpoint()) if manifest.get("project_family") != "predictive_ai" else None,
        "active_tokenizer": resolve_project_tokenizer(slug) if manifest.get("project_family") != "predictive_ai" else None,
        "active_model": resolve_project_predictive_model(slug) if manifest.get("project_family") == "predictive_ai" else None,
        "evaluation_path": resolve_project_predictive_evaluation(slug) if manifest.get("project_family") == "predictive_ai" else None,
    }
    write_json(runtime_path(slug), runtime)
    return runtime


def update_runtime_failure(slug: str, error_text: str) -> None:
    manifest = load_project_manifest(slug)
    runtime = {
        "status": "failed",
        "pid": None,
        "started_at": None,
        "completed_at": utc_now(),
        "error": error_text,
        "log_path": str(project_dir(slug) / "logs" / "train.log"),
        "active_checkpoint": manifest.get("base_checkpoint", effective_default_base_checkpoint()) if manifest.get("project_family") != "predictive_ai" else None,
        "active_tokenizer": resolve_project_tokenizer(slug) if manifest.get("project_family") != "predictive_ai" else None,
        "active_model": resolve_project_predictive_model(slug) if manifest.get("project_family") == "predictive_ai" else None,
        "evaluation_path": resolve_project_predictive_evaluation(slug) if manifest.get("project_family") == "predictive_ai" else None,
    }
    write_json(runtime_path(slug), runtime)


def read_log_tail(slug: str, max_chars: int = 8000) -> str:
    log_path = project_dir(slug) / "logs" / "train.log"
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    return content[-max_chars:]


def load_project_manifest(slug: str) -> dict[str, Any]:
    manifest = read_json(manifest_path(slug), {})
    if not manifest:
        raise FileNotFoundError(f"Unknown project slug: {slug}")
    return manifest


def project_chat_artifacts(slug: str) -> tuple[Path, Path]:
    manifest = load_project_manifest(slug)
    if manifest.get("project_family") == "predictive_ai":
        raise ValueError("Predictive AI projects do not expose chat artifacts.")
    runtime = refresh_runtime(slug)
    checkpoint_rel = runtime.get("active_checkpoint") or manifest.get("base_checkpoint") or effective_default_base_checkpoint()
    tokenizer_rel = runtime.get("active_tokenizer") or manifest.get("tokenizer_model") or DEFAULT_TOKENIZER
    checkpoint = PROJECT_ROOT / checkpoint_rel
    tokenizer = PROJECT_ROOT / tokenizer_rel
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for project {slug}: {checkpoint}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found for project {slug}: {tokenizer}")
    return checkpoint, tokenizer


def predictive_project_artifacts(slug: str) -> tuple[Path, dict[str, Any]]:
    manifest = load_project_manifest(slug)
    if manifest.get("project_family") != "predictive_ai":
        raise ValueError("Project is not a predictive AI project.")
    runtime = refresh_runtime(slug)
    model_rel = runtime.get("active_model") or resolve_project_predictive_model(slug)
    if not model_rel:
        raise FileNotFoundError(f"Predictive model not found for project {slug}.")
    evaluation = read_json(predictive_evaluation_path(slug), {}) if predictive_evaluation_path(slug).exists() else {}
    return PROJECT_ROOT / model_rel, evaluation


def write_builder_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

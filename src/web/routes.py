from __future__ import annotations

import gc
import json
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from src.inference.chat_engine import ChatEngine

web_bp = Blueprint("web", __name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tiny_bilingual.model"
ORIGINAL_SCOPE_PROMPT_PATH = PROJECT_ROOT / "data" / "examples" / "original_scope_prompt.md"
RECOMMENDED_CHECKPOINT = "checkpoints/tiny_cpu_sft_demo_boosted/best.pt"

PRESETS = [
    {"name": "Helpful Assistant", "system": "Answer briefly and clearly."},
    {"name": "SEO Assistant", "system": "Act as a tiny SEO assistant. Answer briefly and practically."},
    {"name": "Hebrew Tutor", "system": "ענה בקצרה ובעברית פשוטה."},
    {"name": "Translation Mode", "system": "Act as a short translation assistant."},
    {"name": "JSON Output Mode", "system": "Respond in compact JSON only."},
]

CHECKPOINT_DOCS = {
    "checkpoints/tiny_cpu_pretrain/best.pt": {
        "label": "Pretrain Base",
        "stage": "Stage 1",
        "description": "המודל אחרי next-token pretraining דו-לשוני על טקסט כללי.",
        "use_case": "מתאים לראות prior שפה בסיסי, אבל לא צ'אט מלוטש.",
        "badge": "Language prior",
    },
    "checkpoints/domain_cpu_adapt/best.pt": {
        "label": "Domain Adapted",
        "stage": "Stage 2",
        "description": "אותו מודל אחרי חידוד על טקסטים מתחום SEO, AI, שיווק ותוכן.",
        "use_case": "טוב יותר לשאלות תחומיות, עדיין פחות יציב כדיאלוג.",
        "badge": "SEO / AI focus",
    },
    "checkpoints/tiny_cpu_sft_demo_boosted/best.pt": {
        "label": "Demo SFT v2 Boosted",
        "stage": "Stage 4",
        "description": "מודל דמו שעבר fine-tuning נוסף על סט SFT מוגדל עם יותר JSON, תרגום, SEO, greetings ו-marketing prompts.",
        "use_case": "זה ה-checkpoint המומלץ החדש לדמו ולהצגה.",
        "badge": "Recommended",
    },
    "checkpoints/tiny_cpu_builder_base/best.pt": {
        "label": "Neutral Builder Base",
        "stage": "Builder Base",
        "description": "checkpoint ניטרלי ל-builder. בנוי לשמש בסיס כללי לפרויקטים חדשים בלי להוריש את אופי ה-SEO/marketing של הדמו הישן.",
        "use_case": "מתאים כבסיס ברירת מחדל ל-Fast Adapter בפרויקטים חדשים.",
        "badge": "Generic base",
    },
}


def list_checkpoints() -> list[str]:
    checkpoints = sorted(str(path.relative_to(PROJECT_ROOT)) for path in CHECKPOINT_ROOT.rglob("best.pt"))
    priority = {RECOMMENDED_CHECKPOINT: 0}
    return sorted(checkpoints, key=lambda item: (priority.get(item, 1), item))


def load_source_manifest() -> list[dict[str, str]]:
    manifest_path = PROJECT_ROOT / "data" / "raw" / "source_manifest.json"
    rows: list[dict[str, str]] = []
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.extend(
        [
            {
                "name": "Curated bilingual SFT examples",
                "url": "data/raw/sft_dialogues.jsonl",
                "license_note": "CC BY-SA 4.0: mixed original, Wikipedia-derived and SQuAD-derived examples. See data/ATTRIBUTION.md.",
            },
            {
                "name": "Boosted demo SFT set",
                "url": "data/raw/sft_dialogues_boosted.jsonl",
                "license_note": "CC BY-SA 4.0: expanded mixed bilingual dataset; see data/ATTRIBUTION.md. No trained checkpoint is distributed.",
            },
            {
                "name": "Compact domain notes",
                "url": "data/raw/domain_notes.txt",
                "license_note": "CC BY-SA 4.0: selected normalized Wikipedia extracts; see the article source manifest.",
            },
            {
                "name": "Neutral builder base SFT set",
                "url": "data/raw/sft_builder_base.jsonl",
                "license_note": "CC BY-SA 4.0: original educational examples mixed with Wikipedia and SQuAD rows; see data/ATTRIBUTION.md.",
            },
        ]
    )
    return rows


def load_original_scope_prompt() -> str:
    if not ORIGINAL_SCOPE_PROMPT_PATH.exists():
        return "Original scope prompt file is missing."
    return ORIGINAL_SCOPE_PROMPT_PATH.read_text(encoding="utf-8").strip()


def process_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return round(int(parts[1]) / 1024, 2)
    return None


def trim_allocator() -> bool:
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


def clear_model_cache_payload() -> dict:
    web_before = get_engine.cache_info().currsize
    builder_before = 0
    predictive_before = 0
    get_engine.cache_clear()
    cleared = ["web.get_engine"]

    try:
        from src.web.builder_routes import get_builder_engine, get_predictive_bundle

        builder_before = get_builder_engine.cache_info().currsize
        get_builder_engine.cache_clear()
        cleared.append("builder.get_builder_engine")
        builder_after = get_builder_engine.cache_info().currsize
        predictive_before = get_predictive_bundle.cache_info().currsize
        get_predictive_bundle.cache_clear()
        cleared.append("builder.get_predictive_bundle")
        predictive_after = get_predictive_bundle.cache_info().currsize
    except Exception:
        builder_after = None
        predictive_after = None

    rss_before = process_rss_mb()
    collected = gc.collect()
    trimmed = trim_allocator()
    rss_after = process_rss_mb()

    return {
        "ok": True,
        "cleared": cleared,
        "cache_sizes_before": {
            "web": web_before,
            "builder": builder_before,
            "predictive": predictive_before,
        },
        "cache_sizes_after": {
            "web": get_engine.cache_info().currsize,
            "builder": builder_after,
            "predictive": predictive_after,
        },
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "gc_collected": collected,
        "malloc_trim": trimmed,
        "note": "Caches are kept in RAM to avoid reloading model weights from disk on every request. After clearing them, the next request may be slower.",
    }


def docs_context() -> dict:
    checkpoints = list_checkpoints()
    checkpoint_cards = []
    for checkpoint in checkpoints:
        doc = CHECKPOINT_DOCS.get(
            checkpoint,
            {
                "label": Path(checkpoint).stem,
                "stage": "Checkpoint",
                "description": "Checkpoint פנימי נוסף שנוצר במהלך האימון.",
                "use_case": "לשימוש ניסויי או השוואתי.",
                "badge": "Internal",
            },
        )
        checkpoint_cards.append({"path": checkpoint, **doc})

    return {
        "pipeline_steps": [
            {
                "title": "1. Raw Data",
                "summary": "איסוף קורפוס קטן וממוקד באנגלית + עברית, עם טקסט כללי, טקסט תחומי ודוגמאות דיאלוג.",
                "metric": "10MB-30MB target",
                "accent": "var(--accent)",
            },
            {
                "title": "2. Tokenization",
                "summary": "SentencePiece בונה vocabulary משותף לשתי השפות כדי לפרק טקסט ליחידות קטנות שהמודל לומד.",
                "metric": "3200 tokens",
                "accent": "var(--accent-2)",
            },
            {
                "title": "3. Pretraining",
                "summary": "המודל לומד לחזות את הטוקן הבא על טקסט כללי. זה בונה prior שפה, לא behavior של assistant.",
                "metric": "5.6M params",
                "accent": "#155e75",
            },
            {
                "title": "4. Domain Adaptation",
                "summary": "שלב ביניים קצר שמטה את המודל לכיוון מושגי SEO, AI, שיווק ותוכן.",
                "metric": "Optional stage",
                "accent": "#b45309",
            },
            {
                "title": "5. SFT / STP",
                "summary": "השלב ההתנהגותי. כאן המודל לומד פורמט של assistant, תרגום, תשובות קצרות ו-JSON פשוט.",
                "metric": "assistant-only loss",
                "accent": "#0f766e",
            },
            {
                "title": "6. Demo Chat UI",
                "summary": "Flask + frontend קל. הצ'אט טוען checkpoint, שולח prompt, ומציג תשובה עם תמיכה ב-RTL.",
                "metric": "localhost / tunnel",
                "accent": "#7c3aed",
            },
        ],
        "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>", "<sys>", "<user>", "<assistant>"],
        "sources": load_source_manifest(),
        "original_scope_prompt": load_original_scope_prompt(),
        "checkpoints": checkpoint_cards,
        "recommended_checkpoint": RECOMMENDED_CHECKPOINT,
        "presets": PRESETS,
        "builder_profiles": [
            {
                "name": "Extraction / JSON",
                "summary": "מיועד להמרת טקסט גולמי ל-JSON קומפקטי לפי schema fields.",
                "example_in": "Schema: customer_name, product, renewal_date\nText: Dana Levi renewed Analytics Pro on 2026-04-01.",
                "example_out": '{"customer_name":"Dana Levi","product":"Analytics Pro","renewal_date":"2026-04-01"}',
            },
            {
                "name": "Internal Knowledge",
                "summary": "מיועד ל-FAQ פנימי ועוזרי ידע המבוססים על קבצים, notes ו-URLs שהוזנו למערכת.",
                "example_in": "Policy note: Employees can work remotely up to 3 days a week.\nQuestion: How many remote days are allowed?",
                "example_out": "Employees can work remotely up to 3 days per week.",
            },
            {
                "name": "Classification / Scoring",
                "summary": "מיועד ל-labeling קצר, triage, ranking, lead scoring או risk scoring.",
                "example_in": "Labels: hot_lead, warm_lead, low_priority\nText: I need a proposal today and want to sign this week.",
                "example_out": '{"label":"hot_lead","score":5,"reason":"urgent buying intent"}',
            },
            {
                "name": "Predictive AI / Forecasting",
                "summary": "מיועד למודלי חיזוי טבלאיים על CSV/XLSX: classification, regression או forecasting עם metrics וגרפים.",
                "example_in": "Columns: carrier, origin, dep_hour, weather_score, previous_leg_delay, delay_minutes\nTarget: delay_minutes",
                "example_out": '{"prediction": 27.4, "summary": "weather_score and previous_leg_delay are the strongest signals."}',
            },
        ],
        "model_size_guide": [
            {
                "name": "~5.6M Safe CPU",
                "fit": "ברירת המחדל המומלצת. תגובות קצרות נשארות מהירות וסבירות על ה-VPS הזה.",
            },
            {
                "name": "~11.2M Balanced CPU",
                "fit": "נקודת איזון טובה אם אתה רוצה קצת יותר capacity בלי להפוך כל תשובה לאיטית מדי.",
            },
            {
                "name": "~15.2M Stretch CPU",
                "fit": "אפשרי למחקר, אבל בדרך כלל כבר פחות נעים לדמו אינטראקטיבי.",
            },
        ],
        "training_stages": [
            {
                "name": "Tokenizer",
                "value": "3200 vocab",
                "width": "72%",
                "text": "vocab קטן מספיק ל-CPU, אבל עדיין שימושי עבור English + Hebrew.",
            },
            {
                "name": "Pretraining",
                "value": "bilingual text",
                "width": "82%",
                "text": "next-token prediction על טקסט כללי ליצירת prior שפה בסיסי.",
            },
            {
                "name": "Domain Adaptation",
                "value": "SEO / AI / marketing",
                "width": "60%",
                "text": "חידוד קצר על אוצר מילים וכתיבה תחומית כדי לחזק את ההתמחות הצרה.",
            },
            {
                "name": "SFT",
                "value": "dialogue behavior",
                "width": "92%",
                "text": "דוגמאות dialogue בפורמט system / user / assistant עם masking נכון על תשובת ה-assistant בלבד.",
            },
        ],
    }


@lru_cache(maxsize=4)
def get_engine(checkpoint_rel_path: str) -> ChatEngine:
    checkpoint_path = PROJECT_ROOT / checkpoint_rel_path
    return ChatEngine(checkpoint_path=checkpoint_path, tokenizer_path=TOKENIZER_PATH, device="cpu")


@web_bp.route("/")
def index():
    return render_template(
        "index.html",
        checkpoints=list_checkpoints(),
        recommended_checkpoint=RECOMMENDED_CHECKPOINT,
        presets=PRESETS,
    )


@web_bp.route("/docs")
def docs():
    return render_template("docs.html", **docs_context())


@web_bp.route("/api/checkpoints")
def api_checkpoints():
    return jsonify({"checkpoints": list_checkpoints()})


@web_bp.route("/api/system/clear-model-cache", methods=["POST"])
def api_clear_model_cache():
    return jsonify(clear_model_cache_payload())


@web_bp.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True)
    checkpoint = payload.get("checkpoint")
    user_text = str(payload.get("message", "")).strip()
    system_text = str(payload.get("system_prompt", "")).strip() or None
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 20))
    max_new_tokens = int(payload.get("max_new_tokens", 64))
    json_mode = bool(payload.get("json_mode", False))

    if not checkpoint:
        return jsonify({"error": "No checkpoint selected."}), 400
    if not user_text:
        return jsonify({"error": "Message is empty."}), 400

    engine = get_engine(checkpoint)
    response = engine.chat(
        user_text=user_text,
        system_text=system_text,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        json_mode=json_mode,
    )
    return jsonify({"response": response})

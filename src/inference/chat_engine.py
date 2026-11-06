from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import torch

from src.data.dialogue_format import render_prompt
from src.inference.generate import sample_tokens
from src.model.config import ModelConfig
from src.model.transformer import TinyGPT
from tokenizer.tokenizer_utils import SentencePieceTokenizer


@dataclass(frozen=True)
class RuleResult:
    text: str
    task: str
    precise: bool = True
    topic: str | None = None
    source_text: str | None = None
    target_language: str | None = None


KNOWLEDGE_ENTRIES = [
    {
        "aliases": ["seo", "search engine optimization", "קידום אתרים", "אופטימיזציה למנועי חיפוש"],
        "topic_en": "SEO",
        "topic_he": "קידום אתרים",
        "summary_en": "SEO is the practice of improving a website so it can appear more clearly in search engine results.",
        "summary_he": "SEO הוא תהליך של שיפור אתר כדי שיופיע טוב יותר בתוצאות של מנועי חיפוש.",
    },
    {
        "aliases": ["ai", "artificial intelligence", "בינה מלאכותית"],
        "topic_en": "Artificial intelligence",
        "topic_he": "בינה מלאכותית",
        "summary_en": "Artificial intelligence is the field of building systems that perform tasks associated with human intelligence.",
        "summary_he": "בינה מלאכותית היא תחום שמפתח מערכות שמבצעות משימות הקשורות לאינטליגנציה אנושית.",
    },
    {
        "aliases": ["machine learning", "ml", "למידת מכונה"],
        "topic_en": "Machine learning",
        "topic_he": "למידת מכונה",
        "summary_en": "Machine learning is a branch of AI that learns patterns from data in order to make predictions or decisions.",
        "summary_he": "למידת מכונה היא תחום בבינה מלאכותית שבו מודלים לומדים דפוסים מנתונים כדי לבצע חיזוי או קבלת החלטות.",
    },
    {
        "aliases": ["nlp", "natural language processing", "עיבוד שפה טבעית"],
        "topic_en": "Natural language processing",
        "topic_he": "עיבוד שפה טבעית",
        "summary_en": "Natural language processing is the field of helping computers understand and generate human language.",
        "summary_he": "עיבוד שפה טבעית הוא תחום שמאפשר למחשבים להבין ולהפיק שפה אנושית.",
    },
    {
        "aliases": ["search engine", "search engines", "מנוע חיפוש", "מנועי חיפוש"],
        "topic_en": "Search engine",
        "topic_he": "מנוע חיפוש",
        "summary_en": "A search engine is a system that indexes information and returns relevant results for a query.",
        "summary_he": "מנוע חיפוש הוא מערכת שמאנדקסת מידע ומחזירה תוצאות רלוונטיות לשאילתה.",
    },
    {
        "aliases": ["google"],
        "topic_en": "Google",
        "topic_he": "גוגל",
        "summary_en": "Google is a technology company best known for its search engine, ads platform, Android, YouTube, and cloud products.",
        "summary_he": "גוגל היא חברת טכנולוגיה שמוכרת בעיקר בזכות מנוע החיפוש שלה, הפרסום הדיגיטלי, אנדרואיד, יוטיוב ושירותי הענן שלה.",
    },
    {
        "aliases": ["json"],
        "topic_en": "JSON",
        "topic_he": "JSON",
        "summary_en": "JSON is a lightweight text format used to store and exchange structured data.",
        "summary_he": "JSON הוא פורמט טקסט קל המשמש לייצוג ולהעברת מידע מובנה.",
    },
    {
        "aliases": ["digital marketing", "שיווק דיגיטלי"],
        "topic_en": "Digital marketing",
        "topic_he": "שיווק דיגיטלי",
        "summary_en": "Digital marketing is the use of online channels to promote products, services, or brands.",
        "summary_he": "שיווק דיגיטלי הוא שימוש בערוצים מקוונים כדי לקדם מוצרים, שירותים או מותגים.",
    },
    {
        "aliases": ["content marketing", "שיווק באמצעות תוכן"],
        "topic_en": "Content marketing",
        "topic_he": "שיווק באמצעות תוכן",
        "summary_en": "Content marketing focuses on creating useful content that attracts and educates a target audience.",
        "summary_he": "שיווק באמצעות תוכן מתמקד ביצירת תוכן מועיל שמושך קהל יעד ומסייע לו.",
    },
    {
        "aliases": ["automation", "אוטומציה"],
        "topic_en": "Automation",
        "topic_he": "אוטומציה",
        "summary_en": "Automation is the use of systems or software to perform tasks with minimal manual work.",
        "summary_he": "אוטומציה היא שימוש במערכות או תוכנה כדי לבצע משימות עם מינימום עבודה ידנית.",
    },
    {
        "aliases": ["website", "site", "אתר", "אתר אינטרנט"],
        "topic_en": "Website",
        "topic_he": "אתר אינטרנט",
        "summary_en": "A website is a collection of web pages published under a shared domain.",
        "summary_he": "אתר אינטרנט הוא אוסף של דפי רשת שמתפרסמים תחת דומיין משותף.",
    },
    {
        "aliases": ["user interface", "ui", "ממשק משתמש"],
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
    ("good morning", "בוקר טוב"),
    ("good evening", "ערב טוב"),
    ("thank you", "תודה"),
]

EN_TO_HE = {source.lower(): target for source, target in TRANSLATION_PAIRS}
HE_TO_EN = {target.lower(): source for source, target in TRANSLATION_PAIRS}


class ChatEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "cpu",
        fallback_paths: list[str | Path] | None = None,
        ruleset: str = "demo",
    ) -> None:
        self.device = device
        self.ruleset = ruleset
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint["config"]
        self.model = TinyGPT(ModelConfig.from_dict(self.config["model"])).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.project_root = Path(__file__).resolve().parents[2]
        resolved_fallback_paths = [Path(path) for path in fallback_paths] if fallback_paths else [
            self.project_root / "data" / "raw" / "sft_dialogues_boosted.jsonl",
            self.project_root / "data" / "raw" / "sft_dialogues.jsonl",
        ]
        self.fallback_examples = self._load_fallback_examples_from_paths(
            resolved_fallback_paths
        )

    def _load_fallback_examples(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if raw:
                    rows.append(json.loads(raw))
        return rows

    def _load_fallback_examples_from_paths(self, paths: list[Path]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for path in paths:
            for row in self._load_fallback_examples(path):
                key = (row.get("system", "") or "", row.get("user", ""), row.get("assistant", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _contains_hebrew(text: str) -> bool:
        return bool(re.search(r"[\u0590-\u05FF]", text))

    def _preferred_language(self, user_text: str, system_text: str | None = None, answer_text: str | None = None) -> str:
        system_norm = self._normalize(system_text or "")
        if "hebrew" in system_norm or "עברית" in system_norm:
            return "he"
        if "english" in system_norm or "אנגלית" in system_norm:
            return "en"
        if answer_text and self._contains_hebrew(answer_text):
            return "he"
        if self._contains_hebrew(user_text):
            return "he"
        return "en"

    @staticmethod
    def _clean_lookup_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().strip("'\"`“”‘’.,!?()[]{}:;")).lower()

    def _is_phatic_prompt(self, user_text: str) -> bool:
        user_norm = self._normalize(user_text).strip("?.!")
        patterns = (
            "how are you",
            "מה שלומך",
            "hello",
            "hi",
            "hey",
            "שלום",
            "היי",
            "אהלן",
            "הי",
        )
        return any(pattern == user_norm or pattern in user_norm for pattern in patterns)

    def _contains_alias(self, text: str, alias: str) -> bool:
        alias_norm = self._clean_lookup_text(alias)
        if not alias_norm:
            return False
        if re.fullmatch(r"[a-z0-9 ]+", alias_norm):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", text))
        return alias_norm in text

    def _is_degenerate(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 2:
            return True
        if stripped.isdigit():
            return True
        if re.fullmatch(r"[0-9\s.,:;!?\"'\-{}[\]]+", stripped):
            return True
        if "summary\": \"summary\"" in stripped:
            return True
        words = re.findall(r"\w+", stripped.lower())
        if words:
            unique_ratio = len(set(words)) / len(words)
            if len(words) >= 6 and unique_ratio < 0.35:
                return True
        max_char_ratio = max(stripped.count(ch) for ch in set(stripped)) / max(1, len(stripped))
        return max_char_ratio > 0.45

    def _fallback_response(self, user_text: str, system_text: str | None = None, min_score: float = 0.48) -> str | None:
        if not self.fallback_examples:
            return None

        user_norm = self._normalize(user_text)
        system_norm = self._normalize(system_text or "")
        best_score = 0.0
        best_answer: str | None = None

        for row in self.fallback_examples:
            candidate_user = self._normalize(row.get("user", ""))
            candidate_system = self._normalize(row.get("system", ""))
            score = SequenceMatcher(None, user_norm, candidate_user).ratio()
            if system_norm and candidate_system:
                score = 0.85 * score + 0.15 * SequenceMatcher(None, system_norm, candidate_system).ratio()
            if user_norm == candidate_user:
                score += 0.3
            if score > best_score:
                best_score = score
                best_answer = row.get("assistant")

        if best_score >= min_score:
            return best_answer
        return None

    def _extract_translation_request(self, user_text: str) -> tuple[str, str] | None:
        text = user_text.strip()
        patterns = [
            r"""translate\s+["']?(.*?)["']?\s+to\s+(hebrew|english)\b""",
            r"""translate\s+(.*?)\s+to\s+(hebrew|english)\b""",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                term = self._clean_lookup_text(match.group(1))
                language = match.group(2).lower()
                if term:
                    return term, language

        he_match = re.search(r"(?:תתרגם|תרגם)\s+לאנגלית\s*[:：]?\s*(.+)$", text)
        if he_match:
            term = self._clean_lookup_text(he_match.group(1))
            if term:
                return term, "english"

        he_match = re.search(r"(?:תתרגם|תרגם)\s+לעברית\s*[:：]?\s*(.+)$", text)
        if he_match:
            term = self._clean_lookup_text(he_match.group(1))
            if term:
                return term, "hebrew"

        return None

    def _translation_response(self, user_text: str) -> RuleResult | None:
        request = self._extract_translation_request(user_text)
        if not request:
            return None

        term, target_language = request
        if target_language == "hebrew":
            translation = EN_TO_HE.get(term)
        else:
            translation = HE_TO_EN.get(term)
        if not translation:
            return None

        return RuleResult(
            text=translation,
            task="translation",
            source_text=term,
            target_language=target_language,
        )

    def _marketing_response(self, user_text: str, preferred_language: str) -> RuleResult | None:
        text = user_text.strip()

        title_match = re.search(r"write a short title for (.+)$", text, flags=re.IGNORECASE)
        if title_match:
            topic = title_match.group(1).strip().rstrip(".!?")
            if preferred_language == "he":
                return RuleResult(text=f"מדריך קצר וברור ל{topic}", task="marketing_title", topic=topic)
            return RuleResult(text=f"Clear {topic.title()} Guide", task="marketing_title", topic=topic)

        he_title_match = re.search(r"כתוב(?: לי)? כותרת קצרה ל(.+)$", text)
        if he_title_match:
            topic = he_title_match.group(1).strip().rstrip(".!?")
            return RuleResult(text=f"מדריך קצר וברור ל{topic}", task="marketing_title", topic=topic)

        meta_match = re.search(r"write a short meta description for (.+)$", text, flags=re.IGNORECASE)
        if meta_match:
            topic = meta_match.group(1).strip().rstrip(".!?")
            if preferred_language == "he":
                return RuleResult(
                    text=f"סקירה קצרה וברורה על {topic}, עם הדגשת הערך, השימושים והיתרונות לקורא.",
                    task="marketing_meta_description",
                    topic=topic,
                )
            return RuleResult(
                text=f"Short clear overview of {topic}, highlighting the value, key use cases, and why it matters.",
                task="marketing_meta_description",
                topic=topic,
            )

        he_meta_match = re.search(r"כתוב(?: לי)? תיאור מטא קצר ל(.+)$", text)
        if he_meta_match:
            topic = he_meta_match.group(1).strip().rstrip(".!?")
            return RuleResult(
                text=f"סקירה קצרה וברורה על {topic}, עם הדגשת הערך, השימושים והיתרונות לקורא.",
                task="marketing_meta_description",
                topic=topic,
            )

        return None

    def _marketing_request_info(self, user_text: str) -> tuple[str, str] | None:
        text = user_text.strip()
        patterns = [
            (r"write a short title for (.+)$", "marketing_title"),
            (r"write a short meta description for (.+)$", "marketing_meta_description"),
            (r"כתוב(?: לי)? כותרת קצרה ל(.+)$", "marketing_title"),
            (r"כתוב(?: לי)? תיאור מטא קצר ל(.+)$", "marketing_meta_description"),
        ]
        for pattern, task in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return task, match.group(1).strip().rstrip(".!?")
        return None

    @staticmethod
    def _sample_variant(variants: list[str], temperature: float) -> str:
        if not variants:
            return ""
        if len(variants) == 1 or temperature < 0.85:
            return variants[0]
        return random.choice(variants)

    def _greeting_or_status_response(
        self,
        user_text: str,
        preferred_language: str,
        system_text: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 64,
    ) -> RuleResult | None:
        user_norm = self._normalize(user_text).strip("?.!")
        system_norm = self._normalize(system_text or "")
        wants_formal = "formal" in system_norm or "פורמלי" in system_norm
        wants_friendly = "friendly" in system_norm or "ידידות" in system_norm or "friendly tone" in system_norm
        wants_brief = any(token in system_norm for token in ("brief", "short", "concise", "בקצרה", "קצר"))
        length_mode = "short" if max_new_tokens <= 16 or wants_brief else "medium" if max_new_tokens <= 40 else "long"

        if user_norm in {"hi", "hello", "hey", "hey there"}:
            if preferred_language == "he":
                variants = {
                    "short": ["שלום.", "היי.", "שלום, כאן העוזר."],
                    "medium": ["שלום. אני כאן כדי לעזור.", "היי. אני מוכן לעזור.", "שלום. אפשר להתחיל."],
                    "long": ["שלום. אני עוזר tiny, אז אני בדרך כלל עונה בקצרה, אבל אני כאן כדי לעזור.", "היי. אני עוזר קטן ופשוט, ומוכן לעזור בצעדים קצרים וברורים."],
                }
                return RuleResult(text=self._sample_variant(variants[length_mode], temperature), task="greeting")
            variants = {
                "short": ["Hello.", "Hi.", "Hey."],
                "medium": ["Hello. I am here to help.", "Hi. Ready when you are.", "Hello. We can start."],
                "long": ["Hello. I am a tiny assistant, so I usually keep things brief, but I am ready to help.", "Hi. I am a small assistant and I can help with short, clear replies."],
            }
            return RuleResult(text=self._sample_variant(variants[length_mode], temperature), task="greeting")
        if user_norm in {"היי", "שלום", "אהלן", "הי"}:
            variants = {
                "short": ["שלום.", "היי.", "אהלן."],
                "medium": ["שלום. אני כאן כדי לעזור.", "היי. איך אפשר לעזור?", "שלום. אפשר להתחיל."],
                "long": ["שלום. אני עוזר tiny, לכן אני בדרך כלל עונה בקצרה, אבל אני כאן כדי לעזור.", "אהלן. אני עוזר קטן ופשוט, ומוכן לעזור בתשובות קצרות וברורות."],
            }
            return RuleResult(text=self._sample_variant(variants[length_mode], temperature), task="greeting")
        if "how are you" in user_norm:
            if preferred_language == "he":
                if wants_formal:
                    variants = {
                        "short": ["אני בטוב.", "אני בסדר."],
                        "medium": ["אני בטוב. כיצד אפשר לסייע?", "אני בסדר. איך אפשר לעזור?"],
                        "long": ["אני בטוב. אני עוזר קטן ולכן אני שומר על תשובות תמציתיות, אבל אני מוכן לסייע.", "אני בסדר. אני נוטה לענות בקצרה, אך עדיין יכול לעזור במשימות ממוקדות."],
                    }
                elif wants_friendly:
                    variants = {
                        "short": ["אני טוב.", "הכול טוב."],
                        "medium": ["אני טוב. מוכן לעזור.", "הכול טוב. מה תרצה לעשות עכשיו?"],
                        "long": ["אני טוב. אני עוזר tiny ולכן בדרך כלל שומר על תשובות קצרות, אבל אני כאן כדי לעזור.", "הכול טוב. אני עונה בקצרה, אבל אפשר להמשיך לכל משימה ממוקדת."],
                    }
                else:
                    variants = {
                        "short": ["אני בסדר.", "הכול בסדר.", "אני טוב."],
                        "medium": ["אני בסדר. אפשר להמשיך.", "הכול בסדר. איך אפשר לעזור?", "אני טוב. מה השאלה הבאה?"],
                        "long": ["אני בסדר. אני עוזר tiny ולכן בדרך כלל עונה בקצרה, אבל אני כאן כדי לעזור.", "הכול בסדר. אני שומר על תשובות קצרות כי המודל קטן, אבל עדיין אפשר לעבוד עם בקשות ממוקדות."],
                    }
            elif wants_formal:
                variants = {
                    "short": ["I am well.", "Doing well."],
                    "medium": ["I am well. How can I assist?", "I am doing well. How may I help?"],
                    "long": ["I am well. I am a small assistant, so I keep answers concise, but I am ready to assist.", "I am doing well. I tend to answer briefly, but I can still help with focused tasks."],
                }
            elif wants_friendly:
                variants = {
                    "short": ["I'm good.", "Doing well."],
                    "medium": ["I'm doing well. Ready to help.", "I'm good. What would you like to do next?"],
                    "long": ["I'm doing well. I'm a tiny assistant, so I usually keep things short, but I'm ready to help.", "I'm good. I keep replies compact, but I can still help with short tasks and questions."],
                }
            else:
                variants = {
                    "short": ["I'm fine.", "Doing well.", "I'm okay."],
                    "medium": ["I'm fine. Ready to help.", "Doing well. What do you need?", "I'm okay. We can continue."],
                    "long": ["I'm fine. I am a tiny assistant, so I usually keep answers short, but I am ready to help.", "Doing well. I keep replies compact because the model is small, but I can still help with focused prompts."],
                }
            return RuleResult(text=self._sample_variant(variants[length_mode], temperature), task="status")
        if "מה שלומך" in user_norm:
            if wants_formal:
                variants = {
                    "short": ["אני בטוב.", "אני בסדר."],
                    "medium": ["אני בטוב. כיצד אפשר לסייע?", "אני בסדר. איך אפשר לעזור?"],
                    "long": ["אני בטוב. אני עוזר קטן ולכן התשובות שלי קצרות יחסית, אבל אפשר להמשיך.", "אני בסדר. אני נוטה לענות בקצרה, אך עדיין יכול לעזור במשימות ממוקדות."],
                }
            elif wants_friendly:
                variants = {
                    "short": ["אני טוב.", "הכול טוב."],
                    "medium": ["אני טוב. מוכן לעזור.", "הכול טוב. מה תרצה לעשות עכשיו?"],
                    "long": ["אני טוב. אני עוזר tiny ולכן בדרך כלל שומר על תשובות קצרות, אבל אני כאן כדי לעזור.", "הכול טוב. אני עונה בקצרה, אבל אפשר להמשיך לכל משימה ממוקדת."],
                }
            else:
                variants = {
                    "short": ["אני בסדר.", "הכול בסדר.", "אני טוב."],
                    "medium": ["אני בסדר. אפשר להמשיך.", "הכול בסדר. איך אפשר לעזור?", "אני טוב. מה השאלה הבאה?"],
                    "long": ["אני בסדר. אני עוזר tiny ולכן בדרך כלל עונה בקצרה, אבל אני כאן כדי לעזור.", "הכול בסדר. אני שומר על תשובות קצרות כי המודל קטן, אבל עדיין אפשר לעבוד עם בקשות ממוקדות."],
                }
            return RuleResult(text=self._sample_variant(variants[length_mode], temperature), task="status")
        return None

    def _definition_topic(self, user_text: str) -> dict | None:
        user_norm = self._normalize(user_text)
        signals = (
            "what is",
            "what's",
            "who is",
            "who are",
            "explain",
            "define",
            "מה זה",
            "מי זה",
            "מי הם",
            "תסביר",
        )
        if not any(signal in user_norm for signal in signals):
            return None

        for entry in sorted(KNOWLEDGE_ENTRIES, key=lambda item: max(len(alias) for alias in item["aliases"]), reverse=True):
            if any(self._contains_alias(user_norm, alias) for alias in entry["aliases"]):
                return entry
        return None

    def _special_response(self, user_text: str, preferred_language: str) -> RuleResult | None:
        user_norm = self._normalize(user_text)
        if ("best search engine" in user_norm or "greatest search" in user_norm) and "engine" in user_norm:
            if preferred_language == "he":
                return RuleResult(text="בפועל גוגל הוא מנוע החיפוש הנפוץ ביותר בעולם, ולכן לרוב מזכירים אותו ראשון.", task="search_engine_ranking")
            return RuleResult(text="Google is generally the most widely used search engine, so it is usually the first example people give.", task="search_engine_ranking")
        return None

    def _knowledge_response(self, user_text: str, preferred_language: str) -> RuleResult | None:
        entry = self._definition_topic(user_text)
        if not entry:
            return None
        if preferred_language == "he":
            return RuleResult(
                text=entry["summary_he"],
                task="definition",
                topic=entry["topic_he"],
            )
        return RuleResult(
            text=entry["summary_en"],
            task="definition",
            topic=entry["topic_en"],
        )

    def _intent_response(
        self,
        user_text: str,
        system_text: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 64,
    ) -> RuleResult | None:
        preferred_language = self._preferred_language(user_text=user_text, system_text=system_text)
        resolvers = [
            self._translation_response,
            lambda text: self._greeting_or_status_response(
                text,
                preferred_language,
                system_text=system_text,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ),
        ]
        if self.ruleset == "demo":
            resolvers.extend(
                [
                    lambda text: self._marketing_response(text, preferred_language),
                    lambda text: self._special_response(text, preferred_language),
                    lambda text: self._knowledge_response(text, preferred_language),
                ]
            )
        for resolver in resolvers:
            result = resolver(user_text)
            if result:
                return result
        return None

    def _model_response(
        self,
        user_text: str,
        system_text: str | None,
        temperature: float,
        top_k: int | None,
        max_new_tokens: int,
    ) -> str:
        prompt_ids = render_prompt(self.tokenizer, user_text=user_text, system_text=system_text)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output_ids = sample_tokens(
            self.model,
            input_ids=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_id,
        )[0].tolist()
        generated = output_ids[len(prompt_ids) :]

        assistant_tok = self.tokenizer.piece_to_id("<assistant>")
        stop_tokens = {self.tokenizer.eos_id, self.tokenizer.pad_id, assistant_tok}
        trimmed: list[int] = []
        for token_id in generated:
            if token_id in stop_tokens:
                break
            trimmed.append(token_id)
        return self.tokenizer.decode(trimmed).strip()

    def _resolve_plain_response(
        self,
        user_text: str,
        system_text: str | None,
        temperature: float,
        top_k: int | None,
        max_new_tokens: int,
    ) -> tuple[str, str, RuleResult | None]:
        strong_fallback = None if self._is_phatic_prompt(user_text) else self._fallback_response(
            user_text=user_text,
            system_text=system_text,
            min_score=0.9,
        )
        if strong_fallback:
            return strong_fallback, "fallback_exact", None

        rule = self._intent_response(
            user_text=user_text,
            system_text=system_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if rule and rule.precise:
            return rule.text, f"rule:{rule.task}", rule

        model_text = self._model_response(
            user_text=user_text,
            system_text=system_text,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        if model_text and not self._is_degenerate(model_text):
            return model_text, "model", rule

        fallback = self._fallback_response(user_text=user_text, system_text=system_text, min_score=0.48)
        if fallback:
            return fallback, "fallback_fuzzy", rule

        if rule:
            return rule.text, f"rule:{rule.task}", rule

        if not model_text:
            return "(empty response)", "empty", None
        return model_text, "model_degenerate", None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _parse_json_candidate(self, text: str) -> dict | list | None:
        candidates: list[str] = []
        stripped = self._strip_code_fences(text)
        if stripped:
            candidates.append(stripped)
        if "{" in stripped and "}" in stripped:
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            candidates.append(stripped[start:end])
        if "[" in stripped and "]" in stripped:
            start = stripped.find("[")
            end = stripped.rfind("]") + 1
            candidates.append(stripped[start:end])

        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
        return None

    def _json_payload_from_rule(self, user_text: str, answer_text: str, rule: RuleResult | None, system_text: str | None) -> dict:
        language = self._preferred_language(user_text=user_text, system_text=system_text, answer_text=answer_text)

        if rule and rule.task == "translation":
            return {
                "task": "translation",
                "source_text": rule.source_text,
                "target_language": rule.target_language,
                "translation": answer_text,
            }

        if rule and rule.task == "definition":
            return {
                "topic": rule.topic,
                "summary": answer_text,
                "language": language,
            }

        if rule and rule.task == "marketing_title":
            return {
                "task": "marketing_title",
                "topic": rule.topic,
                "title": answer_text,
                "language": language,
            }

        if rule and rule.task == "marketing_meta_description":
            return {
                "task": "marketing_meta_description",
                "topic": rule.topic,
                "meta_description": answer_text,
                "language": language,
            }

        request = self._extract_translation_request(user_text)
        if request:
            term, target_language = request
            return {
                "task": "translation",
                "source_text": term,
                "target_language": target_language,
                "translation": answer_text,
            }

        marketing_request = self._marketing_request_info(user_text)
        if marketing_request:
            task, topic = marketing_request
            key = "title" if task == "marketing_title" else "meta_description"
            return {
                "task": task,
                "topic": topic,
                key: answer_text,
                "language": language,
            }

        entry = self._definition_topic(user_text)
        if entry:
            topic = entry["topic_he"] if language == "he" else entry["topic_en"]
            return {
                "topic": topic,
                "summary": answer_text,
                "language": language,
            }

        return {
            "task": "assistant_reply",
            "answer": answer_text,
            "language": language,
        }

    def _json_response(
        self,
        user_text: str,
        system_text: str | None,
        temperature: float,
        top_k: int | None,
        max_new_tokens: int,
    ) -> str:
        json_instruction = "Respond only with a valid JSON object. Do not add markdown fences or prose."
        combined_system = " ".join(part for part in [system_text, json_instruction] if part).strip() or None
        response_text, _, rule = self._resolve_plain_response(
            user_text=user_text,
            system_text=combined_system,
            temperature=min(temperature, 0.35),
            top_k=min(top_k, 8) if top_k is not None else 8,
            max_new_tokens=max(max_new_tokens, 96),
        )

        parsed = self._parse_json_candidate(response_text)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False, indent=2)

        payload = self._json_payload_from_rule(
            user_text=user_text,
            answer_text=response_text,
            rule=rule,
            system_text=system_text,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def chat(
        self,
        user_text: str,
        system_text: str | None = None,
        temperature: float = 0.8,
        top_k: int | None = 20,
        max_new_tokens: int = 64,
        json_mode: bool = False,
    ) -> str:
        if json_mode:
            return self._json_response(
                user_text=user_text,
                system_text=system_text,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )

        response_text, _, _ = self._resolve_plain_response(
            user_text=user_text,
            system_text=system_text,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        return response_text

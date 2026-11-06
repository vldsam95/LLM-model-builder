from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.source_download import fetch_wikipedia_extract, write_source_manifest
from src.web.app import parse_args


class PublicationTest(unittest.TestCase):
    def test_download_preserves_revision_and_resolved_article_credit(self) -> None:
        session = Mock()
        session.get.return_value.json.return_value = {
            "query": {"pages": [{
                "title": "Machine learning", "extract": "Example educational text.",
                "fullurl": "https://en.wikipedia.org/wiki/Machine_learning",
                "revisions": [{"revid": 12345}],
            }]}
        }
        row = fetch_wikipedia_extract(session, "en", "Machine learning")
        self.assertEqual(row["revision_id"], "12345")
        self.assertTrue(row["source_revision_url"].endswith("?oldid=12345"))
        self.assertEqual(row["license"], "CC-BY-SA-4.0")
        self.assertIn("contributors", row["attribution"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wiki_pages_en.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = root / "source_manifest.json"
            write_source_manifest(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            article = next(item for item in manifest if item.get("revision_id") == "12345")
            self.assertEqual(article["provenance_status"], "downloaded_article")
            self.assertEqual(article["url"], row["source_url"])

    def test_historical_inventory_does_not_invent_revision_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source_manifest.json"
            write_source_manifest(output)
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(all(row["license"] == "CC-BY-SA-4.0" for row in rows))
            articles = [row for row in rows if row.get("provenance_status")]
            self.assertTrue(articles)
            self.assertTrue(all(not row["revision_id"] for row in articles))

    def test_educational_web_server_defaults_to_loopback(self) -> None:
        with patch("sys.argv", ["web"]):
            self.assertEqual(parse_args().host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main()

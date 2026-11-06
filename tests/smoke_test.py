from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.dialogue_format import DialogueExample, build_dialogue_example
from src.model.config import ModelConfig
from src.model.transformer import TinyGPT
from tokenizer.tokenizer_utils import SentencePieceTokenizer, train_sentencepiece


class ProjectSmokeTest(unittest.TestCase):
    def test_model_forward_and_dialogue_masking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corpus = tmp_path / "toy.txt"
            corpus.write_text("hello world\nשלום עולם\nwhat is ai\nמה זה ai", encoding="utf-8")
            train_sentencepiece(corpus, tmp_path / "toy_tok", vocab_size=32)
            tokenizer = SentencePieceTokenizer(tmp_path / "toy_tok.model")

            input_ids, labels = build_dialogue_example(
                tokenizer,
                DialogueExample(system="Answer briefly.", user="Hello", assistant="Hello."),
                max_length=32,
            )
            self.assertTrue(any(label != -100 for label in labels))

            model = TinyGPT(
                ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    max_seq_len=32,
                    d_model=64,
                    n_heads=4,
                    n_kv_heads=4,
                    n_layers=2,
                )
            )
            batch = torch.tensor([input_ids], dtype=torch.long)
            label_batch = torch.tensor([labels], dtype=torch.long)
            output = model(batch, labels=label_batch)
            self.assertIn("loss", output)


if __name__ == "__main__":
    unittest.main()

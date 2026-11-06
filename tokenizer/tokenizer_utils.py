from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sys>", "<user>", "<assistant>"]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    def piece_to_id(self, piece: str) -> int:
        return self.processor.piece_to_id(piece)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        safe_ids = [idx for idx in ids if idx >= 0]
        return self.processor.decode(safe_ids)


def train_sentencepiece(
    input_path: str | Path,
    output_prefix: str | Path,
    vocab_size: int,
    model_type: str = "unigram",
) -> None:
    output_prefix = str(output_prefix)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    user_symbols = ",".join(["<sys>", "<user>", "<assistant>"])
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=user_symbols,
        character_coverage=1.0,
        normalization_rule_name="nfkc",
        split_digits=True,
        max_sentence_length=8192,
        shuffle_input_sentence=True,
        input_sentence_size=0,
        hard_vocab_limit=False,
    )

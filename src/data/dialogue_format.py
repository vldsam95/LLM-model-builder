from __future__ import annotations

from dataclasses import dataclass

from tokenizer.tokenizer_utils import SentencePieceTokenizer


@dataclass
class DialogueExample:
    system: str | None
    user: str
    assistant: str


def build_dialogue_example(
    tokenizer: SentencePieceTokenizer,
    example: DialogueExample,
    max_length: int,
) -> tuple[list[int], list[int]]:
    sys_tok = tokenizer.piece_to_id("<sys>")
    user_tok = tokenizer.piece_to_id("<user>")
    assistant_tok = tokenizer.piece_to_id("<assistant>")

    sequence: list[int] = [tokenizer.bos_id]

    if example.system:
        system_tokens = tokenizer.encode(example.system.strip(), add_eos=True)
        sequence.extend([sys_tok] + system_tokens)

    user_tokens = tokenizer.encode(example.user.strip(), add_eos=True)
    sequence.extend([user_tok] + user_tokens)

    assistant_tokens = tokenizer.encode(example.assistant.strip(), add_eos=True)
    assistant_prefix_index = len(sequence)
    sequence.append(assistant_tok)
    sequence.extend(assistant_tokens)

    if len(sequence) < 2:
        raise ValueError("Dialogue sequence is too short.")

    input_ids = sequence[:-1][:max_length]
    next_tokens = sequence[1:][:max_length]
    labels: list[int] = []
    assistant_label_start = assistant_prefix_index
    for idx, token_id in enumerate(next_tokens):
        if idx < assistant_label_start:
            labels.append(-100)
        else:
            labels.append(token_id)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids.extend([tokenizer.pad_id] * pad_len)
        labels.extend([-100] * pad_len)
    return input_ids, labels


def render_prompt(
    tokenizer: SentencePieceTokenizer,
    user_text: str,
    system_text: str | None = None,
) -> list[int]:
    sys_tok = tokenizer.piece_to_id("<sys>")
    user_tok = tokenizer.piece_to_id("<user>")
    assistant_tok = tokenizer.piece_to_id("<assistant>")

    tokens: list[int] = [tokenizer.bos_id]
    if system_text:
        tokens.extend([sys_tok] + tokenizer.encode(system_text.strip(), add_eos=True))
    tokens.extend([user_tok] + tokenizer.encode(user_text.strip(), add_eos=True))
    tokens.append(assistant_tok)
    return tokens

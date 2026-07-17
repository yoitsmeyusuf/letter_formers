"""A small custom byte-pair encoding tokenizer for Turkish names.

This tokenizer keeps newline as a dedicated start/end-of-name token and learns
merge rules from a text file. The learned vocabulary is compact, but the API is
kept close to the old character tokenizer:

    tok = BPETokenizer.from_file("../data/titles.txt", vocab_size=128)
    ids = tok.encode("ali")
    tok.decode(ids)
    tok.newline_id
"""

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self, vocab: list[str], merge_rules: list[tuple[str, str]]):
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.newline_id = self.stoi["\n"]
        self.eos_id = self.newline_id
        self._merge_ranks = {pair: i for i, pair in enumerate(merge_rules)}

    @classmethod
    def from_file(cls, path: str, vocab_size: int = 128) -> "BPETokenizer":
        text = open(path, encoding="utf-8").read()
        if "\n" not in text:
            text += "\n"
        if not text.endswith("\n"):
            text += "\n"

        sequences: list[list[str]] = []
        vocab = {"\n"}

        for chunk in text.splitlines(keepends=True):
            if chunk.endswith("\n"):
                symbols = list(chunk[:-1]) + ["\n"]
            else:
                symbols = list(chunk)
            sequences.append(symbols)
            vocab.update(symbols)

        merge_rules: list[tuple[str, str]] = []
        vocab_list = sorted(vocab)

        while len(vocab_list) < vocab_size:
            pair_freqs: Counter[tuple[str, str]] = Counter()

            for seq in sequences:
                for left, right in zip(seq, seq[1:]):
                    if left == "\n" or right == "\n":
                        continue
                    pair_freqs[(left, right)] += 1

            if not pair_freqs:
                break

            pair, freq = pair_freqs.most_common(1)[0]
            if freq < 2:
                break

            merged_token = pair[0] + pair[1]
            merge_rules.append(pair)
            vocab_list.append(merged_token)

            merged_sequences: list[list[str]] = []
            for seq in sequences:
                merged_seq: list[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                        merged_seq.append(merged_token)
                        i += 2
                    else:
                        merged_seq.append(seq[i])
                        i += 1
                merged_sequences.append(merged_seq)
            sequences = merged_sequences

        return cls(vocab_list, merge_rules)

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(state["vocab"], [tuple(pair) for pair in state["merge_rules"]])

    def state_dict(self) -> dict:
        return {
            "type": "bpe",
            "vocab": self.vocab,
            "merge_rules": [list(pair) for pair in self.merge_rules],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BPETokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state(state)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        if len(symbols) < 2 or not self.merge_rules:
            return symbols

        while True:
            best_index = None
            best_rank = None

            for index, (left, right) in enumerate(zip(symbols, symbols[1:])):
                rank = self._merge_ranks.get((left, right))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index

            if best_index is None:
                break

            symbols = (
                symbols[:best_index]
                + [symbols[best_index] + symbols[best_index + 1]]
                + symbols[best_index + 2:]
            )

        return symbols

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids: list[int] = []
        for chunk in text.splitlines(keepends=True):
            if chunk.endswith("\n"):
                symbols = list(chunk[:-1]) + ["\n"]
            else:
                symbols = list(chunk)

            merged = self._apply_merges(symbols)
            for token in merged:
                ids.append(self.stoi[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
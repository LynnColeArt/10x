from __future__ import annotations

from dataclasses import dataclass


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 256 + BYTE_OFFSET


@dataclass(frozen=True)
class ByteTokenizer:
    pad_token_id: int = PAD_ID
    bos_token_id: int = BOS_ID
    eos_token_id: int = EOS_ID
    unk_token_id: int = UNK_ID
    vocab_size: int = VOCAB_SIZE

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        for value in text.encode("utf-8", errors="replace"):
            tokens.append(value + BYTE_OFFSET)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raw_bytes = bytearray()
        for token_id in token_ids:
            if skip_special_tokens and token_id < BYTE_OFFSET:
                continue
            if token_id >= BYTE_OFFSET:
                raw_bytes.append(token_id - BYTE_OFFSET)
        return raw_bytes.decode("utf-8", errors="replace")


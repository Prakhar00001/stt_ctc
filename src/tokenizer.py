import re
from typing import List

class CTCTokenizer:
    BLANK_ID = 0
    def __init__(self):
        self.chars = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [' ', "'"]
        self.id_to_char = {0: "<BLANK>"}
        self.char_to_id = {"<BLANK>": 0}
        for i, c in enumerate(self.chars, 1):
            self.id_to_char[i] = c
            self.char_to_id[c] = i
        self.vocab_size = len(self.id_to_char)

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z\s']", " ", text.lower().strip())).strip()

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[c] for c in self.normalize(text)]

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.id_to_char[t] for t in token_ids if t != 0])
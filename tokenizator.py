#!/usr/bin/env python

import re
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Used it to phrase json when I was a biginner :(
import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.encode_dict = {token:index for index, token in enumerate(list(set(tokens)))};
        self.encode_dict["<<|UNK|>>"] = len(self.encode_dict);
        self.encode_dict["<<|EOF|>>"] = len(self.encode_dict);

        self.reverse_dict = {index:token for token, index in self.encode_dict.items()};
        self.reverse_dict[self.encode_dict["<<|UNK|>>"]] = "<<|UNK|>>";
        self.reverse_dict[self.encode_dict["<<|EOF|>>"]] = "<<|EOF|>>";

    def encode(self, text: str) -> list[int]:
        tokens = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
        tokens = [self.encode_dict[token] if token in self.encode_dict else self.encode_dict["<<|UNK|>>"] for token in tokens];
        return tokens;

    def decode(self, tokens: list[int]) -> str:
        output = ' '.join([self.reverse_dict[token] for token in tokens]);
        output = re.sub(r'\s+([,.?!"()\'])', r'\1', output);
        return output;

class Data:
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer;
        self.inputs = [];
        self.targets = [];

        tokens = tokenizer.encode(text);
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length];
            target_chunk = tokens[i + 1: i + max_length + 1];
            self.inputs.append(torch.tensor(input_chunk));
            self.targets.append(torch.tensor(target_chunk));

        def __len__(self):
            return len(self.input_ids);

        def __getitem__(self, index: int):
            return self.input_ids[index], self.target_ids[index];

def main() -> None:
    with open('verdict', mode="r", encoding="utf-8") as file:
        text = file.read();

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");

    tokenizer = Tokenizer(tokens);

    input = tokenizer.encode(text);

    contextSize = 10;
    for i in range(contextSize + 1):
        input = input[:i];
        target = input[i];
        print(f'{tokenizer.decode(input)} + {tokenizer.decode([target])}');

if __name__ == "__main__":
    main();


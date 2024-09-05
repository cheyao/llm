#!/usr/bin/env python

import re
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Used it to phrase json when I was a biginner :(
import torch
from torch import Tensor
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

class Data(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int) -> None:
        self.tokenizer = tokenizer;
        self.inputs = [];
        self.targets = [];

        tokens = tokenizer.encode(text);
        for i in range(0, len(tokens) - max_length, stride):
            inputChunk = tokens[i:i + max_length];
            targetChunk = tokens[i + 1: i + max_length + 1];
            self.inputs.append(torch.tensor(inputChunk));
            self.targets.append(torch.tensor(targetChunk));

    def __len__(self) -> int:
        return len(self.inputs);

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index];

def makeLoader(text: str, tokenizer: Tokenizer, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    data = Data(text, tokenizer, max_length, stride);
    dataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last);
    return dataLoader;

def main() -> None:
    with open('verdict', mode="r", encoding="utf-8") as file:
        text = file.read();

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");
    tokenizer = Tokenizer(tokens);
    loader = makeLoader(text, tokenizer);
    it = iter(loader);
    inputs, targets = next(it);
    print(inputs);
    print(targets);

if __name__ == "__main__":
    main();


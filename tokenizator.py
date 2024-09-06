#!/usr/bin/env python

import re
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Used it to phrase json when I was a biginner :(
import torch
from torch import Tensor, batch_norm_update_stats
from torch.nn import Embedding
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
        output: str = ' '.join([self.reverse_dict[token] for token in tokens]);
        output: str = re.sub(r'\s+([,.?!"()\'])', r'\1', output);
        return output;

    def vocabSize(self) -> int:
        return len(self.encode_dict);

class Data(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int) -> None:
        self.tokenizer: Tokenizer = tokenizer;
        self.inputs: list[Tensor] = [];
        self.targets: list[Tensor] = [];

        tokens: list[int] = tokenizer.encode(text);
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
    data: Data = Data(text, tokenizer, max_length, stride);
    dataLoader: DataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last);
    return dataLoader;

def makeContext(batch: Tensor) -> Tensor:
    # For the attentions dot product: Simularity
    # Get it by dot the input n with query and get attention score n
    output: Tensor = torch.empty(batch.shape[0], batch.shape[1], batch.shape[1]);

    for batchNumber, phrase in enumerate(batch):
        output[batchNumber] = torch.softmax((phrase @ phrase.T), dim=-1) @ phrase;

    return output;

def main() -> None:
    with open('verdict', mode="r", encoding="utf-8") as file:
        text: str = file.read();

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");
    tokenizer: Tokenizer = Tokenizer(tokens);
    loader: DataLoader = makeLoader(text, tokenizer);

    token_embedding = Embedding(tokenizer.vocabSize(), 256);
    pos_embedding = Embedding(256, 256);

    for batch_number, batch in enumerate(loader):
        # We shall deal with batches here
        input, target = batch;

        token_embeddings: Tensor = token_embedding(input); # The thing for the word groups
        pos_embeddings: Tensor = pos_embedding(torch.arange(256)); # Add to it the position of the word
        # How tf does llms process these?

        input_embeddings: Tensor = token_embeddings + pos_embeddings;

        context = makeContext(input_embeddings);

        print(context);

        print(f"Processed batch {batch_number}")

        break;

if __name__ == "__main__":
    main();


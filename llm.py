#!/usr/bin/env python

import re
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Used it to phrase json when I was a biginner :(
import torch
import matplotlib
from torch import Tensor, inf
from torch.nn import Embedding, Linear, Module, Dropout
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


# Bruh there is torch.nn.MultiheadAttention
class Attention(Module):
    def __init__(self, inSize: int, outSize: int, contextLength: int, dropout: float, numHeads: int, qkvBias: bool = False) -> None:
        super().__init__();

        assert(outSize % numHeads == 0 and "out vector must be divisible by head number");

        self.outSize: int = outSize;
        self.numHeads: int = numHeads;
        self.headSize: int = outSize // numHeads;

        self.key:   Linear = Linear(inSize, outSize, bias=qkvBias);
        self.query: Linear = Linear(inSize, outSize, bias=qkvBias);
        self.value: Linear = Linear(inSize, outSize, bias=qkvBias);

        # Read: concat
        self.outProj: Linear = Linear(outSize, outSize);
        self.dropout: Dropout = Dropout(dropout); # Gotta hide some stuff from the model
        # Don't let the model see the stuff for the stuff afterwards
        self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1));

    def forward(self, batch: Tensor) -> Tensor:
        assert(len(batch.shape) == 3 and "Did you pass in a phrase?");

        phraseCount, phraseSize, _ = batch.shape;

        keys:    Tensor = self.key(batch);
        queries: Tensor = self.query(batch);
        values:  Tensor  = self.value(batch);

        # Split it up
        keys    = keys.view(phraseCount, phraseSize, self.numHeads, self.headSize);
        values  = values.view(phraseCount, phraseSize, self.numHeads, self.headSize);
        queries = queries.view(phraseCount, phraseSize, self.numHeads, self.headSize);

        # Exchane dim 1 & 2
        keys    = keys.transpose(1, 2);
        queries = queries.transpose(1, 2);
        values  = values.transpose(1, 2)

        # Here we get the score of the thing
        attentionScores: Tensor = queries @ keys.transpose(2, 3);

        # Get a (bit) mask!
        mask = self.mask.bool()[:phraseSize, :phraseSize];
        # Note: softmax -inf = 0
        attentionScores.masked_fill_(mask, -inf);
        
        # Now mask em' and average
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1]**0.5, dim=-1);
        attentionWeights = self.dropout(attentionWeights);

        # Output
        context = (attentionWeights @ values).transpose(1, 2);
        context = context.contiguous().view(phraseCount, phraseSize, self.outSize);
        context = self.outProj(context);

        return context

def makeLoader(text: str, tokenizer: Tokenizer, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    data: Data = Data(text, tokenizer, max_length, stride);
    dataLoader: DataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last);
    return dataLoader;

def main() -> None:
    with open('verdict', mode="r", encoding="utf-8") as file:
        text: str = file.read();

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");
    tokenizer: Tokenizer = Tokenizer(tokens);
    loader: DataLoader = makeLoader(text, tokenizer);

    tokenEmbedding = Embedding(tokenizer.vocabSize() + 1, 256);
    posEmbedding = Embedding(256, 256);

    for batchNumber, batch in enumerate(loader):
        # We shall deal with batches here
        input, target = batch;

        tokenEmbeddings: Tensor = tokenEmbedding(input); # The thing for the word groups
        posEmbeddings: Tensor = posEmbedding(torch.arange(256)); # Add to it the position of the word
        # How tf does llms process these?

        inputEmbeddings: Tensor = tokenEmbeddings + posEmbeddings;
        inputs = torch.tensor(
          [[0.43, 0.15, 0.89], # Your     (x^1)
           [0.55, 0.87, 0.66], # journey  (x^2)
           [0.57, 0.85, 0.64], # starts   (x^3)
           [0.22, 0.58, 0.33], # with     (x^4)
           [0.77, 0.25, 0.10], # one      (x^5)
           [0.05, 0.80, 0.55]] # step     (x^6)
        );

        batch = torch.stack((inputs, inputs), dim=0);

        torch.manual_seed(123);
        attentionModel = Attention(batch.shape[2], 2, batch.shape[1], 0.0, 2);

        print(attentionModel(batch));

        print(f"Processed batch {batchNumber}")

        break;

if __name__ == "__main__":
    main();


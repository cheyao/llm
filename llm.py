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
from torch.nn import Embedding, Linear, Module, Dropout, Sequential, Parameter
from torch.utils.data import Dataset, DataLoader

# Reference thing
CONFIG = {
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

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
    def __init__(self, inSize: int, outSize: int, contextLength: int = 1024, dropout: float = 0.1, numHeads: int = 12, qkvBias: bool = False) -> None:
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

class GPT(Module):
    def __init__(self, vocabSize: int) -> None:
        super().__init__();

        self.tokenEmbedding = Embedding(vocabSize + 1, CONFIG["emb_dim"]);
        self.posEmbedding = Embedding(CONFIG["emb_dim"], CONFIG["emb_dim"]);

        self.dropEmbedding = Dropout(CONFIG["drop_rate"]);
        
        self.blocks = Sequential(*[TransformerBlock() for _ in range(CONFIG["n_layers"])]);
        
        self.finalNorm = LayerNorm(CONFIG["emb_dim"]);
        self.outHead = Linear(CONFIG["emb_dim"], vocabSize + 1, bias=False);

    def forward(self, inIdx: Tensor):
        batch_size, seq_len = inIdx.shape;
        tokenEmbeds = self.tokenEmbedding(inIdx);
        posEmbeds = self.posEmbedding(torch.arange(seq_len, device=inIdx.device));
        x = posEmbeds + tokenEmbeds;
        x = self.dropEmbedding(x);
        x = self.blocks(x);
        x = self.finalNorm(x);
        logits = self.outHead(x);
        return logits;

class TransformerBlock(Module):
    def __init__(self):
        super().__init__();
        # A simple placeholder

    def forward(self, x: Tensor) -> Tensor:
        return x;


class LayerNorm(Module):
    def __init__(self, normalizedShape: int, eps: float = 1e-5):
        super().__init__();

        self.eps: float = eps;
        self.scale: Tensor = Parameter(torch.ones(normalizedShape));
        self.shift: Tensor = Parameter(torch.zeros(normalizedShape));

    def forward(self, batch: Tensor) -> Tensor:
        mean: Tensor = batch.mean(dim=-1, keepdim=True);
        varitation: Tensor = batch.var(dim=-1, keepdim=True, unbiased=False);
        norm: Tensor = (batch - mean) / torch.sqrt(varitation + self.eps);

        return self.scale * norm + self.shift;

def makeLoader(text: str, tokenizer: Tokenizer, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, dropLast: bool = True) -> DataLoader:
    data: Data = Data(text, tokenizer, max_length, stride);
    dataLoader: DataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=dropLast);
    return dataLoader;

def main() -> None:
    with open('verdict', mode="r", encoding="utf-8") as file:
        text: str = file.read();

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");
    tokenizer: Tokenizer = Tokenizer(tokens);
    loader: DataLoader = makeLoader(text, tokenizer);

    model: GPT = GPT(tokenizer.vocabSize());

    tokenEmbedding = Embedding(tokenizer.vocabSize() + 1, CONFIG["emb_dim"]);
    posEmbedding = Embedding(CONFIG["emb_dim"], CONFIG["emb_dim"]);

    for batchNumber, batch in enumerate(loader):
        # We shall deal with batches here
        inputs, targets = batch;

        logits = model(inputs);

        print(inputs.shape);
        print(logits.shape);
        print(logits);

        tokenEmbeddings: Tensor = tokenEmbedding(inputs); # The thing for the word groups
        posEmbeddings: Tensor = posEmbedding(torch.arange(CONFIG["emb_dim"])); # Add to it the position of the word
        # How tf does llms process these? Black magic(tm)

        assert(CONFIG["emb_dim"] == batch.shape[2]);

        # Embed some info in it and the context
        inputs: Tensor = tokenEmbeddings + posEmbeddings;
        attentionModel = Attention(CONFIG["emb_dim"], CONFIG["emb_dim"], CONFIG["context_length"], CONFIG["drop_rate"], CONFIG["n_heads"], CONFIG["qkv_bias"]);

        inputs: Tensor = attentionModel(inputs);

        print(f"Processed batch {batchNumber}")

        break;

if __name__ == "__main__":
    main();


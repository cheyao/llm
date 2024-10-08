#!/usr/bin/env python

from fastapi import FastAPI, Request
from starlette.responses import FileResponse
import torch
from torch import Tensor, inf
from torch.nn import Embedding, Linear, Module, Dropout, Sequential, Parameter
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Reference thing, I don't have a PHD so this is the same as OpenAI
TRAIN = False;
TEST_PROMPT = "three years later";
TOKENS = 20;
CONFIG = {
    "context_length": 256,  # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
};

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Mac! Gotta *borrow* my dad's M1
else:
    device = torch.device("cpu")

if (device == torch.device("cpu")):
    print(f"Warning!: Training on {device}: cuda not avaliable :(");
else:
    print(f"Using {device} device.")

class Tokenizer:
    def __init__(self, text: str) -> None:
        # tokens = sorted(list(set(text)));
        # self.encode_dict = {token:index for index, token in enumerate(tokens)};
        # self.encode_dict["<<|UNK|>>"] = len(self.encode_dict);
        # self.encode_dict["<<|EOF|>>"] = len(self.encode_dict);

        # The one for big model
        self.encode_dict = {'r': 0, 'z': 1, 'R': 2, 'P': 3, '’': 4, ':': 5, '—': 6, 't': 7, 'm': 8, '\u2060': 9, 'X': 10, 'a': 11, '©': 12, '1': 13, 'ç': 14, 'i': 15, "'": 16, '-': 17, 'D': 18, '[': 19, 'é': 20, '5': 21, '~': 22, 'L': 23, 'h': 24, '4': 25, 'n': 26, 'K': 27, 'S': 28, '+': 29, 'j': 30, 'Z': 31, 'C': 32, 'I': 33, 'U': 34, '“': 35, 'g': 36, 'p': 37, '‘': 38, '$': 39, 'o': 40, 'G': 41, 'w': 42, '.': 43, '\n': 44, 'F': 45, 'k': 46, 'u': 47, 'è': 48, 'Q': 49, 'V': 50, '/': 51, ' ': 52, 'T': 53, '@': 54, '”': 55, 'H': 56, 'A': 57, 'd': 58, 'v': 59, '\xad': 60, '8': 61, 'J': 62, '9': 63, '\x1b': 64, 'y': 65, 'Y': 66, ')': 67, 'W': 68, '0': 69, 'q': 70, 'M': 71, 'x': 72, '%': 73, 'à': 74, '3': 75, 'f': 76, 'c': 77, 'l': 78, 's': 79, 'B': 80, '?': 81, '#': 82, 'e': 83, '(': 84, '◆': 85, '*': 86, ',': 87, 'E': 88, '6': 89, 'O': 90, ';': 91, 'b': 92, '&': 93, '2': 94, '7': 95, '!': 96, 'N': 97, '<<|UNK|>>': 98, '<<|EOF|>>': 99};
        print(self.encode_dict);

        self.reverse_dict = {index:token for token, index in self.encode_dict.items()};


    def encode(self, text: str) -> list[int]:
        text = text.replace('"', '“'); # Should have sanitized the input data
        tokens = [self.encode_dict[token] if token in self.encode_dict else self.encode_dict["<<|UNK|>>"] for token in text];

        return tokens;

    def decode(self, tokens: list[int]) -> str:
        output: str = ''.join([self.reverse_dict[token] for token in tokens]);
        return output;

    def decodeTensor(self, tokens: Tensor) -> str:
        return self.decode(tokens.squeeze().tolist());

    def vocabSize(self) -> int:
        return len(self.encode_dict);

class Data(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, maxLength: int, stride: int) -> None:
        self.tokenizer: Tokenizer = tokenizer;
        self.inputs: list[Tensor] = [];
        self.targets: list[Tensor] = [];

        tokens: list[int] = tokenizer.encode(text);
        for i in range(0, len(tokens) - maxLength, stride):
            inputChunk = tokens[i:i + maxLength];
            targetChunk = tokens[i + 1: i + maxLength + 1];
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

        self.tokenEmbedding: Embedding = Embedding(vocabSize, CONFIG["emb_dim"]);
        self.posEmbedding: Embedding = Embedding(CONFIG["emb_dim"], CONFIG["emb_dim"]);

        self.dropEmbedding: Dropout = Dropout(CONFIG["drop_rate"]);
        
        self.blocks: Sequential = Sequential(*[TransformerBlock() for _ in range(CONFIG["n_layers"])]);
        
        self.finalNorm: LayerNorm = LayerNorm(CONFIG["emb_dim"]);
        self.outHead: Linear = Linear(CONFIG["emb_dim"], vocabSize, bias=False);

    def forward(self, batch: Tensor) -> Tensor:
        _, strlen = batch.shape;
        tokenEmbeds = self.tokenEmbedding(batch);
        posEmbeds = self.posEmbedding(torch.arange(strlen, device=batch.device));
        x = posEmbeds + tokenEmbeds;
        x = self.dropEmbedding(x);
        x = self.blocks(x);
        x = self.finalNorm(x);
        logits = self.outHead(x);
        return logits;

class GELU(Module):
    def __init__(self) -> None:
        super().__init__();

    def forward(self, x: Tensor) -> Tensor:
        # Some high-end maths
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ));

class FeedForward(Module):
    def __init__(self) -> None:
        super().__init__();
        self.layers: Sequential = Sequential(
            Linear(CONFIG["emb_dim"], 4 * CONFIG["emb_dim"]),
            GELU(),
            Linear(4 * CONFIG["emb_dim"], CONFIG["emb_dim"]),
        );

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x);

class TransformerBlock(Module):
    def __init__(self) -> None:
        super().__init__();
        
        self.attention = Attention(CONFIG["emb_dim"], CONFIG["emb_dim"], CONFIG["context_length"], CONFIG["drop_rate"], CONFIG["n_heads"], CONFIG["qkv_bias"]);
        self.ff = FeedForward();
        self.norm1 = LayerNorm(CONFIG["emb_dim"]);
        self.norm2 = LayerNorm(CONFIG["emb_dim"]);
        self.norm3 = LayerNorm(CONFIG["emb_dim"]);
        self.drop_shortcut = Dropout(CONFIG["drop_rate"]);

    def forward(self, x: Tensor) -> Tensor:
        # Shortcuts for some gradient stuff
        shortcut: Tensor = x;
        x = self.norm1(x);
        x = self.attention(x);
        x = self.drop_shortcut(x);
        x = x + shortcut;

        shortcut: Tensor = x;
        x = self.norm2(x);
        x = self.ff(x);
        x = self.drop_shortcut(x);
        x = x + shortcut;
        
        shortcut: Tensor = x;
        x = self.norm3(x);
        x = self.ff(x);
        x = self.drop_shortcut(x);
        x = x + shortcut;

        return x;


class LayerNorm(Module):
    def __init__(self, normalizedShape: int, eps: float = 1e-5) -> None:
        super().__init__();

        self.eps: float = eps;
        self.scale: Tensor = Parameter(torch.ones(normalizedShape));
        self.shift: Tensor = Parameter(torch.zeros(normalizedShape));

    def forward(self, batch: Tensor) -> Tensor:
        mean: Tensor = batch.mean(dim=-1, keepdim=True);
        varitation: Tensor = batch.var(dim=-1, keepdim=True, unbiased=False);
        norm: Tensor = (batch - mean) / torch.sqrt(varitation + self.eps);

        return self.scale * norm + self.shift;

def makeLoader(text: str, tokenizer: Tokenizer, batchSize: int = 4, maxLength: int = 256, stride: int = 128, shuffle: bool = True, dropLast: bool = True) -> DataLoader:
    data: Data = Data(text, tokenizer, maxLength, stride);
    dataLoader: DataLoader = DataLoader(data, batch_size=batchSize, shuffle=shuffle, drop_last=dropLast);
    return dataLoader;

def textGenerator(model: GPT, batch: Tensor, maxNewTokens: int, contextSize: int, device) -> Tensor:
    for _ in range(maxNewTokens):
        batch = batch[:, -contextSize:];
        batch = batch.to(device);
    
        # Get the predictions
        with torch.no_grad():
            logits = model(batch);
    
        # Discard token size
        logits = logits[:, -1, :];

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1);
        nextToken = torch.argmax(probas, dim=-1, keepdim=True);

        # Append sampled index to the running sequence
        batch = torch.cat((batch, nextToken), dim=1);

    return batch;

def calcLoss(input: Tensor, target: Tensor, model: GPT, device) -> Tensor:
    input, target = input.to(device), target.to(device);
    logits: Tensor = model(input);
    loss: Tensor = cross_entropy(logits.flatten(0, 1), target.flatten());
    return loss;


def calcLossForLoader(loader: DataLoader, model: GPT, device, batchCount: int | None = None):
    totalLoss = 0.0;

    if len(loader) == 0:
        return float("nan");

    if batchCount is None:
        # Good idea! Train on all the data
        # And at the same time melt my cpu and create a heater :D
        batchCount = len(loader);
    else:
        # Prevent overflow
        batchCount = min(batchCount, len(loader));

    for batchNumber, (input, target) in enumerate(loader):
        if batchNumber >= batchCount:
            break;

        loss = calcLoss(input, target, model, device);
        totalLoss += loss.item();

    return totalLoss / batchCount;

def evaluateModel(model: GPT, trainLoader: DataLoader, testLoader: DataLoader, device, evalIter: int):
    model.eval();

    # Some performance
    with torch.no_grad():
        train_loss = calcLossForLoader(trainLoader, model, device, batchCount=evalIter);
        val_loss = calcLossForLoader(testLoader, model, device, batchCount=evalIter);

    model.train();

    return train_loss, val_loss


def trainModel(model: GPT, trainLoader: DataLoader, testLoader: DataLoader, optimizer, device, numEpochs: int, evalFreq, evalIter, startContext, tokenizer: Tokenizer):
    tokensSeen, step = 0, -1;
    out = textGenerator(
        model=model,
        batch=torch.tensor(tokenizer.encode(startContext)).unsqueeze(0),
        maxNewTokens=80,
        contextSize=CONFIG["context_length"],
        device=device
    );
    print(tokenizer.decodeTensor(out));

    for epoch in range(numEpochs):
        # We are training!
        model.train();
        
        for input, target in trainLoader:
            optimizer.zero_grad(); # Reset
            loss = calcLoss(input, target, model, device);
            loss.backward();
            optimizer.step();
            tokensSeen += input.numel();
            step += 1;

            # Optional evaluation step
            if step % evalFreq == 0:
                trainLoss, testLoss = evaluateModel(model, trainLoader, testLoader, device, evalIter);
                print(f"Cycle {epoch+1} (Step {step:06d}): ");
                print(f"Train loss {trainLoss:.3f}, Test loss {testLoss:.3f}");

        torch.save({"modelState": model.state_dict(), "optimizerState": optimizer.state_dict()}, f"model-{epoch}.pth");

        out = textGenerator(
            model=model,
            batch=torch.tensor(tokenizer.encode(startContext)).unsqueeze(0),
            maxNewTokens=80,
            contextSize=CONFIG["context_length"],
            device=device
        );
        print(tokenizer.decodeTensor(out));

# 1st and 2nd part of Ascendance of a Bookworm
# 814028 tokens
# 3819442 characters
# 17817 unique words
# Read the book, it's peak fiction
with open("ln/full.txt", mode="r", encoding="utf-8") as file:
    data: str = file.read();

tokenizer: Tokenizer = Tokenizer(data);

print(f"The training data is {len(data)} characters long");
print(f"The vocab size of the llm is {tokenizer.vocabSize()}");

# Split up the data
trainRatio = 0.10;
split = int(trainRatio * len(data));
trainData = data[split:];
testData  = data[:split];

checkpoint = torch.load("large-aob.pth", map_location=torch.device('cpu'));
model: GPT = GPT(tokenizer.vocabSize());
model = torch.nn.DataParallel(model)
model = model.to(device);
model.load_state_dict(checkpoint["modelState"]);

if TRAIN:
    model.train();
else:
    model.eval();

params: int = sum(p.numel() for p in model.parameters());

print(f"Total number of parameters is {params}");

if TRAIN:
    trainLoader: DataLoader = makeLoader(trainData, tokenizer=tokenizer, batchSize=15, maxLength=CONFIG["context_length"], stride=CONFIG["context_length"], shuffle=True, dropLast=True);
    testLoader: DataLoader = makeLoader(testData, tokenizer=tokenizer, batchSize=15, maxLength=CONFIG["context_length"], stride=CONFIG["context_length"], shuffle=False, dropLast=False);

    optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=0.1);
    optimizer.load_state_dict(checkpoint["optimizerState"]);
    numEpochs = 10;
    trainModel(model, trainLoader=trainLoader, testLoader=testLoader, optimizer=optimizer, device=device, numEpochs=numEpochs, evalFreq=100, evalIter=1, startContext="three years later", tokenizer=tokenizer);

    torch.save({"modelState": model.state_dict(), "optimizerState": optimizer.state_dict()}, "t.pth");

print(tokenizer.decodeTensor(textGenerator(
        model=model,
        batch=torch.tensor(tokenizer.encode(TEST_PROMPT)).unsqueeze(0),
        maxNewTokens=80,
        contextSize=CONFIG["context_length"],
        device=device
)));

ip_map = {};

app = FastAPI(docs_url=None, redoc_url=None);

@app.get("/api/hidden/api/{string}")
async def read_item(string, request: Request):
    if not request.client == None:
        if not request.client.host in ip_map:
            ip_map[request.client.host] = 0;

        if ip_map[request.client.host] >= TOKENS:
            return {"output": "Insufficient credits! Contact cyao on slack to get more!"};

        ip_map[request.client.host] += 1;

        print(f"Client {request.client.host} has used {ip_map[request.client.host]} tokens");

    if "<<|UNK|>>" in tokenizer.decode(tokenizer.encode(string)):
        return {"output": "Your prompt contains unknown token"};

    print("Request: ", string);
    output: str = tokenizer.decodeTensor(textGenerator(
        model=model,
        batch=torch.tensor(tokenizer.encode(string)).unsqueeze(0),
        maxNewTokens=100,
        contextSize=CONFIG["context_length"],
        device=device
    ));
    print("Output:", output);

    return {"output": output};

@app.get("/")
async def index():
    return FileResponse('index.html');

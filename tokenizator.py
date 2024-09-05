#!/usr/bin/env python

import re
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Regex is not for phrasing json
# Used it to phrase json when I was small :(

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

def main() -> None:
    with open('data.txt') as file:
        text = file.read()

    print(f"The verdict is {len(text)} characters long");

    tokens: list[str] = [token for token in re.split(r'([,.!?:;_"()\']|--|\s)', text) if token.strip()];
    print(f"This text consists of {len(tokens)} tokens");

    tokenizer = Tokenizer(tokens);

    while 1:
        print(tokenizer.decode(tokenizer.encode(input())))

if __name__ == "__main__":
    main();


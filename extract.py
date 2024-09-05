#!/usr/bin/env python

import json

def main():
    with open('messages/index.json') as file:
        index = json.load(file);

    for channel, _ in index.items():
        print(channel);

if __name__ == "__main__":
    main();

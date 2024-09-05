#!/usr/bin/env python

import json

def getMessages(channel: str) -> list[str]:
    with open(f'messages/c{channel}/messages.json') as file:
        messages = json.load(file);
    return [message["Contents"] for message in messages];

def main():
    with open('messages/index.json') as file:
        index = json.load(file);

    channels = [];
    for channel, _ in index.items():
        channels.append(channel);

    # Now to combine the messages
    messages = [getMessages(channel) for channel in channels];
    # Flatten
    messages = [message for channel in messages for message in channel]

    with open('data.txt', "w") as output:
        output.write(' <<|EOF|>> '.join(messages));

if __name__ == "__main__":
    main();

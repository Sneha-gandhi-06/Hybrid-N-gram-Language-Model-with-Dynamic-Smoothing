from collections import Counter
from pipeline.tokenizer import tokenize
import json

def build_vocab(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)
        counter.update(tokens)

    return dict(counter)


def save_vocab(vocab):
    with open("data/vocab.json", "w") as f:
        json.dump(vocab, f)
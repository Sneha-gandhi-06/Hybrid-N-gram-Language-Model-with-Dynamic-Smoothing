from collections import Counter
from person_a.day1.src.tokenizer import tokenize
import json

def build_vocab(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)
        counter.update(tokens)

    return dict(counter)


def save_vocab(vocab):
    with open("person_a/day1/data/vocab.json", "w") as f:
        json.dump(vocab, f)
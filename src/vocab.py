from collections import Counter
from src.tokenizer import tokenize
import json

def build_vocab(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)
        counter.update(tokens)

    vocab = {}
    idx = 0

    for word, freq in counter.items():
        if freq >= 5:
            vocab[word] = idx
            idx += 1

    vocab["<UNK>"] = idx

    return vocab


def save_vocab(vocab):
    with open("data/vocab.json", "w") as f:
        json.dump(vocab, f)
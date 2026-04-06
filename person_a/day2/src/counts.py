from collections import Counter
from person_a.day1.src.tokenizer import tokenize


def build_unigram(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)
        counter.update(tokens)

    return counter


def build_bigram(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)

        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            counter[bigram] += 1

    return counter


def build_trigram(texts):
    counter = Counter()

    for line in texts:
        tokens = tokenize(line)

        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            counter[trigram] += 1

    return counter
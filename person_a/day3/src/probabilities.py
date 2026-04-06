from collections import defaultdict


def build_bigram_prob(bigram_counts, unigram_counts):
    probs = {}

    for (w1, w2), count in bigram_counts.items():
        probs[(w1, w2)] = count / unigram_counts[w1]

    return probs


def build_trigram_prob(trigram_counts, bigram_counts):
    probs = {}

    for (w1, w2, w3), count in trigram_counts.items():
        probs[(w1, w2, w3)] = count / bigram_counts[(w1, w2)]

    return probs
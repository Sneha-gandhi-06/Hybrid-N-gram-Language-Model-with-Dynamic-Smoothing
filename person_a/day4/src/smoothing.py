def add_one_bigram_prob(bigram_counts, unigram_counts, vocab_size):
    probs = {}

    for (w1, w2), count in bigram_counts.items():
        probs[(w1, w2)] = (count + 1) / (unigram_counts[w1] + vocab_size)

    return probs


def get_bigram_prob(w1, w2, bigram_counts, unigram_counts, vocab_size):
    count = bigram_counts.get((w1, w2), 0)
    return (count + 1) / (unigram_counts.get(w1, 0) + vocab_size)
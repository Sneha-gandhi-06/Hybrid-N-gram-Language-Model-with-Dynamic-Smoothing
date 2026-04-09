import math
from pipeline.smoothing_addk import get_bigram_prob
from pipeline.tokenizer import tokenize


def calculate_perplexity(texts, bigram_counts, unigram_counts, vocab_size):
    log_prob_sum = 0
    word_count = 0

    for line in texts:
        tokens = tokenize(line)

        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]

            prob = get_bigram_prob(w1, w2, bigram_counts, unigram_counts, vocab_size)

            log_prob_sum += math.log(prob)
            word_count += 1

    perplexity = math.exp(-log_prob_sum / word_count)
    return perplexity
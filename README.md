# Hybrid N-gram Language Model with Dynamic Smoothing

NLP Mini Project — built in 1 week by 2 people.

## What it does
A language model that predicts the next word given a sequence of words.
Unlike a plain trigram model, it dynamically switches between unigram,
bigram, and trigram based on how much training data exists for each context.
If a trigram context was seen 10+ times it uses trigram probabilities. If
5+ times it falls back to bigram. Otherwise it uses unigram. Probabilities
are smoothed using Kneser-Ney smoothing.

## Results

| Model | Validation Perplexity |
|---|---|
| Static trigram (KN) | XX.XX |
| Hybrid dynamic (KN) | XX.XX |
| Hybrid dynamic — test set | XX.XX |

*(fill in your numbers from day6_b.py output)*

## Dataset
WikiText-103 via HuggingFace datasets. Trained on 3000 lines.

## Key concepts
- **N-gram model** — estimates P(next word | previous N-1 words)
- **Kneser-Ney smoothing** — rewards words appearing in diverse contexts
- **Dynamic switching** — uses context frequency to pick the best N-gram order

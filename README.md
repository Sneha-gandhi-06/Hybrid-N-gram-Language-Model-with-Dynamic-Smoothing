# Hybrid N-gram Language Model with Dynamic Smoothing

NLP Mini Project — built in 1 week by 2 people.

## What it does
A language model that predicts the next word given a sequence of words.
Unlike a plain trigram model, it dynamically switches between unigram,
bigram, and trigram based on how much training data exists for each context.
If a trigram context was seen 10+ times it uses trigram probabilities. If
5+ times it falls back to bigram. Otherwise it uses unigram. Probabilities
are smoothed using Kneser-Ney smoothing.

## Project structure
├── data/
│   └── vocab.json          # shared vocabulary
├── person_a/               # data loading, tokenization, counts
│   ├── day1/src/
│   │   ├── data_loader.py
│   │   ├── tokenizer.py
│   │   └── vocab.py
│   └── day2/src/
│       └── counts.py
├── person_b/               # smoothing, evaluation, charts
│   ├── day3_b.py           # Kneser-Ney smoothing
│   ├── day4_b.py           # perplexity evaluation
│   ├── day5_b.py           # hybrid dynamic switcher
│   └── day6_b.py           # charts and results
└── main.py                 # runs everything end to end
## Setup
```bash
pip install datasets nltk numpy scipy matplotlib
```

## How to run
```bash
# full pipeline
python main.py

# individual days
python person_b/day3_b.py   # Kneser-Ney smoothing
python person_b/day4_b.py   # perplexity evaluation
python person_b/day5_b.py   # hybrid model + demo
python person_b/day6_b.py   # charts
```

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

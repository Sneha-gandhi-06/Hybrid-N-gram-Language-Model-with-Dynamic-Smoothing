# Hybrid N-gram Language Model with Dynamic Smoothing

An NLP project that is a statistical next-word prediction system with a live desktop UI — no deep learning, just math and Python.

---

## What it does

Given a sequence of words, the model predicts the most likely next word and displays it as ghost text in a desktop application — similar to MS Word or VS Code autocomplete.

Unlike a plain trigram model that always uses a fixed context window, this model **dynamically switches** between unigram, bigram, and trigram based on how much training data exists for the current context. This makes it more flexible and robust than traditional fixed N-gram models.

---

## Demo

```
Type:   "in the united "
Ghost:   states

Type:   "he was looking at "
Ghost:   the

Type:   "in the united states "
Ghost:   and
```

Press `Tab` to accept a suggestion. Press `Esc` to clear. The UI shows which N-gram order is being used (Trigram / Bigram / Unigram) live for every prediction.

---

## How it works

### 1. Data loading & tokenization
Raw text from WikiText-103 is lowercased and split into word tokens. Words appearing fewer than 5 times are replaced with `<UNK>` (unknown) to keep the vocabulary manageable.

### 2. N-gram counting
Three count tables are built from the training data:
- **Unigram** — how often each word appears
- **Bigram** — how often each word pair appears, e.g. `("the", "cat") → 30`
- **Trigram** — how often each word triple appears, e.g. `("the", "cat", "sat") → 20`

### 3. Kneser-Ney smoothing
Instead of using raw frequency, Kneser-Ney smoothing assigns probability based on how many **unique contexts** a word appears in. A word like "Francisco" is common but almost always follows "San" — low context diversity, low continuation probability. A word like "the" follows thousands of different words — high continuation probability. This produces much more natural predictions than simple Add-k smoothing.

### 4. Dynamic switching
For every prediction, the model checks how many times the current context was seen in training:

| Context frequency | Model used |
|---|---|
| Trigram context seen ≥ 3 times | Trigram — uses last 2 words |
| Bigram context seen ≥ 2 times | Bigram — uses last 1 word |
| Context rare or unseen | Unigram — uses raw word frequency |

This means the model uses the richest possible context when it has enough data, and falls back gracefully when it doesn't.

### 5. Desktop UI
Built with Python's built-in `tkinter` library. Prediction runs in a background thread so the UI never freezes. Suggestions appear only after a complete word (on spacebar press), never mid-word.

---

## Project structure

```
├── data/
│   └── vocab.json              # shared vocabulary (word → index)
├── pipeline/
│   ├── loader.py               # loads WikiText-103 from HuggingFace
│   ├── tokenizer.py            # lowercases and splits text into tokens
│   ├── vocab_builder.py        # builds and saves vocabulary
│   ├── counts.py               # builds unigram, bigram, trigram counts
│   ├── probabilities.py        # raw probability computation
│   ├── smoothing_addk.py       # Add-k smoothing (baseline)
│   ├── generator.py            # next word generation utilities
│   └── evaluation.py           # perplexity evaluation
├── smoothing/
│   ├── kneser_ney.py           # Kneser-Ney smoothing + get_prob()
│   ├── perplexity.py           # perplexity computation
│   └── switcher.py             # hybrid dynamic switcher + predict_next()
├── ui/
│   └── predictor.py            # ghost text desktop UI (tkinter)
├── main.py                     # single entry point — builds model + launches UI
└── README.md
```

---

## Setup

**Requirements:** Python 3.8+, internet connection (to download dataset on first run)

```bash
pip install datasets nltk numpy scipy matplotlib
```

---

## How to run

```bash
python main.py
```

That's it. On first run it will:
1. Download WikiText-103 (~183MB, one time only)
2. Build count tables from 20,000 training lines (~30-60 seconds)
3. Launch the desktop UI

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | triggers next-word prediction |
| `Tab` | accept the ghost text suggestion |
| `Esc` | clear the input |
| `i` | toggle model info panel |

---

## Results

Evaluated on WikiText-103 validation and test splits (500 lines each):

| Model | Validation Perplexity |
|---|---|
| Static trigram + KN smoothing | 42,315 |
| Hybrid dynamic + KN smoothing | 42,539 |
| Hybrid dynamic — test set | 32,178 |

**Note on perplexity:** These values are high compared to neural models, which is expected for a count-based statistical model trained on only 20,000 lines. Lower perplexity = better. The qualitative predictions (e.g. "in the united" → "states", "kingdom", "nations") are linguistically accurate. Training on the full WikiText-103 dataset (~1.8M lines) would significantly reduce perplexity.

---

## Key concepts

**N-gram model** — estimates the probability of the next word given the previous N-1 words. A trigram uses the last 2 words as context: `P(w3 | w1, w2)`.

**Kneser-Ney smoothing** — assigns probability based on context diversity rather than raw frequency. Prevents zero probabilities for unseen word combinations and produces more natural predictions than Laplace smoothing.

**Dynamic switching** — instead of always using the same N-gram order, the model checks context frequency at runtime and picks the most reliable order. This is what makes the model "hybrid".

**Perplexity** — standard evaluation metric for language models. Measures how surprised the model is by new text. Formula: `PP = exp(-1/N * Σ log P(wi))`. Lower is better.

---

## Limitations

- Trained on Wikipedia — predictions reflect Wikipedia's language style, not everyday conversation
- Context window capped at 3 words (trigram) — cannot capture long-range dependencies
- No semantic understanding — treats words as symbols, not meanings
- High perplexity compared to neural models (LSTM, GPT, etc.)

---

## Future improvements

- Train on the full WikiText-103 dataset for lower perplexity
- Implement interpolation smoothing (blend all three orders on every prediction)
- Add a conversational dataset (BookCorpus, OpenWebText) for more natural predictions
- Compare against a neural baseline (LSTM or fine-tuned GPT-2)
- Cache count tables to disk so startup is instant after first run

---

## Dataset

**WikiText-103** via HuggingFace `datasets` library.
A collection of verified, high-quality Wikipedia articles widely used as an NLP benchmark.

| Split | Lines used |
|---|---|
| Train | 20,000 |
| Validation | 337 |
| Test | 321 |
| Vocabulary | ~159,000 words |

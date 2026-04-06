import json
import sys
import os
from collections import defaultdict, Counter

# ── Add root to path so we can import person_a's code ────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_a.day1.src.data_loader import load_data
from person_a.day1.src.tokenizer import tokenize
from person_a.day2.src.counts import build_unigram, build_bigram, build_trigram

# ── Load vocab ───────────────────────────────────────────────
with open("data/vocab.json") as f:
    vocab = json.load(f)

V = len(vocab)
print(f"Vocab size: {V}")

# ── Build counts (same limit person A used) ──────────────────
print("Loading data...")
train, _, _ = load_data()
train = train[:3000]

print("Building counts...")
uni_counts = build_unigram(train)
bi_raw     = build_bigram(train)
tri_raw    = build_trigram(train)

# ── Convert to defaultdict(Counter) for easy lookup ──────────
# Person A stores bigrams as {(w1,w2): count}
# We need {(w1,): Counter({w2: count})} for context lookup
bi_counts = defaultdict(Counter)
for (w1, w2), count in bi_raw.items():
    bi_counts[(w1,)][w2] += count

tri_counts = defaultdict(Counter)
for (w1, w2, w3), count in tri_raw.items():
    tri_counts[(w1, w2)][w3] += count

print(f"Unigram entries : {len(uni_counts)}")
print(f"Bigram contexts : {len(bi_counts)}")
print(f"Trigram contexts: {len(tri_counts)}")

# ── Step 1: Continuation counts ──────────────────────────────
continuation_counts = Counter()
for context, word_counts in bi_counts.items():
    for word in word_counts:
        continuation_counts[word] += 1

total_continuation = sum(continuation_counts.values())

# ── Step 2: KN unigram ───────────────────────────────────────
def kn_unigram(word):
    return continuation_counts.get(word, 0) / total_continuation

# ── Step 3: KN bigram ────────────────────────────────────────
D = 0.75

def kn_bigram(word, context):
    ctx = context[-1:]
    ctx_total = sum(bi_counts.get(ctx, {}).values())

    if ctx_total == 0:
        return kn_unigram(word)

    word_count  = bi_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(bi_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total

    return discounted + lambda_ctx * kn_unigram(word)

# ── Step 4: KN trigram ───────────────────────────────────────
def kn_trigram(word, context):
    ctx = context[-2:]
    ctx_total = sum(tri_counts.get(ctx, {}).values())

    if ctx_total == 0:
        return kn_bigram(word, context[-1:])

    word_count  = tri_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(tri_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total

    return discounted + lambda_ctx * kn_bigram(word, context[-1:])

# ── Unified get_prob() ───────────────────────────────────────
def get_prob(word: str, context: tuple, order: int) -> float:
    word    = word if word in vocab else "<UNK>"
    context = tuple(w if w in vocab else "<UNK>" for w in context)
    if order == 1:
        return kn_unigram(word)
    elif order == 2:
        return kn_bigram(word, context[-1:])
    else:
        return kn_trigram(word, context[-2:])

# ── Sanity checks ────────────────────────────────────────────
print("\n--- Sanity Checks ---")

sample_vocab = list(vocab.keys())[:2000]
total = sum(get_prob(w, ("on", "the"), 3) for w in sample_vocab)
print(f"Prob sum over 2000 words (trigram): {total:.4f}  ← should be close to 1.0")

print("\nKN trigram probs for context ('on', 'the'):")
for w in ["the", "a", "mat", "floor", "<UNK>"]:
    print(f"  P({w} | on, the) = {get_prob(w, ('on','the'), 3):.8f}")

print("\nAll three orders for word='the':")
print(f"  Unigram : {get_prob('the', (), 1):.8f}")
print(f"  Bigram  : {get_prob('the', ('on',), 2):.8f}")
print(f"  Trigram : {get_prob('the', ('on','the'), 3):.8f}")

print("\nDay 3 complete!")
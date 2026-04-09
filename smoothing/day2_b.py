import json
import pickle
import sys
from collections import defaultdict, Counter

# ── Load vocab ───────────────────────────────────────────────
with open("data/vocab.json") as f:
    vocab = json.load(f)

V = len(vocab)
print(f"Vocab size: {V}")

# ── Load Person A's count tables ─────────────────────────────
# Person A: please push counts.pkl to data/ when done with your Day 2
try:
    with open("../data/counts.pkl", "rb") as f:
        data = pickle.load(f)
        uni_counts = data["unigram"]
        bi_counts  = data["bigram"]
        tri_counts = data["trigram"]
    print(f"Unigram entries: {len(uni_counts)}")
    print(f"Bigram contexts: {len(bi_counts)}")
    print(f"Trigram contexts: {len(tri_counts)}")
    USING_MOCK = False

except FileNotFoundError:
    print("counts.pkl not found yet — using mock counts. Pull again when Person A pushes Day 2.")
    uni_counts = Counter({
        "the": 500, "cat": 80, "sat": 40, "on": 200,
        "mat": 60, "floor": 45, "a": 300, "<UNK>": 150
    })
    bi_counts = defaultdict(Counter, {
        ("the",): Counter({"cat": 30, "mat": 20, "floor": 15}),
        ("cat",): Counter({"sat": 25, "is": 10}),
        ("sat",): Counter({"on": 35}),
        ("on",):  Counter({"the": 40, "a": 12}),
    })
    tri_counts = defaultdict(Counter, {
        ("cat", "sat"): Counter({"on": 20, "there": 5}),
        ("sat", "on"):  Counter({"the": 18, "a": 8}),
        ("on",  "the"): Counter({"mat": 22, "floor": 12}),
    })
    USING_MOCK = True

# ── Add-k Smoothing ──────────────────────────────────────────
k = 1.0

def addk_unigram(word):
    total = sum(uni_counts.values())
    return (uni_counts.get(word, 0) + k) / (total + k * V)

def addk_bigram(word, context):
    ctx = context[-1:]
    ctx_total = sum(bi_counts.get(ctx, {}).values())
    word_count = bi_counts.get(ctx, {}).get(word, 0)
    return (word_count + k) / (ctx_total + k * V)

def addk_trigram(word, context):
    ctx = context[-2:]
    ctx_total = sum(tri_counts.get(ctx, {}).values())
    word_count = tri_counts.get(ctx, {}).get(word, 0)
    return (word_count + k) / (ctx_total + k * V)

# ── Unified get_prob() interface ─────────────────────────────
def get_prob(word: str, context: tuple, order: int) -> float:
    word    = word if word in vocab else "<UNK>"
    context = tuple(w if w in vocab else "<UNK>" for w in context)
    if order == 1:
        return addk_unigram(word)
    elif order == 2:
        return addk_bigram(word, context[-1:])
    else:
        return addk_trigram(word, context[-2:])

# ── Sanity checks ────────────────────────────────────────────
print(f"\n--- Sanity Checks {'(MOCK DATA)' if USING_MOCK else '(REAL DATA)'} ---")

sample_vocab = list(vocab.keys())[:2000]
ctx = ("on", "the")
total = sum(get_prob(w, ctx, 3) for w in sample_vocab)
print(f"Prob sum over 2000 words (trigram): {total:.4f}  ← should be close to 1.0")

print("\nSample trigram probs for context ('on', 'the'):")
for w in ["mat", "floor", "cat", "the", "xqzjw", "<UNK>"]:
    print(f"  P({w} | on, the) = {get_prob(w, ('on','the'), 3):.8f}")

print("\nAll three orders for word='mat':")
print(f"  Unigram:  {get_prob('mat', (), 1):.8f}")
print(f"  Bigram:   {get_prob('mat', ('the',), 2):.8f}")
print(f"  Trigram:  {get_prob('mat', ('on','the'), 3):.8f}")

print("\nDay 2 complete!")
import json
import nltk
from collections import defaultdict, Counter

nltk.download('punkt')

# ── Load vocab from data/ folder ─────────────────────────────
with open("data/vocab.json") as f:
    vocab = json.load(f)

V = len(vocab)
print(f"Vocab loaded: {V} words")

# ── Mock count tables (replace with real counts on Day 2) ────
mock_uni = Counter({
    "the": 500, "cat": 80, "sat": 40, "on": 200,
    "mat": 60, "floor": 45, "a": 300, "<UNK>": 150
})

mock_bi = defaultdict(Counter, {
    ("the",):  Counter({"cat": 30, "mat": 20, "floor": 15}),
    ("cat",):  Counter({"sat": 25, "is": 10}),
    ("sat",):  Counter({"on": 35}),
    ("on",):   Counter({"the": 40, "a": 12}),
})

mock_tri = defaultdict(Counter, {
    ("cat", "sat"): Counter({"on": 20, "there": 5}),
    ("sat", "on"):  Counter({"the": 18, "a": 8}),
    ("on",  "the"): Counter({"mat": 22, "floor": 12}),
})

print(f"Mock tables ready!")
print(f"Unigrams: {len(mock_uni)} | Bigram contexts: {len(mock_bi)} | Trigram contexts: {len(mock_tri)}")

# ── Placeholder get_prob() ───────────────────────────────────
def get_prob(word: str, context: tuple, order: int) -> float:
    """
    Placeholder until real smoothing is built on Day 2 & 3.
    word    = the word whose probability we want
    context = tuple of preceding words e.g. ("on", "the")
    order   = 1 (unigram), 2 (bigram), or 3 (trigram)
    """
    return 1.0 / V

# ── Sanity checks ────────────────────────────────────────────
print("\n--- Sanity Checks ---")

p = get_prob("mat", ("on", "the"), order=3)
print(f"get_prob('mat', ('on','the'), 3) = {p:.6f}")

test_words = ["mat", "floor", "cat", "the", "<UNK>"]
print("\nSample probabilities (all equal — uniform placeholder):")
for w in test_words:
    print(f"  P({w}) = {get_prob(w, ('on', 'the'), 3):.6f}")

ctx = ("the",)
top_words = mock_bi[ctx].most_common(3)
print(f"\nTop words after 'the' in mock data: {top_words}")

print("\nDay 1 complete!")
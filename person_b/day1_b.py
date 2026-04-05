import json
import nltk
from collections import defaultdict, Counter

nltk.download('punkt')

# Mock count tables to practice with until Person A sends real ones
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
    ("cat", "sat"):  Counter({"on": 20, "there": 5}),
    ("sat", "on"):   Counter({"the": 18, "a": 8}),
    ("on",  "the"):  Counter({"mat": 22, "floor": 12}),
})

print("Mock tables ready!")
print(f"Unigrams: {len(mock_uni)} | Bigram contexts: {len(mock_bi)} | Trigram contexts: {len(mock_tri)}")

# The interface Person A's switcher will call — confirm this with them today
def get_prob(word: str, context: tuple, order: int) -> float:
    """Placeholder — returns uniform prob until real smoothing is built."""
    V = 10000  # will be replaced with real vocab size
    return 1.0 / V

print(f"\nTest: get_prob('mat', ('on','the'), 3) = {get_prob('mat', ('on','the'), 3):.6f}")
print("All good — send this function signature to Person A!")
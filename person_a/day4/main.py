from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram
from person_a.day4.src.smoothing import get_bigram_prob

print("Loading data...")
train, _, _ = load_data()

# 🔥 FAST MODE
train = train[:3000]

print("Building counts...")
uni = build_unigram(train)
bi = build_bigram(train)

vocab_size = len(uni)

# 🔥 TEST SMOOTHING
print("\nTesting smoothing...")

print("Seen bigram (i, love):", get_bigram_prob("i", "love", bi, uni, vocab_size))
print("Unseen bigram (i, pizza):", get_bigram_prob("i", "pizza", bi, uni, vocab_size))

print("\nDONE DAY 4 🚀")
from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram, build_trigram
from person_a.day3.src.probabilities import build_bigram_prob, build_trigram_prob

print("Loading data...")
train, _, _ = load_data()

# 🔥 FAST MODE
train = train[:3000]

print("Building counts...")
uni = build_unigram(train)
bi = build_bigram(train)
tri = build_trigram(train)

print("Building probabilities...")
bi_prob = build_bigram_prob(bi, uni)
tri_prob = build_trigram_prob(tri, bi)

# 🔥 SHOW SAMPLE
print("\nSample Bigram Probabilities:", list(bi_prob.items())[:5])
print("Sample Trigram Probabilities:", list(tri_prob.items())[:5])

print("\nDONE DAY 3 🚀")
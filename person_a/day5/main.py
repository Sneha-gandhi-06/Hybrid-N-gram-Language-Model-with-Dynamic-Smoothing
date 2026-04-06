from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram
from person_a.day3.src.probabilities import build_bigram_prob
from person_a.day5.src.generator import generate_sentence

print("Loading data...")
train, _, _ = load_data()

# 🔥 FAST MODE
train = train[:3000]

print("Building counts...")
uni = build_unigram(train)
bi = build_bigram(train)

print("Building probabilities...")
bi_prob = build_bigram_prob(bi, uni)

# 🔥 GENERATE TEXT
print("\nGenerated Sentences:")

print(generate_sentence("i", bi_prob))
print(generate_sentence("the", bi_prob))
print(generate_sentence("he", bi_prob))

print("\nDONE DAY 5 🚀")
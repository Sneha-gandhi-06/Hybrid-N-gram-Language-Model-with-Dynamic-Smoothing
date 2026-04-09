from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram
from person_a.day6.src.evaluation import calculate_perplexity

print("Loading data...")
train, val, _ = load_data()

# 🔥 FAST MODE
train = train[:3000]
val = val[:1000]

print("Building counts...")
uni = build_unigram(train)
bi = build_bigram(train)

vocab_size = len(uni)

print("Calculating perplexity...")
perplexity = calculate_perplexity(val, bi, uni, vocab_size)

print(f"\nPerplexity: {perplexity}")

print("\nDONE DAY 6 🚀")
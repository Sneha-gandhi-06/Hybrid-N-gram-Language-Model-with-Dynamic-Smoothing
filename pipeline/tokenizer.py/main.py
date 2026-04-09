from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram, build_trigram

print("Loading data...")
train, _, _ = load_data()

# 🔥 LIMIT DATA (VERY IMPORTANT)
train = train[:3000]

print("Building unigram...")
uni = build_unigram(train)

print("Building bigram...")
bi = build_bigram(train)

print("Building trigram...")
tri = build_trigram(train)

# 🔥 ONLY PRINT SMALL SAMPLE (NO FILE SAVING)
print("\nSample Unigrams:", list(uni.items())[:5])
print("Sample Bigrams:", list(bi.items())[:5])
print("Sample Trigrams:", list(tri.items())[:5])

print("\nDONE DAY 2 🚀")
from person_a.day1.src.data_loader import load_data
from person_a.day1.src.vocab import build_vocab, save_vocab

print("Loading data...")
train, _, _ = load_data()

# 🔥 FAST MODE (important)
train = train[:5000]

print("Building vocab...")
vocab = build_vocab(train)
save_vocab(vocab)

print("DONE DAY 1 🚀")
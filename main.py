from src.data_loader import load_data
from src.vocab import build_vocab, save_vocab

print("Loading data...")
train, _, _ = load_data()

print("Building vocab...")
vocab = build_vocab(train)

save_vocab(vocab)

print("DONE")
print("Vocab size:", len(vocab))
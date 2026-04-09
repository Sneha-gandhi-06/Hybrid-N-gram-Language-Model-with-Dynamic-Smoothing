import sys
import os
sys.path.insert(0, os.path.abspath("."))

from pipeline.loader import load_data
from pipeline.tokenizer import tokenize
from pipeline.counts import build_unigram, build_bigram, build_trigram
from smoothing.kneser_ney import get_prob, vocab, bi_counts, tri_counts, uni_counts
from smoothing.perplexity import compute_perplexity
from smoothing.switcher import hybrid_get_prob, predict_next

def main():
    print("=" * 50)
    print(" Hybrid N-gram Language Model")
    print("=" * 50)

    # ── Load data ────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train, val, test = load_data()
    train = train[:3000]
    val   = [t for t in val[:500]  if t.strip()]
    test  = [t for t in test[:500] if t.strip()]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # ── Evaluate ─────────────────────────────────────────
    print("\n[2/4] Running evaluation...")
    pp_static = compute_perplexity(val, get_prob,        order=3)
    pp_hybrid = compute_perplexity(val, hybrid_get_prob, order=3)
    pp_test   = compute_perplexity(test, hybrid_get_prob, order=3)

    print(f"\n  Validation perplexity:")
    print(f"    Static KN trigram : {pp_static:.2f}")
    print(f"    Hybrid dynamic    : {pp_hybrid:.2f}")
    print(f"\n  Test perplexity:")
    print(f"    Hybrid dynamic    : {pp_test:.2f}")

    # ── Demo ─────────────────────────────────────────────
    print("\n[3/4] Next-word prediction demo...")
    predict_next("the cat sat on the")
    predict_next("in the united")
    predict_next("he was looking at")

    print("\n[4/4] Done!")
    print("=" * 50)

if __name__ == "__main__":
    main()
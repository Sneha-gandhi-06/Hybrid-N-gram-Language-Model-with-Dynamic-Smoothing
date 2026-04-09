import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.loader import load_data
from pipeline.tokenizer import tokenize
from smoothing.kneser_ney import get_prob, vocab, bi_counts, tri_counts, uni_counts
from smoothing.perplexity import compute_perplexity

T1 = 3
T2 = 2

def tokenize_and_unk(text):
    tokens = tokenize(text)
    return [t if t in vocab else "<UNK>" for t in tokens]

def get_count(context: tuple) -> int:
    if len(context) == 0:
        return sum(uni_counts.values())
    elif len(context) == 1:
        return sum(bi_counts.get(context, {}).values())
    else:
        return sum(tri_counts.get(context[-2:], {}).values())

def hybrid_get_prob(word: str, context: tuple, order=3) -> float:
    tri_ctx = tuple(context[-2:]) if len(context) >= 2 else ()
    bi_ctx  = tuple(context[-1:]) if len(context) >= 1 else ()

    if get_count(tri_ctx) >= T1:
        return get_prob(word, context, 3)
    elif get_count(bi_ctx) >= T2:
        return get_prob(word, context, 2)
    else:
        return get_prob(word, context, 1)

def predict_next(sentence, top_n=5):
    tokens  = tokenize_and_unk(sentence)
    context = tuple(tokens[-2:])
    scores  = {word: hybrid_get_prob(word, context) for word in list(vocab.keys())[:5000]}
    top     = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nInput   : '{sentence}'")
    print(f"Context : {context}")
    print(f"Top {top_n} predictions:")
    for word, prob in top:
        print(f"  {word:<20} {prob:.6f}")

if __name__ == "__main__":
    train, val, test = load_data()
    val_texts = [t for t in val[:500] if t.strip()]

    pp_static = compute_perplexity(val_texts, get_prob,        order=3)
    pp_hybrid = compute_perplexity(val_texts, hybrid_get_prob, order=3)

    print(f"  Static KN trigram : {pp_static:.2f}")
    print(f"  Hybrid dynamic    : {pp_hybrid:.2f}")
    print(f"  Improvement       : {((pp_static - pp_hybrid) / pp_static) * 100:.1f}%")

    predict_next("the cat sat on the")
    predict_next("in the united")
    predict_next("he was looking at")
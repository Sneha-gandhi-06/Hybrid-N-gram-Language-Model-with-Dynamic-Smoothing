import json
import sys
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ── Add root to path ─────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_a.day1.src.data_loader import load_data
from person_a.day1.src.tokenizer import tokenize
from person_b.day3_b import get_prob, vocab, bi_counts, tri_counts, uni_counts
from person_b.day4_b import compute_perplexity
from person_b.day5_b import hybrid_get_prob, get_count, predict_next

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
train, val, test = load_data()
val_texts  = [t for t in val[:500]  if t.strip()]
test_texts = [t for t in test[:500] if t.strip()]

# ── 1. Run all model comparisons ─────────────────────────────
print("Running perplexity comparisons...")

def get_prob_addk(word, context, order=3):
    """Add-k smoothing for comparison baseline."""
    from person_b.day2_b import get_prob as addk
    return addk(word, context, order)

results = {}

print("  evaluating static trigram (KN)...")
results["Static trigram\n(KN)"]     = compute_perplexity(val_texts, get_prob, order=3)

print("  evaluating static bigram (KN)...")
results["Static bigram\n(KN)"]      = compute_perplexity(val_texts, get_prob, order=2)

print("  evaluating hybrid dynamic (KN)...")
results["Hybrid dynamic\n(KN)"]     = compute_perplexity(val_texts, hybrid_get_prob, order=3)

print(f"\nValidation Perplexity Results:")
for name, pp in results.items():
    print(f"  {name.replace(chr(10),' '):30s} : {pp:.2f}")

# ── 2. Final test set evaluation ─────────────────────────────
print("\nRunning on test set (final evaluation)...")
pp_test = compute_perplexity(test_texts, hybrid_get_prob, order=3)
print(f"  Hybrid dynamic (KN) on TEST set: {pp_test:.2f}")

# ── 3. Plot perplexity comparison ────────────────────────────
names  = list(results.keys())
scores = list(results.values())
colors = ["#9FE1CB", "#5DCAA5", "#0F6E56"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, scores, color=colors, width=0.5, edgecolor="white")

ax.set_ylabel("Perplexity", fontsize=12)
ax.set_title("Model comparison", fontsize=13)
ax.set_ylim(0, max(scores) * 1.3)
ax.axhline(y=min(scores), linestyle="--", color="#888780", linewidth=1)
ax.spines[["top", "right"]].set_visible(False)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(scores) * 0.02,
        f"{score:.1f}", ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("person_b/perplexity_comparison.png", dpi=150)
print("\nChart saved → person_b/perplexity_comparison.png")
plt.show()

# ── 4. N-gram usage distribution ─────────────────────────────
print("\nAnalyzing which N-gram order is used per prediction...")

usage = Counter()
T1, T2 = 10, 5

def tokenize_and_unk(text):
    tokens = tokenize(text)
    return [t if t in vocab else "<UNK>" for t in tokens]

for text in val_texts[:200]:
    tokens = tokenize_and_unk(text)
    for i in range(2, len(tokens)):
        context = tuple(tokens[i-2:i])
        tri_ctx = context[-2:]
        bi_ctx  = context[-1:]
        if get_count(tri_ctx) >= T1:
            usage["trigram"] += 1
        elif get_count(bi_ctx) >= T2:
            usage["bigram"] += 1
        else:
            usage["unigram"] += 1

total_preds = sum(usage.values())
print(f"\nN-gram usage breakdown:")
for order in ["trigram", "bigram", "unigram"]:
    pct = usage[order] / total_preds * 100
    print(f"  {order:10s} : {usage[order]:6d} ({pct:.1f}%)")

# Usage pie chart
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.pie(
    [usage["trigram"], usage["bigram"], usage["unigram"]],
    labels=["Trigram", "Bigram", "Unigram"],
    colors=["#0F6E56", "#5DCAA5", "#9FE1CB"],
    autopct="%1.1f%%",
    startangle=140
)
ax2.set_title("N-gram order usage distribution", fontsize=13)
plt.tight_layout()
plt.savefig("person_b/ngram_usage.png", dpi=150)
print("Chart saved → person_b/ngram_usage.png")
plt.show()

# ── 5. Live demo ──────────────────────────────────────────────
print("\n--- Live Predictions ---")
predict_next("the cat sat on the")
predict_next("in the united")
predict_next("he was looking at")

print("\nDay 6 complete! Commit and push:")
print("  git add person_b/day6_b.py person_b/perplexity_comparison.png person_b/ngram_usage.png")
print('  git commit -m "Day 6 done - charts, evaluation, demo complete"')
print("  git push")
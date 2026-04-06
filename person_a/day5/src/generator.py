def generate_next_word(current_word, bigram_probs):
    candidates = {}

    for (w1, w2), prob in bigram_probs.items():
        if w1 == current_word:
            candidates[w2] = prob

    if not candidates:
        return None

    # 🔥 pick word with highest probability
    next_word = max(candidates, key=candidates.get)
    return next_word


def generate_sentence(start_word, bigram_probs, max_length=10):
    sentence = [start_word]

    current_word = start_word

    for _ in range(max_length - 1):
        next_word = generate_next_word(current_word, bigram_probs)

        if not next_word:
            break

        sentence.append(next_word)
        current_word = next_word

    return " ".join(sentence)
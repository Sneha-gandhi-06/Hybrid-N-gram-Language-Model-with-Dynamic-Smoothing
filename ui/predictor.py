import tkinter as tk
import sys
import os
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smoothing.kneser_ney import get_prob, vocab, bi_counts, tri_counts, uni_counts
from smoothing.switcher import hybrid_get_prob, get_count, tokenize_and_unk, T1, T2

# ── Config ───────────────────────────────────────────────────
FONT_MAIN   = ("Arial", 22)
FONT_SMALL  = ("Arial", 11)
FONT_BADGE  = ("Arial", 10, "bold")
FONT_INFO   = ("Arial", 12)

BG          = "#ffffff"
TEXT_COLOR  = "#1a1a1a"
GHOST_COLOR = "#b0b0b0"
BADGE_BG    = "#f0f0f0"
BADGE_FG    = "#555555"
ACCENT      = "#4a90d9"
BORDER      = "#e0e0e0"

TOP_N       = 5

MODEL_INFO = """Hybrid N-gram Language Model
with Dynamic Smoothing

── How it works ──────────────────
Predicts the next word by looking at
the last 1, 2, or 3 words (context).

Trigram   used when context seen 3+ times
Bigram    used when context seen 2+ times
Unigram   fallback — raw word frequency

── Smoothing ─────────────────────
Kneser-Ney smoothing rewards words
that appear in many diverse contexts,
not just frequent ones.

── Dataset ───────────────────────
Trained on WikiText-103 (Wikipedia)
Training lines : 20,000
Vocabulary     : {:,} words

── Shortcuts ─────────────────────
Tab      accept suggestion
Esc      clear input
i        toggle this panel
"""

def get_order_label(context):
    tri_ctx = tuple(context[-2:]) if len(context) >= 2 else ()
    bi_ctx  = tuple(context[-1:]) if len(context) >= 1 else ()
    if get_count(tri_ctx) >= T1:
        return "Trigram", "#2e7d32"
    elif get_count(bi_ctx) >= T2:
        return "Bigram", "#e65100"
    else:
        return "Unigram", "#6a1b9a"

def get_suggestion(text):
    if not text.strip():
        return None, None, None
    tokens  = tokenize_and_unk(text)
    context = tuple(tokens[-2:])
    scores  = {
        word: hybrid_get_prob(word, context)
        for word in list(vocab.keys())[:8000]
    }
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    last = tokens[-1] if tokens else ""
    candidates = [(w, p) for w, p in top if w != last and w != "<UNK>"]
    if not candidates:
        return None, None, None
    best_word, best_prob = candidates[0]
    order_label, order_color = get_order_label(context)
    return best_word, order_label, order_color


class PredictorApp:
    def __init__(self, root):
        self.root          = root
        self.suggestion    = None
        self.info_open     = False
        self._pred_thread  = None
        self._last_typed   = ""

        root.title("N-gram Predictor")
        root.configure(bg=BG)
        root.geometry("720x340")
        root.resizable(True, True)
        root.minsize(500, 280)

        self._build_titlebar()
        self._build_input_area()
        self._build_status_bar()
        self._build_info_panel()

        self.text_var.trace_add("write", self._on_text_change)
        self.entry.bind("<Tab>",       self._on_tab)
        self.entry.bind("<Escape>",    self._on_escape)
        self.root.bind("<KeyPress-i>", self._toggle_info)
        self.entry.focus()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=BG, pady=12, padx=20)
        bar.pack(fill="x")
        tk.Label(
            bar, text="N-gram Predictor",
            font=("Arial", 13, "bold"),
            bg=BG, fg=TEXT_COLOR
        ).pack(side="left")
        info_btn = tk.Label(
            bar, text=" i ",
            font=FONT_BADGE,
            bg=BADGE_BG, fg=ACCENT,
            cursor="hand2", padx=6, pady=2
        )
        info_btn.pack(side="right")
        info_btn.bind("<Button-1>", self._toggle_info)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    def _build_input_area(self):
        wrapper = tk.Frame(self.root, bg=BG, padx=28, pady=28)
        wrapper.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            wrapper, bg=BG, highlightthickness=0, height=44
        )
        self.canvas.pack(fill="x")

        self.text_var = tk.StringVar()
        self.entry = tk.Entry(
            wrapper,
            textvariable=self.text_var,
            font=FONT_MAIN,
            bd=0, relief="flat",
            bg=BG, fg=BG,
            insertbackground=BG,
            width=1
        )
        self.entry.pack(fill="x")

        badge_row = tk.Frame(wrapper, bg=BG)
        badge_row.pack(fill="x", pady=(6, 0))

        self.order_badge = tk.Label(
            badge_row, text="",
            font=FONT_BADGE,
            bg=BADGE_BG, fg=BADGE_FG,
            padx=8, pady=3
        )
        self.order_badge.pack(side="left")

        self.prob_label = tk.Label(
            badge_row, text="",
            font=FONT_SMALL,
            bg=BG, fg=GHOST_COLOR
        )
        self.prob_label.pack(side="left", padx=(8, 0))

    def _build_status_bar(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self.root, bg=BG, padx=20, pady=8)
        bar.pack(fill="x")
        tk.Label(
            bar,
            text="Tab  accept suggestion    Esc  clear    i  model info",
            font=FONT_SMALL,
            bg=BG, fg=GHOST_COLOR
        ).pack(side="left")

    def _build_info_panel(self):
        self.info_frame = tk.Frame(
            self.root, bg="#f8f8f8",
            padx=20, pady=16,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        tk.Label(
            self.info_frame,
            text=MODEL_INFO.format(len(vocab)),
            font=("Arial", 11),
            bg="#f8f8f8", fg=TEXT_COLOR,
            justify="left", anchor="nw"
        ).pack(fill="both", expand=True)

    def _render_text(self):
        self.canvas.delete("all")
        typed  = self.text_var.get()
        import tkinter.font as tkfont
        f      = tkfont.Font(family="Arial", size=22)
        y      = 22
        cursor = "|"

        self.canvas.create_text(
            0, y,
            text=typed + cursor,
            font=FONT_MAIN,
            fill=TEXT_COLOR,
            anchor="w"
        )

        if self.suggestion:
            typed_width = f.measure(typed + cursor + " ")
            self.canvas.create_text(
                typed_width, y,
                text=self.suggestion,
                font=FONT_MAIN,
                fill=GHOST_COLOR,
                anchor="w"
            )

    def _on_text_change(self, *args):
        typed = self.text_var.get()

        # ── only predict after a space (completed word) ──────
        if not typed.endswith(" "):
            self.suggestion = None
            self.order_badge.config(text="")
            self.prob_label.config(text="")
            self._render_text()
            return

        if not typed.strip():
            self.suggestion = None
            self._render_text()
            return

        # ── run prediction in background thread ──────────────
        self._last_typed = typed
        t = threading.Thread(target=self._predict_worker, args=(typed,), daemon=True)
        t.start()

    def _predict_worker(self, typed):
        word, order_label, order_color = get_suggestion(typed)

        # discard if user kept typing while we were predicting
        if typed != self._last_typed:
            return

        # schedule UI update back on main thread
        self.root.after(0, self._update_ui, typed, word, order_label, order_color)

    def _update_ui(self, typed, word, order_label, order_color):
        # discard stale results
        if typed != self._last_typed:
            return

        self.suggestion = word

        if word:
            self.order_badge.config(text=f"  {order_label}  ", fg=order_color)
            tokens  = tokenize_and_unk(typed)
            context = tuple(tokens[-2:])
            prob    = hybrid_get_prob(word, context)
            self.prob_label.config(text=f"p = {prob:.4f}")
        else:
            self.order_badge.config(text="no suggestion", fg=BADGE_FG)
            self.prob_label.config(text="")

        self._render_text()

    def _on_tab(self, event):
        if self.suggestion:
            current = self.text_var.get().rstrip()
            self.text_var.set(current + " " + self.suggestion + " ")
            self.entry.icursor(tk.END)
            self._on_text_change()
        return "break"

    def _on_escape(self, event):
        self.text_var.set("")
        self.suggestion    = None
        self._last_typed   = ""
        self.order_badge.config(text="")
        self.prob_label.config(text="")
        self._render_text()

    def _toggle_info(self, event=None):
        if self.info_open:
            self.info_frame.pack_forget()
            self.root.geometry("720x340")
            self.info_open = False
        else:
            self.info_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))
            self.root.geometry("720x600")
            self.info_open = True


if __name__ == "__main__":
    print("Loading model...")
    root = tk.Tk()
    app  = PredictorApp(root)
    print("Ready!")
    root.mainloop()
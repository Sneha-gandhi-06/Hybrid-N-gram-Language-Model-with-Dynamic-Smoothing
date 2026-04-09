import sys
import os
sys.path.insert(0, os.path.abspath("."))

def main():
    print("Loading model — this takes ~30 seconds...")
    from ui.predictor import PredictorApp
    import tkinter as tk
    root = tk.Tk()
    app  = PredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
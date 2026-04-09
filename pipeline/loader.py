from datasets import load_dataset

def load_data():
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    return dataset["train"]["text"], dataset["validation"]["text"], dataset["test"]["text"]
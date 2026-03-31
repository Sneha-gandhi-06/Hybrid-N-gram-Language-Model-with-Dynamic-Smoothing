from datasets import load_dataset

def load_data():
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    train = dataset["train"]["text"]
    valid = dataset["validation"]["text"]
    test = dataset["test"]["text"]

    return train, valid, test
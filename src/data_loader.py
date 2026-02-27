import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df
import pickle
import os
from preprocess import preprocess

model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "models", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Map model output indices to sentiment labels
# Based on typical LabelEncoder ordering for [negative, neutral, positive]
def get_label(idx):
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping.get(int(idx), "neutral")


def predict(text: str) -> dict:
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    label_idx = model.predict(vec)[0]
    label = get_label(label_idx)
    proba = model.predict_proba(vec).max()
    return {"label": label, "confidence": round(float(proba), 4)}

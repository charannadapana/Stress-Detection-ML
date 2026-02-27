import matplotlib
matplotlib.use("TkAgg")  # Important for Windows

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import vectorize_text
from src.train import train_models
from src.evaluate import evaluate_models
from src.app import launch_app


# ------------------ LOAD DATA ------------------
df = load_data("data/stress_data.csv")

X = df["text"]
y = df["label"]
label_mapping = {0: "Not Stressed", 1: "Stressed"}

label_counts = df["label"].value_counts()

labels = [label_mapping[i] for i in label_counts.index]

plt.figure()
plt.bar(labels, label_counts.values)
plt.title("Label Distribution in Dataset")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------ VECTORIZATION ------------------
tfidf, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)


# ------------------ TRAIN MODELS ------------------
models = train_models(X_train_vec, y_train)


# ------------------ EVALUATE MODELS ------------------
results = evaluate_models(models, X_test_vec, y_test)


# ------------------ FIND BEST MODEL ------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n==============================")
print(f"Best Model: {best_model_name}")
print("==============================")


# ------------------ PLOT ACCURACY GRAPH ------------------
model_names = list(results.keys())
accuracies = list(results.values())

plt.figure()
plt.bar(model_names, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.show()


# ------------------ CONFUSION MATRIX ------------------
ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test_vec,
    y_test
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()


# ------------------ LAUNCH GRADIO APP ------------------
launch_app(best_model, tfidf)
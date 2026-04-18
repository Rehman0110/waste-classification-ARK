# =============================================================
# Minimal Evaluator (Dynamic + Clean)
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


# ── Evaluation ───────────────────────────────────────────────

def evaluate_model(model, val_gen, model_name="model"):

    print(f"\n=== Evaluating: {model_name} ===\n")

    # Get class names dynamically
    class_names = list(val_gen.class_indices.keys())
    print("Classes:", class_names)

    # True labels
    y_true = val_gen.classes

    # Predictions - Handling Binary vs Multi-class
    probs  = model.predict(val_gen, verbose=1)
    
    if probs.shape[1] == 1:
        # Binary classification (Sigmoid)
        y_pred = (probs > 0.5).astype("int32").flatten()
    else:
        # Multi-class classification (Softmax)
        y_pred = np.argmax(probs, axis=1)

    # ── Classification Report ────────────────────────────────
    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}_cm.png"
    plt.savefig(path)
    plt.close()

    print(f"\n📊 Confusion matrix saved → {path}")

    # Accuracy
    acc = np.mean(y_true == y_pred)
    print(f"\n✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    return y_true, y_pred


def predict_single(model, img_array):
    """
    Predict the class of a single preprocessed image.
    Used by Streamlit app and webcam demo.
    """
    probs = model.predict(img_array, verbose=0)[0]
    
    # Handle Binary (1 output unit)
    if len(probs) == 1:
        prob = float(probs[0])
        # Mapping 0 -> Organic, 1 -> Recyclable (standard alphabetical order)
        class_names = ["Organic", "Recyclable"]
        class_idx = 1 if prob > 0.5 else 0
        confidence = prob if class_idx == 1 else (1 - prob)
        all_probs = {
            "Organic": 1 - prob,
            "Recyclable": prob
        }
    # Handle Multi-class
    else:
        class_names = ["Hazardous", "Organic", "Recyclable"]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        all_probs = {class_names[i]: float(probs[i]) for i in range(len(probs))}

    return class_names[class_idx], confidence, all_probs
import os
import sys
import time
import argparse
import numpy as np
import cv2

from tensorflow.keras.models import load_model


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# ⚠️ MUST MATCH YOUR TRAINED MODEL ORDER
CLASS_NAMES = ["Organic", "Recyclable"]

CLASS_COLOURS_BGR = {
    "Recyclable": (200, 150, 50),
    "Organic": (50, 180, 50),
}

CLASS_ICONS = {
    "Recyclable": "[RECYCLE]",
    "Organic": "[ORGANIC]",
}

SUGGESTIONS = {
    "Recyclable": "Send to recycling bin",
    "Organic": "Compost this waste",
}


# ────────────────────────────────────────────────
def preprocess_frame(frame, size=(224, 224)):
    h, w = frame.shape[:2]

    margin = 60
    y1, y2 = margin, max(margin + 10, h - 80)
    x1, x2 = margin, max(margin + 10, w - margin)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        roi = frame

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, size)
    roi = roi.astype(np.float32) / 255.0

    return np.expand_dims(roi, axis=0)


# ────────────────────────────────────────────────
def run_webcam(model_path: str, camera_index: int = 0):

    model = load_model(model_path)
    print("✅ Model loaded")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Camera not found")
        return

    last_pred = None
    last_conf = 0
    last_probs = {}
    last_time = 0

    print("🎥 Webcam started (SPACE = predict, Q = quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # UI box
        cv2.rectangle(frame, (60, 60), (w-60, h-80), (80, 80, 80), 2)

        if last_pred:
            color = CLASS_COLOURS_BGR.get(last_pred, (200, 200, 200))

            cv2.putText(frame,
                        f"{CLASS_ICONS[last_pred]} {last_pred} ({last_conf:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

            cv2.putText(frame,
                        SUGGESTIONS.get(last_pred, ""),
                        (20, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cv2.imshow("Waste Classifier", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):

            x = preprocess_frame(frame)

            t0 = time.time()
            preds = model.predict(x, verbose=0)[0]
            inference = (time.time() - t0) * 1000

            idx = int(np.argmax(preds))

            # SAFE mapping (prevents crash if mismatch)
            if idx >= len(CLASS_NAMES):
                print("⚠️ Model output mismatch")
                continue

            last_pred = CLASS_NAMES[idx]
            last_conf = float(preds[idx])
            last_probs = {
                CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
            }

            last_time = inference

            print(f"{last_pred} | {last_conf:.2f} | {inference:.1f}ms")

    cap.release()
    cv2.destroyAllWindows()


# ────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, "mobilenet_model.h5")

    if not os.path.exists(model_path):
        print("❌ Model not found")
        exit()

    run_webcam(model_path)
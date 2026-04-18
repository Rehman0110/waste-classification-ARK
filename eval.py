from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators
from src.evaluator import evaluate_model

# ── Paths ─────────────────────────────
MODEL_PATH = "models/mobilenet_model.h5"
DATA_DIR = "data"


def main():

    # ── Load trained model ─────────────
    print("\n📦 Loading model...")
    model = load_model(MODEL_PATH)

    # ── Load validation data ───────────
    print("\n📂 Loading dataset...")
    _, val_gen, _ = get_data_generators(DATA_DIR)

    # ── Evaluate ───────────────────────
    print("\n📊 Running evaluation...")
    evaluate_model(model, val_gen, model_name="mobilenet_v2")


if __name__ == "__main__":
    main()
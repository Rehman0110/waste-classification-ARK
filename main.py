# =============================================================
# main.py (Clean Version)
# =============================================================

import os
import argparse

from src.data_loader import get_data_generators
from src.model_builder import build_mobilenet_model
from src.trainer import train_mobilenet_two_phase, plot_training_history
from src.evaluator import evaluate_model

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)

    args = parser.parse_args()

    # ── Load dataset ───────────────────────────────────────────
    print(f"\n📂 Loading dataset from: {args.data_dir}")
    train_gen, val_gen, _ = get_data_generators(args.data_dir)

    # ── Build model ────────────────────────────────────────────
    print("\n🧠 Building MobileNetV2...")
    model, base_model = build_mobilenet_model()

    # ── Train model (2-phase) ─────────────────────────────────
    p1_epochs = int(args.epochs * 0.4)
    p2_epochs = args.epochs - p1_epochs

    history, model = train_mobilenet_two_phase(
        model,
        base_model,
        train_gen,
        val_gen,
        phase1_epochs=p1_epochs,
        phase2_epochs=p2_epochs
    )

    # ── Save model ────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, "mobilenet_model.h5")
    model.save(save_path)
    print(f"\n✅ Training complete. Model saved to: {save_path}")

    # ── Generate Plots ────────────────────────────────────────
    plot_training_history(history, model_name="mobilenet_v2")

    # ── Evaluate ──────────────────────────────────────────────
    evaluate_model(model, val_gen, model_name="mobilenet_v2")
   

if __name__ == "__main__":
    main()
# =============================================================
# utils/trainer.py
# Full training pipeline:
#   - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#   - Phase 1 (frozen base) + Phase 2 (fine-tuning)
#   - Accuracy & Loss plots
# =============================================================

import os
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from datetime import datetime


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Callbacks Factory ─────────────────────────────────────────

def get_callbacks(model_name: str = "model", patience: int = 7):
    """
    Build a list of training callbacks.

    EarlyStopping       → Stop when val_loss stops improving
    ModelCheckpoint     → Save the best model weights automatically
    ReduceLROnPlateau   → Halve LR when val_loss plateaus (helps escape local minima)

    Args:
        model_name : Used in checkpoint filename
        patience   : Epochs to wait before early-stop triggers

    Returns:
        List of Keras callbacks
    """
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")

    callbacks = [
        # ── Stop training early if no improvement ────────────
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,  # Roll back to best epoch
            verbose=1
        ),

        # ── Save the best model to disk ──────────────────────
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        # ── Reduce LR when learning stalls ───────────────────
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,              # New LR = LR × 0.5
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    return callbacks


# ── Training Function ─────────────────────────────────────────

def train_model(model, train_gen, val_gen,
                epochs: int = 30,
                model_name: str = "model"):
    """
    Train the model and save results.

    Args:
        model      : Compiled Keras model
        train_gen  : Training data generator
        val_gen    : Validation data generator
        epochs     : Maximum training epochs
        model_name : Base name for saved files

    Returns:
        history : Keras History object (contains acc/loss per epoch)
    """
    callbacks = get_callbacks(model_name=model_name)

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}  |  Max epochs: {epochs}")
    print(f"{'='*55}\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save final model ──────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    model.save(final_path)
    print(f"\n✅ Model saved → {final_path}")

    return history


# ── Two-Phase Training for MobileNetV2 ───────────────────────

def train_mobilenet_two_phase(model, base_model,
                               train_gen, val_gen,
                               phase1_epochs: int = 15,
                               phase2_epochs: int = 15,
                               fine_tune_at: int = 100):
    """
    Two-phase training for MobileNetV2 transfer learning.

    Phase 1: Train only the new classifier head (base frozen).
    Phase 2: Fine-tune top layers of the base + classifier head.

    Args:
        model          : Full Keras model
        base_model     : MobileNetV2 backbone
        train_gen      : Training generator
        val_gen        : Validation generator
        phase1_epochs  : Epochs for feature-extraction phase
        phase2_epochs  : Additional epochs for fine-tuning
        fine_tune_at   : Unfreeze layers after this index

    Returns:
        combined_history : Dict with merged accuracy/loss lists
    """
    from src.model_builder import unfreeze_for_fine_tuning

    # ── Phase 1: Feature Extraction ───────────────────────────
    print("\n🔵 PHASE 1: Feature Extraction (base frozen)")
    h1 = train_model(model, train_gen, val_gen,
                     epochs=phase1_epochs, model_name="mobilenet_phase1")

    # ── Phase 2: Fine-Tuning ──────────────────────────────────
    print("\n🟠 PHASE 2: Fine-Tuning (unfreezing top layers)")
    model = unfreeze_for_fine_tuning(model, base_model, fine_tune_at=fine_tune_at)

    h2 = train_model(model, train_gen, val_gen,
                     epochs=phase2_epochs, model_name="mobilenet_model")

    # ── Merge histories ───────────────────────────────────────
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]

    return combined, model


# ── Plot Training Curves ──────────────────────────────────────

def plot_training_history(history, model_name: str = "model",
                           save_path: str = None):
    """
    Plot accuracy vs epoch and loss vs epoch side-by-side.

    Args:
        history    : Keras History object OR dict with history keys
        model_name : Title prefix
        save_path  : Where to save the PNG (default: models/)
    """
    # Support both History objects and plain dicts
    h = history.history if hasattr(history, "history") else history

    epochs = range(1, len(h["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History – {model_name}", fontsize=14, fontweight="bold")

    # ── Accuracy plot ─────────────────────────────────────────
    ax1.plot(epochs, h["accuracy"],     "b-o", label="Train Accuracy", markersize=4)
    ax1.plot(epochs, h["val_accuracy"], "r-o", label="Val Accuracy",   markersize=4)
    ax1.set_title("Accuracy vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # ── Loss plot ─────────────────────────────────────────────
    ax2.plot(epochs, h["loss"],     "b-o", label="Train Loss", markersize=4)
    ax2.plot(epochs, h["val_loss"], "r-o", label="Val Loss",   markersize=4)
    ax2.set_title("Loss vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Save figure ───────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"{model_name}_training_plots.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training plots saved → {save_path}")
    return save_path

import os
import sys
import io
import uvicorn
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf

# ── PATH SETUP ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import preprocess_pil_image
from src.evaluator import predict_single

# ── CONSTANTS ────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "mobilenet_model.keras")

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("🔄 Loading model...")

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # Load model using Keras 3 compatible method
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

        print(f"✅ Model loaded successfully from {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        model = None  # ensure safe fallback

    yield

    print("🛑 Shutting down API...")

# ── APP INITIALIZATION ───────────────────────────────────────
app = FastAPI(title="Waste API", lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Smart Waste Segregation API is active."
    }
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    return {"model_loaded": model is not None}



@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on the server."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        img_array = preprocess_pil_image(img)

        # Optional safety: ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        class_name, confidence, probs = predict_single(model, img_array)

        return {
            "filename": file.filename,
            "prediction": class_name,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                k: round(float(v), 4) for k, v in probs.items()
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
#uvicorn api:app --reload

import os
import sys
import io
import uvicorn
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow.keras.models import load_model

# ── PATH SETUP ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import preprocess_pil_image
from src.evaluator import predict_single

# ── CONSTANTS ────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "mobilenet_model.h5")

# Global model variable to hold the loaded weights
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: Load the model ──
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"✅ Model successfully loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Prediction endpoint will return an error until model is present.")
    yield
    # ── Shutdown: Clean up resources if needed ──
    print("Shutting down API...")

# ── APP INITIALIZATION ───────────────────────────────────────
app = FastAPI(title="Waste API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "online", "message": "Smart Waste Segregation API is active."}

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
    
    try:
        # Read uploaded image bytes and convert to PIL Image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess and Predict
        img_array = preprocess_pil_image(img)
        class_name, confidence, probs = predict_single(model, img_array)
        
        return {
            "filename": file.filename,
            "prediction": class_name,
            "confidence": round(float(confidence), 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
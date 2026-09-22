"""FastAPI serving app for the Food-11 classifier.

Loads the model registered in mlflow's Model Registry under the "champion"
alias (see Question 3 for why that's preferred over pointing at a raw .pth
file), and exposes it over HTTP.

Run locally (from the repo root, with the mlflow tracking server already
running):
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" \
        http://127.0.0.1:8000/predict
"""

import io
import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from .data import CATEGORIES
from .train import IMAGENET_MEAN, IMAGENET_STD

MODEL_URI = "models:/food11@champion"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = [CATEGORIES[i] for i in range(len(CATEGORIES))]

_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

_model = None  # populated at startup, once, in the lifespan handler below


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Loading {MODEL_URI} from tracking server at {tracking_uri} ...")
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded, ready to serve.")
    yield
    _model = None


app = FastAPI(title="food11-api", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    input_tensor = _transform(image).unsqueeze(0).numpy()  # shape (1, 3, 128, 128)

    raw_output = _model.predict(input_tensor)
    logits = raw_output.values if hasattr(raw_output, "values") else np.asarray(raw_output)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]

    predicted_index = int(probabilities.argmax())
    return {
        "category": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }

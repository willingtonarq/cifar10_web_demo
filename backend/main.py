from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cifar10_dense_model.keras"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="API para clasificar imágenes con una red densa entrenada en CIFAR-10.",
    version="1.0.0",
)

# Solo para desarrollo local. En producción, limita los orígenes permitidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Copia cifar10_dense_model.keras dentro de la carpeta backend."
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/")
def root():
    return {
        "message": "API CIFAR-10 funcionando",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Sube un archivo de imagen.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        return {
            "predicted_class": CLASS_NAMES[int(top_indices[0])],
            "confidence": float(probabilities[top_indices[0]]),
            "top3": [
                {
                    "class_name": CLASS_NAMES[int(i)],
                    "probability": float(probabilities[int(i)]),
                }
                for i in top_indices
            ],
            "note": (
                "El modelo fue entrenado con imágenes CIFAR-10 de 32x32 píxeles. "
                "Las fotos reales pueden clasificarse con poca precisión."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar la imagen: {exc}")
    finally:
        await file.close()

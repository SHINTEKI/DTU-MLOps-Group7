import json
import cv2
import numpy as np
import torch
from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse
from loguru import logger
# from opentelemetry import trace
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel
from pathlib import Path

# Tracing setup
# provider = TracerProvider()
# processor = BatchSpanProcessor(OTLPSpanExporter())
# provider.add_span_processor(processor)
# trace.set_tracer_provider(provider)
# tracer = trace.get_tracer(__name__)

app = FastAPI()
# FastAPIInstrumentor.instrument_app(app)

class PredictionResult(BaseModel):
    filename: str
    label: str

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.post("/predict/")
async def predict(filepath: str = Form(...)):
    path = Path(filepath)

    if not path.exists():
        logger.error(f"File not found: {filepath}")
        return {"error": f"Path '{filepath}' does not exist."}

    results = []

    if path.is_file():
        label = predict_step(path)
        results.append(PredictionResult(filename=path.name, label=label))

    elif path.is_dir():
        image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        if not image_files:
            return {"error": f"No images found in directory '{filepath}'."}

        for image_file in image_files:
            label = predict_step(image_file)
            results.append(PredictionResult(filename=image_file.name, label=label))

    return results

def predict_step(image_path: Path):
    model = torch.jit.load("models/deployable_model.pt")
    model.eval()

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.from_numpy(img).float()

    y_pred = model(img)
    ps = torch.exp(y_pred)
    _, top_class = ps.topk(1, dim=1)

    return mapping_to_label(top_class.item())

def mapping_to_label(top_class: int):
    with open("app/index_to_name.json") as f:
        data = json.load(f)
    return data[str(top_class)][1]

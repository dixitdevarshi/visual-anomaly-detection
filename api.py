import io
import time
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms

from src.model.extractor import DINOv2Extractor
from src.model.patchcore import PatchCore

app = FastAPI(title="Visual Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

extractor = None
memory_banks = {}


@app.on_event("startup")
async def startup():
    global extractor
    print("Loading DINOv2 extractor...")
    extractor = DINOv2Extractor()
    print("Extractor loaded.")

    for cat in CATEGORIES:
        bank_path = Path(f"artifacts/{cat}_memory_bank.pt")
        if bank_path.exists():
            pc = PatchCore(extractor=extractor)
            pc.load(str(bank_path))
            memory_banks[cat] = pc
            print(f"Loaded memory bank: {cat}")
        else:
            print(f"No memory bank found for {cat}, skipping.")


def image_to_base64(img_array: np.ndarray) -> str:
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heatmap_to_base64(heatmap: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def patch_scores_to_heatmap(patch_scores: torch.Tensor, image_size: int = 224, patch_size: int = 14) -> np.ndarray:
    n_patches = image_size // patch_size
    scores_2d = patch_scores.reshape(n_patches, n_patches).numpy()
    heatmap = torch.nn.functional.interpolate(
        torch.tensor(scores_2d).unsqueeze(0).unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cmap = cm.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * image).astype(np.uint8)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_categories": list(memory_banks.keys()),
        "n_loaded": len(memory_banks)
    }


@app.get("/categories")
async def categories():
    return {"categories": list(memory_banks.keys())}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    if category not in memory_banks:
        raise HTTPException(
            status_code=400,
            detail=f"Memory bank not found for '{category}'. Run run_pipeline.py first."
        )

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    image_np = np.array(image)

    start = time.time()
    tensor = TRANSFORM(image).unsqueeze(0)
    image_scores, patch_scores = memory_banks[category].predict(tensor)
    inference_ms = round((time.time() - start) * 1000, 1)

    score = float(image_scores[0].item())
    patch_score = patch_scores[0]

    heatmap = patch_scores_to_heatmap(patch_score)
    overlay = overlay_heatmap(image_np, heatmap)

    return JSONResponse({
        "score": score,
        "category": category,
        "inference_time_ms": inference_ms,
        "is_anomaly": score > 30,
        "original": image_to_base64(image_np),
        "heatmap": heatmap_to_base64(heatmap),
        "overlay": image_to_base64(overlay),
    })
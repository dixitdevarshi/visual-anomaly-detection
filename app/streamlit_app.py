import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.model.extractor import DINOv2Extractor
from src.model.patchcore import PatchCore
from src.visualization.gradcam import patch_scores_to_heatmap, overlay_heatmap

# page config
st.set_page_config(
    page_title="Visual Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Visual Anomaly Detection")
st.markdown("Upload a product image to detect defects using DINOv2 + PatchCore")

# sidebar
st.sidebar.header("Settings")
category = st.sidebar.selectbox(
    "Product Category",
    ["bottle", "leather", "hazelnut", "cable", "capsule",
     "carpet", "grid", "metal_nut", "pill", "screw",
     "tile", "toothbrush", "transistor", "wood", "zipper"]
)
threshold = st.sidebar.slider(
    "Anomaly Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# load models
@st.cache_resource
def load_extractor():
    return DINOv2Extractor()

@st.cache_resource
def load_patchcore(category: str):
    extractor = load_extractor()
    patchcore = PatchCore(extractor=extractor)
    memory_bank_path = f"artifacts/{category}_memory_bank.pt"

    if not Path(memory_bank_path).exists():
        st.error(f"Memory bank not found for {category}. Run run_pipeline.py first.")
        st.stop()

    patchcore.load(memory_bank_path)
    return patchcore

# main interface
uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file:
    col1, col2, col3 = st.columns(3)

    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    image_np = np.array(image)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with st.spinner("Running anomaly detection..."):
        extractor = load_extractor()
        patchcore = load_patchcore(category)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tensor = transform(image).unsqueeze(0)
        image_score, patch_scores = patchcore.predict(tensor)

        score = image_score[0].item()
        patch_score = patch_scores[0]

        heatmap = patch_scores_to_heatmap(patch_score)
        overlaid = overlay_heatmap(image_np, heatmap)

    with col2:
        st.subheader("Anomaly Heatmap")
        st.image(heatmap, use_column_width=True, clamp=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlaid, use_column_width=True)

    st.divider()

    # result
    is_anomaly = score > threshold
    if is_anomaly:
        st.error(f"⚠️ Anomaly Detected | Score: {score:.4f}")
    else:
        st.success(f"✅ Normal | Score: {score:.4f}")

    st.metric("Anomaly Score", f"{score:.4f}")
    st.progress(min(score, 1.0))
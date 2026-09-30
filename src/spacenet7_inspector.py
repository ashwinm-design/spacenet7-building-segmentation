"""
SpaceNet 7 Cadastral Dataset Inspector — Streamlit Application.

Run with:
    streamlit run src/spacenet7_inspector.py
"""

import os
import sys
import random
import json
import numpy as np
import cv2
from PIL import Image

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed in the current environment.")
    print("Use `python src/visualize_spacenet7.py` for the standalone visualizer.")
    sys.exit(0)


st.set_page_config(
    page_title="SpaceNet 7 Cadastral Dataset Inspector",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ SpaceNet 7 Cadastral Dataset Inspector")
st.markdown("Visual verification of aerial imagery and rasterized building footprint masks for SIH 2026.")

DATASET_DIR = "data/processed_spacenet7"
OUTPUT_DIR = "results/visualizations/spacenet7"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_data
def load_splits():
    splits = {"train": [], "val": [], "test": []}
    for s in ["train", "val", "test"]:
        sp = os.path.join(DATASET_DIR, "splits", f"{s}.txt")
        if os.path.exists(sp):
            with open(sp) as f:
                lines = [l.strip().split(",") for l in f.readlines() if l.strip()]
            splits[s] = [{"img": l[0], "mask": l[1], "name": os.path.splitext(os.path.basename(l[0]))[0], "split": s} for l in lines]
    return splits


splits_data = load_splits()
all_samples = splits_data["train"] + splits_data["val"] + splits_data["test"]

if not all_samples:
    st.error("No processed SpaceNet 7 dataset found in data/processed_spacenet7. Run preparation first.")
    st.stop()

# Sidebar
st.sidebar.header("Controls & Filters")
selected_split = st.sidebar.selectbox("Filter Split", ["All", "Train", "Validation", "Test"])

if selected_split == "Train":
    current_pool = splits_data["train"]
elif selected_split == "Validation":
    current_pool = splits_data["val"]
elif selected_split == "Test":
    current_pool = splits_data["test"]
else:
    current_pool = all_samples

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

col_nav1, col_nav2, col_nav3 = st.sidebar.columns(3)
if col_nav1.button("← Prev"):
    st.session_state.sample_idx = (st.session_state.sample_idx - 1) % len(current_pool)
if col_nav2.button("🎲 Random"):
    st.session_state.sample_idx = random.randint(0, len(current_pool) - 1)
if col_nav3.button("Next →"):
    st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(current_pool)

sample_idx = min(st.session_state.sample_idx, len(current_pool) - 1)
current_sample = current_pool[sample_idx]

# Alpha slider
alpha = st.sidebar.slider("Overlay Mask Opacity", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

# Load current sample
img_bgr = cv2.imread(current_sample["img"])
mask_gray = cv2.imread(current_sample["mask"], cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Compute metrics
b_pixels = int(np.sum(mask_gray > 0))
total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
coverage_pct = round(b_pixels / total_pixels * 100, 2)
num_labels, _ = cv2.connectedComponents((mask_gray > 0).astype(np.uint8))
num_buildings = max(0, num_labels - 1)

# Dataset Information Card
st.sidebar.markdown("---")
st.sidebar.subheader("Sample Information")
st.sidebar.write(f"**Sample:** `{current_sample['name']}`")
st.sidebar.write(f"**Split:** `{current_sample['split'].upper()}`")
st.sidebar.write(f"**Dimensions:** `{img_rgb.shape[1]}x{img_rgb.shape[0]}`")
st.sidebar.write(f"**Building Pixels:** `{b_pixels:,}` px")
st.sidebar.write(f"**Coverage:** `{coverage_pct}%`")
st.sidebar.write(f"**Buildings Count:** `{num_buildings}`")

# Generate views
mask_rgb = np.zeros_like(img_rgb)
mask_rgb[mask_gray > 0] = [255, 255, 255]

color_fill = np.zeros_like(img_rgb)
color_fill[mask_gray > 0] = [255, 50, 50]
overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, color_fill, alpha, 0)
contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (40, 255, 120), 1, lineType=cv2.LINE_AA)

# Side-by-side view
st.subheader("Side-by-Side Comparison")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(img_rgb, caption="1. Original Aerial Image (256x256)", use_container_width=True)
with col2:
    st.image(mask_rgb, caption="2. Ground Truth Mask", use_container_width=True)
with col3:
    st.image(overlay, caption="3. Cadastral Overlay ⭐", use_container_width=True)

# Save visualization button
if st.sidebar.button("💾 Save Sample Visualization"):
    Image.fromarray(img_rgb).save(os.path.join(OUTPUT_DIR, f"{current_sample['name']}_original.png"))
    Image.fromarray(mask_rgb).save(os.path.join(OUTPUT_DIR, f"{current_sample['name']}_mask.png"))
    Image.fromarray(overlay).save(os.path.join(OUTPUT_DIR, f"{current_sample['name']}_overlay.png"))
    st.sidebar.success(f"Saved to {OUTPUT_DIR}/")

# Section: Multiple Random Samples
st.markdown("---")
st.subheader("Multiple Random Samples Gallery")
if st.button("🎲 Generate Random Samples (6 Examples)"):
    st.session_state.gallery_samples = [
        random.choice(splits_data["train"]),
        random.choice(splits_data["train"]),
        random.choice(splits_data["val"]),
        random.choice(splits_data["val"]),
        random.choice(splits_data["test"]),
        random.choice(splits_data["test"]),
    ]

if "gallery_samples" in st.session_state:
    cols = st.columns(3)
    for idx, s in enumerate(st.session_state.gallery_samples):
        with cols[idx % 3]:
            im = cv2.cvtColor(cv2.imread(s["img"]), cv2.COLOR_BGR2RGB)
            mk = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
            ov = im.copy()
            cf = np.zeros_like(im)
            cf[mk > 0] = [255, 50, 50]
            ov = cv2.addWeighted(ov, 0.65, cf, 0.35, 0)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (40, 255, 120), 1)
            st.image(ov, caption=f"[{s['split'].upper()}] {s['name']}", use_container_width=True)

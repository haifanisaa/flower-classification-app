# app.py
import os
import io
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

# =========================
# Config
# =========================
MODEL_PATH = "best_model_transfer.keras"   # change if the name is different
ENCODER_PATH = "encoder.pkl"               # change if the name/location is different
IMG_SIZE = 224                             # match with your model input size
TOP_K = 3                                  # how many top labels to display

st.set_page_config(
    page_title="Flower Classification App",
    page_icon="🌸",
    layout="centered",
)

# =========================
# Helpers
# =========================
@st.cache_resource
def load_keras_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at: {os.path.abspath(path)}\n"
            "Make sure the .keras file is in the same folder as app.py "
            "or update MODEL_PATH with the correct location."
        )
    return load_model(path)

@st.cache_resource
def load_encoder(path: str):
    if not os.path.exists(path):
        # fallback if encoder.pkl is missing
        st.warning("encoder.pkl not found. Using fallback class names ['class_0', 'class_1', ...].")
        return None
    try:
        enc = joblib.load(path)
        _ = enc.classes_  # LabelEncoder has attribute classes_
        return enc
    except Exception as e:
        st.warning(f"Failed to load encoder.pkl: {e}\nUsing fallback class names.")
        return None

def get_class_names(encoder, num_classes: int):
    if encoder is not None:
        return list(encoder.classes_)
    # generic fallback
    return [f"class_{i}" for i in range(num_classes)]

def preprocess_pil(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

def predict_topk(model, x: np.ndarray, k: int = TOP_K):
    probs = model.predict(x, verbose=0)[0]  # shape (C,)
    idxs = probs.argsort()[::-1][:k]
    return idxs, probs[idxs], probs  # top-k index, top-k probs, all probs

# =========================
# UI
# =========================
st.markdown(
    "<h1 style='text-align:center;'>Flower Classification App 🌸</h1>",
    unsafe_allow_html=True,
)
st.write("Upload a flower image and the model will predict its class")
st.divider()

# Sidebar info
with st.sidebar:
    st.header("Settings")
    st.write("⚙️ Model & Preprocess")
    st.code(f"MODEL_PATH = '{MODEL_PATH}'\nENCODER_PATH = '{ENCODER_PATH}'\nIMG_SIZE = {IMG_SIZE}\nTOP_K = {TOP_K}", language="python")
    st.info("Tip: use **use_container_width=True** to make images responsive.")

# Load model & encoder (cached)
try:
    model = load_keras_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# infer number of classes from output layer
try:
    num_classes = model.output_shape[-1]
except Exception:
    num_classes = 3  # safe default
encoder = load_encoder(ENCODER_PATH)
CLASS_NAMES = get_class_names(encoder, num_classes)

# =========================
# Uploader
# =========================
uploaded_files = st.file_uploader(
    "Choose a flower image...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        st.write("---")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            # show image
            img = Image.open(io.BytesIO(file.read()))
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Predicting..."):
                x = preprocess_pil(img, IMG_SIZE)
                top_idxs, top_probs, all_probs = predict_topk(model, x, TOP_K)

            # Main prediction result
            pred_idx = int(top_idxs[0])
            pred_name = CLASS_NAMES[pred_idx]
            conf = float(top_probs[0])

            st.subheader("Prediction")
            st.markdown(f"**{pred_name}**  \nConfidence: **{conf*100:.2f}%**")

            # Progress bar for confidence
            st.progress(min(max(conf, 0.0), 1.0))

            # Top-k table
            st.write("Top predictions:")
            for i, (idx, p) in enumerate(zip(top_idxs, top_probs), start=1):
                st.write(f"{i}. {CLASS_NAMES[int(idx)]} — {p*100:.2f}%")

            # (optional) show all probabilities in a bar chart
            with st.expander("Show all class probabilities"):
                probs_dict = {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))}

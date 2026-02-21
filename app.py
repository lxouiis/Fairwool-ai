import streamlit as st
import torch
import timm
from PIL import Image
import numpy as np
import joblib
from torchvision import transforms

st.set_page_config(page_title="FairWool AI", layout="centered")

CLASS_NAMES = ["A", "B", "C"]

@st.cache_resource
def load_classifier(ckpt_path: str):
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        map_location = "cpu"
    else:
        # Load to CPU first to avoid issues if GPU not available or different device
        map_location = "cpu"
        
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location)
    except FileNotFoundError:
        st.error(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
        return None, None, None

    model_name = ckpt["model_name"]
    model = timm.create_model(model_name, pretrained=False)
    
    # Check explicitly if we need to reset classifier based on the checkpoint logic used training
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=3)
    else:
        in_features = model.get_classifier().in_features
        model.set_classifier(torch.nn.Linear(in_features, 3))
        
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Use MPS if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    
    tfm = transforms.Compose([
        transforms.Resize(int(ckpt["image_size"] * 1.15)),
        transforms.CenterCrop(ckpt["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    return model, tfm, device

@st.cache_resource
def load_price_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

st.title("FairWool AI — Grading & Fair Price Prototype")

# Sidebar for configuration
st.sidebar.header("Configuration")
ckpt_path = st.sidebar.text_input("Classifier checkpoint path", value="fairwool_deit_small.pt")
price_model_path = st.sidebar.text_input("Price model path", value="price_reg.joblib")

uploaded = st.file_uploader("Upload a raw wool image", type=["jpg", "jpeg", "png"])

grade = None
conf = None

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)
    
    model, tfm, device = load_classifier(ckpt_path)
    
    if model:
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            idx = int(np.argmax(probs))
            grade = CLASS_NAMES[idx]
            conf = float(probs[idx])
        
        st.subheader("Predicted Grade")
        st.write(f"**Grade:** {grade}")
        st.write(f"**Confidence:** {conf:.2f}")
        
        # Display probability bars
        st.bar_chart({name: p for name, p in zip(CLASS_NAMES, probs)})

st.divider()
st.subheader("Fair Price Suggestion (Prototype)")

manual_grade = None
if grade is None:
    st.info("No image uploaded. You can manually select a grade for simulation.")
    manual_grade = st.selectbox("Select Grade Manually", ["A", "B", "C"])

col1, col2 = st.columns(2)
with col1:
    weight_kg = st.number_input("Weight (kg)", min_value=0.1, value=1.0, step=0.1)
    season = st.selectbox("Season", ["monsoon", "normal", "winter"])
with col2:
    distance_km = st.number_input("Distance to buyer/center (km)", min_value=0.0, value=10.0, step=1.0)
    contam = st.slider("Contamination (0 clean → 1 dirty)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

season_idx = {"monsoon": -0.05, "normal": 0.0, "winter": 0.10}[season]

if st.button("Compute Price"):
    target_grade = grade if grade else manual_grade
    
    if target_grade:
        grade_num = {"C": 1, "B": 2, "A": 3}[target_grade]
        
        reg = load_price_model(price_model_path)
        if reg:
            X = np.array([[grade_num, weight_kg, season_idx, distance_km, contam]], dtype=float)
            price_per_kg = float(reg.predict(X)[0])
            total = price_per_kg * weight_kg
            
            st.success(f"Suggested price for Grade {target_grade}: ₹{price_per_kg:.2f}/kg")
            st.metric("Estimated Total Value", f"₹{total:.2f}")
        else:
            st.error("Price model not found. Please run generate_price_data.py first.")
    else:
        st.warning("Please upload an image or select a grade.")

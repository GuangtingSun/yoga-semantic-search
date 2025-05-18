import streamlit as st
import faiss
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import os
import gdown

# ---- Load pre-trained ResNet50 model ----
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC layer
    model.eval()
    return model

model = load_model()

# ---- Image preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(img):
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(x).squeeze().numpy()
    return vec / np.linalg.norm(vec)

# ---- Load FAISS index and paths ----
@st.cache_resource
def load_index():
    faiss_path = "yoga_index.faiss"
    pkl_path = "yoga_img_paths.pkl"

    # Download FAISS index if it doesn't exist
    if not os.path.exists(faiss_path):
        st.write("⬇️ Downloading FAISS index from Google Drive...")
        file_id = "1S10nI3dEIi15tvsLTeQsaB2U5VBbTqiP"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", faiss_path, quiet=False)

    # Load index and image paths
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        img_paths = pickle.load(f)
    return index, img_paths

index, img_paths = load_index()

# ---- Streamlit UI ----
st.title("🧘 Yoga Pose Semantic Image Search")
uploaded_file = st.file_uploader("Upload a yoga pose image", type=["jpg", "png"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_column_width=True)

    # Extract features and query
    vec = extract_feature(query_img).astype("float32").reshape(1, -1)
    D, I = index.search(vec, k=5)

    st.subheader("🔍 Top 5 Most Similar Poses:")
    for idx in I[0]:
        st.image(img_paths[idx], width=200)

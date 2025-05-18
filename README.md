# 🧘 Yoga Pose Semantic Image Search

This Streamlit app allows you to upload a yoga pose image and returns the 5 most visually similar poses from a training dataset.

### 🔍 Features
- Uses a pre-trained ResNet50 model to extract image embeddings
- FAISS for fast similarity search
- Downloads the large FAISS index file dynamically from Google Drive
- Streamlit Cloud deployable

---

### 📁 Files in this Repository

| File | Description |
|------|-------------|
| `streamlit_app.py` | The main app script |
| `yoga_img_paths.pkl` | Pickle file with image paths corresponding to embeddings |
| `requirements.txt` | Required Python libraries |

---

### 📦 Setup Instructions (Local)

1. Clone this repo  
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app.py
```

---

### 🌐 Deploy to Streamlit Cloud

This app is fully compatible with [Streamlit Cloud](https://streamlit.io/cloud).  
The large `yoga_index.faiss` file is automatically downloaded at runtime from Google Drive using `gdown`.

---

### 🧠 Acknowledgements

- ResNet50 from PyTorch torchvision models
- FAISS similarity search
- Streamlit for UI

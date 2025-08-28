# 🌸 Flower Classification with Transfer Learning

This repo contains:
- **baseline.ipynb** → baseline implementation (adapted from [AmanxAI Blog](https://amanxai.com/2020/11/24/flower-recognition-with-python/))  
- **model modified.ipynb** → modified version with my own improvements and experiments  
- **app.py** → Streamlit web app for testing the trained model  

## ⚙️ Features (App)
- Upload one or multiple images (`.jpg`, `.jpeg`, `.png`).
- Model outputs **Top-1 prediction** with confidence and progress bar.
- Shows **Top-k predictions** (default = 3).
- Option to expand and see **all class probabilities**.
- Uses `encoder.pkl` for class names (fallback available if missing).

## 🛠️ How to Run the App
1. Install dependencies:
   ```bash
   pip install streamlit tensorflow keras pillow numpy joblib

2. Make sure these files are in the repo:
- app.py
- best_model_transfer.keras (your trained model)
- encoder.pkl (class label encoder)

3. Run the app:
   ```bash
   streamlit run app.py

📖 Reference
Flower Recognition with Python – AmanxAI Blog (https://amanxai.com/2020/11/24/flower-recognition-with-python/)

Author: Haifa Nisa Anwari


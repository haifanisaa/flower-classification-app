# 🌸 Flower Classification App

This project classifies flowers using **Convolutional Neural Networks (CNNs)**.  
It contains two models:
- **baseline.ipynb** → simple CNN (reference model).  
- **model_modified.ipynb** → improved version with data augmentation and deeper architecture.

---

## 📂 Dataset
The dataset is not stored in this repo (too large).  
👉 Download here: [Dataset.zip](https://drive.google.com/file/d/1RPBLceezEUPWwn1B4e2BnPj5bQbxKRT8/view?usp=sharing)  
After download, extract it into a folder named `dataset/`.

---

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python seaborn
2. Open either notebook in Jupyter/Colab:
   baseline.ipynb
   model_modified.ipynb

   📊 Notes
   Baseline = adapted from tutorial.
   Modified = my own improvement (better accuracy + evaluation).

📖 Reference
[Flower Recognition with Python – AmanxAI Blog](https://amanxai.com/2020/11/24/flower-recognition-with-python/#google_vignette)

👩‍💻 Author: Haifa Nisa Anwari

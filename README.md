# 🍅 Tomato Leaf Disease Classification using MobileNetV2 and Grad-CAM

A web-based application for tomato leaf disease classification using the MobileNetV2 deep learning model with Explainable Artificial Intelligence (XAI) through Grad-CAM visualization.

This project was developed as an undergraduate thesis to classify tomato leaf diseases from image input and provide visual explanations of the model's prediction using Grad-CAM.

---

## 📌 Features

- Upload tomato leaf images
- Automatic disease classification using MobileNetV2
- Prediction confidence score
- Confidence level interpretation
- Probability bar chart for all classes
- Feature Map visualization
- Grad-CAM heatmap visualization
- Overlay visualization between original image and Grad-CAM
- Simple and interactive Streamlit dashboard

---

## 🍅 Disease Classes

The model classifies images into the following 10 classes:

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus

---

## 🧠 Deep Learning Model

- Model : MobileNetV2
- Framework : TensorFlow / Keras
- Transfer Learning
- Fine-Tuning
- Explainable AI : Grad-CAM

---

## 🖥️ Dashboard

The application was developed using Streamlit to provide an interactive interface for disease classification.

Main functionalities:

- Upload leaf image
- Automatic preprocessing
- Disease prediction
- Confidence score visualization
- Grad-CAM explanation

---

## 📂 Project Structure

```
tomato-leaf-disease-classification/
│
├── app.py
├── requirements.txt
├── MobileNetV2.keras
├── assets/
├── images/
├── README.md
```

---

## ⚙️ Installation

Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/tomato-leaf-disease-classification.git
```

Move into project directory

```bash
cd tomato-leaf-disease-classification
```

Install required libraries

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📊 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

## 🎓 Research Information

**Title**

Tomato Leaf Disease Classification using MobileNetV2 with Explainable Artificial Intelligence (Grad-CAM)

**Methodology**

CRISP-DM

---

## 👩‍💻 Author

**Azzahra Dania Indriyani**

Bachelor of Information Systems

Gunadarma University

---

## 📄 License

This project was developed for academic and research purposes.

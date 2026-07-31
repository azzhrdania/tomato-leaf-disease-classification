# 🍅 Tomato Leaf Disease Classification Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV2-green)](https://keras.io/api/applications/mobilenet/)
[![XAI](https://img.shields.io/badge/XAI-Grad--CAM-purple)](https://keras.io/examples/vision/grad_cam/)

> **End-to-End Deep Learning Project** untuk mengklasifikasikan penyakit pada daun tomat menggunakan model **MobileNetV2** serta memberikan interpretasi hasil prediksi menggunakan **Explainable Artificial Intelligence (XAI) dengan Grad-CAM**.

### 🔗 Project Links

- 🌐 **Live Dashboard:** [Tomato Leaf Disease Classification](https://YOUR-APP-NAME.streamlit.app/)
- 📦 **GitHub Repository:** [Tomato Leaf Disease Classification](https://github.com/YOUR_USERNAME/tomato-leaf-disease-classification)

---

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk:

- Mengembangkan model deep learning untuk melakukan klasifikasi penyakit pada citra daun tomat
- Mengklasifikasikan citra daun tomat ke dalam beberapa kelas penyakit dan kondisi sehat
- Memanfaatkan **Transfer Learning** menggunakan arsitektur **MobileNetV2**
- Meningkatkan kemampuan model melalui proses **Fine-Tuning**
- Memberikan interpretasi terhadap hasil prediksi model menggunakan **Grad-CAM**
- Menyediakan aplikasi berbasis web yang memungkinkan pengguna melakukan klasifikasi citra daun tomat secara interaktif
- Mengimplementasikan model yang telah dilatih ke dalam dashboard menggunakan **Streamlit**

---

## 🍅 Kelas Klasifikasi

Model digunakan untuk mengklasifikasikan citra daun tomat ke dalam **10 kelas**, yaitu:

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

---

## 👥 Target Pengguna Dashboard

Dashboard ini ditujukan untuk:

- **Petani / pengguna tanaman tomat**
- **Mahasiswa dan akademisi**
- **Peneliti di bidang pertanian dan machine learning**
- **Pengguna yang membutuhkan identifikasi awal penyakit daun tomat**

Dashboard ini dikembangkan sebagai **alat bantu klasifikasi berbasis citra**, bukan sebagai pengganti diagnosis ahli di bidang pertanian.

---

## 🧠 Cara Kerja Sistem (High-Level)

1. Pengguna mengunggah citra daun tomat melalui dashboard
2. Citra diproses sesuai dengan preprocessing yang digunakan pada model
3. Citra diberikan kepada model **MobileNetV2** untuk proses klasifikasi
4. Model menghasilkan probabilitas untuk masing-masing kelas
5. Kelas dengan nilai probabilitas tertinggi ditentukan sebagai hasil prediksi
6. Dashboard menampilkan:
   - Hasil klasifikasi
   - Confidence score
   - Distribusi probabilitas setiap kelas
   - Feature Map
   - Visualisasi Grad-CAM
   - Overlay Grad-CAM pada citra asli
7. Visualisasi Grad-CAM digunakan untuk membantu memahami area citra yang menjadi perhatian model dalam menghasilkan prediksi

---

## 🧠 Model yang Digunakan

Model klasifikasi dibangun menggunakan **MobileNetV2** dengan pendekatan **Transfer Learning**.

Tahapan pengembangan model meliputi:

- Preprocessing citra
- Pembagian dataset
- Pembentukan arsitektur MobileNetV2
- Transfer Learning
- Compile model
- Training model
- Penerapan Class Weight
- Fine-Tuning
- Evaluasi model
- Implementasi Explainable AI menggunakan Grad-CAM

### Konfigurasi Model

| Komponen | Konfigurasi |
|---|---|
| Arsitektur | MobileNetV2 |
| Pendekatan | Transfer Learning |
| Fine-Tuning | 50 layer terakhir |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Learning Rate Fine-Tuning | 3 × 10⁻⁶ |
| Output | 10 kelas |
| Explainability | Grad-CAM |

---

## 🔍 Explainable Artificial Intelligence (Grad-CAM)

Untuk memberikan interpretasi terhadap hasil klasifikasi, sistem menggunakan metode **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM menghasilkan heatmap yang menunjukkan area pada citra yang memiliki kontribusi terhadap prediksi model.

Pada dashboard, hasil Grad-CAM ditampilkan dalam bentuk:

- Feature Map
- Grad-CAM Heatmap
- Overlay antara Grad-CAM dengan citra asli

Visualisasi tersebut digunakan untuk membantu pengguna memahami bagian daun yang menjadi perhatian model ketika menentukan kelas penyakit.

---

## 📊 Evaluasi Model

Evaluasi model dilakukan untuk mengetahui kemampuan model dalam melakukan klasifikasi penyakit daun tomat.

Metode evaluasi yang digunakan meliputi:

- **Accuracy**
- **Loss**
- **Confusion Matrix**
- **Classification Report**
- **Precision**
- **Recall**
- **F1-Score**

Classification Report digunakan untuk mengetahui performa model pada masing-masing kelas, sedangkan Confusion Matrix digunakan untuk melihat distribusi prediksi benar dan kesalahan klasifikasi antar kelas.

---

## 📈 Hasil Pelatihan Model

Model melalui dua tahap utama dalam proses pembelajaran:

### Training

Pada tahap awal, backbone MobileNetV2 digunakan dengan sebagian besar layer dalam kondisi tidak dapat dilatih (*frozen*).

### Fine-Tuning

Pada tahap berikutnya, **50 layer terakhir MobileNetV2** dibuka untuk proses fine-tuning menggunakan learning rate yang lebih kecil agar model dapat menyesuaikan representasi fitur dengan karakteristik citra daun tomat.

Hasil proses fine-tuning menunjukkan peningkatan performa validasi selama proses pelatihan.

---

## 🖥️ Dashboard Streamlit

Aplikasi dikembangkan sebagai **single-page dashboard** menggunakan Streamlit.

Fitur utama dashboard:

### 📤 Upload Citra

Pengguna dapat mengunggah citra daun tomat melalui interface dashboard.

### 🔮 Prediksi

Model memberikan hasil klasifikasi berdasarkan citra yang diunggah.

### 📊 Confidence Score

Dashboard menampilkan nilai confidence dari hasil prediksi serta distribusi probabilitas untuk masing-masing kelas.

### 🔍 Explainable AI

Dashboard menyediakan visualisasi Feature Map dan Grad-CAM untuk membantu menginterpretasikan hasil prediksi model.

---

## 🚀 Deployment

Aplikasi di-deploy menggunakan **Streamlit Community Cloud** dengan source code yang dikelola melalui **GitHub**.

Arsitektur deployment:

```text
GitHub Repository
       │
       ▼
   Streamlit Cloud
       │
       ▼
   app.py
       │
       ▼
 MobileNetV2 Model
       │
       ▼
 Prediction + Grad-CAM
       │
       ▼
   Dashboard

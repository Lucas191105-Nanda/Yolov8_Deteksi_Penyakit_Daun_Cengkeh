# YOLOv8 Deteksi Penyakit Daun Cengkeh

Sistem deteksi otomatis untuk mengidentifikasi penyakit pada daun cengkeh menggunakan algoritma YOLOv8.

## 📋 Deskripsi Project

Project ini mengimplementasikan model deep learning YOLOv8 untuk mendeteksi berbagai jenis penyakit yang menyerang daun tanaman cengkeh. Sistem ini dapat membantu petani dan peneliti untuk melakukan deteksi dini penyakit tanaman secara cepat dan akurat.

## 🎯 Fitur Utama

- Deteksi real-time penyakit daun cengkeh
- Klasifikasi multi-class untuk berbagai jenis penyakit
- Akurasi tinggi dengan model YOLOv8
- Mudah digunakan dan di-deploy

## 📊 Dataset

Dataset yang digunakan berasal dari Roboflow Universe:
- **Sumber**: [Dataset Anotasi Gambar Cengkeh](https://universe.roboflow.com/dataset-cengkeh/anotasi-gambar/dataset)
- Dataset sudah dianotasi dan siap digunakan untuk training

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **YOLOv8** (Ultralytics)
- **OpenCV** untuk image processing
- **PyTorch** sebagai framework deep learning
- **Roboflow** untuk manajemen dataset

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Lucas191105-Nanda/Yolov8_Deteksi_Penyakit_Daun_Cengkeh.git
cd Yolov8_Deteksi_Penyakit_Daun_Cengkeh
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

Gunakan requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download dataset secara manual dari Roboflow:

🔗 **Link Dataset**: https://universe.roboflow.com/dataset-cengkeh/anotasi-gambar/dataset

**Langkah-langkah:**
1. Buka link dataset di atas
2. Pilih format **YOLOv8**
3. Klik tombol **Download**
4. Extract file zip yang sudah didownload
5. Pindahkan folder dataset ke dalam folder `datasets/`

### 5. Struktur Folder

Pastikan struktur folder seperti ini:

```
Yolov8_Deteksi_Penyakit_Daun_Cengkeh/
│
├── datasets/
│   └── anotasi-gambar/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── models/
│   └── best.pt  # Model hasil training
│
├── train.py
├── predict.py
└── README.md
```

## 🚀 Cara Menggunakan

### Training Model

```python
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO('yolov8n.pt')  # atau yolov8s.pt, yolov8m.pt

# Training
results = model.train(
    data='datasets/anotasi-gambar/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Prediksi/Deteksi

```python
from ultralytics import YOLO

# Load model yang sudah di-train
model = YOLO('models/best.pt')

# Prediksi pada gambar
results = model.predict(
    source='path/to/image.jpg',
    save=True,
    conf=0.5
)
```

### Validasi Model

```python
# Evaluasi performa model
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 📈 Hasil Model

- **Precision**: (sesuaikan dengan hasil training Anda)
- **Recall**: (sesuaikan dengan hasil training Anda)
- **mAP50**: (sesuaikan dengan hasil training Anda)
- **mAP50-95**: (sesuaikan dengan hasil training Anda)

## 💡 Tips

- Gunakan GPU untuk training lebih cepat
- Sesuaikan hyperparameter (epochs, batch size) sesuai spesifikasi komputer
- Lakukan augmentasi data untuk meningkatkan performa model
- Gunakan model YOLOv8n untuk training cepat, YOLOv8m/l untuk akurasi lebih tinggi

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan buat pull request atau buka issue untuk saran dan perbaikan.

## 📝 Lisensi

[Sesuaikan dengan lisensi project Anda]

## 📧 Kontak

- **Developer**: Lucas Nanda
- **GitHub**: [@Lucas191105-Nanda](https://github.com/Lucas191105-Nanda)

## 🙏 Acknowledgments

- Dataset dari Roboflow Universe
- YOLOv8 by Ultralytics
- Komunitas open-source yang mendukung project ini

---

⭐ Jangan lupa untuk memberikan star jika project ini membantu Anda!

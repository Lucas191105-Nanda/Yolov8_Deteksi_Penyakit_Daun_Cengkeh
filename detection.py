"""
YOLOv8 Inference Script - Uji Coba Model Daun Cengkeh
Untuk dijalankan di Visual Studio Code setelah training

Cara Setup:
1. Download file best.pt dari Google Colab
2. Letakkan di folder project Anda
3. Install ultralytics: pip install ultralytics opencv-python
4. Jalankan script ini
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

# ========== KONFIGURASI ==========
MODEL_PATH = "best.pt"  # Path ke model Anda
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence untuk deteksi
CLASS_NAMES = ['bukan-daun-penyakit-cengkeh', 'cacar', 'sehat']


# ========== FUNGSI INFERENCE ==========

def predict_single_image(model, image_path, conf=0.5, save=True):
    """
    Prediksi pada 1 gambar
    
    Args:
        model: YOLO model
        image_path: Path ke gambar
        conf: Confidence threshold
        save: Simpan hasil atau tidak
    """
    print(f"\n🔍 Memprediksi: {image_path}")
    
    # Prediksi
    results = model.predict(image_path, conf=conf, save=save)
    
    # Tampilkan hasil
    for r in results:
        boxes = r.boxes
        
        if len(boxes) == 0:
            print("  ❌ Tidak ada deteksi")
            continue
        
        print(f"  ✓ Ditemukan {len(boxes)} deteksi:")
        
        for idx, box in enumerate(boxes):
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            label = CLASS_NAMES[cls]
            
            # Koordinat bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"    {idx+1}. {label} - {confidence:.2%} confidence")
            print(f"       Koordinat: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
    
    return results


def predict_multiple_images(model, folder_path, conf=0.5, save_dir="results"):
    """
    Prediksi pada banyak gambar dalam folder
    
    Args:
        model: YOLO model
        folder_path: Path ke folder berisi gambar
        conf: Confidence threshold
        save_dir: Folder untuk menyimpan hasil
    """
    print(f"\n📁 Memproses folder: {folder_path}")
    
    # Ekstensi gambar yang didukung
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # Ambil semua file gambar
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print("  ❌ Tidak ada gambar ditemukan!")
        return
    
    print(f"  📊 Total gambar: {len(image_files)}")
    
    # Buat folder hasil
    os.makedirs(save_dir, exist_ok=True)
    
    # Prediksi semua gambar
    results = model.predict(image_files, conf=conf, save=True, project=save_dir)
    
    # Ringkasan hasil
    print(f"\n📊 Ringkasan Hasil:")
    detection_count = {'sehat': 0, 'cacat': 0, 'bukan_daun': 0}
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = CLASS_NAMES[cls]
            detection_count[label] += 1
    
    print(f"  🌿 Sehat: {detection_count['sehat']}")
    print(f"  ⚠️  Cacat: {detection_count['cacat']}")
    print(f"  ❌ Bukan Daun: {detection_count['bukan_daun']}")
    print(f"\n  💾 Hasil disimpan di: {save_dir}/")
    
    return results


def predict_webcam(model, conf=0.5):
    """
    Prediksi realtime dari webcam
    
    Args:
        model: YOLO model
        conf: Confidence threshold
    """
    print("\n📹 Membuka webcam...")
    print("Tekan 'q' untuk keluar")
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Tidak bisa membuka webcam!")
        return
    
    print("✓ Webcam terbuka!")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error membaca frame")
            break
        
        # Prediksi
        results = model.predict(frame, conf=conf, verbose=False)
        
        # Gambar hasil pada frame
        annotated_frame = results[0].plot()
        
        # Tampilkan FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tampilkan frame
        cv2.imshow('YOLOv8 - Deteksi Daun Cengkeh', annotated_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam ditutup")


def predict_video(model, video_path, conf=0.5, save_path="output_video.mp4"):
    """
    Prediksi pada video
    
    Args:
        model: YOLO model
        video_path: Path ke video
        conf: Confidence threshold
        save_path: Path untuk menyimpan video hasil
    """
    print(f"\n🎥 Memproses video: {video_path}")
    
    # Prediksi
    results = model.predict(video_path, conf=conf, save=True)
    
    print(f"✓ Video hasil disimpan di: runs/detect/predict/")
    
    return results


def get_detailed_predictions(model, image_path, conf=0.5):
    """
    Mendapatkan prediksi detail dalam format dictionary
    
    Args:
        model: YOLO model
        image_path: Path ke gambar
        conf: Confidence threshold
    
    Returns:
        List of dictionaries dengan detail prediksi
    """
    results = model.predict(image_path, conf=conf, verbose=False)
    
    predictions = []
    
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            pred_dict = {
                'class_id': cls,
                'class_name': CLASS_NAMES[cls],
                'confidence': confidence,
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                }
            }
            
            predictions.append(pred_dict)
    
    return predictions


def visualize_with_custom_labels(model, image_path, conf=0.5, save_path="result.jpg"):
    """
    Visualisasi dengan custom label dan warna
    
    Args:
        model: YOLO model
        image_path: Path ke gambar
        conf: Confidence threshold
        save_path: Path untuk menyimpan hasil
    """
    # Baca gambar
    img = cv2.imread(image_path)
    
    # Prediksi
    results = model.predict(image_path, conf=conf, verbose=False)
    
    # Warna untuk setiap kelas (BGR format)
    colors = {
        'sehat': (0, 255, 0),      # Hijau
        'cacat': (0, 165, 255),    # Orange
        'bukan_daun': (0, 0, 255)  # Merah
    }
    
    # Gambar bounding box
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            label = CLASS_NAMES[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Warna
            color = colors[label]
            
            # Gambar rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label text
            text = f"{label}: {confidence:.2%}"
            
            # Background untuk text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Text
            cv2.putText(img, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Simpan hasil
    cv2.imwrite(save_path, img)
    print(f"✓ Hasil disimpan di: {save_path}")
    
    return img


# ========== MAIN PROGRAM ==========

def main():
    print("=" * 70)
    print("🌿 YOLOv8 Inference - Deteksi Daun Cengkeh")
    print("=" * 70)
    
    # Check apakah model ada
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model tidak ditemukan di: {MODEL_PATH}")
        print("Pastikan file best.pt ada di folder yang sama dengan script ini.")
        return
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Model berhasil di-load!")
    
    # Menu
    while True:
        print("\n" + "=" * 70)
        print("PILIH MODE INFERENSI:")
        print("=" * 70)
        print("1. Prediksi pada 1 gambar")
        print("2. Prediksi pada banyak gambar (folder)")
        print("3. Prediksi realtime dari webcam")
        print("4. Prediksi pada video")
        print("5. Prediksi dengan detail (JSON format)")
        print("6. Prediksi dengan custom visualization")
        print("0. Keluar")
        print("=" * 70)
        
        choice = input("\nPilih (0-6): ").strip()
        
        if choice == "1":
            image_path = input("Masukkan path gambar: ").strip()
            if os.path.exists(image_path):
                predict_single_image(model, image_path, conf=CONFIDENCE_THRESHOLD)
            else:
                print(f"❌ File tidak ditemukan: {image_path}")
        
        elif choice == "2":
            folder_path = input("Masukkan path folder: ").strip()
            if os.path.exists(folder_path):
                predict_multiple_images(model, folder_path, conf=CONFIDENCE_THRESHOLD)
            else:
                print(f"❌ Folder tidak ditemukan: {folder_path}")
        
        elif choice == "3":
            predict_webcam(model, conf=CONFIDENCE_THRESHOLD)
        
        elif choice == "4":
            video_path = input("Masukkan path video: ").strip()
            if os.path.exists(video_path):
                predict_video(model, video_path, conf=CONFIDENCE_THRESHOLD)
            else:
                print(f"❌ File tidak ditemukan: {video_path}")
        
        elif choice == "5":
            image_path = input("Masukkan path gambar: ").strip()
            if os.path.exists(image_path):
                predictions = get_detailed_predictions(model, image_path, conf=CONFIDENCE_THRESHOLD)
                print("\n📊 Hasil Prediksi (JSON):")
                import json
                print(json.dumps(predictions, indent=2))
            else:
                print(f"❌ File tidak ditemukan: {image_path}")
        
        elif choice == "6":
            image_path = input("Masukkan path gambar: ").strip()
            save_path = input("Simpan hasil di (default: result.jpg): ").strip() or "result.jpg"
            if os.path.exists(image_path):
                visualize_with_custom_labels(model, image_path, conf=CONFIDENCE_THRESHOLD, save_path=save_path)
            else:
                print(f"❌ File tidak ditemukan: {image_path}")
        
        elif choice == "0":
            print("\n👋 Terima kasih telah menggunakan YOLOv8 Inference!")
            break
        
        else:
            print("❌ Pilihan tidak valid!")


if __name__ == "__main__":
    # Install dependencies jika belum
    # pip install ultralytics opencv-python
    
    main()


# ========== CONTOH PENGGUNAAN SINGKAT ==========
"""
# 1. PREDIKSI CEPAT (1 BARIS)
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('gambar.jpg', conf=0.5, save=True)


# 2. PREDIKSI & AMBIL HASIL
results = model.predict('gambar.jpg')
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {cls}, Confidence: {conf:.2f}")


# 3. PREDIKSI WEBCAM (1 BARIS)
model.predict(source=0, show=True)


# 4. PREDIKSI VIDEO
model.predict('video.mp4', save=True)


# 5. PREDIKSI BATCH
model.predict('folder_gambar/', save=True)
"""
# ============================================================
# YOLOv8 Training Script for Daun Cengkeh (Windows Safe)
# ============================================================

import os
import yaml
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FUNGSI TRAINING & VALIDASI
# ============================================================

def train_yolo(dataset_path):

    # ---------- SETUP ----------
    print("=" * 60)
    print("🚀 Setup YOLOv8 Training Environment")
    print("=" * 60)

    print(f"\n✓ PyTorch version : {torch.__version__}")
    print(f"✓ CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU detected    : {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU tidak terdeteksi — pakai CPU (lebih lambat)")

    # ---------- CEK STRUKTUR ----------
    def check_dataset(path):
        needed = [
            "train/images",
            "train/labels",
            "val/images",
            "val/labels"
        ]
        for d in needed:
            full = os.path.join(path, d)
            if not os.path.exists(full):
                print(f"❌ Missing folder: {full}")
                return False
            print(f"✓ {d} OK ({len(os.listdir(full))} files)")
        return True

    if not check_dataset(dataset_path):
        print("\n❌ Dataset tidak valid.")
        return

    # ---------- BUAT data.yaml ----------
    classes = ['sehat', 'cacat', 'bukan_daun_cengkeh']

    yaml_file = os.path.join(dataset_path, "data.yaml")
    yaml_content = {
        "path": dataset_path,
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": classes
    }

    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    print(f"\n✓ data.yaml created at: {yaml_file}")

    # ---------- TRAIN ----------
    print("\n" + "=" * 60)
    print("🏋️ TRAINING DIMULAI")
    print("=" * 60)

    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_file,
        epochs=100,
        batch=16,
        imgsz=640,
        project="runs_daun",
        name="exp",
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        save=True
    )

    print("\n✓ Training selesai!")

    best_model = "runs_daun/exp/weights/best.pt"
    model = YOLO(best_model)

    # ---------- VALIDASI ----------
    print("\n" + "=" * 60)
    print("📊 VALIDASI")
    print("=" * 60)

    metrics = model.val()

    print(f"- mAP50     : {metrics.box.map50:.4f}")
    print(f"- mAP50-95  : {metrics.box.map:.4f}")
    print(f"- Precision : {metrics.box.mp:.4f}")
    print(f"- Recall    : {metrics.box.mr:.4f}")

    # ============================================================
    # CONFUSION MATRIX + CLASSIFICATION REPORT
    # ============================================================

    print("\n📌 Generating confusion matrix & classification report...")

    # ambil prediksi & label
    preds = []
    trues = []

    for r in metrics.results:
        preds.append(r.boxes.cls.cpu().numpy().astype(int))
        trues.append(r.boxes.data[:, 5].cpu().numpy().astype(int))

    preds = [p[0] for p in preds]
    trues = [t[0] for t in trues]

    # --- Classification Report ---
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(trues, preds, target_names=classes))

    # --- Confusion Matrix ---
    cm = confusion_matrix(trues, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✓ confusion_matrix.png saved.")

    print("\nSelesai semua!")


# ============================================================
# MAIN PROGRAM UNTUK WINDOWS (WAJIB)
# ============================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # WAJIB

    print("\nMasukkan path dataset lokal:")
    print("Contoh:  C:/Nanda Workspace/dataset_daun_cengkeh")

    dataset_path = input("\nPath dataset: ").strip()
    dataset_path = os.path.normpath(dataset_path)

    if not os.path.exists(dataset_path):
        print("❌ ERROR: Path dataset tidak ditemukan!")
        exit()

    train_yolo(dataset_path)

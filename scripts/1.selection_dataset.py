import os
import cv2

# Lokasi folder dataset
dataset_path = r"D:\SELEKSI DATA"

jumlah_valid = 0
jumlah_rusak = 0

for root, dirs, files in os.walk(dataset_path):

    # Hanya memeriksa folder images
    if os.path.basename(root).lower() != "images":
        continue

    print(f"\nMemeriksa folder: {root}")

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            file_path = os.path.join(root, file)

            image = cv2.imread(file_path)

            if image is None:
                jumlah_rusak += 1
                print(f"[RUSAK] {file_path}")
            else:
                jumlah_valid += 1

print("\n===== HASIL VALIDASI DATASET =====")
print(f"Jumlah citra valid : {jumlah_valid}")
print(f"Jumlah citra rusak : {jumlah_rusak}")
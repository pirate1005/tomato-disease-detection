import os
from PIL import Image

dataset_path = r"D:\SELEKSI DATA"

jumlah_konversi = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.lower().endswith(".jpeg"):

            file_path = os.path.join(root, file)

            # Membuka gambar
            img = Image.open(file_path).convert("RGB")

            # Nama file baru
            new_file = os.path.splitext(file_path)[0] + ".jpg"

            # Simpan sebagai JPG
            img.save(new_file, "JPEG")

            # Hapus file lama (.jpeg)
            os.remove(file_path)

            jumlah_konversi += 1
            print(f"Berhasil dikonversi : {new_file}")

print(f"\nTotal file yang dikonversi : {jumlah_konversi}")
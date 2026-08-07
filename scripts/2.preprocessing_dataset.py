import os

# Lokasi folder dataset
dataset_path = r"D:\SELEKSI DATA"

jumlah_jpg = 0
jumlah_format_lain = 0

for root, dirs, files in os.walk(dataset_path):

    for file in files:
        ext = os.path.splitext(file)[1].lower()

        if ext == ".jpg":
            jumlah_jpg += 1
        else:
            jumlah_format_lain += 1
            print(f"Format berbeda : {os.path.join(root, file)}")

print("\n===== HASIL PEMERIKSAAN FORMAT FILE =====")
print(f"Jumlah file JPG : {jumlah_jpg}")
print(f"Jumlah file selain JPG : {jumlah_format_lain}")

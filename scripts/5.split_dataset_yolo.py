from pathlib import Path
import random
import shutil

# ==========================================================
# KONFIGURASI
# ==========================================================

SOURCE_ROOT = Path(r"D:\Dataset_Tomat_Balanced")
OUTPUT_ROOT = Path(r"D:\Dataset_Tomat_YOLO")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

random.seed(42)

# ==========================================================
# URUTAN CLASS ID YOLO
# ==========================================================
# Penting:
# Urutan ini harus sama dengan CLASS_ID_MAP pada balance_dataset_tomat.py
# dan harus sama dengan data.yaml saat training YOLO.

CLASS_NAMES = {
    "bacterial_spot_leaf": 0,
    "early_blight_leaf": 1,
    "healthy_leaf": 2,
    "late_blight_leaf": 3,
    "mosaic_virus": 4,
    "septoria_leaf_spot": 5,
    "bacterial_spot_fruit": 6,
    "blossom_end_rot": 7,
    "catface": 8,
    "healthy_fruit": 9,
    "serangan_hama": 10
}


# ==========================================================
# FUNGSI BANTUAN
# ==========================================================

def get_image_files(images_dir):
    image_files = []

    if not images_dir.exists():
        return image_files

    for file_path in images_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)

    return sorted(image_files)


def validate_ratios():
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Total rasio harus 1.0. "
            f"Saat ini: TRAIN + VAL + TEST = {total_ratio}"
        )


def read_yolo_label(source_label_path):
    """
    Membaca label YOLO:
    class_id x_center y_center width height

    Fungsi ini hanya mengambil koordinat bounding box.
    class_id akan ditulis ulang berdasarkan nama folder kelas.
    """
    labels = []

    with open(source_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            return None

        try:
            old_class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            return None

        # Validasi koordinat YOLO
        if not (0 <= x_center <= 1):
            return None
        if not (0 <= y_center <= 1):
            return None
        if not (0 < width <= 1):
            return None
        if not (0 < height <= 1):
            return None

        labels.append([old_class_id, x_center, y_center, width, height])

    if len(labels) == 0:
        return None

    return labels


def rewrite_label_class_id(source_label_path, output_label_path, new_class_id):
    """
    Menulis ulang label YOLO agar class_id sesuai urutan CLASS_NAMES.

    Format YOLO:
    class_id x_center y_center width height
    """
    labels = read_yolo_label(source_label_path)

    if labels is None:
        return False

    new_lines = []

    for label in labels:
        old_class_id, x_center, y_center, width, height = label

        new_line = (
            f"{new_class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        new_lines.append(new_line)

    with open(output_label_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    return True


def prepare_output_folders():
    """
    Membuat folder output YOLO.
    Struktur akhir:
    OUTPUT_ROOT/images/train
    OUTPUT_ROOT/images/val
    OUTPUT_ROOT/images/test
    OUTPUT_ROOT/labels/train
    OUTPUT_ROOT/labels/val
    OUTPUT_ROOT/labels/test
    """
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_pairs():
    """
    Mengumpulkan semua pasangan gambar-label dari dataset balanced.

    Struktur sumber:
    D:\Dataset_Tomat_Balanced
    ├── Penyakit Buah Tomat
    │   ├── bacterial_spot_fruit
    │   │   ├── images
    │   │   └── labels
    │   └── ...
    └── Penyakit Daun Tomat
        ├── bacterial_spot_leaf
        │   ├── images
        │   └── labels
        └── ...
    """
    all_pairs = []
    skipped_unknown_class = []
    missing_labels = []
    invalid_labels = []

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {SOURCE_ROOT}")

    for group_dir in SOURCE_ROOT.iterdir():
        if not group_dir.is_dir():
            continue

        for class_dir in group_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name not in CLASS_NAMES:
                skipped_unknown_class.append(class_name)
                continue

            class_id = CLASS_NAMES[class_name]

            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            if not images_dir.exists():
                print(f"Peringatan: folder images tidak ditemukan: {images_dir}")
                continue

            if not labels_dir.exists():
                print(f"Peringatan: folder labels tidak ditemukan: {labels_dir}")
                continue

            image_files = get_image_files(images_dir)

            for image_path in image_files:
                label_path = labels_dir / f"{image_path.stem}.txt"

                if not label_path.exists():
                    missing_labels.append(str(image_path))
                    continue

                labels = read_yolo_label(label_path)

                if labels is None:
                    invalid_labels.append(str(label_path))
                    continue

                all_pairs.append({
                    "image_path": image_path,
                    "label_path": label_path,
                    "class_name": class_name,
                    "class_id": class_id
                })

    if skipped_unknown_class:
        print("\nPeringatan: kelas berikut tidak dikenali dan dilewati:")
        for class_name in sorted(set(skipped_unknown_class)):
            print(f"- {class_name}")

    if missing_labels:
        print(f"\nPeringatan: {len(missing_labels)} gambar tidak memiliki label.")

    if invalid_labels:
        print(f"\nPeringatan: {len(invalid_labels)} file label tidak valid dan dilewati.")

    return all_pairs


def split_data(pairs):
    """
    Membagi data menjadi train, val, dan test.
    Pembagian dilakukan per kelas agar distribusi tetap seimbang.
    """
    grouped = {}

    for item in pairs:
        class_name = item["class_name"]

        if class_name not in grouped:
            grouped[class_name] = []

        grouped[class_name].append(item)

    split_result = {
        "train": [],
        "val": [],
        "test": []
    }

    print("\nREKAP SPLIT PER KELAS")
    print("-" * 70)

    for class_name in sorted(grouped.keys(), key=lambda name: CLASS_NAMES[name]):
        items = grouped[class_name]
        random.shuffle(items)

        total = len(items)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        test_count = total - train_count - val_count

        train_items = items[:train_count]
        val_items = items[train_count:train_count + val_count]
        test_items = items[train_count + val_count:]

        split_result["train"].extend(train_items)
        split_result["val"].extend(val_items)
        split_result["test"].extend(test_items)

        print(
            f"{CLASS_NAMES[class_name]:>2} - {class_name:<25} "
            f"total={total:<5} "
            f"train={len(train_items):<5} "
            f"val={len(val_items):<5} "
            f"test={len(test_items):<5}"
        )

    return split_result


def copy_split_files(split_result):
    """
    Menyalin gambar dan label ke folder YOLO akhir.
    Label ditulis ulang agar class_id sesuai CLASS_NAMES.
    """
    for split_name, items in split_result.items():
        print(f"\nMenyalin data {split_name}: {len(items)} item")

        for index, item in enumerate(items):
            image_path = item["image_path"]
            label_path = item["label_path"]
            class_name = item["class_name"]
            class_id = item["class_id"]

            # Nama file dibuat unik dan tetap menunjukkan kelas.
            new_stem = f"{class_name}_{index:05d}"

            new_image_path = (
                OUTPUT_ROOT /
                "images" /
                split_name /
                f"{new_stem}{image_path.suffix.lower()}"
            )

            new_label_path = (
                OUTPUT_ROOT /
                "labels" /
                split_name /
                f"{new_stem}.txt"
            )

            shutil.copy2(image_path, new_image_path)

            success = rewrite_label_class_id(
                source_label_path=label_path,
                output_label_path=new_label_path,
                new_class_id=class_id
            )

            if not success:
                print(f"Peringatan: gagal menulis label untuk {label_path}")


def create_data_yaml():
    """
    Membuat file data.yaml untuk YOLO.
    """
    yaml_path = OUTPUT_ROOT / "data.yaml"

    class_list = [None] * len(CLASS_NAMES)

    for class_name, class_id in CLASS_NAMES.items():
        class_list[class_id] = class_name

    if any(name is None for name in class_list):
        raise ValueError("Ada class_id yang kosong pada CLASS_NAMES.")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUTPUT_ROOT.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write("names:\n")

        for index, class_name in enumerate(class_list):
            f.write(f"  {index}: {class_name}\n")

    print(f"\ndata.yaml dibuat di: {yaml_path}")


def count_label_class_ids(label_dir):
    """
    Menghitung jumlah class_id pada folder label tertentu.
    """
    class_counter = {}

    if not label_dir.exists():
        return class_counter

    for label_path in label_dir.glob("*.txt"):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))
                class_counter[class_id] = class_counter.get(class_id, 0) + 1

    return class_counter


def show_final_count():
    """
    Menampilkan jumlah akhir file.
    """
    print("\n" + "=" * 70)
    print("REKAP DATASET YOLO AKHIR")
    print("=" * 70)

    total_images_all = 0
    total_labels_all = 0

    for split in ["train", "val", "test"]:
        image_count = len(get_image_files(OUTPUT_ROOT / "images" / split))
        label_count = len(list((OUTPUT_ROOT / "labels" / split).glob("*.txt")))

        total_images_all += image_count
        total_labels_all += label_count

        print(f"{split}: {image_count} gambar, {label_count} label")

    print("-" * 70)
    print(f"Total gambar: {total_images_all}")
    print(f"Total label : {total_labels_all}")

    print("\nDistribusi class_id pada setiap split:")

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")

        class_counter = count_label_class_ids(OUTPUT_ROOT / "labels" / split)

        for class_name, class_id in sorted(CLASS_NAMES.items(), key=lambda x: x[1]):
            count = class_counter.get(class_id, 0)
            print(f"{class_id:>2} - {class_name:<25}: {count}")


def check_expected_classes(pairs):
    """
    Mengecek apakah semua kelas yang diharapkan ada di dataset.
    """
    found_classes = sorted(set(item["class_name"] for item in pairs))
    expected_classes = sorted(CLASS_NAMES.keys())

    print("\nKELAS YANG DITEMUKAN:")
    for class_name in found_classes:
        print(f"- {class_name}")

    missing_classes = set(expected_classes) - set(found_classes)

    if missing_classes:
        print("\nPeringatan: kelas berikut tidak ditemukan:")
        for class_name in sorted(missing_classes):
            print(f"- {class_name}")
    else:
        print("\nSemua kelas pada CLASS_NAMES ditemukan.")


# ==========================================================
# PROGRAM UTAMA
# ==========================================================

def main():
    print("=" * 70)
    print("SPLIT DATASET TOMAT KE FORMAT YOLO")
    print("=" * 70)

    validate_ratios()

    print(f"Folder sumber : {SOURCE_ROOT}")
    print(f"Folder output : {OUTPUT_ROOT}")

    print("\nMenyiapkan folder output YOLO...")
    prepare_output_folders()

    print("\nMengumpulkan pasangan gambar-label...")
    pairs = collect_pairs()

    print(f"\nTotal pasangan gambar-label valid ditemukan: {len(pairs)}")

    if len(pairs) == 0:
        print("Tidak ada data yang bisa diproses.")
        return

    check_expected_classes(pairs)

    print("\nMembagi dataset menjadi train, val, dan test...")
    split_result = split_data(pairs)

    print("\nMenyalin file ke format YOLO...")
    copy_split_files(split_result)

    print("\nMembuat file data.yaml...")
    create_data_yaml()

    show_final_count()

    print("\nSelesai.")
    print(f"Dataset YOLO siap digunakan di: {OUTPUT_ROOT}")
    print(f"File konfigurasi YOLO: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
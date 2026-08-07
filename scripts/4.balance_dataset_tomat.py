from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

# ==========================================================
# KONFIGURASI FOLDER DATASET
# ==========================================================

SOURCE_ROOT = Path(r"D:\DATASET FINAL")

LEAF_ROOT = SOURCE_ROOT / "Penyakit Daun Tomat"
FRUIT_ROOT = SOURCE_ROOT / "Penyakit Buah Tomat"

OUTPUT_ROOT = Path(r"D:\Dataset_Tomat_Balanced")

# Jika None, target otomatis mengikuti kelas terbanyak.
# Berdasarkan data terbaru, target otomatisnya adalah 420.
TARGET_PER_CLASS = None

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

random.seed(42)

# ==========================================================
# URUTAN CLASS ID
# ==========================================================
# Penting:
# Urutan ini harus sama dengan data.yaml pada proses training YOLO.

CLASS_ID_MAP = {
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
    "serangan_hama": 10,
}

# Jika True, class_id pada file label akan disesuaikan otomatis
# berdasarkan nama folder kelas.
FORCE_CLASS_ID_FROM_FOLDER = True


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


def get_label_path(labels_dir, image_path):
    label_path = labels_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        return label_path

    return None


def read_yolo_label(label_path, forced_class_id=None):
    """
    Membaca label YOLO format:
    class_id x_center y_center width height

    Jika forced_class_id diberikan, class_id lama diganti
    dengan class_id berdasarkan nama folder kelas.
    """
    labels = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

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

        # Validasi nilai koordinat YOLO
        if not (0 <= x_center <= 1):
            return None
        if not (0 <= y_center <= 1):
            return None
        if not (0 < width <= 1):
            return None
        if not (0 < height <= 1):
            return None

        if forced_class_id is not None:
            class_id = forced_class_id
        else:
            class_id = old_class_id

        labels.append([class_id, x_center, y_center, width, height])

    if len(labels) == 0:
        return None

    return labels


def write_yolo_label(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            f.write(
                f"{int(class_id)} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


def horizontal_flip_labels(labels):
    flipped_labels = []

    for label in labels:
        class_id, x_center, y_center, width, height = label
        new_x_center = 1.0 - x_center
        flipped_labels.append([class_id, new_x_center, y_center, width, height])

    return flipped_labels


def vertical_flip_labels(labels):
    flipped_labels = []

    for label in labels:
        class_id, x_center, y_center, width, height = label
        new_y_center = 1.0 - y_center
        flipped_labels.append([class_id, x_center, new_y_center, width, height])

    return flipped_labels


def get_class_id_from_folder(class_dir):
    class_name = class_dir.name

    if class_name not in CLASS_ID_MAP:
        raise ValueError(
            f"Nama kelas '{class_name}' belum ada di CLASS_ID_MAP. "
            f"Tambahkan dulu class_id untuk kelas tersebut."
        )

    return CLASS_ID_MAP[class_name]


def get_image_label_pairs(class_dir):
    """
    Mengambil pasangan gambar dan label dari satu folder kelas.

    Struktur yang digunakan:
    class_dir/images
    class_dir/labels
    """
    images_dir = class_dir / "images"
    labels_dir = class_dir / "labels"

    forced_class_id = None

    if FORCE_CLASS_ID_FROM_FOLDER:
        forced_class_id = get_class_id_from_folder(class_dir)

    image_files = get_image_files(images_dir)

    pairs = []
    missing_labels = []
    invalid_labels = []

    for image_path in image_files:
        label_path = get_label_path(labels_dir, image_path)

        if label_path is None:
            missing_labels.append(image_path.name)
            continue

        labels = read_yolo_label(label_path, forced_class_id=forced_class_id)

        if labels is None:
            invalid_labels.append(label_path.name)
            continue

        pairs.append((image_path, label_path, labels))

    if missing_labels:
        print(
            f"Peringatan: {len(missing_labels)} gambar tidak memiliki label "
            f"di kelas {class_dir.name}"
        )

    if invalid_labels:
        print(
            f"Peringatan: {len(invalid_labels)} label tidak valid "
            f"di kelas {class_dir.name}"
        )

    return pairs


def augment_image_and_label(image_path, labels):
    image = Image.open(image_path).convert("RGB")

    augment_type = random.choice([
        "hflip",
        "vflip",
        "brightness",
        "contrast",
        "sharpness",
        "color",
        "blur"
    ])

    augmented_labels = [label.copy() for label in labels]

    if augment_type == "hflip":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_labels = horizontal_flip_labels(augmented_labels)

    elif augment_type == "vflip":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        augmented_labels = vertical_flip_labels(augmented_labels)

    elif augment_type == "brightness":
        factor = random.uniform(0.85, 1.15)
        image = ImageEnhance.Brightness(image).enhance(factor)

    elif augment_type == "contrast":
        factor = random.uniform(0.85, 1.15)
        image = ImageEnhance.Contrast(image).enhance(factor)

    elif augment_type == "sharpness":
        factor = random.uniform(0.85, 1.25)
        image = ImageEnhance.Sharpness(image).enhance(factor)

    elif augment_type == "color":
        factor = random.uniform(0.85, 1.15)
        image = ImageEnhance.Color(image).enhance(factor)

    elif augment_type == "blur":
        radius = random.uniform(0.2, 0.6)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    return image, augmented_labels, augment_type


def copy_original_pairs(pairs, out_images_dir, out_labels_dir):
    """
    Menyalin gambar asli dan menulis ulang label YOLO
    agar class_id sesuai CLASS_ID_MAP.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path, labels in pairs:
        new_image_path = out_images_dir / image_path.name
        new_label_path = out_labels_dir / label_path.name

        shutil.copy2(image_path, new_image_path)
        write_yolo_label(new_label_path, labels)


def balance_one_class(class_dir, out_class_dir, target_count):
    """
    Membalancing satu kelas penyakit.
    """
    images_dir = out_class_dir / "images"
    labels_dir = out_class_dir / "labels"

    pairs = get_image_label_pairs(class_dir)

    print("\n" + "-" * 60)
    print(f"Kelas: {class_dir.name}")
    print(f"Class ID: {CLASS_ID_MAP.get(class_dir.name, 'BELUM TERDAFTAR')}")
    print(f"Jumlah pasangan gambar-label valid: {len(pairs)}")

    if len(pairs) == 0:
        print("Dilewati karena tidak ada pasangan gambar dan label valid.")
        return

    copy_original_pairs(pairs, images_dir, labels_dir)

    current_count = len(get_image_files(images_dir))

    print(f"Jumlah awal disalin: {current_count}")
    print(f"Target balancing: {target_count}")

    if current_count >= target_count:
        print("Jumlah sudah mencapai/melebihi target. Tidak dilakukan augmentasi.")
        return

    augment_number = 1

    while current_count < target_count:
        selected_image, selected_label, selected_labels = random.choice(pairs)

        augmented_image, augmented_labels, augment_type = augment_image_and_label(
            selected_image,
            selected_labels
        )

        new_stem = f"aug_{augment_number:04d}_{augment_type}_{selected_image.stem}"

        new_image_path = images_dir / f"{new_stem}.jpg"
        new_label_path = labels_dir / f"{new_stem}.txt"

        augmented_image.save(new_image_path, quality=95)
        write_yolo_label(new_label_path, augmented_labels)

        current_count += 1
        augment_number += 1

    print(f"Jumlah akhir: {current_count}")


def collect_all_class_dirs():
    """
    Mengumpulkan folder kelas dari penyakit daun dan buah.
    """
    class_dirs = []

    for root in [LEAF_ROOT, FRUIT_ROOT]:
        if root.exists():
            for class_dir in root.iterdir():
                if class_dir.is_dir():
                    class_dirs.append(class_dir)
        else:
            print(f"Folder tidak ditemukan: {root}")

    # Urutkan berdasarkan CLASS_ID_MAP agar konsisten
    class_dirs = sorted(
        class_dirs,
        key=lambda x: CLASS_ID_MAP.get(x.name, 999)
    )

    return class_dirs


def determine_target_count(class_dirs):
    """
    Menentukan target balancing otomatis berdasarkan kelas terbanyak.
    """
    max_count = 0

    print("\nMengecek jumlah data awal setiap kelas:")

    for class_dir in class_dirs:
        pairs = get_image_label_pairs(class_dir)
        count = len(pairs)
        print(f"{class_dir.name}: {count}")

        if count > max_count:
            max_count = count

    return max_count


def show_final_result():
    """
    Menampilkan rekap hasil balancing.
    """
    print("\n" + "=" * 60)
    print("REKAP HASIL BALANCING")
    print("=" * 60)

    total_images_all = 0
    total_labels_all = 0

    for group_dir in OUTPUT_ROOT.iterdir():
        if group_dir.is_dir():
            print(f"\n{group_dir.name}")

            for class_dir in sorted(group_dir.iterdir()):
                if class_dir.is_dir():
                    images_dir = class_dir / "images"
                    labels_dir = class_dir / "labels"

                    total_images = len(get_image_files(images_dir))
                    total_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

                    total_images_all += total_images
                    total_labels_all += total_labels

                    print(f"{class_dir.name}: {total_images} gambar, {total_labels} label")

    print("\n" + "-" * 60)
    print(f"Total semua gambar: {total_images_all}")
    print(f"Total semua label : {total_labels_all}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("BALANCING DATASET TOMAT YOLO")
    print("=" * 60)

    print(f"Folder sumber daun : {LEAF_ROOT}")
    print(f"Folder sumber buah : {FRUIT_ROOT}")
    print(f"Folder output      : {OUTPUT_ROOT}")

    class_dirs = collect_all_class_dirs()

    if len(class_dirs) == 0:
        print("Tidak ada folder kelas yang ditemukan.")
        return

    print("\nDaftar kelas ditemukan:")
    for class_dir in class_dirs:
        print(f"{CLASS_ID_MAP.get(class_dir.name, 'NA')} - {class_dir.name}")

    if TARGET_PER_CLASS is None:
        target_count = determine_target_count(class_dirs)
    else:
        target_count = TARGET_PER_CLASS

    print("\n" + "=" * 60)
    print(f"TARGET BALANCING SETIAP KELAS: {target_count}")
    print("=" * 60)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        group_name = class_dir.parent.name
        out_class_dir = OUTPUT_ROOT / group_name / class_dir.name

        balance_one_class(
            class_dir=class_dir,
            out_class_dir=out_class_dir,
            target_count=target_count
        )

    show_final_result()

    print("\nSelesai.")
    print(f"Dataset hasil balancing tersimpan di: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
USE deteksi_tomat_db;

INSERT INTO hasil_deteksi
(
    gambar,
    label,
    confidence,
    iou,
    precision_score,
    recall_score,
    map50,
    map5095,
    gejala
)
VALUES
(
    'contoh_early_blight_leaf.jpg',
    'early_blight_leaf',
    97.82,
    95.74,
    98.41,
    97.36,
    98.52,
    97.11,
    'Daun menunjukkan bercak coklat berbentuk melingkar (konsentris) yang menyebar pada permukaan daun. Jika tidak ditangani, bercak akan membesar dan menyebabkan daun mengering.'
),
(
    'contoh_blossom_end_rot.jpg',
    'blossom_end_rot',
    95.67,
    96.18,
    99.12,
    98.54,
    99.35,
    98.71,
    'Buah tomat mengalami busuk pada bagian ujung bawah (blossom end). Gejala ditandai bercak coklat hingga hitam yang cekung akibat kekurangan kalsium.'
);
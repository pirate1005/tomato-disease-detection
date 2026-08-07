from __future__ import annotations

import html
import tempfile
import textwrap
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2
import pandas as pd
import pymysql
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# ==========================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Deteksi Penyakit Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# PATH PROJECT
#
# DATA SISTEM/
# ├── models/
# │   └── best.pt
# ├── result/
# │   └── cek_target_90_persen_perkelas_test.csv
# ├── src/
# │   ├── app.py
# │   └── static/
# │       └── style.css
# └── requirements.txt
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
STYLE_PATH = BASE_DIR / "static" / "style.css"

METRIC_CANDIDATES = [
    PROJECT_DIR
    / "result"
    / "cek_target_90_persen_perkelas_test.csv",

    PROJECT_DIR
    / "results"
    / "cek_target_90_persen_perkelas_test.csv",
]

METRIC_FILE = next(
    (
        path
        for path in METRIC_CANDIDATES
        if path.exists()
    ),
    METRIC_CANDIDATES[0],
)


# ==========================================================
# KONFIGURASI KAMERA DAN INFERENSI
# ==========================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

CONF_THRESHOLD = 0.90
IMGSZ = 640
FRAME_SKIP = 3
JPEG_QUALITY = 80


# ==========================================================
# CLASS NAMES 11 KELAS
# ==========================================================
CLASS_NAMES = {
    0: "bacterial_spot_leaf",
    1: "early_blight_leaf",
    2: "healthy_leaf",
    3: "late_blight_leaf",
    4: "mosaic_virus",
    5: "septoria_leaf_spot",
    6: "bacterial_spot_fruit",
    7: "blossom_end_rot",
    8: "catface",
    9: "healthy_fruit",
    10: "serangan_hama",
}


# ==========================================================
# DESKRIPSI GEJALA
# ==========================================================
GEJALA = {
    "bacterial_spot_leaf": (
        "Gejala berupa bercak kecil berwarna coklat hingga "
        "kehitaman pada permukaan daun. Bercak dapat menyebar "
        "dan menyebabkan jaringan daun mengering."
    ),

    "early_blight_leaf": (
        "Gejala berupa bercak coklat pada daun yang sering "
        "membentuk pola melingkar atau konsentris. Daun dapat "
        "menguning dan mengering apabila serangan semakin parah."
    ),

    "healthy_leaf": (
        "Daun tampak sehat, berwarna hijau normal, dan tidak "
        "menunjukkan bercak atau kerusakan visual."
    ),

    "late_blight_leaf": (
        "Gejala berupa bercak gelap atau area nekrosis pada daun. "
        "Serangan dapat menyebar cepat dan membuat jaringan daun "
        "tampak membusuk atau menghitam."
    ),

    "mosaic_virus": (
        "Gejala berupa perubahan warna daun seperti pola mosaik, "
        "belang hijau muda dan hijau tua, serta dapat disertai "
        "perubahan bentuk daun."
    ),

    "septoria_leaf_spot": (
        "Gejala berupa bercak kecil pada daun dengan bagian tengah "
        "yang lebih terang dan tepi lebih gelap. Bercak biasanya "
        "muncul dalam jumlah banyak."
    ),

    "bacterial_spot_fruit": (
        "Gejala berupa bercak kecil berwarna gelap pada permukaan "
        "buah tomat. Bercak dapat tampak kasar dan menyebar pada "
        "kulit buah."
    ),

    "blossom_end_rot": (
        "Gejala berupa area busuk berwarna coklat hingga hitam "
        "pada bagian ujung bawah buah tomat."
    ),

    "catface": (
        "Gejala berupa bentuk buah yang tidak normal, berlekuk, "
        "retak, atau mengalami deformasi pada permukaan buah."
    ),

    "healthy_fruit": (
        "Buah tampak sehat, bentuk normal, warna merata, dan tidak "
        "menunjukkan bercak, busuk, atau deformasi."
    ),

    "serangan_hama": (
        "Gejala berupa kerusakan permukaan akibat gigitan, lubang, "
        "bekas serangan, atau kerusakan jaringan oleh organisme "
        "pengganggu."
    ),
}


# ==========================================================
# STATE DETEKSI KOSONG
# ==========================================================
def empty_detection_state(
    message: str = "Tidak terdeteksi",
) -> dict[str, Any]:
    return {
        "label": message,
        "class_id": "-",
        "confidence": 0.0,
        "iou": "-",
        "precision": "-",
        "recall": "-",
        "map50": "-",
        "map5095": "-",
        "gejala": "-",
    }


# ==========================================================
# NORMALISASI NILAI METRIK
# ==========================================================
def normalize_metric_value(
    value: Any,
) -> float | str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "-"

    if numeric_value <= 1:
        numeric_value *= 100

    return round(numeric_value, 2)


def display_metric(
    value: Any,
) -> str:
    if value in (
        None,
        "",
        "-",
    ):
        return "-"

    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def nullable_metric(
    value: Any,
) -> float | None:
    if value in (
        None,
        "",
        "-",
    ):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ==========================================================
# LOAD FILE METRIK
# ==========================================================
@st.cache_data(show_spinner=False)
def load_metrics(
    metric_path: str,
) -> dict[str, dict[str, float | str]]:
    path = Path(metric_path)

    metrics_lookup: dict[
        str,
        dict[str, float | str],
    ] = {}

    if not path.exists():
        return metrics_lookup

    dataframe = pd.read_csv(path)

    required_columns = [
        "class_name",
        "IoU",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom file metrik tidak lengkap. "
            "Kolom yang tidak ditemukan: "
            + ", ".join(missing_columns)
        )

    for _, row in dataframe.iterrows():
        class_label = str(
            row["class_name"]
        ).strip()

        metrics_lookup[class_label] = {
            "iou": normalize_metric_value(
                row["IoU"]
            ),

            "precision": normalize_metric_value(
                row["Precision"]
            ),

            "recall": normalize_metric_value(
                row["Recall"]
            ),

            "map50": normalize_metric_value(
                row["mAP50"]
            ),

            "map5095": normalize_metric_value(
                row["mAP50-95"]
            ),
        }

    return metrics_lookup


# ==========================================================
# LOAD MODEL YOLO
# Model disimpan dalam cache agar tidak dimuat berulang.
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(
    model_path: str,
):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}"
        )

    loaded_model = YOLO(str(path))

    if torch.cuda.is_available():
        selected_device: int | str = 0
    else:
        selected_device = "cpu"

    try:
        loaded_model.fuse()
    except Exception:
        pass

    inference_lock = threading.Lock()

    return (
        loaded_model,
        selected_device,
        inference_lock,
    )


# ==========================================================
# MEMBENTUK DATA DETEKSI
# ==========================================================
def update_detection_from_box(
    best_box: Any,

    metrics_lookup: dict[
        str,
        dict[str, float | str],
    ],
) -> dict[str, Any]:
    class_id = int(
        best_box.cls[0]
    )

    confidence = float(
        best_box.conf[0]
    )

    label = CLASS_NAMES.get(
        class_id,
        "unknown",
    )

    detection: dict[str, Any] = {
        "label": label,

        "class_id": class_id,

        "confidence": round(
            confidence * 100,
            2,
        ),

        "iou": "-",
        "precision": "-",
        "recall": "-",
        "map50": "-",
        "map5095": "-",

        "gejala": GEJALA.get(
            label,
            "-",
        ),
    }

    if label in metrics_lookup:
        detection.update(
            metrics_lookup[label]
        )

    return detection


# ==========================================================
# PENYIMPAN STATE DETEKSI
#
# streamlit-webrtc menjalankan callback pada thread terpisah,
# sehingga state harus dilindungi menggunakan Lock.
# ==========================================================
class DetectionStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.frame_counter = 0

        self.current_detection = (
            empty_detection_state(
                "Menunggu deteksi..."
            )
        )

        self.last_annotated_frame = None
        self.last_jpeg: bytes | None = None

    def increment_frame(
        self,
    ) -> int:
        with self.lock:
            self.frame_counter += 1

            return self.frame_counter

    def get_last_annotated_frame(
        self,
    ):
        with self.lock:
            if self.last_annotated_frame is None:
                return None

            return self.last_annotated_frame.copy()

    def set_result(
        self,

        detection: dict[str, Any],

        annotated_frame,

        jpeg_bytes: bytes | None,
    ) -> None:
        with self.lock:
            self.current_detection = (
                detection.copy()
            )

            if annotated_frame is None:
                self.last_annotated_frame = None
            else:
                self.last_annotated_frame = (
                    annotated_frame.copy()
                )

            self.last_jpeg = jpeg_bytes

    def snapshot(
        self,
    ) -> tuple[
        dict[str, Any],
        bytes | None,
    ]:
        with self.lock:
            detection = (
                self.current_detection.copy()
            )

            jpeg_bytes = self.last_jpeg

        return (
            detection,
            jpeg_bytes,
        )


# ==========================================================
# KONFIGURASI DATABASE AIVEN MYSQL
#
# Konfigurasi diambil dari Streamlit Secrets:
#
# [mysql]
# host = "..."
# port = 12345
# user = "avnadmin"
# password = "..."
# database = "defaultdb"
#
# ca_cert = """
# -----BEGIN CERTIFICATE-----
# ...
# -----END CERTIFICATE-----
# """
# ==========================================================
def get_database_config() -> dict[str, Any]:
    if "mysql" not in st.secrets:
        raise RuntimeError(
            "Konfigurasi [mysql] belum tersedia "
            "di Streamlit Secrets."
        )

    mysql_secret = st.secrets["mysql"]

    required_keys = [
        "host",
        "port",
        "user",
        "password",
        "database",
    ]

    missing_keys = [
        key
        for key in required_keys
        if not mysql_secret.get(key)
    ]

    if missing_keys:
        raise RuntimeError(
            "Konfigurasi database belum lengkap: "
            + ", ".join(missing_keys)
        )

    config: dict[str, Any] = {
        "host": str(
            mysql_secret["host"]
        ),

        "port": int(
            mysql_secret["port"]
        ),

        "user": str(
            mysql_secret["user"]
        ),

        "password": str(
            mysql_secret["password"]
        ),

        "database": str(
            mysql_secret["database"]
        ),

        "cursorclass": (
            pymysql.cursors.DictCursor
        ),

        "autocommit": True,
        "charset": "utf8mb4",

        "connect_timeout": 15,
        "read_timeout": 30,
        "write_timeout": 30,
    }

    ca_certificate = (
        mysql_secret.get("ca_cert")
        or mysql_secret.get(
            "ca_certificate"
        )
    )

    if ca_certificate:
        ca_path = (
            Path(tempfile.gettempdir())
            / "aiven-mysql-ca.pem"
        )

        clean_certificate = (
            textwrap.dedent(
                str(ca_certificate)
            ).strip()
            + "\n"
        )

        ca_path.write_text(
            clean_certificate,
            encoding="utf-8",
        )

        config["ssl"] = {
            "ca": str(ca_path),
            "check_hostname": True,
        }

    else:
        # Tetap menggunakan koneksi SSL.
        # Disarankan tetap mengisi ca_cert dari Aiven.
        config["ssl"] = {
            "check_hostname": False,
        }

    return config


def open_database_connection():
    return pymysql.connect(
        **get_database_config()
    )


# ==========================================================
# MEMBUAT / MENYESUAIKAN TABEL HASIL DETEKSI
#
# Jika tabel sudah ada, data lama tidak dihapus.
# Kolom yang belum tersedia akan ditambahkan.
# ==========================================================
def ensure_detection_table(
    connection,
) -> None:
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS hasil_deteksi (
            id BIGINT UNSIGNED
                NOT NULL
                AUTO_INCREMENT,

            gambar VARCHAR(255)
                NULL,

            gambar_blob LONGBLOB
                NULL,

            label VARCHAR(100)
                NOT NULL,

            confidence DECIMAL(8,2)
                NULL,

            iou DECIMAL(8,2)
                NULL,

            precision_score DECIMAL(8,2)
                NULL,

            recall_score DECIMAL(8,2)
                NULL,

            map50 DECIMAL(8,2)
                NULL,

            map5095 DECIMAL(8,2)
                NULL,

            gejala TEXT
                NULL,

            created_at TIMESTAMP
                NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (id)
        )
        ENGINE=InnoDB
        DEFAULT CHARSET=utf8mb4;
    """

    required_columns = {
        "gambar": (
            "VARCHAR(255) NULL"
        ),

        "gambar_blob": (
            "LONGBLOB NULL"
        ),

        "label": (
            "VARCHAR(100) NULL"
        ),

        "confidence": (
            "DECIMAL(8,2) NULL"
        ),

        "iou": (
            "DECIMAL(8,2) NULL"
        ),

        "precision_score": (
            "DECIMAL(8,2) NULL"
        ),

        "recall_score": (
            "DECIMAL(8,2) NULL"
        ),

        "map50": (
            "DECIMAL(8,2) NULL"
        ),

        "map5095": (
            "DECIMAL(8,2) NULL"
        ),

        "gejala": (
            "TEXT NULL"
        ),

        "created_at": (
            "TIMESTAMP NOT NULL "
            "DEFAULT CURRENT_TIMESTAMP"
        ),
    }

    with connection.cursor() as cursor:
        cursor.execute(
            create_table_sql
        )

        cursor.execute(
            "SHOW COLUMNS "
            "FROM hasil_deteksi"
        )

        existing_columns = {
            str(row["Field"])
            for row in cursor.fetchall()
        }

        for (
            column_name,
            column_definition,
        ) in required_columns.items():

            if (
                column_name
                not in existing_columns
            ):
                alter_query = (
                    "ALTER TABLE "
                    "hasil_deteksi "
                    f"ADD COLUMN "
                    f"`{column_name}` "
                    f"{column_definition}"
                )

                cursor.execute(
                    alter_query
                )


# ==========================================================
# INISIALISASI DATABASE
# ==========================================================
@st.cache_resource(show_spinner=False)
def initialize_database() -> bool:
    connection = None

    try:
        connection = (
            open_database_connection()
        )

        ensure_detection_table(
            connection
        )

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 AS connection_test"
            )

            cursor.fetchone()

        return True

    finally:
        if connection is not None:
            connection.close()


# ==========================================================
# SIMPAN HASIL DETEKSI KE AIVEN MYSQL
#
# Gambar disimpan ke kolom gambar_blob.
# Tidak bergantung pada folder uploads lokal.
# ==========================================================
def save_detection_to_database(
    detection: dict[str, Any],

    jpeg_bytes: bytes | None,
) -> tuple[bool, str]:
    invalid_labels = {
        "Menunggu deteksi...",
        "Tidak terdeteksi",
        "Error prediksi",
        "Kamera tidak terbaca",
    }

    if (
        detection.get("label")
        in invalid_labels
    ):
        return (
            False,
            "Belum ada hasil deteksi "
            "yang dapat disimpan.",
        )

    if not jpeg_bytes:
        return (
            False,
            "Gambar hasil deteksi "
            "belum tersedia.",
        )

    connection = None

    try:
        connection = (
            open_database_connection()
        )

        ensure_detection_table(
            connection
        )

        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".jpg"
        )

        insert_sql = """
            INSERT INTO hasil_deteksi (
                gambar,
                gambar_blob,
                label,
                confidence,
                iou,
                precision_score,
                recall_score,
                map50,
                map5095,
                gejala
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """

        with connection.cursor() as cursor:
            cursor.execute(
                insert_sql,

                (
                    filename,

                    jpeg_bytes,

                    detection["label"],

                    nullable_metric(
                        detection["confidence"]
                    ),

                    nullable_metric(
                        detection["iou"]
                    ),

                    nullable_metric(
                        detection["precision"]
                    ),

                    nullable_metric(
                        detection["recall"]
                    ),

                    nullable_metric(
                        detection["map50"]
                    ),

                    nullable_metric(
                        detection["map5095"]
                    ),

                    detection["gejala"],
                ),
            )

        return (
            True,
            "Hasil deteksi berhasil "
            "disimpan ke database.",
        )

    except Exception as error:
        return (
            False,
            "Gagal menyimpan ke database: "
            f"{error}",
        )

    finally:
        if connection is not None:
            connection.close()


# ==========================================================
# CALLBACK VIDEO REALTIME
#
# Kamera berasal dari browser pengguna.
# Model diproses setiap FRAME_SKIP frame.
# ==========================================================
def build_video_callback(
    store: DetectionStore,

    model,

    device: int | str,

    inference_lock: threading.Lock,

    metrics_lookup: dict[
        str,
        dict[str, float | str],
    ],
):
    def video_frame_callback(
        frame: av.VideoFrame,
    ) -> av.VideoFrame:
        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.resize(
            image,

            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),
        )

        frame_number = (
            store.increment_frame()
        )

        previous_annotated = (
            store.get_last_annotated_frame()
        )

        if previous_annotated is None:
            display_frame = image.copy()
        else:
            display_frame = (
                previous_annotated
            )

        if (
            frame_number
            % FRAME_SKIP
            == 0
        ):
            try:
                with inference_lock:
                    results = model.predict(
                        source=image,
                        imgsz=IMGSZ,
                        conf=CONF_THRESHOLD,
                        device=device,
                        verbose=False,
                    )

                result = results[0]
                boxes = result.boxes

                if (
                    boxes is not None
                    and len(boxes) > 0
                ):
                    best_box = max(
                        boxes,

                        key=lambda box: float(
                            box.conf[0]
                        ),
                    )

                    best_confidence = float(
                        best_box.conf[0]
                    )

                    if (
                        best_confidence
                        >= CONF_THRESHOLD
                    ):
                        detection = (
                            update_detection_from_box(
                                best_box,
                                metrics_lookup,
                            )
                        )

                        annotated_frame = (
                            result.plot(
                                line_width=2,
                                font_size=0.6,
                            )
                        )

                        (
                            encode_success,
                            encoded_image,
                        ) = cv2.imencode(
                            ".jpg",

                            annotated_frame,

                            [
                                int(
                                    cv2.IMWRITE_JPEG_QUALITY
                                ),

                                JPEG_QUALITY,
                            ],
                        )

                        if encode_success:
                            jpeg_bytes = (
                                encoded_image.tobytes()
                            )
                        else:
                            jpeg_bytes = None

                        store.set_result(
                            detection,
                            annotated_frame,
                            jpeg_bytes,
                        )

                        display_frame = (
                            annotated_frame
                        )

                    else:
                        store.set_result(
                            empty_detection_state(),
                            None,
                            None,
                        )

                        display_frame = (
                            image.copy()
                        )

                else:
                    store.set_result(
                        empty_detection_state(),
                        None,
                        None,
                    )

                    display_frame = (
                        image.copy()
                    )

            except Exception as error:
                print(
                    "Error saat prediksi: "
                    f"{error}"
                )

                store.set_result(
                    empty_detection_state(
                        "Error prediksi"
                    ),
                    None,
                    None,
                )

                display_frame = (
                    image.copy()
                )

        output_frame = (
            av.VideoFrame.from_ndarray(
                display_frame,
                format="bgr24",
            )
        )

        output_frame.pts = frame.pts
        output_frame.time_base = (
            frame.time_base
        )

        return output_frame

    return video_frame_callback


# ==========================================================
# LOAD CSS UI LAMA
# ==========================================================
def load_existing_css() -> str:
    if not STYLE_PATH.exists():
        raise FileNotFoundError(
            f"File CSS tidak ditemukan: {STYLE_PATH}"
        )

    return STYLE_PATH.read_text(
        encoding="utf-8"
    )


# ==========================================================
# CSS ADAPTER STREAMLIT
#
# CSS utama tetap berasal dari:
# src/static/style.css
# ==========================================================
STREAMLIT_ADAPTER_CSS = """
/* Hilangkan komponen bawaan Streamlit */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display: none !important;
}

/* Background dan container utama */
html,
body,
[data-testid="stAppViewContainer"],
.stApp {
    background: #eef3f9 !important;
    color: #222;
}

.block-container {
    width: 96% !important;
    max-width: 1800px !important;
    margin: 0 auto !important;
    padding: 15px 0 30px 0 !important;
}

/* Jarak layout utama */
div[data-testid="stHorizontalBlock"] {
    gap: 18px !important;
    align-items: stretch !important;
}

/* Hilangkan border bawaan container Streamlit */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .streamlit-webcam-marker
),
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .streamlit-right-marker
) {
    height: 100%;
    background: white;
    padding: 18px;
    border: none !important;
    border-radius: 20px;
    box-shadow:
        0 8px 20px
        rgba(0, 0, 0, 0.08);
    box-sizing: border-box;
}

/* Marker hanya untuk selector CSS */
.streamlit-webcam-marker,
.streamlit-right-marker {
    display: none;
}

/* Judul kamera */
.streamlit-section-title {
    color: #2f80ed;
    font-size: 22px;
    margin: 0 0 12px 0;
    font-weight: 700;
}

/* Komponen kamera */
div[data-testid="stCustomComponentV1"] {
    width: 100%;
    border-radius: 18px;
    border: 3px solid #dce8ff;
    overflow: hidden;
    background: #f7fbff;
    box-sizing: border-box;
}

div[data-testid="stCustomComponentV1"] iframe {
    width: 100% !important;
    min-height: 470px !important;
    border: 0 !important;
    border-radius: 15px !important;
    background: #f7fbff !important;
}

/* Tombol simpan hasil */
div[data-testid="stButton"] {
    margin-top: 16px;
}

div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 70px;
    background: #f7fbff;
    border: 2px solid #dbe9ff;
    border-radius: 16px;
    color: #2f80ed;
    font-size: 20px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
}

div[data-testid="stButton"] > button:hover {
    background: #2f80ed;
    color: white;
    border-color: #2f80ed;
}

div[data-testid="stButton"] > button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    background: #f7fbff;
    color: #7f9bc3;
    border-color: #dbe9ff;
}

/* Notifikasi simpan */
div[data-testid="stAlert"] {
    margin-top: 12px;
    border-radius: 16px;
}

/* Spinner */
div[data-testid="stSpinner"] {
    margin: 10px 0;
}

/* Responsive */
@media (max-width: 1200px) {
    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }

    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}

@media (max-width: 768px) {
    .block-container {
        width: 94% !important;
    }

    div[data-testid="stCustomComponentV1"] iframe {
        min-height: 360px !important;
    }
}
"""


# ==========================================================
# PASANG CSS
# ==========================================================
try:
    EXISTING_CSS = load_existing_css()

except Exception as error:
    st.error(
        f"CSS gagal dimuat: {error}"
    )

    st.stop()


st.markdown(
    f"""
    <style>
        {EXISTING_CSS}

        {STREAMLIT_ADAPTER_CSS}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div class="hero">
        <h1>
            🍅 Sistem Deteksi Penyakit Tanaman Tomat
        </h1>

        <p>
            Deteksi realtime menggunakan YOLOv11
            berbasis webcam
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOAD MODEL
# ==========================================================
try:
    with st.spinner(
        "Memuat model YOLOv11..."
    ):
        (
            model,
            device,
            model_lock,
        ) = load_yolo_model(
            str(MODEL_PATH)
        )

except Exception as error:
    st.error(
        f"Model gagal dimuat: {error}"
    )

    st.stop()


# ==========================================================
# LOAD METRIK
# ==========================================================
try:
    metrics_lookup = load_metrics(
        str(METRIC_FILE)
    )

except Exception as error:
    st.error(
        f"File metrik gagal dibaca: {error}"
    )

    st.stop()


if not metrics_lookup:
    st.warning(
        "File metrik tidak ditemukan atau kosong. "
        "Deteksi tetap berjalan, tetapi nilai "
        "evaluasi akan tampil '-'."
    )


# ==========================================================
# HUBUNGKAN DATABASE AIVEN
# ==========================================================
try:
    with st.spinner(
        "Menghubungkan aplikasi ke database..."
    ):
        database_connected = (
            initialize_database()
        )

except Exception as error:
    st.error(
        "Koneksi database Aiven gagal: "
        f"{error}"
    )

    st.info(
        "Tambahkan konfigurasi [mysql] "
        "melalui Streamlit App Settings > Secrets, "
        "kemudian reboot aplikasi."
    )

    st.stop()


# ==========================================================
# SESSION STATE
# ==========================================================
if (
    "detection_store"
    not in st.session_state
):
    st.session_state.detection_store = (
        DetectionStore()
    )


if (
    "save_feedback"
    not in st.session_state
):
    st.session_state.save_feedback = None


store: DetectionStore = (
    st.session_state.detection_store
)


# ==========================================================
# CALLBACK VIDEO
# ==========================================================
video_callback = build_video_callback(
    store=store,

    model=model,

    device=device,

    inference_lock=model_lock,

    metrics_lookup=metrics_lookup,
)


# ==========================================================
# LAYOUT UTAMA
# ==========================================================
left_column, right_column = (
    st.columns(
        [2, 1],
        gap="medium",
    )
)


# ==========================================================
# KAMERA REALTIME
# ==========================================================
with left_column:
    with st.container(
        border=True
    ):
        st.markdown(
            '<span class="streamlit-webcam-marker"></span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <h2 class="streamlit-section-title">
                📷 Kamera Realtime
            </h2>
            """,
            unsafe_allow_html=True,
        )

        webrtc_context = webrtc_streamer(
            key="tomato-yolo-realtime",

            mode=WebRtcMode.SENDRECV,

            video_frame_callback=(
                video_callback
            ),

            media_stream_constraints={
                "video": {
                    "width": {
                        "ideal": CAMERA_WIDTH,
                    },

                    "height": {
                        "ideal": CAMERA_HEIGHT,
                    },

                    "frameRate": {
                        "ideal": CAMERA_FPS,
                        "max": 20,
                    },
                },

                "audio": False,
            },

            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:"
                            "stun.l.google.com:"
                            "19302"
                        ]
                    }
                ]
            },

            media_toggle_controls=False,

            async_processing=True,
        )


# ==========================================================
# PANEL HASIL DETEKSI
# ==========================================================
with right_column:

    @st.fragment(
        run_every="1s"
    )
    def detection_panel() -> None:
        (
            detection,
            jpeg_bytes,
        ) = store.snapshot()

        raw_label = str(
            detection.get(
                "label",
                "Menunggu deteksi...",
            )
        )

        safe_label = html.escape(
            raw_label
        )

        try:
            confidence = float(
                detection.get(
                    "confidence",
                    0,
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        precision = display_metric(
            detection.get(
                "precision"
            )
        )

        recall = display_metric(
            detection.get(
                "recall"
            )
        )

        map50 = display_metric(
            detection.get(
                "map50"
            )
        )

        map5095 = display_metric(
            detection.get(
                "map5095"
            )
        )

        with st.container(
            border=True
        ):
            st.markdown(
                '<span class="streamlit-right-marker"></span>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <h2>
                        📋 Hasil Deteksi
                    </h2>

                    <div class="result-info">
                        <div class="result-item">
                            <span class="label">
                                Penyakit / Kondisi
                            </span>

                            <span id="detection-label">
                                {safe_label}
                            </span>
                        </div>

                        <div class="result-item">
                            <span class="label">
                                Confidence
                            </span>

                            <span id="detection-confidence">
                                {confidence:.2f}%
                            </span>
                        </div>
                    </div>
                </div>

                <div class="metrics-box">
                    <h2>
                        📊 Evaluasi Model YOLOv11
                    </h2>

                    <div class="metric-grid">
                        <div class="metric-item">
                            <h3>
                                Precision
                            </h3>

                            <p>
                                {precision}
                            </p>
                        </div>

                        <div class="metric-item">
                            <h3>
                                Recall
                            </h3>

                            <p>
                                {recall}
                            </p>
                        </div>

                        <div class="metric-item">
                            <h3>
                                mAP50
                            </h3>

                            <p>
                                {map50}
                            </p>
                        </div>

                        <div class="metric-item">
                            <h3>
                                mAP50-95
                            </h3>

                            <p>
                                {map5095}
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            invalid_labels = {
                "Menunggu deteksi...",
                "Tidak terdeteksi",
                "Error prediksi",
                "Kamera tidak terbaca",
            }

            can_save = (
                raw_label
                not in invalid_labels
                and jpeg_bytes is not None
                and database_connected
            )

            if st.button(
                "💾 Simpan Hasil",

                key="save-detection-button",

                type="primary",

                use_container_width=True,

                disabled=not can_save,
            ):
                (
                    success,
                    message,
                ) = save_detection_to_database(
                    detection,
                    jpeg_bytes,
                )

                st.session_state.save_feedback = (
                    success,
                    message,
                )

            feedback = (
                st.session_state.save_feedback
            )

            if feedback:
                (
                    success,
                    message,
                ) = feedback

                if success:
                    st.success(
                        message
                    )
                else:
                    st.error(
                        message
                    )

    detection_panel()


# ==========================================================
# STATUS SISTEM
# ==========================================================
@st.fragment(
    run_every="1s"
)
def status_panel() -> None:
    if webrtc_context.state.playing:
        status_text = (
            "Model aktif - "
            "webcam realtime berjalan"
        )
    else:
        status_text = (
            "Model aktif - "
            "tekan START untuk menjalankan webcam"
        )

    st.markdown(
        f"""
        <div class="card status-box">
            <h2>
                🟢 Status Sistem
            </h2>

            <p>
                {html.escape(status_text)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


status_panel()
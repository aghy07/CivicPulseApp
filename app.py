import os
import re
import sqlite3
from importlib import import_module

import joblib

flask = import_module("flask")
Flask = flask.Flask
redirect = flask.redirect
render_template = flask.render_template
request = flask.request
url_for = flask.url_for

try:
    mysql_connector = import_module("mysql.connector")
except ModuleNotFoundError:
    mysql_connector = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "civicpulse_model_final.joblib")
DB_PATH = os.path.join(BASE_DIR, "civicpulse.db")

app = Flask(__name__)

# Muat model NLP sekali saat server Flask dijalankan.
# Model file berada di folder proyek yang sama, jadi path absolut lebih aman.
model = joblib.load(MODEL_PATH)

LABEL_TAMPILAN = {
    "jalan_rusak": "Kerusakan Jalan & Trotoar",
    "kebersihan_dan_drainase": "Sampah, Kebersihan & Drainase",
    "taman_penataan_lingkungan": "Taman & Penataan Lingkungan",
    "utilitas_air_pdam": "Gangguan Air Bersih (PDAM)",
    "utilitas_listrik": "Gangguan Listrik & PJU",
}


def get_db_connection():
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "")
    mysql_database = os.getenv("MYSQL_DATABASE", "civicpulse_db")

    if mysql_host and mysql_connector is not None:
        try:
            conn = mysql_connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
                port=int(os.getenv("MYSQL_PORT", "3306")),
            )
            return conn
        except mysql_connector.Error:
            pass

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def setup_database():
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reporter_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    area TEXT,
                    predicted_category TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    verified_category TEXT,
                    status TEXT DEFAULT 'Menunggu Verifikasi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        else:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    reporter_name VARCHAR(100) NOT NULL,
                    title VARCHAR(150) NOT NULL,
                    description TEXT NOT NULL,
                    area VARCHAR(100) DEFAULT NULL,
                    predicted_category VARCHAR(50) NOT NULL,
                    confidence_score DECIMAL(5,2) NOT NULL,
                    verified_category VARCHAR(50) DEFAULT NULL,
                    status ENUM('Menunggu Verifikasi', 'Terverifikasi', 'Dalam Tindak Lanjut', 'Selesai') DEFAULT 'Menunggu Verifikasi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            cursor.close()
    finally:
        conn.close()


setup_database()


def bersihkan_input(teks):
    teks = re.sub(r"http\S+|www\.\S+", "", str(teks))
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


@app.route("/", methods=["GET", "POST"])
def index():
    hasil_prediksi = None

    if request.method == "POST":
        nama = request.form.get("reporter_name", "").strip()
        judul = request.form.get("title", "").strip()
        deskripsi = request.form.get("description", "").strip()
        area = request.form.get("area", "").strip()

        teks_gabungan = bersihkan_input(f"{judul}. {deskripsi}")
        probs = model.predict_proba([teks_gabungan])[0]
        idx_max = probs.argmax()

        kode_kategori = str(model.classes_[idx_max])
        confidence = round(float(probs[idx_max]) * 100, 2)
        status_awal = "Terverifikasi" if confidence >= 60.0 else "Menunggu Verifikasi"

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if isinstance(conn, sqlite3.Connection):
                cursor.execute(
                    """
                    INSERT INTO complaints
                    (reporter_name, title, description, area, predicted_category, confidence_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (nama, judul, deskripsi, area, kode_kategori, confidence, status_awal),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO complaints
                    (reporter_name, title, description, area, predicted_category, confidence_score, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (nama, judul, deskripsi, area, kode_kategori, confidence, status_awal),
                )
            conn.commit()
        finally:
            conn.close()

        hasil_prediksi = {
            "judul": judul,
            "kode": kode_kategori,
            "nama_kategori": LABEL_TAMPILAN.get(kode_kategori, kode_kategori),
            "confidence": confidence,
            "status": status_awal,
        }

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 20")
            daftar_laporan = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = [dict(row) for row in cursor.fetchall()]
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 20")
            daftar_laporan = cursor.fetchall()
            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = cursor.fetchall()
    finally:
        conn.close()

    return render_template(
        "index.html",
        hasil=hasil_prediksi,
        laporan=daftar_laporan,
        statistik=statistik,
        label_map=LABEL_TAMPILAN,
    )


@app.route("/verifikasi/<int:complaint_id>", methods=["POST"])
def verifikasi_laporan(complaint_id):
    kategori_baru = request.form.get("verified_category") or None
    status_baru = request.form.get("status", "Terverifikasi")

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cursor.execute(
                """
                UPDATE complaints
                SET verified_category = ?, status = ?
                WHERE id = ?
                """,
                (kategori_baru, status_baru, complaint_id),
            )
        else:
            cursor.execute(
                """
                UPDATE complaints
                SET verified_category = %s, status = %s
                WHERE id = %s
                """,
                (kategori_baru, status_baru, complaint_id),
            )
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
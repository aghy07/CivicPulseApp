import os
import re
import sqlite3
from functools import wraps

import joblib
import mysql.connector
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "civicpulse_model_final.joblib")
DB_PATH = os.path.join(BASE_DIR, "civicpulse.db")

app = Flask(__name__)
app.secret_key = "civicpulse-sdg11-secret-key-ubah-saat-produksi"

# Muat model NLP sekali saat server menyala
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

    if mysql_host:
        try:
            conn = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
                port=int(os.getenv("MYSQL_PORT", "3306")),
            )
            return conn
        except mysql.connector.Error:
            pass

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_nullable_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_geo_rows(rows):
    geo_rows = []
    for row in rows:
        latitude = row.get("latitude")
        longitude = row.get("longitude")

        if latitude is None or longitude is None:
            continue

        try:
            lat = float(latitude)
            lng = float(longitude)
        except (TypeError, ValueError):
            continue

        geo_rows.append({
            "id": row.get("id"),
            "title": row.get("title"),
            "reporter_name": row.get("reporter_name"),
            "area": row.get("area"),
            "latitude": lat,
            "longitude": lng,
            "kategori": row.get("verified_category") or row.get("predicted_category"),
            "status": row.get("status"),
        })
    return geo_rows


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
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin'
                )
                """
            )
            columns = [row[1] for row in conn.execute("PRAGMA table_info(complaints)").fetchall()]
            if "latitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN latitude REAL")
            if "longitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN longitude REAL")
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
                    latitude DECIMAL(10,7) NULL,
                    longitude DECIMAL(10,7) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(150) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(30) NOT NULL DEFAULT 'admin'
                )
                """
            )

            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'latitude'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN latitude DECIMAL(10,7) NULL")

            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'longitude'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN longitude DECIMAL(10,7) NULL")

            conn.commit()
            cursor.close()
        conn.commit()
    finally:
        conn.close()


def init_default_admin():
    """Membuat akun admin bawaan otomatis jika tabel users masih kosong."""
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
            admin_ada = cursor.fetchone()
            if not admin_ada:
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    ("Administrator CivicPulse", "admin@civicpulse.id", generate_password_hash("admin123"), "admin"),
                )
                conn.commit()
            cursor.close()
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
            admin_ada = cursor.fetchone()
            if not admin_ada:
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    ("Administrator CivicPulse", "admin@civicpulse.id", generate_password_hash("admin123"), "admin"),
                )
                conn.commit()
            cursor.close()
    finally:
        conn.close()


def admin_required(f):
    """Dekorator untuk melindungi rute khusus administrator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "admin":
            flash("Silakan login sebagai administrator untuk mengakses dashboard.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def bersihkan_input(teks):
    teks = re.sub(r"http\S+|www\.\S+", "", str(teks))
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


# Inisialisasi database saat import agar tabel selalu ada.
setup_database()


# ==========================================================
# 1. HALAMAN MASYARAKAT (PUBLIK)
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    hasil_prediksi = None

    if request.method == "POST":
        nama = request.form.get("reporter_name", "").strip()
        judul = request.form.get("title", "").strip()
        deskripsi = request.form.get("description", "").strip()
        area = request.form.get("area", "").strip()
        latitude = parse_nullable_float(request.form.get("latitude"))
        longitude = parse_nullable_float(request.form.get("longitude"))

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
                    (reporter_name, title, description, area, predicted_category, confidence_score, status, latitude, longitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (nama, judul, deskripsi, area, kode_kategori, confidence, status_awal, latitude, longitude),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO complaints
                    (reporter_name, title, description, area, predicted_category, confidence_score, status, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (nama, judul, deskripsi, area, kode_kategori, confidence, status_awal, latitude, longitude),
                )
            conn.commit()
        finally:
            conn.close()

        hasil_prediksi = {
            "judul": judul,
            "nama_kategori": LABEL_TAMPILAN.get(kode_kategori, kode_kategori),
            "confidence": confidence,
            "status": status_awal,
        }

    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM complaints
                ORDER BY created_at DESC
                LIMIT 15
                """
            )
            daftar_laporan = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = [dict(row) for row in cursor.fetchall()]
            cursor.close()
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT *
                FROM complaints
                ORDER BY created_at DESC
                LIMIT 15
                """
            )
            daftar_laporan = cursor.fetchall()
            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = cursor.fetchall()
            cursor.close()
    finally:
        conn.close()

    laporan_geo = normalize_geo_rows(daftar_laporan)

    return render_template(
        "index.html",
        hasil=hasil_prediksi,
        laporan=daftar_laporan,
        statistik=statistik,
        laporan_geo=laporan_geo,
        label_map=LABEL_TAMPILAN,
    )


# ==========================================================
# 2. AUTENTIKASI ADMINISTRATOR (LOGIN & LOGOUT)
# ==========================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        try:
            if isinstance(conn, sqlite3.Connection):
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ? LIMIT 1", (email,))
                user = cursor.fetchone()
                if user:
                    user = {
                        "id": user[0],
                        "name": user[1],
                        "email": user[2],
                        "password_hash": user[3],
                        "role": user[4],
                    }
                cursor.close()
            else:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s LIMIT 1", (email,))
                user = cursor.fetchone()
                cursor.close()
        finally:
            conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("admin_dashboard"))
        flash("Email atau kata sandi salah.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Anda telah keluar dari sesi administrator.", "info")
    return redirect(url_for("login"))


# ==========================================================
# 3. DASHBOARD & VERIFIKASI ADMINISTRATOR (TERPROTEKSI)
# ==========================================================
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC")
            semua_laporan = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = [dict(row) for row in cursor.fetchall()]
            cursor.close()
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC")
            semua_laporan = cursor.fetchall()
            cursor.execute(
                """
                SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total
                FROM complaints
                GROUP BY kategori
                """
            )
            statistik = cursor.fetchall()
            cursor.close()
    finally:
        conn.close()

    ringkasan_status = {
        "total": len(semua_laporan),
        "menunggu": sum(1 for r in semua_laporan if r["status"] == "Menunggu Verifikasi"),
        "proses": sum(1 for r in semua_laporan if r["status"] in ["Terverifikasi", "Dalam Tindak Lanjut"]),
        "selesai": sum(1 for r in semua_laporan if r["status"] == "Selesai"),
    }

    return render_template(
        "admin.html",
        laporan=semua_laporan,
        statistik=statistik,
        ringkasan=ringkasan_status,
        laporan_geo=normalize_geo_rows(semua_laporan),
        label_map=LABEL_TAMPILAN,
    )


@app.route("/admin/verifikasi/<int:complaint_id>", methods=["POST"])
@admin_required
def verifikasi_laporan(complaint_id):
    kategori_baru = request.form.get("verified_category")
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

    flash(f"Laporan #{complaint_id} berhasil diperbarui.", "success")
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    init_default_admin()
    app.run(debug=True)
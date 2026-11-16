import hmac
import math
import os
import re
import secrets
import sqlite3
from functools import wraps

import joblib
import mysql.connector
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "civicpulse_model_final.joblib")
DB_PATH = os.path.join(BASE_DIR, "civicpulse.db")

app = Flask(__name__)
secret_key = os.getenv("FLASK_SECRET_KEY")
if not secret_key:
    if os.getenv("FLASK_ENV", "").lower() == "production":
        raise RuntimeError("FLASK_SECRET_KEY harus dikonfigurasi untuk production.")
    secret_key = secrets.token_urlsafe(32)
elif len(secret_key) < 32:
    raise RuntimeError("FLASK_SECRET_KEY harus memiliki panjang minimal 32 karakter.")
app.secret_key = secret_key
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv(
        "SESSION_COOKIE_SECURE",
        "true" if os.getenv("FLASK_ENV", "").lower() == "production" else "false",
    ).lower() in {"1", "true", "yes"},
)


def create_csrf_token():
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": create_csrf_token}


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        session_token = session.get("_csrf_token", "")
        if not submitted_token or not session_token or not hmac.compare_digest(submitted_token, session_token):
            abort(400)


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
        return mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
            port=int(os.getenv("MYSQL_PORT", "3306")),
        )

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_nullable_float(value, minimum, maximum):
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def normalize_geo_rows(rows, include_reporter=False):
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
        if not math.isfinite(lat) or not math.isfinite(lng) or not -90 <= lat <= 90 or not -180 <= lng <= 180:
            continue

        geo_row = {
            "id": row.get("id"),
            "title": row.get("title"),
            "area": row.get("area"),
            "latitude": lat,
            "longitude": lng,
            "kategori": row.get("verified_category") or row.get("predicted_category"),
            "status": row.get("status"),
        }
        if include_reporter:
            geo_row["reporter_name"] = row.get("reporter_name")
        geo_rows.append(geo_row)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    role TEXT NOT NULL DEFAULT 'admin',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = [row[1] for row in conn.execute("PRAGMA table_info(complaints)").fetchall()]
            if "latitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN latitude REAL")
            if "longitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN longitude REAL")
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN updated_at TIMESTAMP")
                conn.execute("UPDATE complaints SET updated_at = created_at WHERE updated_at IS NULL")
            conn.execute("UPDATE complaints SET verified_category = NULL WHERE TRIM(verified_category) = ''")
            user_columns = [row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
            if "created_at" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP")
                conn.execute("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS complaints_set_updated_at_on_insert
                AFTER INSERT ON complaints
                FOR EACH ROW
                WHEN NEW.updated_at IS NULL
                BEGIN
                    UPDATE complaints
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS complaints_updated_at
                AFTER UPDATE ON complaints
                FOR EACH ROW
                WHEN NEW.updated_at = OLD.updated_at
                BEGIN
                    UPDATE complaints
                    SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                    WHERE id = NEW.id;
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS users_set_created_at_on_insert
                AFTER INSERT ON users
                FOR EACH ROW
                WHEN NEW.created_at IS NULL
                BEGIN
                    UPDATE users
                    SET created_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END
                """
            )
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
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
                    role VARCHAR(30) NOT NULL DEFAULT 'admin',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'latitude'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN latitude DECIMAL(10,7) NULL")

            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'longitude'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN longitude DECIMAL(10,7) NULL")

            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'updated_at'")
            if cursor.fetchone() is None:
                cursor.execute(
                    "ALTER TABLE complaints ADD COLUMN updated_at TIMESTAMP "
                    "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                )

            cursor.execute("SHOW COLUMNS FROM users LIKE 'created_at'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

            conn.commit()
            cursor.close()
        conn.commit()
    finally:
        conn.close()


def init_default_admin():
    """Create the initial admin only when an explicit password is configured."""
    admin_password = os.getenv("CIVICPULSE_ADMIN_PASSWORD")
    admin_email = os.getenv("CIVICPULSE_ADMIN_EMAIL", "admin@civicpulse.id")
    admin_name = os.getenv("CIVICPULSE_ADMIN_NAME", "Administrator CivicPulse")
    if admin_password is not None and len(admin_password) < 12:
        raise RuntimeError("CIVICPULSE_ADMIN_PASSWORD harus memiliki panjang minimal 12 karakter.")
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash FROM users WHERE role = 'admin' LIMIT 1")
            admin_ada = cursor.fetchone()
            if not admin_ada:
                if not admin_password:
                    app.logger.warning("Admin belum dibuat; set CIVICPULSE_ADMIN_PASSWORD untuk inisialisasi.")
                    return
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (admin_name, admin_email, generate_password_hash(admin_password), "admin"),
                )
                conn.commit()
            elif admin_password:
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (generate_password_hash(admin_password), admin_ada["id"]),
                )
                conn.commit()
            cursor.close()
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, password_hash FROM users WHERE role = 'admin' LIMIT 1")
            admin_ada = cursor.fetchone()
            if not admin_ada:
                if not admin_password:
                    app.logger.warning("Admin belum dibuat; set CIVICPULSE_ADMIN_PASSWORD untuk inisialisasi.")
                    return
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    (admin_name, admin_email, generate_password_hash(admin_password), "admin"),
                )
                conn.commit()
            elif admin_password:
                cursor.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (generate_password_hash(admin_password), admin_ada["id"]),
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
init_default_admin()


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
        raw_latitude = request.form.get("latitude", "")
        raw_longitude = request.form.get("longitude", "")
        latitude = parse_nullable_float(raw_latitude, -90, 90)
        longitude = parse_nullable_float(raw_longitude, -180, 180)
        if (raw_latitude or raw_longitude) and (latitude is None or longitude is None):
            latitude = None
            longitude = None
            flash("Koordinat tidak valid; laporan disimpan tanpa titik peta.", "warning")

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
            cursor.execute(
                """
                SELECT *
                FROM complaints
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY created_at DESC
                """
            )
            laporan_geo_rows = [dict(row) for row in cursor.fetchall()]
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
            cursor.execute(
                """
                SELECT *
                FROM complaints
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY created_at DESC
                """
            )
            laporan_geo_rows = cursor.fetchall()
            cursor.close()
    finally:
        conn.close()

    laporan_geo = normalize_geo_rows(laporan_geo_rows)

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
            session.clear()
            session["_csrf_token"] = secrets.token_urlsafe(32)
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("admin_dashboard"))
        flash("Email atau kata sandi salah.", "danger")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
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
        laporan_geo=normalize_geo_rows(semua_laporan, include_reporter=True),
        label_map=LABEL_TAMPILAN,
    )


@app.route("/admin/verifikasi/<int:complaint_id>", methods=["POST"])
@admin_required
def verifikasi_laporan(complaint_id):
    kategori_baru = request.form.get("verified_category") or None
    status_baru = request.form.get("status", "Terverifikasi")
    allowed_statuses = {
        "Menunggu Verifikasi",
        "Terverifikasi",
        "Dalam Tindak Lanjut",
        "Selesai",
    }

    if kategori_baru is not None and kategori_baru not in LABEL_TAMPILAN:
        abort(400)
    if status_baru not in allowed_statuses:
        abort(400)

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


@app.route("/admin/export-csv")
@admin_required
def export_csv():
    import csv
    import io
    from flask import Response

    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, description, area, predicted_category, confidence_score,
                       verified_category, COALESCE(verified_category, predicted_category) AS final_category,
                       status, latitude, longitude, created_at
                FROM complaints
                WHERE status != 'Menunggu Verifikasi'
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, title, description, area, predicted_category, confidence_score,
                       verified_category, COALESCE(verified_category, predicted_category) AS final_category,
                       status, latitude, longitude, created_at
                FROM complaints
                WHERE status != 'Menunggu Verifikasi'
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "title", "description", "area", "predicted_category",
        "confidence_score", "verified_category", "final_category",
        "status", "latitude", "longitude", "created_at"
    ])

    for row in rows:
        writer.writerow([
            row["id"], row["title"], row["description"], row["area"],
            row["predicted_category"], row["confidence_score"],
            row["verified_category"], row["final_category"],
            row["status"], row["latitude"], row["longitude"], row["created_at"]
        ])

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=civicpulse_verified_dataset.csv"
    return response


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_enabled)
import csv
import hmac
import io
import math
import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

import joblib
import mysql.connector
from PIL import Image, UnidentifiedImageError
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "civicpulse_model_final.joblib")
DB_PATH = os.path.join(BASE_DIR, "civicpulse.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB Max Upload
)

# Configuration for Flask-Limiter
storage_uri = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=storage_uri,
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
        submitted_token = request.form.get("csrf_token") or request.headers.get("X-CSRFToken", "")
        session_token = session.get("_csrf_token", "")
        if not submitted_token or not session_token or not hmac.compare_digest(submitted_token, session_token):
            abort(400)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org https://unpkg.com;"
    )
    return response


# Muat model NLP
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


@app.template_filter("wita")
def format_wita(value):
    if not value:
        return ""
    if isinstance(value, str):
        try:
            # handle sqlite format
            dt = datetime.strptime(value.split(".")[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value
    else:
        dt = value
    
    # Database default assumption: UTC
    wita_tz = timezone(timedelta(hours=8))
    dt_utc = dt.replace(tzinfo=timezone.utc)
    dt_wita = dt_utc.astimezone(wita_tz)
    
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    month_name = months[dt_wita.month - 1]
    return dt_wita.strftime(f"%d {month_name} %Y, %H:%M WITA")


def parse_nullable_float(value, minimum, maximum):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def generate_ticket_code(conn):
    year = datetime.now().year
    cursor = conn.cursor()
    if isinstance(conn, sqlite3.Connection):
        cursor.execute("SELECT COUNT(*) FROM complaints WHERE ticket_code LIKE ?", (f"CP-{year}-%",))
        row = cursor.fetchone()
    else:
        cursor.execute("SELECT COUNT(*) FROM complaints WHERE ticket_code LIKE %s", (f"CP-{year}-%",))
        row = cursor.fetchone()
    count = row[0] if not isinstance(row, dict) else row["COUNT(*)"]
    return f"CP-{year}-{(count + 1):04d}"


def process_photo(file):
    if not file or not file.filename:
        return None

    allowed_mimes = {"image/jpeg", "image/png", "image/webp"}
    if file.mimetype not in allowed_mimes:
        raise ValueError("Format foto harus JPEG, PNG, atau WebP.")

    ext = "webp" if file.mimetype == "image/webp" else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image = Image.open(file.stream)
        expected_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[file.mimetype]
        if image.format != expected_format or image.width * image.height > 20_000_000:
            raise ValueError("Foto tidak valid atau resolusinya terlalu besar.")
        image.load()
        if image.mode not in ("RGB", "L") or file.mimetype != "image/webp":
            image = image.convert("RGB")

        max_width = 1280
        if image.width > max_width:
            ratio = max_width / float(image.width)
            new_height = int(float(image.height) * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        image.save(filepath, optimize=True, quality=85)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("Foto tidak dapat dibaca. Pilih berkas JPEG, PNG, atau WebP yang valid.") from error
    return f"static/uploads/{filename}"


def record_status_event(conn, complaint_id, status, event_type):
    cursor = conn.cursor()
    if isinstance(conn, sqlite3.Connection):
        cursor.execute(
            """
            INSERT INTO complaint_status_history (complaint_id, status, event_type)
            VALUES (?, ?, ?)
            """,
            (complaint_id, status, event_type),
        )
    else:
        cursor.execute(
            """
            INSERT INTO complaint_status_history (complaint_id, status, event_type)
            VALUES (%s, %s, %s)
            """,
            (complaint_id, status, event_type),
        )


def check_duplication(conn, lat, lng, category):
    if lat is None or lng is None:
        return None

    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    
    if isinstance(conn, sqlite3.Connection):
        cursor.execute(
            """
            SELECT ticket_code, latitude, longitude FROM complaints 
            WHERE predicted_category = ? AND status != 'Selesai' AND created_at >= ?
            AND latitude IS NOT NULL AND longitude IS NOT NULL
            """,
            (category, seven_days_ago)
        )
        rows = [dict(r) for r in cursor.fetchall()]
    else:
        cursor.execute(
            """
            SELECT ticket_code, latitude, longitude FROM complaints 
            WHERE predicted_category = %s AND status != 'Selesai' AND created_at >= %s
            AND latitude IS NOT NULL AND longitude IS NOT NULL
            """,
            (category, seven_days_ago)
        )
        rows = cursor.fetchall()
        
    for row in rows:
        dist = haversine(lat, lng, float(row["latitude"]), float(row["longitude"]))
        if dist <= 100:  # 100 meters
            return row["ticket_code"]
            
    return None


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
            "ticket_code": row.get("ticket_code"),
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
                    ticket_code TEXT UNIQUE NOT NULL,
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
                    photo_path TEXT,
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
            if "ticket_code" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN ticket_code TEXT")
                # Backfill ticket_code for old data
                rows = conn.execute("SELECT id FROM complaints WHERE ticket_code IS NULL").fetchall()
                year = datetime.now().year
                for idx, row in enumerate(rows, 1):
                    conn.execute("UPDATE complaints SET ticket_code = ? WHERE id = ?", (f"CP-{year}-{idx:04d}", row[0]))
            if "photo_path" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN photo_path TEXT")
            if "latitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN latitude REAL")
            if "longitude" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN longitude REAL")
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE complaints ADD COLUMN updated_at TIMESTAMP")
                conn.execute("UPDATE complaints SET updated_at = created_at WHERE updated_at IS NULL")
            conn.execute("UPDATE complaints SET verified_category = NULL WHERE TRIM(verified_category) = ''")
        else:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticket_code VARCHAR(20) UNIQUE NOT NULL,
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
                    photo_path VARCHAR(255) DEFAULT NULL,
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
            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'ticket_code'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN ticket_code VARCHAR(20) UNIQUE NOT NULL")
            cursor.execute("SHOW COLUMNS FROM complaints LIKE 'photo_path'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE complaints ADD COLUMN photo_path VARCHAR(255) DEFAULT NULL")
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
            cursor.close()
    finally:
        conn.close()


@app.cli.command("init-db")
def init_db_command():
    """Perintah CLI untuk inisialisasi skema database dan admin default."""
    setup_database()
    init_default_admin()
    print("Database CivicPulse berhasil diinisialisasi.")


# Auto-init hanya di mode Non-Production jika database SQLite belum ada
if os.getenv("FLASK_ENV", "").lower() != "production" and not os.path.exists(DB_PATH) and not os.getenv("MYSQL_HOST"):
    setup_database()
    init_default_admin()


def admin_required(f):
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


# ==========================================================
# 1. HALAMAN MASYARAKAT & ENDPOINT API
# ==========================================================
@app.route("/", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def index():
    hasil_prediksi = None

    if request.method == "POST":
        nama = request.form.get("reporter_name", "").strip()
        judul = request.form.get("title", "").strip()
        deskripsi = request.form.get("description", "").strip()
        area = request.form.get("area", "").strip()
        raw_latitude = request.form.get("latitude", "")
        raw_longitude = request.form.get("longitude", "")

        # Validasi Panjang Input
        if len(judul) < 5:
            flash("Judul laporan terlalu pendek (minimal 5 karakter).", "warning")
            return redirect(url_for("index"))
        if len(deskripsi) < 10:
            flash("Deskripsi kejadian terlalu pendek (minimal 10 karakter).", "warning")
            return redirect(url_for("index"))

        latitude = parse_nullable_float(raw_latitude, -90, 90)
        longitude = parse_nullable_float(raw_longitude, -180, 180)

        # NLP Inference
        teks_gabungan = bersihkan_input(f"{judul}. {deskripsi}")
        probs = model.predict_proba([teks_gabungan])[0]
        idx_max = probs.argmax()
        kode_kategori = str(model.classes_[idx_max])
        confidence = round(float(probs[idx_max]) * 100, 2)
        status_awal = "Terverifikasi" if confidence >= 60.0 else "Menunggu Verifikasi"

        conn = get_db_connection()
        photo_path = None
        try:
            # Check Image Upload
            if "photo" in request.files:
                photo_path = process_photo(request.files["photo"])

            # Spatial Duplicate Check
            duplicate_ticket = check_duplication(conn, latitude, longitude, kode_kategori)
            if duplicate_ticket:
                flash(f"Peringatan: Terdapat laporan serupa di lokasi terdekat yang sedang diproses dengan Nomor Tiket: {duplicate_ticket}.", "warning")

            ticket_code = generate_ticket_code(conn)

            cursor = conn.cursor()
            if isinstance(conn, sqlite3.Connection):
                cursor.execute(
                    """
                    INSERT INTO complaints
                    (ticket_code, reporter_name, title, description, area, predicted_category, confidence_score, status, latitude, longitude, photo_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ticket_code, nama, judul, deskripsi, area, kode_kategori, confidence, status_awal, latitude, longitude, photo_path),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO complaints
                    (ticket_code, reporter_name, title, description, area, predicted_category, confidence_score, status, latitude, longitude, photo_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (ticket_code, nama, judul, deskripsi, area, kode_kategori, confidence, status_awal, latitude, longitude, photo_path),
                )
            conn.commit()
        finally:
            conn.close()

        hasil_prediksi = {
            "ticket_code": ticket_code,
            "judul": judul,
            "nama_kategori": LABEL_TAMPILAN.get(kode_kategori, kode_kategori),
            "confidence": confidence,
            "status": status_awal,
        }

    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 15")
            daftar_laporan = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total FROM complaints GROUP BY kategori")
            statistik = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT * FROM complaints WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY created_at DESC")
            laporan_geo_rows = [dict(row) for row in cursor.fetchall()]
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 15")
            daftar_laporan = cursor.fetchall()
            cursor.execute("SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total FROM complaints GROUP BY kategori")
            statistik = cursor.fetchall()
            cursor.execute("SELECT * FROM complaints WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY created_at DESC")
            laporan_geo_rows = cursor.fetchall()
    finally:
        conn.close()

    return render_template(
        "index.html",
        hasil=hasil_prediksi,
        laporan=daftar_laporan,
        statistik=statistik,
        laporan_geo=normalize_geo_rows(laporan_geo_rows),
        label_map=LABEL_TAMPILAN,
    )


@app.route("/api/tiket/<ticket_code>", methods=["GET"])
def api_tiket(ticket_code):
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints WHERE ticket_code = ?", (ticket_code,))
            row = cursor.fetchone()
            if row:
                row = dict(row)
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM complaints WHERE ticket_code = %s", (ticket_code,))
            row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({"status": "error", "message": "Nomor tiket tidak ditemukan."}), 404

    # Masking nama pelapor: misal "Ahmad Subagyo" -> "A***"
    reporter = row.get("reporter_name", "")
    masked_reporter = f"{reporter[0]}***" if reporter else "Anonim"

    kategori_final = row.get("verified_category") or row.get("predicted_category")

    return jsonify({
        "status": "success",
        "data": {
            "ticket_code": row["ticket_code"],
            "title": row["title"],
            "reporter_init": masked_reporter,
            "area": row.get("area") or "Wilayah umum",
            "kategori": LABEL_TAMPILAN.get(kategori_final, kategori_final),
            "status": row["status"],
            "created_at": format_wita(row["created_at"]),
            "updated_at": format_wita(row["updated_at"]),
        }
    })


# ==========================================================
# 2. AUTENTIKASI ADMINISTRATOR
# ==========================================================
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per 15 minutes", methods=["POST"])
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
                    user = dict(user)
            else:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s LIMIT 1", (email,))
                user = cursor.fetchone()
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
# 3. DASHBOARD ADMINISTRATOR
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
            cursor.execute("SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total FROM complaints GROUP BY kategori")
            statistik = [dict(row) for row in cursor.fetchall()]
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC")
            semua_laporan = cursor.fetchall()
            cursor.execute("SELECT COALESCE(verified_category, predicted_category) AS kategori, COUNT(*) AS total FROM complaints GROUP BY kategori")
            statistik = cursor.fetchall()
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
    allowed_statuses = {"Menunggu Verifikasi", "Terverifikasi", "Dalam Tindak Lanjut", "Selesai"}

    if kategori_baru is not None and kategori_baru not in LABEL_TAMPILAN:
        abort(400)
    if status_baru not in allowed_statuses:
        abort(400)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cursor.execute("UPDATE complaints SET verified_category = ?, status = ? WHERE id = ?", (kategori_baru, status_baru, complaint_id))
        else:
            cursor.execute("UPDATE complaints SET verified_category = %s, status = %s WHERE id = %s", (kategori_baru, status_baru, complaint_id))
        conn.commit()
    finally:
        conn.close()

    flash(f"Laporan #{complaint_id} berhasil diperbarui.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/export-csv")
@admin_required
def export_csv():
    conn = get_db_connection()
    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, ticket_code, title, description, area, predicted_category, confidence_score,
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
                SELECT id, ticket_code, title, description, area, predicted_category, confidence_score,
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
        "id", "ticket_code", "title", "description", "area", "predicted_category",
        "confidence_score", "verified_category", "final_category",
        "status", "latitude", "longitude", "created_at"
    ])

    for row in rows:
        writer.writerow([
            row["id"], row["ticket_code"], row["title"], row["description"], row["area"],
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

# CivicPulse

CivicPulse adalah aplikasi Flask untuk menerima dan mengelola laporan masalah
layanan publik di Kota Kendari. Klasifikasi laporan menggunakan model LSTM
Keras, dengan SQLite sebagai database default dan MySQL sebagai opsi deployment.

## Menjalankan secara lokal

Gunakan Python 3.12 atau versi yang didukung TensorFlow CPU:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Aplikasi lokal berjalan di `http://127.0.0.1:5000`.

## Deployment

Set `FLASK_SECRET_KEY` melalui secret/environment variable platform deployment
dengan nilai acak minimal 32 karakter. Untuk menjalankan dengan Gunicorn:

```sh
gunicorn --bind 0.0.0.0:${PORT:-8000} app:app
```

SQLite digunakan jika `MYSQL_HOST` tidak diatur. Untuk MySQL, atur
`MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, dan opsional
`MYSQL_PORT`. Pastikan direktori unggahan `static/uploads` tersedia pada storage
yang persisten jika foto laporan perlu dipertahankan setelah restart.

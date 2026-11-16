CREATE DATABASE IF NOT EXISTS civicpulse_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE civicpulse_db;

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
);
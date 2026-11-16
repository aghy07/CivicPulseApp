-- CivicPulse Database Schema (MySQL Compatible)
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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_ticket (ticket_code),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

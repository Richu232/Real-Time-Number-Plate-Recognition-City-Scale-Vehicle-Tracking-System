-- -- Create database
-- CREATE DATABASE IF NOT EXISTS vehicle_tracking;
-- USE vehicle_tracking;

-- -- Camera table to store camera information
-- CREATE TABLE IF NOT EXISTS camera (
--     camera_id VARCHAR(50) PRIMARY KEY,
--     location_name VARCHAR(100) NOT NULL,
--     latitude DECIMAL(10, 7) NOT NULL,
--     longitude DECIMAL(10, 7) NOT NULL,
--     status ENUM('active', 'inactive') DEFAULT 'active',
--     last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- -- VehicleDetection table to store plate detections
-- CREATE TABLE IF NOT EXISTS vehicle_detection (
--     detection_id INT AUTO_INCREMENT PRIMARY KEY,
--     plate_number VARCHAR(20) NOT NULL,
--     camera_id VARCHAR(50) NOT NULL,    -- Foreign key to camera table
--     detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     latitude DECIMAL(10, 7) NOT NULL,
--     longitude DECIMAL(10, 7) NOT NULL,
--     plate_image_path VARCHAR(255) NOT NULL,
--     FOREIGN KEY (camera_id) REFERENCES camera(camera_id)
-- );

-- -- Create indexes for optimized queries
-- CREATE INDEX idx_plate_detected ON vehicle_detection(plate_number, detected_at);
-- CREATE INDEX idx_latitude ON vehicle_detection(latitude);
-- CREATE INDEX idx_longitude ON vehicle_detection(longitude);

-- -- Insert sample camera (optional)
-- INSERT INTO camera (camera_id, location_name, latitude, longitude, status, last_heartbeat)
-- VALUES ('laptop_cam_001', 'Main Entrance', 37.7749, -122.4194, 'active', NOW())
-- ON DUPLICATE KEY UPDATE
--     location_name = VALUES(location_name),
--     latitude = VALUES(latitude),
--     longitude = VALUES(longitude),
--     status = VALUES(status),
--     last_heartbeat = VALUES(last_heartbeat);





--  Create Database
CREATE DATABASE IF NOT EXISTS edai_project_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

--  Create User & Grant Permissions
CREATE USER IF NOT EXISTS 'edai_project_user'@'localhost' IDENTIFIED BY 'strongPassword123';
GRANT ALL PRIVILEGES ON edai_project_db.* TO 'edai_project_user'@'localhost';
FLUSH PRIVILEGES;

--  Use the Database
USE edai_project_db;

--  Camera Table
CREATE TABLE IF NOT EXISTS camera (
    camera_id VARCHAR(50) PRIMARY KEY,
    location_name VARCHAR(100) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    status BOOLEAN DEFAULT TRUE,
    last_heartbeat DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_location (latitude, longitude),
    INDEX idx_status (status)
);

--  Vehicle Detection Table
CREATE TABLE IF NOT EXISTS vehicle_detection (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    plate_image_path VARCHAR(255) NOT NULL,
    
    INDEX idx_plate (plate_number),
    INDEX idx_detected_at (detected_at),
    INDEX idx_camera (camera_id),
    INDEX idx_geo (latitude, longitude),
    
    FOREIGN KEY (camera_id) REFERENCES camera(camera_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

--  Plate Events Table (for movement & Haversine analysis)
CREATE TABLE IF NOT EXISTS plate_events (
    event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    from_camera_id VARCHAR(50) NOT NULL,
    to_camera_id VARCHAR(50) NOT NULL,
    distance_km DECIMAL(8,3) NOT NULL,
    travel_time_seconds INT NOT NULL,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_plate_event (plate_number, detected_at),
    FOREIGN KEY (from_camera_id) REFERENCES camera(camera_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (to_camera_id) REFERENCES camera(camera_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

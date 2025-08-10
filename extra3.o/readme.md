#  Real-Time Number Plate Recognition & City-Scale Vehicle Tracking System

This project implements a **production-grade vehicle tracking system** that combines **YOLOv8**, **EasyOCR**, **ESP32-CAM live streaming**, **MySQL (B+Tree indexes)**, **geo-location logic (Haversine distance)**, **Parquet analytics storage**, and a **Flask REST API** integrated with a **map-based web dashboard**.

---

##  Features

- **ESP32-CAM Live Video Capture** (supports multiple cameras)
- **Dynamic Camera Registration** via API from ESP32-CAM (location, ID, status)
- **Real-Time License Plate Detection** with YOLOv8
- **OCR Plate Recognition** using EasyOCR with pre-processing & normalization
- **Duplicate Detection Filtering** using similarity checks + time thresholds
- **City-Scale Geo-Logic** with Haversine formula for movement & distance tracking
- **MySQL Storage** with B+Tree indexing for fast queries
- **Parquet Analytics Database** for large-scale historical queries
- **REST API** (Flask) for real-time vehicle tracking & geo-filters
- **Interactive Web Dashboard** for live tracking and searching detections

---

##  System Architecture

The system is made up of **four major components**:

1. **ESP32-CAM Units** → Streams live MJPEG video & sends metadata (ID, location)
2. **Plate Detection Pipeline (`plate_detection_final.py`)** → Detects plates, extracts text, stores results
3. **Database Layer (MySQL + Parquet)** → Stores live and historical detection data
4. **API & Dashboard (`api_server.py`)** → Serves tracking data to browser-based dashboard

---

##  Setup Instructions

###  Install Python Dependencies
```bash
python -m venv env
source env/bin/activate   # macOS/Linux
env\Scripts\activate      # Windows

pip install -r requirements.txt
```

###  MySQL Database Setup

1. Install MySQL  
2. Create the database & tables:  
```bash
mysql -u root -p < database_setup.sql
```

**`database_setup.sql` includes:**
- `camera` → Stores camera info (ID, location, heartbeat)
- `vehicle_detection` → Stores each detection event
- `plate_events` → Stores movement events between cameras (with Haversine distance & time)

Sample:
```sql
INSERT INTO camera (camera_id, location_name, latitude, longitude)
VALUES ('esp32_cam_001', 'Main Entrance', 37.7749, -122.4194);
```

###  Configure Database in Python
Update in **`plate_detection_final.py`** and **`api_server.py`**:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'edai_project_user',
    'password': 'strongPassword123',
    'database': 'edai_project_db'
}
```

---

##  Running the System

###  1. Start Plate Detection Pipeline
```bash
python plate_detection_final.py
```
- Connects to ESP32-CAM MJPEG stream
- Runs YOLOv8 plate detection
- Extracts plate text with OCR
- Saves to MySQL & Parquet (with deduplication & geo-tagging)

###  2. Start API & Dashboard
```bash
python api_server.py
```
- REST endpoint for vehicle search:
```
GET /track-vehicle/?plate_number=ABC123&start_date=2025-01-01&radius_km=5
```
- Interactive dashboard at:
```
http://localhost:5000/
```

---

##  API Parameters

- **`plate_number`** *(Required)* → The license plate number to search.  
- **`start_date`** / **`end_date`** *(Optional)* → Filter results by date range (`YYYY-MM-DD`).  
- **`latitude`**, **`longitude`**, **`radius_km`** *(Optional)* → Search for detections within a given radius from a point.  
- **`include_movements`** *(Optional)* → If set, returns camera-to-camera movement history along with detections.  

---

## Data Flow

1. ESP32-CAM sends **location + video stream**
2. Python pipeline detects plate, runs OCR, normalizes text
3. Checks for **recent duplicates** in cache + DB
4. Saves detection to **MySQL** (for live queries) & **Parquet** (for analytics)
5. Computes **Haversine distance** for movement between cameras → stores in `plate_events`
6. Flask API serves results to dashboard

---

##  Performance Optimizations

- **B+Tree MySQL Indexing** on `plate_number`, `detected_at`, `(latitude, longitude)`
- **In-memory recent plate cache** to skip duplicates within 5 min
- **Adaptive OCR preprocessing** to improve accuracy
- **Parquet compression** for historical analytics (40% query latency reduction)

---

##  Extending the System

- **Multiple ESP32-CAM units**: Just power on, they auto-register in DB
- **Custom YOLO training** for your region's plate format
- **Advanced dashboard** with heatmaps, live map tracking, and alerts

---

##  Troubleshooting

- **ESP32-CAM stream not opening** → Check `ESP_CAM_STREAM_URL` and ensure MJPEG mode
- **DB connection error** → Verify credentials in `DB_CONFIG`
- **OCR misses plates** → Adjust adaptive thresholding in `extract_plate_text`

---

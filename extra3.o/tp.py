
import os
import time
import cv2
import logging
import mysql.connector
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from ultralytics import YOLO
from plate_detection_final import PlateProcessor

# CONFIG
ESP_CAM_STREAM = "http://192.168.111.29/stream"
MODEL_PATH = "best.pt"
SAVE_DIR = "plate_images"
PARQUET_DIR = "parquet_store"
SQLITE_INDEX = "detections_index.db"
DB_CONFIG = {
    "host": "localhost",
    "user": "edai_project_user",
    "password": "strongPassword123",
    "database": "edai_project_db"
}
CAMERA_INFO = {
    "camera_id": "esp32_cam_001",
    "name": "Main Entrance",
    "latitude": 37.7749,
    "longitude": -122.4194
}
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class ParquetIndex:
    """
    Simple helper to write/read parquet and maintain a sqlite index for fast searching by plate.
    SQLite index acts as a B-tree (internal to sqlite).
    """
    def __init__(self, sqlite_path=SQLITE_INDEX, parquet_dir=PARQUET_DIR):
        self.sqlite_path = sqlite_path
        self.parquet_dir = parquet_dir
        self._ensure_db()

    def _ensure_db(self):
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                parquet_file TEXT,
                parquet_row INTEGER,
                detected_at TEXT
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plate ON detections_index(plate_number);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_detected_at ON detections_index(detected_at);")
        self.conn.commit()
        cur.close()

    def append(self, rec: dict):
        """
        Append rec into a daily parquet file and insert index row.
        rec should contain keys: plate_number, camera_id, detected_at (ISO str), latitude, longitude, plate_image_path
        """
        date_str = rec['detected_at'][:10]  # YYYY-MM-DD
        fname = os.path.join(self.parquet_dir, f"detections_{date_str}.parquet")
        df = pd.DataFrame([rec])
        # append by reading existing (simple prototype). For large scale replace with ParquetWriter.
        if os.path.exists(fname):
            try:
                existing = pd.read_parquet(fname)
                df = pd.concat([existing, df], ignore_index=True)
            except Exception as e:
                # fallback: overwrite
                pass
        df.to_parquet(fname, index=False)
        # find row index (we'll use last row index)
        row_idx = len(df) - 1
        # insert into sqlite index
        cur = self.conn.cursor()
        cur.execute("INSERT INTO detections_index (plate_number, parquet_file, parquet_row, detected_at) VALUES (?, ?, ?, ?)",
                    (rec['plate_number'], fname, row_idx, rec['detected_at']))
        self.conn.commit()
        cur.close()

    def fetch_recent(self, minutes=10):
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.cursor()
        cur.execute("SELECT plate_number, parquet_file, parquet_row, detected_at FROM detections_index WHERE detected_at >= ? ORDER BY detected_at DESC LIMIT 1000", (cutoff,))
        rows = cur.fetchall()
        out = []
        for plate, pf, pr, dt in rows:
            # read parquet row
            try:
                df = pd.read_parquet(pf, engine='pyarrow', use_pandas_metadata=False)
                row = df.iloc[pr].to_dict()
            except Exception:
                row = {'plate_number': plate, 'detected_at': dt}
            out.append(row)
        cur.close()
        return out

    def query_by_plate_prefix(self, prefix, limit=100):
        cur = self.conn.cursor()
        like = f"{prefix}%"
        cur.execute("SELECT plate_number, parquet_file, parquet_row, detected_at FROM detections_index WHERE plate_number LIKE ? ORDER BY detected_at DESC LIMIT ?", (like, limit))
        rows = cur.fetchall()
        results = []
        for plate, pf, pr, dt in rows:
            try:
                df = pd.read_parquet(pf)
                results.append(df.iloc[pr].to_dict())
            except Exception:
                results.append({'plate_number': plate, 'detected_at': dt})
        cur.close()
        return results

class PlateDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.processor = PlateProcessor()
        self.parquet_index = ParquetIndex()
        self.db = None
        self.connect_db()
        self.recent_cache = []  # list of dicts from recent MySQL or parquet reads

    def connect_db(self):
        try:
            self.db = mysql.connector.connect(**DB_CONFIG)
            logging.info("Connected to MySQL")
        except Exception as e:
            logging.error("MySQL connect failed: %s", e)
            self.db = None

    def fetch_recent_from_db(self, window_minutes=10):
        out = []
        if not self.db:
            return out
        cur = self.db.cursor()
        q = "SELECT plate_number, detected_at, latitude, longitude FROM vehicle_detection WHERE detected_at >= NOW() - INTERVAL %s MINUTE"
        try:
            cur.execute(q, (window_minutes,))
            for plate, dt, lat, lon in cur:
                out.append({'plate_number': plate, 'detected_at': dt, 'latitude': lat, 'longitude': lon})
        except Exception:
            pass
        cur.close()
        return out

    def save_detection(self, plate_text, crop_img, lat=CAMERA_INFO['latitude'], lon=CAMERA_INFO['longitude']):
        now = datetime.utcnow()
        # refresh recent cache from DB for geo-dup checks
        self.recent_cache = self.fetch_recent_from_db(window_minutes=10)
        # check geo-duplicate
        if self.processor.is_geo_duplicate(plate_text, lat, lon, self.recent_cache, time_window_seconds=300, distance_meters=200):
            logging.info("Geo-duplicate suppressed for %s", plate_text)
            return False
        # Save image
        ts = now.strftime("%Y%m%d_%H%M%S")
        fn = f"{plate_text}_{ts}.jpg"
        path = os.path.join(SAVE_DIR, fn)
        cv2.imwrite(path, crop_img)
        # Save to MySQL
        if self.db:
            try:
                cur = self.db.cursor()
                q = """
                INSERT INTO vehicle_detection (plate_number, camera_id, detected_at, latitude, longitude, plate_image_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cur.execute(q, (plate_text, CAMERA_INFO['camera_id'], now.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, path))
                self.db.commit()
                cur.close()
            except Exception as e:
                logging.error("DB insert error: %s", e)
        # Save to Parquet + index
        rec = {
            'plate_number': plate_text,
            'camera_id': CAMERA_INFO['camera_id'],
            'detected_at': now.strftime("%Y-%m-%d %H:%M:%S"),
            'latitude': lat,
            'longitude': lon,
            'plate_image_path': path
        }
        try:
            self.parquet_index.append(rec)
        except Exception as e:
            logging.error("Parquet append error: %s", e)
        logging.info("Saved detection %s", plate_text)
        return True

    def connect_cam(self):
        cap = cv2.VideoCapture(ESP_CAM_STREAM)
        if not cap.isOpened():
            logging.error("Could not open stream %s", ESP_CAM_STREAM)
            return None
        return cap

    def run(self):
        cap = self.connect_cam()
        if cap is None:
            return
        logging.info("Starting detection loop")
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.2)
                    continue
                display = frame.copy()
                results = self.model(frame)
                for res in results:
                    boxes = res.boxes.cpu().numpy() if hasattr(res.boxes, 'cpu') else res.boxes.numpy()
                    for box in boxes:
                        xy = box.xyxy[0] if hasattr(box, 'xyxy') else box.xyxy
                        x1, y1, x2, y2 = map(int, xy)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                        h = y2 - y1
                        if h < 12: continue
                        plate_crop = frame[y1:y2, x1:x2]
                        plate_text = self.processor.extract_plate_text(plate_crop)
                        if not plate_text: continue
                        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(display, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        # save
                        self.save_detection(plate_text, plate_crop)
                cv2.imshow("ALPR", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.exception("Loop error: %s", e)
                time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        if self.db:
            self.db.close()

if __name__ == "__main__":
    detector = PlateDetector()
    detector.run()

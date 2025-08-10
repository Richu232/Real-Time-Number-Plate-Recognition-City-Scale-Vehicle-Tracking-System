# # import cv2
# # import torch
# # import mysql.connector
# # import time
# # import os
# # import easyocr
# # import numpy as np
# # from datetime import datetime
# # from ultralytics import YOLO
# # from mysql.connector import Error

# # # Configuration
# # DB_CONFIG = {
# #     "host": "localhost",
# #     "user": "edai_project_user",  # MySQL username
# #     "password": "strongPassword123",  # MySQL password
# #     "database": "edai_project_db"  # MySQL database name
# # }

# # CAMERA_ID = "laptop_cam_001"
# # CAMERA_LOCATION = {
# #     'name': 'Main Entrance',
# #     'latitude': 37.7749,  # Example coordinates (San Francisco)
# #     'longitude': -122.4194
# # }
# # SAVE_DIR = 'plate_images'

# # # Ensure directory exists for storing plate images
# # os.makedirs(SAVE_DIR, exist_ok=True)

# # class PlateDetectionSystem:
# #     def __init__(self):
# #         self.model = YOLO('best.pt')  # Load your trained YOLOv8 model for license plates
# #         self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR
# #         self.db_connection = None
# #         self.connect_to_database()
# #         self.register_camera()
        
# #     def connect_to_database(self):
# #         """Establish connection to MySQL database"""
# #         try:
# #             self.db_connection = mysql.connector.connect(**DB_CONFIG)
# #             print("Database connection established successfully")
# #         except Error as e:
# #             print(f"Error connecting to MySQL database: {e}")
            
# #     def register_camera(self):
# #         """Register or update camera in the database"""
# #         if not self.db_connection or not self.db_connection.is_connected():
# #             self.connect_to_database()
            
# #         try:
# #             cursor = self.db_connection.cursor()
            
# #             # Check if camera exists
# #             query = """
# #             SELECT camera_id FROM camera WHERE camera_id = %s
# #             """
# #             cursor.execute(query, (CAMERA_ID,))
# #             result = cursor.fetchone()
            
# #             if result:
# #                 # Update existing camera
# #                 query = """
# #                 UPDATE camera
# #                 SET location_name = %s, latitude = %s, longitude = %s, 
# #                     status = TRUE, last_heartbeat = NOW()
# #                 WHERE camera_id = %s
# #                 """
# #                 cursor.execute(query, (
# #                     CAMERA_LOCATION['name'],
# #                     CAMERA_LOCATION['latitude'],
# #                     CAMERA_LOCATION['longitude'],
# #                     CAMERA_ID
# #                 ))
# #             else:
# #                 # Insert new camera
# #                 query = """
# #                 INSERT INTO camera 
# #                 (camera_id, location_name, latitude, longitude, status, last_heartbeat)
# #                 VALUES (%s, %s, %s, %s, TRUE, NOW())
# #                 """
# #                 cursor.execute(query, (
# #                     CAMERA_ID,
# #                     CAMERA_LOCATION['name'],
# #                     CAMERA_LOCATION['latitude'],
# #                     CAMERA_LOCATION['longitude']
# #                 ))
                
# #             self.db_connection.commit()
# #             print(f"Camera {CAMERA_ID} registered successfully")
# #         except Error as e:
# #             print(f"Error registering camera: {e}")
            
# #     def preprocess_for_ocr(self, plate_img):
# #         """Preprocess the plate image to improve OCR accuracy"""
# #         # Convert to grayscale
# #         gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
# #         # Apply adaptive thresholding
# #         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
# #                                       cv2.THRESH_BINARY, 11, 2)
        
# #         # Optional: Remove noise
# #         blur = cv2.GaussianBlur(thresh, (5, 5), 0)
        
# #         return blur
    
# #     def extract_plate_text(self, plate_img):
# #         """Extract text from license plate using EasyOCR"""
# #         # Preprocess the image
# #         processed_img = self.preprocess_for_ocr(plate_img)
        
# #         # Run OCR
# #         results = self.reader.readtext(processed_img)
        
# #         # Extract the text with highest confidence
# #         if results:
# #             # Sort by confidence and get the highest one
# #             results.sort(key=lambda x: x[2], reverse=True)
# #             text = results[0][1]
            
# #             # Clean the text (remove spaces, keep alphanumeric)
# #             plate_text = ''.join(char for char in text if char.isalnum())
            
# #             return plate_text
        
# #         return None
        
# #     def save_detection(self, plate_number, plate_img):
# #         """Save detection to database and image to disk"""
# #         if not self.db_connection or not self.db_connection.is_connected():
# #             self.connect_to_database()
            
# #         try:
# #             # Generate filename with timestamp
# #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #             filename = f"{plate_number}_{timestamp}.jpg"
# #             filepath = os.path.join(SAVE_DIR, filename)
            
# #             # Save image
# #             cv2.imwrite(filepath, plate_img)
            
# #             # Save to database
# #             cursor = self.db_connection.cursor()
# #             query = """
# #             INSERT INTO vehicle_detection
# #             (plate_number, camera_id, detected_at, latitude, longitude, plate_image_path)
# #             VALUES (%s, %s, NOW(), %s, %s, %s)
# #             """
# #             cursor.execute(query, (
# #                 plate_number,
# #                 CAMERA_ID,
# #                 CAMERA_LOCATION['latitude'],
# #                 CAMERA_LOCATION['longitude'],
# #                 filepath
# #             ))
            
# #             self.db_connection.commit()
# #             print(f"Saved detection for plate {plate_number}")
# #             return True
# #         except Error as e:
# #             print(f"Error saving detection: {e}")
# #             return False
            
# #     def run_detection(self):
# #         """Run the main detection loop"""
# #         cap = cv2.VideoCapture(0)  # Use default camera (0)
        
# #         if not cap.isOpened():
# #             print("Error: Could not open camera.")
# #             return
            
# #         print("Starting plate detection. Press 'q' to quit.")
        
# #         last_detection_time = 0
# #         detection_cooldown = 2  # Seconds between detections to avoid duplicates
        
# #         while True:
# #             ret, frame = cap.read()
            
# #             if not ret:
# #                 print("Error: Failed to capture frame.")
# #                 break
                
# #             # Display the original frame
# #             display_frame = frame.copy()
            
# #             # Run YOLOv8 detection
# #             results = self.model(frame)  # Use your trained model for license plate detection
            
# #             current_time = time.time()
            
# #             # Process results
# #             for result in results:
# #                 boxes = result.boxes.cpu().numpy()
# #                 for box in boxes:
# #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
# #                     # Extract the potential plate region
# #                     plate_img = frame[y1:y2, x1:x2]
# #                     if plate_img.size == 0:
# #                         continue
                        
# #                     # Draw bounding box
# #                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
# #                     # Only process every few seconds to avoid duplicate detections
# #                     if current_time - last_detection_time > detection_cooldown:
# #                         # Extract plate text
# #                         plate_number = self.extract_plate_text(plate_img)
                        
# #                         if plate_number and len(plate_number) >= 5:  # Minimum plate length check
# #                             cv2.putText(display_frame, plate_number, (x1, y1-10), 
# #                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            
# #                             # Save the detection
# #                             self.save_detection(plate_number, plate_img)
# #                             last_detection_time = current_time
            
# #             # Show frame
# #             cv2.imshow('License Plate Detection', display_frame)
            
# #             # Check for exit key
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break
                
# #         # Release resources
# #         cap.release()
# #         cv2.destroyAllWindows()
        
# #         # Close database connection
# #         if self.db_connection and self.db_connection.is_connected():
# #             self.db_connection.close()
# #             print("Database connection closed.")

# # if __name__ == "__main__":
# #     detector = PlateDetectionSystem()
# #     detector.run_detection()




# import cv2
# import torch
# import mysql.connector
# import time
# import os
# import easyocr
# import re
# from datetime import datetime, timedelta
# from ultralytics import YOLO
# from mysql.connector import Error
# from collections import defaultdict

# # Configuration
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "edai_project_user",  # MySQL username
#     "password": "strongPassword123",  # MySQL password
#     "database": "edai_project_db"  # MySQL database name
# }

# CAMERA_ID = "laptop_cam_001"
# CAMERA_LOCATION = {
#     'name': 'Main Entrance',
#     'latitude': 37.7749,
#     'longitude': -122.4194
# }
# SAVE_DIR = 'plate_images'
# os.makedirs(SAVE_DIR, exist_ok=True)

# class PlateDetectionSystem:
#     def __init__(self):
#         self.model = YOLO('best.pt')  # Load your trained YOLOv8 model for license plates
#         self.reader = easyocr.Reader(['en'])
#         self.db_connection = None
#         self.recent_plates = defaultdict(datetime)  # In-memory cache of recent plates
#         self.connect_to_database()
#         self.register_camera()
        
#     def connect_to_database(self):
#         """Establish connection to MySQL database"""
#         try:
#             self.db_connection = mysql.connector.connect(**DB_CONFIG)
#             print("Database connection established")
#         except Error as e:
#             print(f"Database connection error: {e}")
            
#     def register_camera(self):
#         """Register or update camera in the database"""
#         if not self.db_connection or not self.db_connection.is_connected():
#             self.connect_to_database()
            
#         try:
#             cursor = self.db_connection.cursor()
#             query = """
#             INSERT INTO camera 
#             (camera_id, location_name, latitude, longitude, status, last_heartbeat)
#             VALUES (%s, %s, %s, %s, TRUE, NOW())
#             ON DUPLICATE KEY UPDATE
#             location_name = VALUES(location_name),
#             latitude = VALUES(latitude),
#             longitude = VALUES(longitude),
#             status = VALUES(status),
#             last_heartbeat = VALUES(last_heartbeat)
#             """
#             cursor.execute(query, (
#                 CAMERA_ID,
#                 CAMERA_LOCATION['name'],
#                 CAMERA_LOCATION['latitude'],
#                 CAMERA_LOCATION['longitude']
#             ))
#             self.db_connection.commit()
#         except Error as e:
#             print(f"Camera registration error: {e}")
            
#     def normalize_plate_text(self, text):
#         """Normalize plate text for consistent comparison"""
#         # Remove all non-alphanumeric characters and convert to uppercase
#         cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).upper()
        
#         # Common OCR error corrections (customize based on your plates)
#         corrections = {
#             '0': 'O',
#             '1': 'I',
#             '5': 'S',
#             '8': 'B'
#         }
        
#         # Apply character substitutions
#         normalized = ''.join([corrections.get(c, c) for c in cleaned])
        
#         return normalized
    
#     def is_similar_plate(self, plate1, plate2):
#         """Check if two plates are likely the same with minor OCR differences"""
#         # Simple similarity check - adjust threshold as needed
#         from difflib import SequenceMatcher
#         return SequenceMatcher(None, plate1, plate2).ratio() > 0.7
    
#     def extract_plate_text(self, plate_img):
#         """Improved plate text extraction with validation"""
#         try:
#             # Preprocessing
#             gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#             thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                          cv2.THRESH_BINARY, 11, 2)
            
#             # Run OCR
#             results = self.reader.readtext(thresh)
            
#             if not results:
#                 return None
                
#             # Get best result by confidence
#             best_result = max(results, key=lambda x: x[2])
#             text, confidence = best_result[1], best_result[2]
            
#             # Validate result
#             if confidence < 0.6:  # Increased confidence threshold
#                 return None
                
#             normalized = self.normalize_plate_text(text)
            
#             # Length validation (adjust for your plate format)
#             if not (5 <= len(normalized) <= 10):
#                 return None
                
#             return normalized
            
#         except Exception as e:
#             print(f"OCR error: {e}")
#             return None
    
#     def is_recent_detection(self, plate_number):
#         """Check if this plate was recently detected using cache and database"""
#         # First check in-memory cache
#         now = datetime.now()
#         for cached_plate, timestamp in list(self.recent_plates.items()):
#             # Remove expired entries (older than 5 minutes)
#             if now - timestamp > timedelta(minutes=5):
#                 del self.recent_plates[cached_plate]
#             elif self.is_similar_plate(plate_number, cached_plate):
#                 return True
                
#         # Then check database if not found in cache
#         try:
#             cursor = self.db_connection.cursor()
#             query = """
#             SELECT plate_number FROM vehicle_detection
#             WHERE detected_at >= NOW() - INTERVAL 5 MINUTE
#             """
#             cursor.execute(query)
#             for (db_plate,) in cursor:
#                 if self.is_similar_plate(plate_number, db_plate):
#                     return True
#         except Error as e:
#             print(f"Database query error: {e}")
            
#         return False
    
#     def save_detection(self, plate_number, plate_img):
#         """Save detection only if not a recent duplicate"""
#         if self.is_recent_detection(plate_number):
#             print(f"Duplicate detected: {plate_number}")
#             return False
            
#         try:
#             # Generate filename
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"{plate_number}_{timestamp}.jpg"
#             filepath = os.path.join(SAVE_DIR, filename)
            
#             # Save image
#             cv2.imwrite(filepath, plate_img)
            
#             # Save to database
#             cursor = self.db_connection.cursor()
#             query = """
#             INSERT INTO vehicle_detection
#             (plate_number, camera_id, detected_at, latitude, longitude, plate_image_path)
#             VALUES (%s, %s, NOW(), %s, %s, %s)
#             """
#             cursor.execute(query, (
#                 plate_number,
#                 CAMERA_ID,
#                 CAMERA_LOCATION['latitude'],
#                 CAMERA_LOCATION['longitude'],
#                 filepath
#             ))
#             self.db_connection.commit()
            
#             # Add to recent plates cache
#             self.recent_plates[plate_number] = datetime.now()
            
#             print(f"Saved new detection: {plate_number}")
#             return True
            
#         except Error as e:
#             print(f"Save detection error: {e}")
#             return False
    
#     def run_detection(self):
#         """Main detection loop with improved duplicate prevention"""
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("Error: Could not open camera.")
#             return
            
#         print("Starting plate detection. Press 'q' to quit.")
        
#         # Track last processed plate and time
#         last_plate = None
#         last_plate_time = 0
#         min_processing_interval = 1.5  # Seconds between processing same plate
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture frame.")
#                 break
                
#             display_frame = frame.copy()
#             current_time = time.time()
            
#             # Run detection
#             results = self.model(frame)
            
#             for result in results:
#                 for box in result.boxes.cpu().numpy():
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     plate_img = frame[y1:y2, x1:x2]
                    
#                     if plate_img.size == 0:
#                         continue
                        
#                     # Draw bounding box
#                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
#                     # Process plate if enough time has passed
#                     if current_time - last_plate_time > min_processing_interval:
#                         plate_number = self.extract_plate_text(plate_img)
                        
#                         if plate_number:
#                             # Display plate text
#                             cv2.putText(display_frame, plate_number, (x1, y1-10),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            
#                             # Only process if different from last plate or enough time passed
#                             if (last_plate is None or 
#                                 not self.is_similar_plate(plate_number, last_plate) or
#                                 current_time - last_plate_time > 30):  # 30 second cooldown for same plate
                                
#                                 self.save_detection(plate_number, plate_img)
#                                 last_plate = plate_number
#                                 last_plate_time = current_time
            
#             cv2.imshow('License Plate Detection', display_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         cap.release()
#         cv2.destroyAllWindows()
#         if self.db_connection and self.db_connection.is_connected():
#             self.db_connection.close()

# if __name__ == "__main__":
#     detector = PlateDetectionSystem()
#     detector.run_detection()        


    
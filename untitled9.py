"""
Real-time Object Detection (YOLOv8) + Face Recognition (face_recognition)
--------------------------------------------------------------
Features:
- Detects general objects (YOLOv8).
- Detects and identifies KNOWN faces (face_recognition) within 'person' bounding boxes.
- Draws bounding boxes, class names, and confidence scores.
- Displays FPS counter.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition # <-- NEW LIBRARY
import os # <-- NEW LIBRARY

# ==============================
# USER SETTINGS
# ==============================

MODEL_NAME = "yolov8n.pt"        # Choose from yolov8n/s/m/l/x based on accuracy vs speed
CONFIDENCE_THRESHOLD = 0.35      # Minimum confidence for displaying detections
WEBCAM_INDEX = 0                 # 0 = default camera
SELECTED_CLASSES = {"person", "cell phone", "laptop"}  # Only detect these general objects
FACE_DATA_FOLDER = "known_faces" # <-- NEW: Folder containing your known face images
MAX_FACE_DISTANCE = 0.6          # <-- NEW: Tolerance for face recognition (lower=stricter)

# Font & display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2

# ==============================
# 1. LOAD FACE RECOGNITION DATA (NEW SECTION)
# ==============================
known_face_encodings = []
known_face_names = []
print(f"👤 Loading known faces from: {FACE_DATA_FOLDER}")

if not os.path.isdir(FACE_DATA_FOLDER):
    print(f"⚠️ Warning: Face data folder '{FACE_DATA_FOLDER}' not found. Face recognition will be skipped.")
else:
    for filename in os.listdir(FACE_DATA_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # The person's name is the part of the filename before the first '.'
            name = os.path.splitext(filename)[0]
            try:
                # Load image and get face encoding
                image_path = os.path.join(FACE_DATA_FOLDER, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Check if a face is actually found in the image
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"    - Loaded: {name}")
                else:
                    print(f"    - Skipped: {filename} (No face found or face not clear)")
                    
            except Exception as e:
                print(f"    - Error loading {filename}: {e}")

print(f"✅ Loaded {len(known_face_encodings)} known faces.")


# ==============================
# 2. LOAD YOLO MODEL
# ==============================
print(f"🔄 Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)  # Downloads automatically if not available

# Get class names
names = model.names if hasattr(model, "names") else model.model.names
idx_to_name = {int(i): name for i, name in names.items()}
PERSON_CLASS_ID = next((i for i, n in idx_to_name.items() if n == "person"), None) # Get ID for "person"

# Prepare selected class indices (if filtering)
if SELECTED_CLASSES is not None:
    selected_idx = {i for i, n in idx_to_name.items() if n in SELECTED_CLASSES}
    if not selected_idx:
        print("⚠️ Warning: None of the SELECTED_CLASSES matched model classes.")
        selected_idx = None
else:
    selected_idx = None

# ==============================
# 3. INITIALIZE WEBCAM
# ==============================
print("🎥 Opening webcam...")
cap = cv2.VideoCapture(0)  # 0,1,2 for index
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open webcam (index {WEBCAM_INDEX}).")

print("✅ Webcam opened successfully. Press 'q' to quit.")
prev_time = 0
fps = 0

# ==============================
# 4. MAIN LOOP
# ==============================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read from webcam. Exiting...")
            break
        
        # NOTE: face_recognition uses RGB format, while OpenCV uses BGR.
        # We only convert when needed to save time.
        rgb_frame = None

        # --- Run YOLO inference on frame ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]

        # --- Draw bounding boxes and labels ---
        if hasattr(r, "boxes") and r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = r.boxes.conf.cpu().numpy()  # Confidence scores
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # Class IDs

            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
                # Filter by selected classes if applicable
                if selected_idx is not None and cid not in selected_idx:
                    continue

                # Convert coordinates to integers for OpenCV
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Default label is from YOLO
                label = idx_to_name.get(cid, f"class {cid}")
                display_text = f"{label} {conf:.2f}"
                color = (0, 255, 0) # Default color for general objects

                # --- NEW: FACE RECOGNITION LOGIC ---
                if PERSON_CLASS_ID is not None and cid == PERSON_CLASS_ID:
                    # 5. Extract the "person" region from the frame
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Convert to RGB if not already done for the frame
                    if rgb_frame is None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # The 'face_recognition' library expects coordinates in (top, right, bottom, left)
                    # format relative to the original frame, NOT the ROI.
                    
                    # Find all face locations and encodings in the person ROI (or just the ROI coords in the frame)
                    # We will use the ROI coords relative to the whole frame for accuracy
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    # Filter face_locations to only those that fall within the YOLO person box
                    # This is a critical optimization step
                    matched_face_location = None
                    for (top, right, bottom, left) in face_locations:
                        # Check if the face location is largely contained within the YOLO box
                        if x1 <= left and x2 >= right and y1 <= top and y2 >= bottom:
                             matched_face_location = (top, right, bottom, left)
                             break
                    
                    if matched_face_location:
                        # 6. Encode the detected face
                        face_encodings = face_recognition.face_encodings(rgb_frame, [matched_face_location])

                        if face_encodings:
                            face_encoding = face_encodings[0]
                            
                            # 7. Compare face with known faces
                            matches = face_recognition.compare_faces(
                                known_face_encodings, 
                                face_encoding, 
                                tolerance=MAX_FACE_DISTANCE # Use the user-defined tolerance
                            )
                            
                            face_name = "Unknown"
                            
                            # If a match is found, find the best match (smallest distance)
                            if True in matches:
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    face_name = known_face_names[best_match_index]
                                    
                            # Update display text and color for the identified person
                            color = (255, 165, 0) if face_name == "Unknown" else (255, 0, 255) # Orange for Unknown, Magenta for Known
                            display_text = f"👤 {face_name}"
                    # --- END FACE RECOGNITION LOGIC ---
                    
                    # If it's a 'person' but no face was found by the library
                    if cid == PERSON_CLASS_ID and "👤" not in display_text:
                        # Use the original YOLO person label
                        color = (0, 0, 255) # Red for generic person box
                        
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

                # Draw filled rectangle for text background
                (tw, th), _ = cv2.getTextSize(display_text, FONT, FONT_SCALE, THICKNESS)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)

                # Put label text in white
                cv2.putText(frame, display_text, (x1, y1 - 4),
                            FONT, FONT_SCALE, (255, 255, 255),
                            THICKNESS - 1, cv2.LINE_AA)

        # --- FPS Calculation & Overlay ---
        cur_time = time.time()
        dt = cur_time - prev_time if prev_time > 0 else 0
        prev_time = cur_time
        fps = 1.0 / dt if dt > 0 else fps

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Display frame ---
        cv2.imshow("🔍 Real-time Detection & Recognition", frame)

        # --- Exit key ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Exiting...")
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam and windows closed properly.")
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")
model.fuse = lambda *args, **kwargs: model  # Prevent fusion error

# Risk classification thresholds
HIGH_RISK_SIZE_RATIO = 0.2  # Proportion of frame size for high risk
MEDIUM_RISK_SIZE_RATIO = 0.1  # Proportion of frame size for medium risk
MOVEMENT_THRESHOLD = 10  # Pixels moved per frame to classify as "moving"

# Store previous pedestrian positions
previous_positions = {}
pedestrian_id = 0  # Unique ID counter for tracking

def detect_pedestrian(frame):
    global previous_positions, pedestrian_id
    results = model(frame)  # Run segmentation model
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)  # Empty overlay
    new_positions = {}  # Store positions of detected pedestrians
    frame_area = frame.shape[0] * frame.shape[1]
    
    for result in results:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])  # Get class ID
            if class_id == 0:  # Only detect persons (ID = 0)
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_size = (x2 - x1) * (y2 - y1)  # Approximate area of bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Assign a unique ID for tracking
                pedestrian_id += 1
                new_positions[pedestrian_id] = (cx, cy)
                
                # Normalize risk classification by frame size
                if bbox_size / frame_area > HIGH_RISK_SIZE_RATIO:
                    risk_label = "HIGH RISK"
                    color = (0, 0, 255)  # Red
                elif bbox_size / frame_area > MEDIUM_RISK_SIZE_RATIO:
                    risk_label = "MEDIUM RISK"
                    color = (0, 255, 255)  # Yellow
                else:
                    risk_label = "LOW RISK"
                    color = (0, 255, 0)  # Green
                
                # Track movement to adjust risk
                if pedestrian_id in previous_positions:
                    px, py = previous_positions[pedestrian_id]
                    movement = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if movement > MOVEMENT_THRESHOLD:
                        risk_label = "HIGH RISK (Moving)"
                        color = (0, 0, 255)

                # Draw bounding box & risk label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, risk_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Apply segmentation mask
                mask = result.masks.data[i].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Create translucent overlay
                green_overlay = np.zeros_like(frame, dtype=np.uint8)
                green_overlay[:, :] = (0, 255, 0)
                mask_indices = mask_resized > 0
                mask_overlay[mask_indices] = green_overlay[mask_indices]

    previous_positions = new_positions  # Update positions for tracking
    alpha = 0.5  # Adjust transparency level
    blended_frame = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)
    return blended_frame

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_pedestrian(frame)  # Process frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

running = False

def start_webcam():
    global running
    running = True
    generate_frames()  # Start webcam stream

def stop_webcam():
    global running
    running = False  # Stop loop

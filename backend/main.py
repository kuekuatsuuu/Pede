import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")
model.fuse = lambda *args, **kwargs: model  # Prevent fusion error

def detect_pedestrian(frame):
    results = model(frame)  # Run segmentation model

    mask_overlay = np.zeros_like(frame, dtype=np.uint8)  # Empty overlay

    for result in results:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])  # Get class ID
            if class_id == 0:  # Only detect persons (ID = 0)
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Apply segmentation mask
                mask = result.masks.data[i].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

                # Resize mask to frame dimensions
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Create translucent overlay (Green)
                green_overlay = np.zeros_like(frame, dtype=np.uint8)
                green_overlay[:, :] = (0, 255, 0)  # Green color
                mask_indices = mask_resized > 0

                # Apply mask to overlay
                mask_overlay[mask_indices] = green_overlay[mask_indices]

    # Blend original frame with overlay for transparency effect
    alpha = 0.5  # Adjust transparency level (0 = invisible, 1 = solid)
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

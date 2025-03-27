import cv2
from ultralytics import YOLO
from utils import visualize_fusion_pedestrians_only

model = YOLO('yolov8s-seg.pt')

def detect_pedestrian(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Check the file path.")
        return None

    results = model(image)

    # Generate the output image path
    output_path = image_path.replace("uploads", "processed")
    
    return visualize_fusion_pedestrians_only(image, results, output_path)

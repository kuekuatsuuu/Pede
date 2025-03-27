from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from main import detect_pedestrian  # Import the detection function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def home():
    return "Flask is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Run pedestrian detection
    detected_image_path = detect_pedestrian(file_path)

    if detected_image_path is None:
        return jsonify({"error": "Image processing failed"}), 500

    # Convert processed image to base64 for frontend display
    with open(detected_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return jsonify({"image": encoded_image})

if __name__ == "__main__":
    app.run(debug=True)

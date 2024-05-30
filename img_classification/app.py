from flask import Flask, request, jsonify
import os
from object_detector import ObjectDetector

app = Flask(__name__)
detector = ObjectDetector(model_path="yolov8n.pt", results_dir="results")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(image_path)

    detections = detector.detect_objects(image_path)
    if detections.empty:
        return jsonify({"message": "No cars or persons detected"}), 200

    detector.save_detections_json(image_path, detections)
    detector.draw_boxes(image_path, detections)

    result_image_path = os.path.join("results", file.filename)
    json_path = os.path.join("results", f"{os.path.splitext(file.filename)[0]}.json")

    return jsonify({
        "image_path": result_image_path,
        "json_path": json_path
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

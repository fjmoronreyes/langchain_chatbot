# Object Detection Project

## Overview

This project contains scripts and resources for an object detection system using YOLOv8. The main components are:
- **`object_detector.py`**: Defines the `ObjectDetector` class for detecting objects (cars and people) in images, drawing bounding boxes around detected objects, and saving detection results.
- **`call_detector.py`**: A script to send an image to a running Flask server for object detection and print the detection results.

## Scripts

### `object_detector.py`
- **Purpose**: Detect objects in images, draw bounding boxes around detected objects, and save detection results in JSON format.
- **Key Components**:
  - `ObjectDetector`: A class encapsulating the object detection logic using YOLOv8.
  - `detect_objects(image_path: str)`: Detects objects in the given image and returns the results as a DataFrame.
  - `draw_boxes(image_path: str, detections: pd.DataFrame)`: Draws bounding boxes around detected objects and saves the image with the boxes.
  - `save_detections_json(image_path: str, detections: pd.DataFrame)`: Saves the detection results in JSON format.
  - `process_image(image_path: str)`: Detects objects in the image, draws bounding boxes, and saves the results.

### `call_detector.py`
- **Purpose**: Sends an image to a running Flask server for object detection and prints the response.
- **Key Components**:
  - `send_image_for_detection(url: str, image_path: str) -> Dict[str, str]`: Sends the image to the specified URL for detection and returns the response.

## Requirements

The project uses the following key libraries:
- `os`
- `pandas`
- `ultralytics`
- `PIL (Python Imaging Library)`
- `requests`

Make sure these libraries are installed in your Python environment. You can install them using `pip`:

## Docker

### Building the Docker Image

To build the Docker image, navigate to the project directory and run:

```
docker build -t object-detector .
```

### Running the Docker Container

To run the Docker container with the object detector service, execute:

On Windows

```
docker run -p 5000:5000 -v ${PWD}/results:/app/results object-detector #windows tested
```

On Linux or macOS

```
docker run -p 5000:5000 -v $(pwd)/results:/app/results object-detector
```


This command does the following:

- Maps port 5000 of the Docker container to port 5000 on your host machine.
- Mounts the results directory on your host machine to the /app/results directory in the Docker container to store results.

## Using call_detector.py

Once the Docker container is running, you can use call_detector.py to send an image for detection:

Ensure the Docker container is running and the Flask server is accessible at http://localhost:5000/detect.
Run the script to send an image and print the detection results:

```
python call_detector.py
```

The script will send the image located at ../imgs/img1.jpg to the Flask server for detection and print the JSON response with the results.

## Notes

- Make sure the results directory exists in the project root before running the Docker container.
- Adjust the image_path in call_detector.py if your image is located elsewhere.
import os
from typing import Any
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", results_dir: str = "results") -> None:
        self.model = YOLO(model_path)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def detect_objects(self, image_path: str) -> pd.DataFrame:
        results = self.model(image_path)[0]
        boxes = results.boxes

        data = boxes.data.cpu().numpy()
        df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

        df['class'] = df['class'].apply(lambda x: results.names[int(x)])
        df = df[df['class'].isin(['car', 'person'])]

        return df

    def draw_boxes(self, image_path: str, detections: pd.DataFrame) -> None:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for _, detection in detections.iterrows():
            x, y, w, h = (
                detection['xmin'],
                detection['ymin'],
                detection['xmax'],
                detection['ymax']
            )
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            draw.rectangle([x, y, w, h], outline="red", width=3)
            draw.text((x, y), label, fill="red")

        image_with_boxes_path = os.path.join(self.results_dir, os.path.basename(image_path))
        image.save(image_with_boxes_path)

    def save_detections_json(self, image_path: str, detections: pd.DataFrame) -> None:
        detections_json = detections.to_json(orient="records")
        json_file_path = os.path.join(
            self.results_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}.json"
        )
        with open(json_file_path, 'w') as json_file:
            json_file.write(detections_json)

    def process_image(self, image_path: str) -> None:
        detections = self.detect_objects(image_path)
        if not detections.empty:
            self.save_detections_json(image_path, detections)
            self.draw_boxes(image_path, detections)
            print(f"Detections and results saved in '{self.results_dir}'.")
        else:
            print("No people or cars detected")


if __name__ == "__main__":
    IMAGE_PATH = "../imgs/img1.jpg"

    detector = ObjectDetector()
    detector.process_image(IMAGE_PATH)
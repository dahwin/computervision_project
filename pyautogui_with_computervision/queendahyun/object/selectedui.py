import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
import cv2
import numpy as np
import json
from typing import List, Tuple


def filter_annotations(objects: List[str], bboxes: List[List[float]], img_path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Filter out annotations where the bounding box area is >= two-thirds of the image area.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image could not be loaded. Please check the file path.")
    height, width, _ = img.shape
    total_area = width * height
    two_third_area = (2 / 3) * total_area

    filtered_objects = []
    filtered_bboxes = []
    for obj, bbox in zip(objects, bboxes):
        x_min, y_min, x_max, y_max = bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)
        if bbox_area < two_third_area:
            filtered_objects.append(obj)
            filtered_bboxes.append(bbox)
    return filtered_objects, filtered_bboxes


class BoundingBoxApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_label_mousePressEvent

        self.load_button = QPushButton("Load Image and Annotations")
        self.load_button.clicked.connect(self.load_image_and_annotations)

        self.image_path_input = QLineEdit()
        self.image_path_input.setText(r"C:\Users\ALL USER\Desktop\e\ui\p.png")  # Default image path
        self.image_path_input.setReadOnly(True)

        self.annotations_path_input = QLineEdit()
        self.annotations_path_input.setText("annotations.json")
        self.annotations_path_input.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_path_input)
        layout.addWidget(self.annotations_path_input)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
        self.setWindowTitle("Bounding Box App")
        self.resize(1920, 1080)

        self.original_image = None
        self.image_with_bboxes = None
        self.objects = []
        self.bboxes = []
        self.bbox_colors = []
        self.selected_bbox_index = -1

        self.load_image_and_annotations()

    def load_image_and_annotations(self):
        image_path = self.image_path_input.text()
        self.objects, self.bboxes = self.load_annotations_from_json(self.annotations_path_input.text())
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Error: Could not load image at {image_path}")
            return
        self.original_image = cv2.resize(self.original_image, (1920, 1080))
        self.bbox_colors = [
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            for _ in range(len(self.objects))
        ]
        self.image_with_bboxes = self.process_image_with_bboxes(self.original_image.copy(), self.objects, self.bboxes, self.bbox_colors)
        self.update_image_label()

    def load_annotations_from_json(self, json_path: str) -> Tuple[List[str], List[List[float]]]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            objects = data['objects']
            bboxes = data['bboxes']
            objects, bboxes = filter_annotations(objects, bboxes, self.image_path_input.text())
            return objects, bboxes
        except FileNotFoundError:
            print(f"Error: Annotations file not found at {json_path}")
            return [], []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            return [], []

    def process_image_with_bboxes(self, img, objects, bboxes, colors):
        for obj, bbox, color in zip(objects, bboxes, colors):
            img = self.draw_bbox_and_label(img, bbox, obj, color)
        return img

    def draw_bbox_and_label(self, img, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3), font, font_scale, (255, 255, 255), thickness)
        return img

    def update_image_label(self):
        if self.image_with_bboxes is not None:
            height, width, channel = self.image_with_bboxes.shape
            bytes_per_line = 3 * width
            qimg = QImage(self.image_with_bboxes.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(1920, 1080, Qt.KeepAspectRatio))
        else:
            print("Error: image_with_bboxes is None. Cannot update image label.")

    def image_label_mousePressEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()

        # Map mouse coordinates to original image size
        scaled_x = int(x * (self.original_image.shape[1] / self.image_label.width()))
        scaled_y = int(y * (self.original_image.shape[0] / self.image_label.height()))

        self.selected_bbox_index = self.find_closest_bbox_index(scaled_x, scaled_y)
        self.update_image_with_selection()

    def find_closest_bbox_index(self, x, y):
        tolerance = 10  # Expand bounding box area slightly for hit testing
        closest_index = -1
        min_distance = float('inf')

        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            if x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
        return closest_index

    def update_image_with_selection(self):
        if self.original_image is None:
            print("Error: original_image is None. Cannot update image with selection.")
            return
        self.image_with_bboxes = self.original_image.copy()
        if self.selected_bbox_index != -1:
            x1, y1, x2, y2 = map(int, self.bboxes[self.selected_bbox_index])
            color = self.bbox_colors[self.selected_bbox_index]
            cv2.rectangle(self.image_with_bboxes, (x1, y1), (x2, y2), color, -1)
        self.image_with_bboxes = self.process_image_with_bboxes(self.image_with_bboxes, self.objects, self.bboxes, self.bbox_colors)
        self.update_image_label()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())
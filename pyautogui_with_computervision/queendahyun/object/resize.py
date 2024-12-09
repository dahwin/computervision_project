import sys
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QLinearGradient, QBrush
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QSizePolicy, QHBoxLayout
import cv2
import numpy as np
import json
from typing import List, Tuple
import httpx
import asyncio
from pathlib import Path
import nest_asyncio
from PIL import Image
import io
import os
import requests

nest_asyncio.apply()

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

        # Initialize panning-related attributes
        self._panning = False
        self._panStartX = 0
        self._panStartY = 0
        self._panStartXValue = 0
        self._panStartYValue = 0

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_label_mousePressEvent
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMaximumSize(1280, 720)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image_wrapper)
        self.upload_button.setFixedSize(120, 40)  # Smaller button size

        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image_wrapper)
        self.process_button.setEnabled(False)  # Initially disabled
        self.process_button.setFixedSize(120, 40)  # Smaller button size

        self.image_path_input = QLineEdit()
        self.image_path_input.setReadOnly(True)
        self.image_path_input.setFixedHeight(30)  # Smaller input height

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("Threshold (default: 0.15)")
        self.threshold_input.setText("0.15")  # Default threshold value
        self.threshold_input.setFixedSize(120, 50)  # Smaller input size

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(self.threshold_input)
        control_layout.addWidget(self.process_button)
        layout.addLayout(control_layout)

        self.setLayout(layout)
        self.setWindowTitle("Bounding Box App")
        self.resize(1280, 720)  # Set the default size to 1280x720

        self.original_image = None
        self.image_with_bboxes = None
        self.objects = []
        self.bboxes = []
        self.bbox_colors = []
        self.selected_bbox_index = -1

        self.scale_factor = 1.0
        self.translation_x = 0
        self.translation_y = 0

        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #000000, stop:1 #B2B5D3);
                color: white;
                font-family: Arial, sans-serif;
            }

            QPushButton {
                padding: 5px 10px;
                border: 2px solid #243689;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                         stop:0 #000000, stop:1 #B2B5D3);
                color: white;
                font-weight: bold;
                margin-bottom: 10px;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                         stop:0 #B2B5D3, stop:1 #000000);
            }

            QLineEdit {
                padding: 5px;
                border: 2px solid #4287f5;
                border-radius: 10px;
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                margin-bottom: 10px;
            }

            QLabel {
                font-size: 14px;
            }
        """)
    def upload_image_wrapper(self):
        asyncio.run(self.upload_image())

    async def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.image_path_input.setText(file_path)
            await self.upload_and_process_image()

    async def upload_and_process_image(self):
        base_url = "https://7ae1-34-136-52-218.ngrok-free.app"
        endpoint = "/upload_process/"
        server_url = f"{base_url}{endpoint}"

        try:
            file_path = self.image_path_input.text()
            threshold_value = float(self.threshold_input.text())

            file = Path(file_path)
            if not file.exists() or not file.is_file():
                print(f"File not found: {file_path}")
                return

            # Open the image using PIL
            with Image.open(file) as img:
                # Convert the image to RGB (in case it's not already in RGB format)
                img = img.convert("RGB")
                
                # Compress and save the image to a BytesIO stream
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)  # Adjust quality as needed
                buffer.seek(0)  # Reset the stream position
                file_path= os.path.basename(file_path)
                print(f'file {file_path}')
                # Send the image data as a binary stream
                response = requests.post(
                    server_url,
                    files={"file": (file_path, buffer, "image/jpeg")},
                    data={"threshold": str(threshold_value)},  # Send additional form data
                )

            # Check the response
            if response.status_code == 200:
                print("Image uploaded successfully")
                self.display_uploaded_image()  # Display the uploaded image immediately
                self.process_button.setEnabled(True)  # Enable the process button after upload
            else:
                print("Error:", response.status_code, response.text)

        except Exception as e:
            print(f"An error occurred: {e}")

    def display_uploaded_image(self):
        if self.image_path_input.text():
            pixmap = QPixmap(self.image_path_input.text())
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_image_wrapper(self):
        asyncio.run(self.get_results())

    async def get_results(self):
        base =  "https://7ae1-34-136-52-218.ngrok-free.app"
        url = f"{base}/full/"

        try:
            response = httpx.post(url)
            response.raise_for_status()

            results = response.json()
            print("Received results from /full/ endpoint:")
            top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, top_middle_side, bottom_middle_side, left_middle_side, right_middle_side, center_point, filtered_results, all_object, all_b = results['results']

            self.objects, self.bboxes = filter_annotations(all_object, all_b, self.image_path_input.text())
            self.original_image = cv2.imread(self.image_path_input.text())
            self.original_image = cv2.resize(self.original_image, (1920, 1080))
            self.bbox_colors = [
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                for _ in range(len(self.objects))
            ]
            self.image_with_bboxes = self.process_image_with_bboxes(self.original_image.copy(), self.objects, self.bboxes, self.bbox_colors)
            self.update_image_label()

        except httpx.RequestError as e:
            print(f"An error occurred while communicating with the API: {e}")

    def update_image_label(self):
        if self.image_with_bboxes is not None:
            height, width, channel = self.image_with_bboxes.shape
            bytes_per_line = 3 * width
            qimg = QImage(self.image_with_bboxes.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(pixmap.width() * self.scale_factor, pixmap.height() * self.scale_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.center_image()
        else:
            print("Error: image_with_bboxes is None. Cannot update image label.")

    def center_image(self):
        if self.image_label.pixmap() is not None:
            pixmap_width = self.image_label.pixmap().width()
            pixmap_height = self.image_label.pixmap().height()
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            x_offset = (label_width - pixmap_width) // 2
            y_offset = (label_height - pixmap_height) // 2
            self.image_label.move(x_offset + self.translation_x, y_offset + self.translation_y)

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

    def image_label_mousePressEvent(self, event):
        # Use event.position() and convert it to QPoint
        click_position = event.position().toPoint()

        # Get QLabel dimensions
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Get the scaled pixmap dimensions
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            print("Error: No pixmap found in QLabel.")
            return

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate aspect ratio offsets
        x_offset = max((label_width - pixmap_width) // 2, 0)  # Horizontal padding
        y_offset = max((label_height - pixmap_height) // 2, 0)  # Vertical padding

        # Debug logs
        print(f"Mouse click: {click_position}")
        print(f"QLabel size: {label_width}x{label_height}, Pixmap size: {pixmap_width}x{pixmap_height}")
        print(f"Offsets: x_offset={x_offset}, y_offset={y_offset}")

        # Ensure the click is within the image area
        if not (x_offset <= click_position.x() <= x_offset + pixmap_width and
                y_offset <= click_position.y() <= y_offset + pixmap_height):
            print("Click is outside the displayed image area.")
            return

        # Map the click position to the original image size
        scaled_x = int((click_position.x() - x_offset) * (self.original_image.shape[1] / pixmap_width))
        scaled_y = int((click_position.y() - y_offset) * (self.original_image.shape[0] / pixmap_height))

        # Debug logs for scaled coordinates
        print(f"Mapped coordinates: scaled_x={scaled_x}, scaled_y={scaled_y}")

        # Find the closest bounding box
        self.selected_bbox_index = self.find_closest_bbox_index(scaled_x, scaled_y)

        # Update the image display with the selection
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

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self.scale_factor += 0.1
        self.update_image_label()

    def zoom_out(self):
        if self.scale_factor > 0.1:
            self.scale_factor -= 0.1
            self.update_image_label()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panStartX = event.position().toPoint().x()
            self._panStartY = event.position().toPoint().y()
            self._panStartXValue = self.translation_x
            self._panStartYValue = self.translation_y
            self._panning = True

    def mouseMoveEvent(self, event):
        if self._panning:
            pos = event.position().toPoint()
            x = pos.x()
            y = pos.y()
            self.translation_x = self._panStartXValue + (x - self._panStartX)
            self.translation_y = self._panStartYValue + (y - self._panStartY)
            self.center_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())
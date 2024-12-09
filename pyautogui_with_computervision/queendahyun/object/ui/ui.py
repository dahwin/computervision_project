import sys
import httpx
import asyncio
from pathlib import Path
import nest_asyncio
import cv2
import numpy as np
import requests
import time
from PIL import Image
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from PySide6.QtGui import QPixmap, QImage, QWheelEvent
from PySide6.QtCore import Qt, QSize

nest_asyncio.apply()

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update_image()

    def update_image(self):
        if self.image is not None:
            scaled_image = self.image.scaled(self.size() * self.scale_factor, Qt.KeepAspectRatio)
            self.setPixmap(QPixmap.fromImage(scaled_image))

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        self.update_image()

    def resizeEvent(self, event):
        self.update_image()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Upload and Process")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image_wrapper)
        self.layout.addWidget(self.upload_button)

        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image_wrapper)
        self.process_button.setEnabled(False)  # Initially disabled
        self.layout.addWidget(self.process_button)

        self.image_viewer = ImageViewer()
        self.layout.addWidget(self.image_viewer)

        self.img_path = None

    def upload_image_wrapper(self):
        asyncio.run(self.upload_image())

    async def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.img_path = file_path
            await self.upload_and_process_image()

    async def upload_and_process_image(self):
        base_url =  "https://1450-34-30-134-19.ngrok-free.app"
        endpoint = "/process/"
        server_url = f"{base_url}{endpoint}"

        try:
            file = Path(self.img_path)
            if not file.exists() or not file.is_file():
                print(f"File not found: {self.img_path}")
                return

            async with httpx.AsyncClient(http2=True) as client:
                with file.open("rb") as f:
                    response = await client.post(
                        server_url,
                        files={"file": (file.name, f)},
                    )

                    if response.status_code == 200:
                        print("Image uploaded and processed successfully")
                        self.display_uploaded_image()
                        self.process_button.setEnabled(True)  # Enable the process button after upload
                    else:
                        print("Error:", response.status_code, response.text)
        except Exception as e:
            print(f"An error occurred: {e}")

    def display_uploaded_image(self):
        if self.img_path:
            pixmap = QPixmap(self.img_path)
            self.image_viewer.set_image(pixmap.toImage())

    def process_image_wrapper(self):
        asyncio.run(self.get_results())

    async def get_results(self):
        base =  "https://1450-34-30-134-19.ngrok-free.app"
        url = f"{base}/full/"

        try:
            response = requests.post(url)
            response.raise_for_status()

            results = response.json()
            print("Received results from /full/ endpoint:")
            top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, top_middle_side, bottom_middle_side, left_middle_side, right_middle_side, center_point, filtered_results, all_object, all_b = results['results']

            img = self.process_image_with_bboxes(
                self.img_path,
                all_object,
                all_b,
                output_path=None
            )

            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_viewer.set_image(q_image)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while communicating with the API: {e}")

    def process_image_with_bboxes(self, image_path, objects, bboxes, output_path=None):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (1920, 1080))

        for obj, bbox in zip(objects, bboxes):
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            img = self.draw_bbox_and_label(img, bbox, obj, color)

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def draw_bbox_and_label(self, img, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(img,
                      (x1, y1 - label_height - 5),
                      (x1 + label_width, y1),
                      color,
                      -1)

        cv2.putText(img,
                    label,
                    (x1, y1 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness)

        return img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

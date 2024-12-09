import sys
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QLinearGradient, QBrush
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QSizePolicy, QHBoxLayout, QScrollArea
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

        # Create a QScrollArea to contain the image label
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setMaximumSize(1280, 720)

        # Create the image label
        self.canva = QLabel()
        self.canva.setAlignment(Qt.AlignCenter)
        self.canva.mousePressEvent = self.canva_label_mousePressEvent
        self.canva.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Changed to Fixed
        self.canva.setMaximumSize(1280, 720)

        # Create the image label
        self.canva2 = QLabel()
        self.canva2.setAlignment(Qt.AlignCenter)
        self.canva2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Changed to Fixed
        self.canva2.setMaximumSize(600, 600)

        # Add the image label to the scroll area
        self.scroll_area.setWidget(self.canva)

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
        layout.addWidget(self.scroll_area)

        full_process_layout = QHBoxLayout()
        full_process_layout.addWidget(self.upload_button)
        full_process_layout.addWidget(self.threshold_input)
        full_process_layout.addWidget(self.process_button)
        layout.addLayout(full_process_layout)

        last_layout = QHBoxLayout()
        last_layout.addLayout(layout)  # Use addLayout instead of addWidget
        last_layout.addWidget(self.canva2)

        self.setLayout(last_layout)
        self.setWindowTitle("Bounding Box App")
        self.resize(1280, 720)  # Set the default size to 1280x720

        self.original_image = None
        self.image_with_bboxes = None
        self.objects = []
        self.bboxes = []
        self.bbox_colors = []
        self.selected_bbox_index = -1

        self.scale_factor = 1.0
        self.canva2_scale_factor = 1.0  # Add new scale factor for canva2
        self.make_canva2_rounded()

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
    def make_canva2_rounded(self):
        # Set the rounded rectangle mask for canva2
        size = self.canva2.size()
        rounded_rect = QImage(size, QImage.Format_ARGB32)
        rounded_rect.fill(Qt.transparent)  # Make the background transparent

        painter = QPainter(rounded_rect)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)  # This sets the fill color for the mask
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, size.width(), size.height(), 20, 20)  # Adjust radius here
        painter.end()

        # Set the rounded rectangle as the widget mask
        self.canva2.setMask(QPixmap.fromImage(rounded_rect).mask())

    async def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.image_path_input.setText(file_path)
            await self.upload_and_process_image()

    async def upload_and_process_image(self):
        base_url =  "https://fdb1-34-172-135-61.ngrok-free.app" 
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
            # Load and set the original image
            self.original_image = cv2.imread(self.image_path_input.text())
            
            # Reset the processed image
            self.image_with_bboxes = None
            
            # Update the image label to display the image
            self.update_image_label()

    def process_image_wrapper(self):
        asyncio.run(self.get_results())

    async def get_results(self):
        base =    "https://fdb1-34-172-135-61.ngrok-free.app" 
        url = f"{base}/full_process/"

        try:
            response = httpx.post(url)
            response.raise_for_status()

            results = response.json()
            print("Received results from /full/ endpoint:")
            top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, top_middle_side, bottom_middle_side, left_middle_side, right_middle_side, center_point, filtered_results, all_object, all_b = results['results']

            self.objects, self.bboxes = filter_annotations(all_object, all_b, self.image_path_input.text())
            self.original_image = cv2.imread(self.image_path_input.text())
            # self.original_image = cv2.resize(self.original_image, (1920, 1080))
            self.bbox_colors = [
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                for _ in range(len(self.objects))
            ]
            self.image_with_bboxes = self.process_image_with_bboxes(self.original_image.copy(), self.objects, self.bboxes, self.bbox_colors)
            self.update_image_label()

        except httpx.RequestError as e:
            print(f"An error occurred while communicating with the API: {e}")

    def update_image_label(self):
        if self.image_with_bboxes is not None or self.original_image is not None:
            # Get the image to display
            img = self.image_with_bboxes if self.image_with_bboxes is not None else self.original_image
            
            # Convert to QImage
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            
            # Calculate scaled dimensions
            scaled_width = int(width * self.scale_factor)
            scaled_height = int(height * self.scale_factor)
            
            # Scale the pixmap
            scaled_pixmap = pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Set the pixmap
            self.canva.setPixmap(scaled_pixmap)
            
            # Update the image label size to match the scaled image
            self.canva.setFixedSize(scaled_width, scaled_height)
            
            # Ensure scroll area updates its scrollbars
            self.scroll_area.setWidget(self.canva)
            
            # Print debug information
            print(f"Scale Factor: {self.scale_factor}")
            print(f"Scaled Size: {scaled_width}x{scaled_height}")
            print(f"Viewport Size: {self.scroll_area.viewport().width()}x{self.scroll_area.viewport().height()}")
        else:
            print("Error: No image to display")

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

    def canva_label_mousePressEvent(self, event):
        click_position = event.position().toPoint()
        
        # Get the current pixmap
        pixmap = self.canva.pixmap()
        if pixmap is None:
            return

        # Get the actual displayed image size (after scaling)
        displayed_width = int(self.original_image.shape[1] * self.scale_factor)
        displayed_height = int(self.original_image.shape[0] * self.scale_factor)

        # Calculate padding/offset to center the image
        x_offset = (self.canva.width() - displayed_width) // 2
        y_offset = (self.canva.height() - displayed_height) // 2

        # Adjust click position by removing the padding offset
        image_x = click_position.x() - x_offset
        image_y = click_position.y() - y_offset

        # Convert click position back to original image coordinates
        original_x = int(image_x / self.scale_factor)
        original_y = int(image_y / self.scale_factor)

        # Debug information
        print(f"Click Position: ({click_position.x()}, {click_position.y()})")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"Displayed Size: {displayed_width}x{displayed_height}")
        print(f"Offset: ({x_offset}, {y_offset})")
        print(f"Adjusted Position: ({image_x}, {image_y})")
        print(f"Original Image Position: ({original_x}, {original_y})")

        # Check if click is within image bounds
        if (0 <= original_x <= self.original_image.shape[1] and 
            0 <= original_y <= self.original_image.shape[0]):
            
            # Find the closest bounding box with adjusted tolerance based on zoom
            tolerance = int(10 / self.scale_factor)  # Adjust tolerance based on zoom level
            self.selected_bbox_index = self.find_closest_bbox_index(original_x, original_y, tolerance)
            
            if self.selected_bbox_index != -1:
                self.update_image_with_selection()

    def find_closest_bbox_index(self, x, y, tolerance):
        closest_index = -1
        min_distance = float('inf')

        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bbox area by tolerance
            x1 -= tolerance
            y1 -= tolerance
            x2 += tolerance
            y2 += tolerance
            
            # Check if point is inside expanded bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Calculate distance to center of bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
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

            # Extract and resize the selected bbox
            roi = self.extract_selected_bbox()
            resized_roi = self.resize_roi_to_fit_canva2(roi)

            if resized_roi is not None:
                # Use update_canva2 instead of directly setting the pixmap
                self.canva2_scale_factor = 1.0  # Reset scale factor when selecting new bbox
                self.update_canva2()

        self.image_with_bboxes = self.process_image_with_bboxes(self.image_with_bboxes, self.objects, self.bboxes, self.bbox_colors)
        self.update_image_label()

    def extract_selected_bbox(self):
        if self.selected_bbox_index == -1:
            return None
        x1, y1, x2, y2 = map(int, self.bboxes[self.selected_bbox_index])
        roi = self.original_image[y1:y2, x1:x2]
        return roi

    def resize_roi_to_fit_canva2(self, roi):
        if roi is None:
            return None
        max_width = self.canva2.width()
        max_height = self.canva2.height()
        roi_height, roi_width, _ = roi.shape

        # Calculate the scaling factor to fit within canva2
        scale_factor = min(max_width / roi_width, max_height / roi_height)

        # Ensure the ROI is not enlarged unnecessarily
        if scale_factor > 1.0:
            scale_factor = 1.0

        # Resize the ROI
        resized_roi = cv2.resize(roi, (int(roi_width * scale_factor), int(roi_height * scale_factor)))
        return resized_roi

    def wheelEvent(self, event):
        # Determine which widget the mouse is over
        pos = event.position()
        widget_under_mouse = self.childAt(int(pos.x()), int(pos.y()))

        if widget_under_mouse == self.canva:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        elif widget_under_mouse == self.canva2:
            if event.angleDelta().y() > 0:
                self.zoom_in_canva2()
            else:
                self.zoom_out_canva2()

    # Add new zoom methods for canva2
    def zoom_in_canva2(self):
        self.canva2_scale_factor += 0.1
        self.update_canva2()

    def zoom_out_canva2(self):
        if self.canva2_scale_factor > 0.1:
            self.canva2_scale_factor -= 0.1
            self.update_canva2()

    def update_canva2(self):
        if self.selected_bbox_index != -1:
            roi = self.extract_selected_bbox()
            if roi is not None:
                resized_roi = self.resize_roi_to_fit_canva2(roi)
                if resized_roi is not None:
                    # Get the base size after fitting to canva2
                    height, width, channel = resized_roi.shape
                    
                    # Apply additional scaling based on canva2_scale_factor
                    scaled_width = int(width * self.canva2_scale_factor)
                    scaled_height = int(height * self.canva2_scale_factor)
                    
                    # Resize the ROI with the new scale
                    scaled_roi = cv2.resize(resized_roi, (scaled_width, scaled_height))
                    
                    # Convert to QImage and display
                    bytes_per_line = 3 * scaled_width
                    qimg = QImage(scaled_roi.data, scaled_width, scaled_height, 
                                bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(qimg)
                    self.canva2.setPixmap(pixmap)

    def zoom_in(self):
        self.scale_factor += 0.1
        self.update_image_label()

    def zoom_out(self):
        if self.scale_factor > 0.1:
            self.scale_factor -= 0.1
            self.update_image_label()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())


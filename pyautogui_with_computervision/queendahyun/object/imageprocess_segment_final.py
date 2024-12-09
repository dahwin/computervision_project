import sys
from PySide6.QtCore import Qt, QPoint, QPointF, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QLinearGradient, QBrush
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QSizePolicy, QHBoxLayout, QScrollArea, QComboBox, QCheckBox, QSpinBox
from PySide6.QtCore import Property, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

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
import pyautogui

nest_asyncio.apply()

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

def resize_to_fit_screen(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def show_box(image, box):
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
    return image

def show_points(image, coords, labels, marker_size=10):
    for point, label in zip(coords, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(image, tuple(point), marker_size, color, thickness=-1)
    return image

def show_res(masks, scores, input_point, input_label, input_box, image):
    outputs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        output = show_mask(mask, image.copy())
        if input_box is not None:
            output = show_box(output, input_box[i])
        if input_point is not None and input_label is not None:
            output = show_points(output, input_point, input_label)
        outputs.append(output)
        print(f"Score: {score:.3f}")
    return outputs


def show_mask(mask, image, random_color=False):
    """
    Overlay the mask on the image with a specific or random BGR color.
    """
    overlay = image.copy()
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1).astype(np.uint8)

    if random_color:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)  # Random BGR color
    else:
        color = np.array([255, 144, 30], dtype=np.uint8)  # Fixed BGR for blue

    # Apply the mask color
    for c in range(3):  # Blend each color channel
        overlay[:, :, c] = np.where(mask.squeeze() == 1, 
                                    overlay[:, :, c] * 0.4 + color[c] * 0.6, 
                                    overlay[:, :, c])
    return overlay

class ToggleButton(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self._toggle = False
        self._margin = 3
        self._thumb_radius = (self.height() - 2 * self._margin) / 2
        self._thumb_position = self._margin
        self._anim = QPropertyAnimation(self, b"thumb_position", self)
        self._anim.setDuration(200)

    def thumb_position(self):
        return self._thumb_position

    def set_thumb_position(self, pos):
        self._thumb_position = pos
        self.update()

    thumb_position = Property(float, thumb_position, set_thumb_position)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_color = QColor("#4CD964") if self._toggle else QColor("#E9E9EA")
        thumb_color = Qt.white
        text_color = Qt.white if self._toggle else QColor("#9B9B9B")

        # Draw track
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height() / 2, self.height() / 2)

        # Draw thumb
        painter.setBrush(thumb_color)
        painter.drawEllipse(QRectF(self._thumb_position, self._margin, self._thumb_radius * 2, self._thumb_radius * 2))

        # Draw text
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Arial", 8))
        text = "ON" if self._toggle else "OFF"
        text_width = painter.fontMetrics().horizontalAdvance(text)
        painter.drawText(self.width() / 2 - text_width / 2, self.height() / 2 + 5, text)

    def mouseReleaseEvent(self, event):
        self._toggle = not self._toggle
        self._anim.setStartValue(self._thumb_position)
        self._anim.setEndValue(self.width() - 2 * self._margin - self._thumb_radius * 2 if self._toggle else self._margin)
        self._anim.start()

    def is_on(self):
        return self._toggle

class ProcessingWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, url, data):
        super().__init__()
        self.url = url
        self.data = data
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            if not self._is_running:
                return

            # Add timeout to prevent hanging
            response = requests.post(self.url, json=self.data, timeout=300)  # 5 minutes timeout
            
            if not self._is_running:
                return
                
            if response.status_code == 200:
                self.finished.emit(response.json())
            else:
                self.error.emit(f"Error: {response.status_code} - {response.text}")
        except requests.Timeout:
            self.error.emit("Request timed out after 5 minutes")
        except Exception as e:
            self.error.emit(str(e))

class BoundingBoxApp(QWidget):
    def __init__(self):
        super().__init__()
        self.url =    "https://ef15-34-57-170-62.ngrok-free.app"
        # Initialize all widgets first
        # Create a QScrollArea to contain the image label
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setMinimumSize(800, 600)
        self.scroll_area.setMaximumSize(1280,720)

        # Create the image label (canva)
        self.canva = QLabel()
        self.canva.setAlignment(Qt.AlignCenter)
        self.canva.mousePressEvent = self.canva_label_mousePressEvent
        self.canva.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canva.setMinimumSize(800, 600)
        self.canva.setMaximumSize(1280,720)


        # Add the image label to the scroll area
        self.scroll_area.setWidget(self.canva)

        # Create the second image label (canva2)
        self.canva2 = QLabel()
        self.canva2.setAlignment(Qt.AlignCenter)
        self.canva2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canva2.setMinimumSize(300, 300)
        self.canva2.setMaximumSize(600,620)

        # Initialize buttons and inputs
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image_wrapper)
        self.upload_button.setFixedSize(120, 40)

        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image_wrapper)
        self.process_button.setEnabled(False)
        self.process_button.setFixedSize(120, 40)

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("Threshold (default: 0.15)")
        self.threshold_input.setText("0.15")
        self.threshold_input.setFixedSize(120, 50)


        # Create main horizontal layout to hold everything
        main_layout = QHBoxLayout()

        # Create left side layout for canvas and controls
        left_layout = QVBoxLayout()

        # Add scroll area (main canvas)
        left_layout.addWidget(self.scroll_area, stretch=2)



        # Add layout selector
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["Full Process", "Specific Process"])
        self.layout_selector.currentIndexChanged.connect(self.switch_layout)
        left_layout.addWidget(self.layout_selector)

        # Create container widgets for layouts
        self.full_process_container = QWidget()
        self.specific_process_container = QWidget()

        # Create and setup the layouts
        self.full_process_layout = QHBoxLayout()
        self.specific_process_layout = QVBoxLayout()

        # Set layouts to their containers
        self.full_process_container.setLayout(self.full_process_layout)
        self.specific_process_container.setLayout(self.specific_process_layout)

        # Add widgets to full_process_layout
        self.full_process_layout.addWidget(self.upload_button)
        self.full_process_layout.addWidget(self.threshold_input)
        self.full_process_layout.addWidget(self.process_button)

        # Create and populate specific_process_layout
        self.create_specific_process_layout()

        # Add containers to left layout
        left_layout.addWidget(self.full_process_container)
        left_layout.addWidget(self.specific_process_container)

        # Initially hide specific process container
        self.specific_process_container.hide()

        # Add left layout to main layout
        main_layout.addLayout(left_layout)

        # Add canva2 (right side) to main layout
        main_layout.addWidget(self.canva2, stretch=1)

        # Set the main layout
        self.setLayout(main_layout)

        # Initialize other class variables
        self.original_image = None
        self.image_with_bboxes = None
        self.objects = []
        self.bboxes = []
        self.bbox_colors = []
        self.selected_bbox_index = -1
        self.scale_factor = 1.0
        self.canva2_scale_factor = 1.0
        self.image_path = None  # Store image path as a class variable instead of QLineEdit

        # Make canva2 rounded
        self.make_canva2_rounded()

        # Apply stylesheet
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

        # Connect the specific process button to the async function
        self.specific_process_button.clicked.connect(self.process_specific_image_wrapper)
        self.resize(1280, 720)  # Set the default size to 1280x720

        # Add loading indicator
        self.loading_label = QLabel("Processing...")
        self.loading_label.setStyleSheet("color: white; font-weight: bold;")
        self.loading_label.hide()
        self.specific_process_layout.addWidget(self.loading_label)
        
        # Initialize worker as None
        self.worker = None

        # Add segment toggle to full process layout
        self.full_segment_toggle = ToggleButton()
        segment_label = QLabel("Segment:")
        segment_label.setFixedWidth(75)  # Reduced width
        segment_label.setFixedHeight(20)  # Fixed height
        segment_label.setAlignment(Qt.AlignCenter)  # Center alignment
        segment_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #000000, stop:1 #B2B5D3);
                color: white;
                border: 1px solid #243689;
                border-radius: 5px;
                padding: 2px;
                font-weight: bold;
            }
        """)
        
        # Create vertical layout for segment controls
        segment_layout = QVBoxLayout()
        segment_layout.setSpacing(2)  # Reduce spacing between label and toggle
        segment_layout.addWidget(segment_label)
        segment_layout.addWidget(self.full_segment_toggle)
        segment_layout.setAlignment(Qt.AlignCenter)
        self.full_process_layout.addLayout(segment_layout)

        # Add segment toggle to specific process layout
        self.specific_segment_toggle = ToggleButton()
        segment_label2 = QLabel("Segment:")
        segment_label2.setFixedWidth(75)  # Reduced width
        segment_label2.setFixedHeight(20)  # Fixed height
        segment_label2.setAlignment(Qt.AlignCenter)  # Center alignment
        segment_label2.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #000000, stop:1 #B2B5D3);
                color: white;
                border: 1px solid #243689;
                border-radius: 5px;
                padding: 2px;
                font-weight: bold;
            }
        """)
        
        # Create vertical layout for segment controls
        segment_layout2 = QVBoxLayout()
        segment_layout2.setSpacing(2)  # Reduce spacing between label and toggle
        segment_layout2.addWidget(segment_label2)
        segment_layout2.addWidget(self.specific_segment_toggle)
        segment_layout2.setAlignment(Qt.AlignCenter)
        self.specific_process_layout.addLayout(segment_layout2)

        # Add variables for segmentation
        self.current_masks = None
        self.current_scores = None

        # Add variable for storing segmented output
        self.current_segmented_output = None


    def send_bbox_and_get_masks(self,bbox):
        base_url = self.url
        url = f"{base_url}/predict/"
        response = requests.post(url, json={"box": bbox})
        if response.status_code == 200:
            # Save the received compressed data to a file
            with open("output_data_compressed.npz", "wb") as f:
                f.write(response.content)
            # Load the arrays from the compressed file
            with np.load("output_data_compressed.npz") as data:
                masks = data["masks"]
                scores = data["scores"]
            return masks, scores
        else:
            print(f"Error: {response.json()}")
            return None, None

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
            self.image_path = file_path  # Store the path in the class variable
            await self.upload_and_process_image()

    async def upload_and_process_image(self):
        base_url =self.url
        endpoint = "/upload_process/"
        server_url = f"{base_url}{endpoint}"

        try:
            if self.image_path:
                # Reset all previous data
                self.objects = []
                self.bboxes = []
                self.bbox_colors = []
                self.selected_bbox_index = -1
                self.current_masks = None
                self.current_scores = None
                self.current_segmented_output = None
                self.image_with_bboxes = None
                self.canva2.clear()  # Clear canva2

                threshold_value = float(self.threshold_input.text())

                file = Path(self.image_path)
                if not file.exists() or not file.is_file():
                    print(f"File not found: {self.image_path}")
                    return

                # Open the image using PIL
                with Image.open(file) as img:
                    # Convert the image to RGB (in case it's not already in RGB format)
                    img = img.convert("RGB")
                    
                    # Compress and save the image to a BytesIO stream
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=90)  # Adjust quality as needed
                    buffer.seek(0)  # Reset the stream position
                    file_path = os.path.basename(self.image_path)
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
                self.display_uploaded_image(self.image_path)  # Display the uploaded image immediately
                self.process_button.setEnabled(True)  # Enable the process button after upload
            else:
                print("Error:", response.status_code, response.text)

        except Exception as e:
            print(f"An error occurred: {e}")
    def display_uploaded_image(self, file_path):
        if file_path:
            # Load and set the original image
            self.original_image = cv2.imread(file_path)
            
            # Reset the processed image
            self.image_with_bboxes = None
            
            # Update the image label to display the image
            self.update_image_label()

    def process_image_wrapper(self):
        # Reset previous processing data
        self.objects = []
        self.bboxes = []
        self.bbox_colors = []
        self.selected_bbox_index = -1
        self.current_masks = None
        self.current_scores = None
        self.current_segmented_output = None
        self.canva2.clear()  # Clear canva2
        
        # Process the new image
        asyncio.run(self.get_results())

    async def get_results(self):
        base =    self.url
        url = f"{base}/full_process/"

        try:
            response = httpx.post(url)
            response.raise_for_status()

            results = response.json()
            print("Received results from /full/ endpoint:")
            top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, top_middle_side, bottom_middle_side, left_middle_side, right_middle_side, center_point, filtered_results, all_object, all_b = results['results']

            self.objects, self.bboxes = all_object, all_b
            self.original_image = cv2.imread(self.image_path)  # Use self.image_path here
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

        # Check if click is within image bounds
        if (0 <= original_x <= self.original_image.shape[1] and 
            0 <= original_y <= self.original_image.shape[0]):
            
            # Find the closest bounding box with adjusted tolerance based on zoom
            tolerance = int(10 / self.scale_factor)  # Adjust tolerance based on zoom level
            clicked_bbox_index = self.find_closest_bbox_index(original_x, original_y, tolerance)
            
            # Toggle selection if clicking the same bbox
            if clicked_bbox_index == self.selected_bbox_index:
                self.selected_bbox_index = -1  # Unselect
                self.canva2.clear()  # Clear the zoomed view
            else:
                self.selected_bbox_index = clicked_bbox_index  # Select new bbox
            
            # Check if segmentation is enabled
            is_segment_enabled = (
                (self.layout_selector.currentIndex() == 0 and self.full_segment_toggle.is_on()) or
                (self.layout_selector.currentIndex() == 1 and self.specific_segment_toggle.is_on())
            )
            
            if is_segment_enabled:
                # Get the bbox coordinates
                bbox = self.bboxes[clicked_bbox_index]
                # Get segmentation masks
                masks, scores = self.get_segmentation_masks(bbox)
                if masks is not None:
                    self.current_masks = masks
                    self.current_scores = scores
                    self.update_image_with_segmentation()
            else:
                self.current_masks = None
                self.current_scores = None
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
        
        # Only highlight and show zoomed view if a bbox is selected
        if self.selected_bbox_index != -1:
            x1, y1, x2, y2 = map(int, self.bboxes[self.selected_bbox_index])
            color = self.bbox_colors[self.selected_bbox_index]
            cv2.rectangle(self.image_with_bboxes, (x1, y1), (x2, y2), color, -1)

            # Extract and resize the selected bbox
            roi = self.extract_selected_bbox()
            resized_roi = self.resize_roi_to_fit_canva2(roi)

            if resized_roi is not None:
                self.canva2_scale_factor = 1.0  # Reset scale factor when selecting new bbox
                self.update_canva2()
        else:
            # Clear canva2 when no bbox is selected
            self.canva2.clear()

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
            if hasattr(self, 'current_segmented_output') and self.current_segmented_output is not None:
                # Get the bbox coordinates
                x1, y1, x2, y2 = map(int, self.bboxes[self.selected_bbox_index])
                
                # Extract the segmented region from the stored output
                roi = self.current_segmented_output[y1:y2, x1:x2].copy()
                
                # Resize ROI to fit canva2
                resized_roi = self.resize_roi_to_fit_canva2(roi)
                if resized_roi is not None:
                    # Get dimensions
                    height, width, channel = resized_roi.shape
                    
                    # Scale based on canva2_scale_factor
                    scaled_width = int(width * self.canva2_scale_factor)
                    scaled_height = int(height * self.canva2_scale_factor)
                    
                    # Resize the ROI
                    scaled_roi = cv2.resize(resized_roi, (scaled_width, scaled_height))
                    
                    # Convert to QImage and display
                    bytes_per_line = 3 * scaled_width
                    qimg = QImage(scaled_roi.data, scaled_width, scaled_height, 
                                bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(qimg)
                    self.canva2.setPixmap(pixmap)
            else:
                # If no segmentation, show regular ROI
                roi = self.extract_selected_bbox()
                if roi is not None:
                    resized_roi = self.resize_roi_to_fit_canva2(roi)
                    if resized_roi is not None:
                        height, width, channel = resized_roi.shape
                        scaled_width = int(width * self.canva2_scale_factor)
                        scaled_height = int(height * self.canva2_scale_factor)
                        scaled_roi = cv2.resize(resized_roi, (scaled_width, scaled_height))
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

    def create_specific_process_layout(self):
        # Filter Range
        filter_range_layout = QHBoxLayout()
        self.filter_range_combo = QComboBox()
        self.filter_range_combo.addItems([
            'top_left_corner', 'top_right_corner', 'bottom_left_corner',
            'bottom_right_corner', 'top_middle_side', 'bottom_middle_side',
            'left_middle_side', 'right_middle_side', 'center_point'
        ])
        filter_range_layout.addWidget(QLabel("Filter Range:"))
        filter_range_layout.addWidget(self.filter_range_combo)
        filter_range_layout.addStretch()
        self.specific_process_layout.addLayout(filter_range_layout)

        # Object Input
        object_layout = QHBoxLayout()
        self.object_input = QLineEdit()
        self.object_input.setPlaceholderText("Enter object name")
        self.object_input.setFixedWidth(200)
        object_layout.addWidget(QLabel("Object:"))
        object_layout.addWidget(self.object_input)
        object_layout.addStretch()
        self.specific_process_layout.addLayout(object_layout)

        # Close Input
        close_layout = QHBoxLayout()
        self.close_input = QLineEdit()
        self.close_input.setPlaceholderText("Enter close value (optional)")
        self.close_input.setFixedWidth(200)
        close_layout.addWidget(QLabel("Close:"))
        close_layout.addWidget(self.close_input)
        close_layout.addStretch()
        self.specific_process_layout.addLayout(close_layout)

        # Verify Checkbox
        verify_layout = QHBoxLayout()
        self.verify_checkbox = QCheckBox("Verify")
        verify_layout.addWidget(self.verify_checkbox)
        verify_layout.addStretch()
        self.specific_process_layout.addLayout(verify_layout)

        # N SpinBox
        n_layout = QHBoxLayout()
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setRange(1, 10)
        n_layout.addWidget(QLabel("N:"))
        n_layout.addWidget(self.n_spinbox)
        n_layout.addStretch()
        self.specific_process_layout.addLayout(n_layout)

        # Process Button
        self.specific_process_button = QPushButton("Process Image")
        self.specific_process_button.clicked.connect(self.process_specific_image_wrapper)
        process_layout = QHBoxLayout()
        process_layout.addWidget(self.specific_process_button)
        process_layout.addStretch()
        self.specific_process_layout.addLayout(process_layout)

    def switch_layout(self, index):
        if index == 0:  # Full Process selected
            self.full_process_container.show()
            self.specific_process_container.hide()
        else:  # Specific Process selected
            self.full_process_container.hide()
            self.specific_process_container.show()

    def process_specific_image(self):
        try:
            # Clean up any existing worker
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait()
                self.worker.deleteLater()
                self.worker = None

            base_url =   self.url
            url = f"{base_url}/specific_process/"

            file_path = self.image_path
            if not file_path:
                print("No image uploaded.")
                return

            close_value = self.close_input.text()
            if not close_value:
                close_value = None

            data = {
                "filter_range": self.filter_range_combo.currentText(),
                "object": self.object_input.text(),
                "close": close_value,
                "direction": None,
                "n": self.n_spinbox.value(),
                "verify": self.verify_checkbox.isChecked()
            }

            # Disable UI elements
            self.specific_process_button.setEnabled(False)
            self.loading_label.show()

            # Create and setup worker
            self.worker = ProcessingWorker(url, data)
            self.worker.finished.connect(self.handle_processing_result)
            self.worker.error.connect(self.handle_processing_error)
            self.worker.start()

        except Exception as e:
            print(f"An error occurred: {e}")
            self.cleanup_worker()

    def handle_processing_result(self, response_data):
        try:
            if not response_data or 'results' not in response_data:
                raise ValueError("Invalid response data")

            bbox = response_data['results']
            
            # Reset previous bboxes and objects
            self.bboxes = [bbox]
            self.objects = [self.object_input.text()]
            self.bbox_colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))]
            
            # Load and process the image
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError("Failed to load image")

            self.image_with_bboxes = self.process_image_with_bboxes(
                self.original_image.copy(), 
                self.objects, 
                self.bboxes, 
                self.bbox_colors
            )
            
            # Update the display
            self.update_image_label()
        except Exception as e:
            print(f"Error processing result: {e}")
        finally:
            self.cleanup_worker()

    def handle_processing_error(self, error_message):
        print(f"Processing error: {error_message}")
        self.cleanup_worker()

    def cleanup_worker(self):
        # Re-enable UI elements
        self.specific_process_button.setEnabled(True)
        self.loading_label.hide()
        
        # Clean up worker
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event):
        # Clean up worker when closing the application
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        event.accept()

    def process_specific_image_wrapper(self):
        self.process_specific_image()

    def get_segmentation_masks(self, bbox):
        try:
            # Convert bbox to integers
            bbox = [round(value) for value in bbox]
            return self.send_bbox_and_get_masks(bbox)
        except Exception as e:
            print(f"Error getting segmentation masks: {e}")
            return None, None

    def update_image_with_segmentation(self):
        if self.original_image is None or self.current_masks is None:
            return

        # Create copy of original image
        self.image_with_bboxes = self.original_image.copy()
        
        if self.selected_bbox_index != -1:
            # Prepare inputs for show_res
            bbox = self.bboxes[self.selected_bbox_index]
            input_box = np.expand_dims(np.array(bbox), axis=0)
            
            # Get segmentation results
            outputs = show_res(
                self.current_masks,
                self.current_scores,
                input_point=None,
                input_label=None,
                input_box=input_box,
                image=self.original_image.copy()
            )
            
            # Store the segmented output for canva2
            if outputs:
                self.current_segmented_output = outputs[0]
                x1, y1, x2, y2 = map(int, bbox)
                # Put the segmented region into the main image
                self.image_with_bboxes[y1:y2, x1:x2] = self.current_segmented_output[y1:y2, x1:x2]

        # Draw all bboxes
        self.image_with_bboxes = self.process_image_with_bboxes(
            self.image_with_bboxes, self.objects, self.bboxes, self.bbox_colors
        )
        
        # Update displays
        self.update_image_label()
        self.update_canva2()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())
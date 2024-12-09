import sys
import requests
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLineEdit,
    QCheckBox, QSpinBox, QPushButton, QLabel, QFileDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class ImageProcessorUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 1280, 900)  # Set the main window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # GraphicsView to display the image
        self.canva = QGraphicsView()
        self.canva.setFixedSize(1280, 720)  # Set the fixed size of the QGraphicsView
        self.layout.addWidget(self.canva)

        self.scene = QGraphicsScene()
        self.canva.setScene(self.scene)

        # Create a new layout for the widgets below the canvas
        self.specific_process_layout = QVBoxLayout()
        self.layout.addLayout(self.specific_process_layout)

        # ComboBox for filter_range
        filter_range_layout = QHBoxLayout()
        self.filter_range_combo = QComboBox()
        self.filter_range_combo.addItems([
            'top_left_corner', 'top_right_corner', 'bottom_left_corner',
            'bottom_right_corner', 'top_middle_side', 'bottom_middle_side',
            'left_middle_side', 'right_middle_side', 'center_point'
        ])
        filter_range_layout.addWidget(QLabel("Filter Range:"))
        filter_range_layout.addWidget(self.filter_range_combo)
        filter_range_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(filter_range_layout)

        # LineEdit for object
        object_layout = QHBoxLayout()
        self.object_input = QLineEdit()
        self.object_input.setPlaceholderText("Enter object name")
        self.object_input.setFixedWidth(200)  # Set fixed width for the QLineEdit
        object_layout.addWidget(QLabel("Object:"))
        object_layout.addWidget(self.object_input)
        object_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(object_layout)

        # LineEdit for close
        close_layout = QHBoxLayout()
        self.close_input = QLineEdit()
        self.close_input.setPlaceholderText("Enter close value (optional)")
        self.close_input.setFixedWidth(200)  # Set fixed width for the QLineEdit
        close_layout.addWidget(QLabel("Close:"))
        close_layout.addWidget(self.close_input)
        close_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(close_layout)

        # CheckBox for verify
        verify_layout = QHBoxLayout()
        self.verify_checkbox = QCheckBox("Verify")
        verify_layout.addWidget(self.verify_checkbox)
        verify_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(verify_layout)

        # SpinBox for n
        n_layout = QHBoxLayout()
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setRange(1, 10)
        n_layout.addWidget(QLabel("N:"))
        n_layout.addWidget(self.n_spinbox)
        n_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(n_layout)

        # Button to process image
        process_layout = QHBoxLayout()
        self.process_button = QPushButton("Process Image")
        process_layout.addWidget(self.process_button)
        process_layout.addStretch()  # Add stretch factor to push widgets to the left
        self.specific_process_layout.addLayout(process_layout)

        self.process_button.clicked.connect(self.process_image)

    def process_image(self):
        base_url =   "https://8e27-34-172-135-61.ngrok-free.app"
        url = f"{base_url}/specific_process/"

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

        response = requests.post(url, json=data)

        if response.status_code == 200:
            response_data = response.json()
            bbox = response_data['results']
            print(bbox)
            x_min, y_min, x_max, y_max = map(int, bbox)

            file_path = r"C:\Users\ALL USER\Downloads\scroll bar1.png"
            image = cv2.imread(file_path)

            color = tuple(np.random.randint(0, 256, size=3).tolist())
            thickness = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Resize the image to 1280x720
            resized_image = cv2.resize(image, (1280, 720))

            height, width, channel = resized_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)

            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.canva.setSceneRect(pixmap.rect())
        else:
            print(f"Error: {response.json()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorUI()
    window.show()
    sys.exit(app.exec())

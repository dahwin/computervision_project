import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.last_point = None
        self.thickness = 2
        self.img = None
        self.mask = None
        self.setMouseTracking(True)

    def setImage(self, img):
        self.img = img.copy()
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.updatePixmap()

    def updatePixmap(self):
        height, width, channel = self.img.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing:
            current_point = event.position().toPoint()
            cv2.line(self.img, (self.last_point.x(), self.last_point.y()),
                     (current_point.x(), current_point.y()), (0, 255, 0), self.thickness)
            cv2.line(self.mask, (self.last_point.x(), self.last_point.y()),
                     (current_point.x(), current_point.y()), 255, self.thickness)
            self.last_point = current_point
            self.updatePixmap()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def setThickness(self, value):
        self.thickness = value

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Canvas")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas)

        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(2)
        self.thickness_slider.valueChanged.connect(self.canvas.setThickness)
        layout.addWidget(self.thickness_slider)

        # Load the image
        img = cv2.imread('al.png')
        if img is None:
            print("Error: Could not load image 'al.png'.")
            sys.exit()

        self.canvas.setImage(img)

        # Set up a timer to periodically update the mask display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateMaskDisplay)
        self.timer.start(100)  # Update every 100 ms

        self.mask_label = QLabel()
        layout.addWidget(self.mask_label)

    def updateMaskDisplay(self):
        if self.canvas.mask is not None:
            height, width = self.canvas.mask.shape
            bytes_per_line = width
            q_img = QImage(self.canvas.mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            self.mask_label.setPixmap(pixmap.scaled(self.mask_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.updatePixmap()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
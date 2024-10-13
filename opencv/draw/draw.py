import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QColorDialog, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QSlider
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal

class ImageUpdateThread(QThread):
    update_signal = pyqtSignal(QImage)

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        q_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], 
                         self.image.shape[1] * 3, QImage.Format_RGB888)
        self.update_signal.emit(q_image)

class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('OpenCV Drawing App')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()

        # Load the image
        self.image = cv2.imread(r"C:\Users\ALL USER\Desktop\computervision_project\opencv\draw\image.png")
        self.drawing = False
        self.last_point = QPoint()
        self.current_color = QColor(255, 0, 0)  # Default color: red
        self.current_thickness = 2  # Default thickness

        # Convert image to RGB for display
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Create QImage and QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Initialize the image
        self.update_image_threaded()

        # Mouse events for drawing
        self.image_label.mousePressEvent = self.start_drawing
        self.image_label.mouseMoveEvent = self.draw
        self.image_label.mouseReleaseEvent = self.stop_drawing

        main_layout.addWidget(self.image_label)

        # Save button
        save_button = QPushButton('Save Image', self)
        save_button.clicked.connect(self.save_image)
        button_layout.addWidget(save_button)

        # Color selection button
        color_button = QPushButton('Select Color', self)
        color_button.clicked.connect(self.select_color)
        button_layout.addWidget(color_button)

        main_layout.addLayout(button_layout)

        # Thickness slider
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(20)
        self.thickness_slider.setValue(self.current_thickness)
        self.thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.thickness_slider.setTickInterval(1)
        self.thickness_slider.valueChanged.connect(self.update_thickness)

        self.thickness_label = QLabel(f"Thickness: {self.current_thickness}")
        
        slider_layout.addWidget(QLabel("Thickness:"))
        slider_layout.addWidget(self.thickness_slider)
        slider_layout.addWidget(self.thickness_label)

        main_layout.addLayout(slider_layout)

        self.setLayout(main_layout)

    def start_drawing(self, event):
        self.drawing = True
        self.last_point = self.get_image_coordinates(event.pos())

    def draw(self, event):
        if self.drawing:
            current_point = self.get_image_coordinates(event.pos())
            painter = QPainter(self.q_image)
            painter.setPen(QPen(self.current_color, self.current_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update_image_label()

    def stop_drawing(self, event):
        self.drawing = False

    def get_image_coordinates(self, pos):
        label_size = self.image_label.size()
        image_size = self.q_image.size()
        x_offset = (label_size.width() - image_size.width()) // 2
        y_offset = (label_size.height() - image_size.height()) // 2

        x = pos.x() - x_offset
        y = pos.y() - y_offset

        x = max(0, min(x, image_size.width() - 1))
        y = max(0, min(y, image_size.height() - 1))

        return QPoint(x, y)

    def update_image_label(self):
        self.image_label.setPixmap(QPixmap.fromImage(self.q_image))

    def update_image_threaded(self):
        self.image_thread = ImageUpdateThread(self.rgb_image)
        self.image_thread.update_signal.connect(self.on_image_updated)
        self.image_thread.start()

    def on_image_updated(self, q_image):
        self.q_image = q_image
        self.update_image_label()

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
        if file_path:
            self.q_image.save(file_path)

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color

    def update_thickness(self, value):
        self.current_thickness = value
        self.thickness_label.setText(f"Thickness: {self.current_thickness}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DrawingApp()
    ex.show()
    sys.exit(app.exec_())
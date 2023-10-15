'''dahyun+darwin = dahwin'''
# with webcam
# with webcam
import sys
import datetime
from PIL import ImageGrab
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from win32api import GetSystemMetrics
from PyQt5 import QtCore, QtWidgets

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.width = GetSystemMetrics(0)
        self.height = GetSystemMetrics(1)
        self.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.file_name = f'{self.time_stamp}.mp4'
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.captured_video = cv2.VideoWriter(self.file_name, self.fourcc, 20.0, (self.width, self.height))

        self.ratul_webcam = cv2.VideoCapture(0)

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start(self):
        self.timer.start(10)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop(self):
        self.timer.stop()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.captured_video.release()

    def update_frame(self):
        img = ImageGrab.grab(bbox=(0, 0, self.width, self.height))
        img_np = np.array(img)
        img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        try:
            _, frame = self.ratul_webcam.read()
            fr_height, fr_width, _ = frame.shape
            img_final[0:fr_height, 0:fr_width:] = frame[0:fr_height, 0:fr_width, :]
            cv2.imshow('ratul_webcam', frame)
        except:
            None
        cv2.imshow('Dahwin', img_final)
        self.captured_video.write(img_final)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
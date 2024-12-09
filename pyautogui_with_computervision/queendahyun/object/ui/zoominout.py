import sys
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QPixmap, QImage, QBrush, QPen, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QVBoxLayout, QWidget

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.pixmap_item = None
        self.zoom_level = 1
        self._panStartX = 0
        self._panStartY = 0
        self._panStartXValue = 0
        self._panStartYValue = 0
        self._panning = False
        self._cursorX = 0
        self._cursorY = 0
        self._dragging = False
        self._dragStartX = 0
        self._dragStartY = 0

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image File", '.', "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.pixmap_item = QGraphicsPixmapItem(QPixmap(filename))
            self.scene.addItem(self.pixmap_item)
            
            # Scale the image to half of its original size
            self.pixmap_item.setScale(0.5)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self.zoom_level += 0.1
        self.resetTransform()
        self.scale(self.zoom_level, self.zoom_level)

    def zoom_out(self):
        if self.zoom_level > 0.1:
            self.zoom_level -= 0.1
            self.resetTransform()
            self.scale(self.zoom_level, self.zoom_level)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            self._panStartX = pos.x()
            self._panStartY = pos.y()
            self._panStartXValue = self.horizontalScrollBar().value()
            self._panStartYValue = self.verticalScrollBar().value()
            self._panning = True
            self._cursorX = pos.x()
            self._cursorY = pos.y()
            self._dragging = True
            self._dragStartX = pos.x()
            self._dragStartY = pos.y()

    def mouseMoveEvent(self, event):
        if self._panning:
            pos = event.position().toPoint()
            x = pos.x()
            y = pos.y()
            self.horizontalScrollBar().setValue(self._panStartXValue - (x - self._panStartX))
            self.verticalScrollBar().setValue(self._panStartYValue - (y - self._panStartY))
            self._cursorX = x
            self._cursorY = y

        if self._dragging:
            pos = event.position().toPoint()
            x = pos.x()
            y = pos.y()
            # Normalize the dragging speed by dividing the movement by the zoom level
            dx = (x - self._dragStartX) / self.zoom_level
            dy = (y - self._dragStartY) / self.zoom_level

            # Move the image by normalized deltas
            self.pixmap_item.setPos(self.pixmap_item.pos().x() + dx, self.pixmap_item.pos().y() + dy)
            
            # Update drag start coordinates
            self._dragStartX = x
            self._dragStartY = y

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self._dragging = False

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - 10)
        elif event.key() == Qt.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + 10)
        elif event.key() == Qt.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - 10)
        elif event.key() == Qt.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + 10)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.image_viewer = ImageViewer()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.setLayout(self.layout)

        self.image_viewer.open_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1920, 1080)
    window.move(1920 // 2 - window.width() // 2, 1080 // 2 - window.height() // 2)
    window.show()
    sys.exit(app.exec())
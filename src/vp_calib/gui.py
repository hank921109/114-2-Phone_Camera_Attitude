import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional, Tuple
from vp_calib.engine import main as calib_main, read_image

class CalibrationApp(QtWidgets.QMainWindow):
    """
    Main GUI application for Vanishing Point Camera Calibration.
    Refactored to avoid global variables and follow PEP 8.
    """
    def __init__(self):
        super(CalibrationApp, self).__init__()
        self.img_name: Optional[str] = None
        self.rows_prop: float = 1.0
        self.cols_prop: float = 1.0
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.camera_h: float = 0.0
        
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Vanishing Point Camera Calibration")
        self.resize(1250, 680)
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        
        # Left: Image Display
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 800, 600))
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setScaledContents(True)
        
        # Right: Controls
        self.btn_open = QtWidgets.QPushButton("Open Image", self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(815, 10, 430, 50))
        self.btn_open.clicked.connect(self.handle_open_image)
        
        self.btn_height = QtWidgets.QPushButton("Input Camera Height", self.centralwidget)
        self.btn_height.setGeometry(QtCore.QRect(815, 70, 430, 50))
        self.btn_height.clicked.connect(self.handle_input_height)
        
        self.lbl_height_val = QtWidgets.QLabel("Height: 0.0 m", self.centralwidget)
        self.lbl_height_val.setGeometry(QtCore.QRect(820, 125, 420, 30))
        self.lbl_height_val.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)

        self.lbl_pos_info = QtWidgets.QLabel("(Click image to set origin point)", self.centralwidget)
        self.lbl_pos_info.setGeometry(QtCore.QRect(820, 160, 430, 30))
        
        self.lbl_origin_coords = QtWidgets.QLabel("Origin (px): X=0.0, Y=0.0", self.centralwidget)
        self.lbl_origin_coords.setGeometry(QtCore.QRect(820, 195, 420, 30))
        self.lbl_origin_coords.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        
        self.btn_calibrate = QtWidgets.QPushButton("Start Calibration", self.centralwidget)
        self.btn_calibrate.setGeometry(QtCore.QRect(815, 250, 430, 50))
        self.btn_calibrate.clicked.connect(self.handle_calibration)
        
        # Results Section
        self.lbl_results_header = QtWidgets.QLabel("Results:", self.centralwidget)
        self.lbl_results_header.setGeometry(QtCore.QRect(820, 310, 430, 30))
        self.lbl_results_header.setStyleSheet("font-weight: bold;")
        
        self.lbl_focal = QtWidgets.QLabel("Focal (px): -", self.centralwidget)
        self.lbl_focal.setGeometry(QtCore.QRect(820, 345, 425, 30))
        self.lbl_focal.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        
        self.lbl_rotation = QtWidgets.QLabel("Rotation Matrix:\n-", self.centralwidget)
        self.lbl_rotation.setGeometry(QtCore.QRect(820, 385, 425, 120))
        self.lbl_rotation.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self.lbl_rotation.setWordWrap(True)
        
        self.lbl_translation = QtWidgets.QLabel("Translation (m): -", self.centralwidget)
        self.lbl_translation.setGeometry(QtCore.QRect(820, 515, 425, 30))
        self.lbl_translation.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

    def handle_open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose Image", "data/samples/", "*.jpg;;*.png;;All Files(*)"
        )
        if not path:
            return
            
        self.img_name = path
        pixmap = QtGui.QPixmap(path).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(pixmap)
        
        image = read_image(path)
        if image is not None:
            rows, cols = image.shape[:2]
            self.rows_prop = rows / self.label_image.height()
            self.cols_prop = cols / self.label_image.width()
            self.label_image.mousePressEvent = self.handle_mouse_click
            self.statusbar.showMessage(f"Loaded: {path}")

    def handle_input_height(self):
        val, ok = QtWidgets.QInputDialog.getDouble(self, "Camera Height", "Input height (meters):", 1.5, 0, 100, 2)
        if ok:
            self.camera_h = val
            self.lbl_height_val.setText(f"Height: {val:.2f} m")

    def handle_mouse_click(self, event):
        self.origin_x = event.pos().x() * self.cols_prop
        self.origin_y = event.pos().y() * self.rows_prop
        self.lbl_origin_coords.setText(f"Origin (px): X={self.origin_x:.1f}, Y={self.origin_y:.1f}")

    def handle_calibration(self):
        if not self.img_name:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        if self.camera_h <= 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please input a valid camera height!")
            return
            
        self.statusbar.showMessage("Calibrating... please wait.")
        QtWidgets.QApplication.processEvents()
        
        try:
            focal, rot_matrix, trans_vec = calib_main(
                self.img_name, self.origin_x, self.origin_y, self.camera_h, iterations=1000
            )
            
            self.lbl_focal.setText(f"Focal (px): {focal:.2f}")
            self.lbl_rotation.setText(f"Rotation Matrix:\n{np.array2string(rot_matrix, precision=4)}")
            self.lbl_translation.setText(f"Translation (m): {np.array2string(trans_vec, precision=4)}")
            self.statusbar.showMessage("Calibration complete.")
            
            # Show a message box with the result
            QtWidgets.QMessageBox.information(self, "Success", f"Calibration Finished!\nFocal: {focal:.1f} px")
            
        except Exception as e:
            self.statusbar.showMessage("Calibration failed.")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during calibration:\n{str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CalibrationApp()
    window.show()
    sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog, QVBoxLayout, QMessageBox, QWidget, \
    QHBoxLayout, QFileDialog
import subprocess
import cv2
import os
from ultralytics import YOLO
import cv2
import tensorflow as tf

class MyApp(QWidget):
    window_size_x = 300
    window_size_y = 300
    video_path = r"C:\Users\USER\Downloads\Harvesting Palm Oil Using a Machine.mp4"
    model = YOLO(r"C:\Users\USER\source\repos\YOLO\model\best_prev.pt")
    cap = cv2.VideoCapture(video_path)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, self.window_size_x, self.window_size_y)
        self.setWindowTitle('Let\'s find some Palm Oil Fruits!')

        detection_btn = QPushButton('Detection', self)
        detection_btn.setToolTip('Click to show a message box')
        detection_btn.move(self.window_size_x // 3, self.window_size_y // 3)
        detection_btn.clicked.connect(self.capture_video)

        segmentation_btn = QPushButton('Segmentation', self)
        segmentation_btn.setToolTip('Click to open segmentation window')
        segmentation_btn.move(self.window_size_x // 3, self.window_size_y // 3 + int(0.2 * self.window_size_y))
        segmentation_btn.clicked.connect(self.open_new_window)

    def showMessageBox(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Message Box')
        msg_box.setText('You clicked the Detection button!')
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def process_frame(self, frame):
        # Predict objects in the frame
        predictions = self.model.predict(frame, show=True, conf=0.5)

        print(predictions[0].boxes.conf)
        arr = predictions[0].boxes.conf.numpy()
        if arr.size > 0:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def capture_video(self):

        while True:
            # Read a frame from the video
            ret, frame = self.cap.read()

            if not ret:
                break  # Break the loop if no frame is captured

            # Process the frame to detect objects
            self.process_frame(frame)

        self.cap.release()  # Release the video capture object

    def open_new_window(self):
        self.new_window = NewWindow()
        self.new_window.show()


class NewWindow(QDialog):
    window_size_x = 300
    window_size_y = 300

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, self.window_size_x, self.window_size_y)
        self.setWindowTitle('Segmentation')

        # Create a vertical layout for the dialog window
        main_layout = QVBoxLayout()

        # Create a horizontal layout for positioning the "Close" button
        top_layout = QHBoxLayout()

        # Add a stretchable space to push the "Close" button to the right
        top_layout.addStretch(1)

        # Create the "Close" button and add it to the horizontal layout
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.close)
        top_layout.addWidget(close_btn)

        # Add the horizontal layout to the main vertical layout
        main_layout.addLayout(top_layout)

        # Create buttons for Mask and Rodent and add them to the main layout
        mask_btn = QPushButton('Mask')
        mask_btn.setToolTip('Click to show a message box')
        mask_btn.clicked.connect(self.openFileDialog)
        main_layout.addWidget(mask_btn)

        rodent_btn = QPushButton('Rodent')
        rodent_btn.setToolTip('Click to open segmentation window')
        rodent_btn.clicked.connect(self.showRodentMessageBox)
        main_layout.addWidget(rodent_btn)

        # Set the main layout for the dialog window
        self.setLayout(main_layout)

    def showRodentMessageBox(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Message Box')
        msg_box.setText('You clicked the Rodent button!')
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def showSegmentMaskMessageBox(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Message Box')
        msg_box.setText('You clicked the Mask button!')
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def openFileDialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose a file", "", "All Files (*);;Python Files (*.py)",
                                                   options=options)

        if file_path:
            self.file_path_label.setText(f'Selected File: {file_path}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())

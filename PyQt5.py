import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedLayout,QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import subprocess
import cv2
import os
from ultralytics import YOLO
import cv2

class MainWindow(QWidget):
    video_path = r"C:\Users\USER\Downloads\Harvesting Palm Oil Using a Machine.mp4"
    model = YOLO(r"C:\Users\USER\source\repos\YOLO\model\best_prev.pt")
    cap = cv2.VideoCapture(video_path)
    continue_detection = True
    segmentation_data_path = "frames/frame.jpg"

    def __init__(self):
        super().__init__()

        # Initialize the window properties
        self.setWindowTitle("Borderless Window with Key Toggle")
        self.setGeometry(0, 0, 400, 300)
        screen_geometry = QDesktopWidget().screenGeometry()

        # Calculate window position
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2

        # Set window position
        self.move(x, y)
        self.stacked_layout = QStackedLayout()
        self.create_initial_layout()
        # Create stacked layout
        self.stacked_layout = QStackedLayout()

        # Create initial layout
        self.create_initial_layout()

        # Create new layout
        self.create_detection_layout()
        self.create_segmentation_layout()

        # Add layouts to stacked layout
        self.stacked_layout.addWidget(self.initial_widget)
        self.stacked_layout.addWidget(self.detection_widget)
        self.stacked_layout.addWidget(self.segmentation_widget)

        # Set stacked layout for the window
        self.setLayout(self.stacked_layout)

    def create_initial_layout(self):
        """Create initial layout"""
        detection_btn = QPushButton("Detection")
        segmentation_btn = QPushButton("Segmentation")

        detection_btn.setFixedSize(300, 200)
        segmentation_btn.setFixedSize(300, 200)

        detection_btn.setStyleSheet("""
                            background-color: #4CAF50;  /* Green background color */
                            color: white;               /* White text color */
                            font-size: 30px;            /* Font size */
                            font-family: Arial;         /* Font family */
                            padding: 10px 20px;         /* Padding */
                            border-radius: 25px;        /* Rounded corners */
                        """)

        segmentation_btn.setStyleSheet("""
                                    background-color: #941010;  /* Green background color */
                                    color: white;               /* White text color */
                                    font-size: 30px;            /* Font size */
                                    font-family: Arial;         /* Font family */
                                    padding: 10px 20px;         /* Padding */
                                    border-radius: 25px;        /* Rounded corners */
                                """)

        detection_btn.clicked.connect(self.capture_video)
        segmentation_btn.clicked.connect(self.switch_to_segmentation_layout)

        # Create a QHBoxLayout
        h_layout = QHBoxLayout()
        h_layout.addWidget(detection_btn)
        h_layout.addWidget(segmentation_btn)

        self.initial_widget = QWidget()
        self.initial_widget.setLayout(h_layout)


    def create_detection_layout(self):
            """Create new layout"""
            self.detection_widget = QWidget()
            layout = QHBoxLayout()
            back_button = QPushButton("Go Back")
            segment_button = QPushButton("Segment")
            segment_button.setFixedSize(300,200)
            back_button.setFixedSize(300, 200)
            segment_button.setStyleSheet("""
                                        background-color: #4CAF50;  /* Green background color */
                                        color: white;               /* White text color */
                                        font-size: 30px;            /* Font size */
                                        font-family: Arial;         /* Font family */
                                        padding: 10px 20px;         /* Padding */
                                        border-radius: 25px;        /* Rounded corners */
                                    """)

            back_button.setStyleSheet("""
                                                background-color: #941010;  /* Green background color */
                                                color: white;               /* White text color */
                                                font-size: 30px;            /* Font size */
                                                font-family: Arial;         /* Font family */
                                                padding: 10px 20px;         /* Padding */
                                                border-radius: 25px;        /* Rounded corners */
                                            """)

            segment_button.clicked.connect(self.put_detection_frame)
            back_button.clicked.connect(self.back_to_initial_layout)
            layout.addWidget(back_button)
            layout.addWidget(segment_button)

            self.detection_widget.setLayout(layout)

    def create_segmentation_layout(self):
            """Create new layout"""
            self.segmentation_widget = QWidget()
            btn_layout = QHBoxLayout()
            detected_pic_layout = QVBoxLayout()
            segmented_pic_layout = QVBoxLayout()
            pics_layout = QHBoxLayout()

            back_button = QPushButton("Go Back")
            back_button.setFixedSize(300, 200)
            choose_button = QPushButton("Choose Pic")
            choose_button.setFixedSize(300, 200)
            detected_pic = QLabel(self)
            detected_label = QLabel("Detected frame")
            pixmap = QPixmap("frames/frame.jpg")  # Replace with your image file path
            detected_pic.setPixmap(pixmap)
            detected_pic_layout.addWidget(detected_pic)
            detected_pic_layout.addWidget(detected_label)

            segmented_pic = QLabel(self)
            segmented_label = QLabel("Segmented frame")
            pixmap = QPixmap("frames/frame.jpg")  # Replace with your image file path
            segmented_pic.setPixmap(pixmap)
            segmented_pic_layout.addWidget(segmented_pic)
            segmented_pic_layout.addWidget(segmented_label)

            back_button.setStyleSheet("""
                                                background-color: #941010;  /* Green background color */
                                                color: white;               /* White text color */
                                                font-size: 30px;            /* Font size */
                                                font-family: Arial;         /* Font family */
                                                padding: 10px 20px;         /* Padding */
                                                border-radius: 25px;        /* Rounded corners */
                                            """)

            choose_button.setStyleSheet("""
                                                            background-color: #941010;  /* Green background color */
                                                            color: white;               /* White text color */
                                                            font-size: 30px;            /* Font size */
                                                            font-family: Arial;         /* Font family */
                                                            padding: 10px 20px;         /* Padding */
                                                            border-radius: 25px;        /* Rounded corners */
                                                        """)
            back_button.clicked.connect(self.back_to_initial_layout)
            btn_layout.addWidget(back_button)
            btn_layout.addWidget(choose_button)

            buttons_widget = QWidget()  # Create a QWidget to hold the QHBoxLayout
            buttons_widget.setLayout(btn_layout)  # Set the QHBoxLayout as the layout for the QWidget

            detected_pics_widget = QWidget()
            detected_pics_widget.setLayout(detected_pic_layout)

            segmented_pics_widget = QWidget()
            segmented_pics_widget.setLayout(segmented_pic_layout)

            pics_layout.addWidget(detected_pics_widget)
            pics_layout.addWidget(segmented_pics_widget)

            pics_widget = QWidget()
            pics_widget.setLayout(pics_layout)
            segmentation_layout = QVBoxLayout()
            # Add widgets to the QVBoxLayout
            segmentation_layout.addWidget(pics_widget, alignment=Qt.AlignCenter)
            segmentation_layout.addWidget(buttons_widget, alignment=Qt.AlignCenter)


            self.segmentation_widget.setLayout(segmentation_layout)

    def put_detection_frame(self):
        cv2.destroyAllWindows()
        self.continue_detection = False
        self.switch_to_segmentation_layout()

    def switch_to_detection_layout(self):
        """Switch layout to new layout"""
        self.stacked_layout.setCurrentIndex(1)  # Show new layout

    def switch_to_segmentation_layout(self):
        """Switch layout to new layout"""
        self.stacked_layout.setCurrentIndex(2)  # Show new layout

    def back_to_initial_layout(self):
        """Switch layout back to initial layout"""
        self.stacked_layout.setCurrentIndex(0)  # Show initial layout

    def capture_video(self):
        self.continue_detection = True
        self.switch_to_detection_layout()
        cap = cv2.VideoCapture(self.video_path)
        while self.continue_detection:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break  # Break the loop if no frame is captured

            # Process the frame to detect objects
            self.process_frame(frame)

        cap.release()  # Release the video capture object

    def process_frame(self, frame):
        # Predict objects in the frame
        predictions = self.model.predict(frame, show=True, conf=0.5)

        print(predictions[0].boxes.conf)
        arr = predictions[0].boxes.conf.numpy()
        if arr.size > 0:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowFlags(window.windowFlags() | 0x00000001)  # Set flag for borderless window
    window.show()
    sys.exit(app.exec_())

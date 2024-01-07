import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedLayout,QDesktopWidget, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from UNET import Unet
from ultralytics import YOLO
import cv2

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget

detected_frame_path = "detection/detected_frame.jpg"
mask_path = "detection/mask.jpg"
masked_img_path = "detection/masked_img.jpg"

class Segmentation(QWidget):

    def __init__(self, parent=None):
        global detected_frame_path, mask_path, masked_img_path
        self.model_path = r"C:\Users\USER\source\repos\Pytorch-UNet\checkpoints\checkpoint_epoch145.pth"
        self.in_files = [detected_frame_path]
        self.out_files = ["detection/mask.jpg"]
        super(Segmentation, self).__init__(parent)

        layout = QVBoxLayout()
        """Create new layout"""
        self.segmentation_widget = QWidget()
        btn_layout = QHBoxLayout()
        pic_layout = QHBoxLayout()
        pics_layout = QHBoxLayout()

        rodent_button = QPushButton("Rodent")
        rodent_button.setFixedSize(300, 200)
        segment_palm_oil_button = QPushButton("Palm Oil")
        segment_palm_oil_button.setFixedSize(300, 200)
        self.detected_pic = QLabel(self)
        detected_label = QLabel("Input frame")
        pixmap = QPixmap(detected_frame_path)  # Replace with your image file path
        self.detected_pic.setPixmap(pixmap)
        pic_layout.addWidget(self.detected_pic)
        pic_layout.addWidget(detected_label)

        segment_palm_oil_button.setStyleSheet("""
                                                                    background-color: #941010;  /* Green background color */
                                                                    color: white;               /* White text color */
                                                                    font-size: 30px;            /* Font size */
                                                                    font-family: Arial;         /* Font family */
                                                                    padding: 10px 20px;         /* Padding */
                                                                    border-radius: 25px;        /* Rounded corners */
                                                                """)

        rodent_button.setStyleSheet("""
                                                                            background-color: #941010;  /* Green background color */
                                                                            color: white;               /* White text color */
                                                                            font-size: 30px;            /* Font size */
                                                                            font-family: Arial;         /* Font family */
                                                                            padding: 10px 20px;         /* Padding */
                                                                            border-radius: 25px;        /* Rounded corners */
                                                                        """)
        btn_layout.addWidget(segment_palm_oil_button)
        btn_layout.addWidget(rodent_button)
        rodent_button.clicked.connect(self.show_file_dialog)
        segment_palm_oil_button.clicked.connect(self.show_file_dialog)

        self.mask = QLabel(self)
        mask_label = QLabel("Mask")
        pixmap = QPixmap(mask_path)  # Replace with your image file path
        self.mask.setPixmap(pixmap)
        pic_layout.addWidget(self.mask)
        pic_layout.addWidget(mask_label)

        self.segmented_pic = QLabel(self)
        segmented_label = QLabel("Segmented frame")
        pixmap = QPixmap(masked_img_path)  # Replace with your image file path
        self.segmented_pic.setPixmap(pixmap)
        pic_layout.addWidget(self.segmented_pic)
        layout.addWidget(segmented_label)

        buttons_widget = QWidget()  # Create a QWidget to hold the QHBoxLayout
        buttons_widget.setLayout(btn_layout)  # Set the QHBoxLayout as the layout for the QWidget

        detected_pics_widget = QWidget()
        detected_pics_widget.setLayout(pic_layout)

        segmented_pics_widget = QWidget()
        segmented_pics_widget.setLayout(pic_layout)

        pics_layout.addWidget(detected_pics_widget)
        pics_layout.addWidget(segmented_pics_widget)

        pics_widget = QWidget()
        pics_widget.setLayout(pics_layout)
        segmentation_layout = QVBoxLayout()
        # Add widgets to the QVBoxLayout
        segmentation_layout.addWidget(pics_widget, alignment=Qt.AlignCenter)
        segmentation_layout.addWidget(buttons_widget, alignment=Qt.AlignCenter)

        self.segmentation_widget.setLayout(segmentation_layout)

        self.setLayout(segmentation_layout)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)

    def performSegmentation(self):
        unet = Unet(self.model_path, self.in_files, self.out_files)
        unet.run()

    def performSegmentation(self, in_file):
        unet = Unet(self.model_path, in_file, self.out_files)
        unet.run(in_file)

    def show_file_dialog(self):
        global detected_frame_path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                   options=options)

        if file_path:
            self.performSegmentation(file_path)
            # Load the selected image using QPixmap
            input = cv2.imread(file_path)
            cv2.imwrite(detected_frame_path, input)
            detected_frame_path = file_path
            pixmap = QPixmap(file_path)  # Replace with your image file path
            self.detected_pic.setPixmap(pixmap)
            masked_image = cv2.bitwise_and(cv2.imread(detected_frame_path), cv2.imread(mask_path))
            cv2.imwrite(masked_img_path, masked_image)
            pixmap = QPixmap(masked_img_path)  # Replace with your image file path
            self.segmented_pic.setPixmap(pixmap)

            pixmap = QPixmap(mask_path)  # Replace with your image file path
            self.mask.setPixmap(pixmap)

class Detection(QWidget):
    video_path = r"C:\Users\USER\Downloads\Harvesting Palm Oil Using a Machine.mp4"
    model = YOLO(r"C:\Users\USER\source\repos\YOLO\model\best_prev.pt")
    cap = cv2.VideoCapture(video_path)
    continue_detection = True
    detected_frame_path = "detection/detected_frame.jpg"

    def __init__(self, parent=None):
        super(Detection, self).__init__(parent)

        """Create new layout"""
        self.detection_widget = QWidget()
        self.layout = QVBoxLayout()
        self.btn_layout = QHBoxLayout()
        segment_button = QPushButton("Segment")
        detect_button = QPushButton("Detect")
        segment_button.setFixedSize(300, 200)
        detect_button.setFixedSize(300, 200)
        segment_button.setStyleSheet("""
                                                background-color: #4CAF50;  /* Green background color */
                                                color: white;               /* White text color */
                                                font-size: 30px;            /* Font size */
                                                font-family: Arial;         /* Font family */
                                                padding: 10px 20px;         /* Padding */
                                                border-radius: 25px;        /* Rounded corners */
                                            """)


        detect_button.setStyleSheet("""
                                                                background-color: #941010;  /* Green background color */
                                                                color: white;               /* White text color */
                                                                font-size: 30px;            /* Font size */
                                                                font-family: Arial;         /* Font family */
                                                                padding: 10px 20px;         /* Padding */
                                                                border-radius: 25px;        /* Rounded corners */
                                                            """)

        detect_button.clicked.connect(self.capture_video)
        segment_button.clicked.connect(self.put_detection_frame)

        self.btn_layout.addWidget(segment_button)
        self.btn_layout.addWidget(detect_button)
        self.btn_widget = QWidget()
        self.btn_widget.setLayout(self.btn_layout)

        self.original_pic = QLabel(self)
        self.original_img_pixmap = QPixmap()  # Replace with your image file path
        self.original_pic.setPixmap(self.original_img_pixmap)

        self.mask = QLabel(self)
        self.mask_img = QPixmap()  # Replace with your image file path
        self.mask.setPixmap(self.mask_img)

        self.segmented_pic = QLabel(self)
        self.masked_img_pixmap = QPixmap()  # Replace with your image file path
        self.segmented_pic.setPixmap(self.masked_img_pixmap)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.original_pic)
        self.h_layout.addWidget(self.segmented_pic)
        self.h_layout.addWidget(self.mask)

        self.pics_widget = QWidget()
        self.pics_widget.setLayout(self.h_layout)

        self.layout.addWidget(self.pics_widget, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.btn_widget, alignment=Qt.AlignCenter)
        self.setLayout(self.layout)

    def capture_video(self):
        self.continue_detection = True
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
            cv2.imwrite(detected_frame_path, frame)
            cv2.waitKey(0)

    def put_detection_frame(self):
        cv2.destroyAllWindows()
        self.continue_detection = False
        self.original_img_pixmap = QPixmap(detected_frame_path)
        self.original_pic.setPixmap(self.original_img_pixmap)
        segmentation = Segmentation()
        segmentation.performSegmentation(detected_frame_path)
        masked_image = cv2.bitwise_and(cv2.imread(detected_frame_path), cv2.imread(mask_path))
        cv2.imwrite(masked_img_path, masked_image)

        self.mask_pixmap = QPixmap(mask_path)  # Replace with your image file path
        self.mask.setPixmap(self.mask_pixmap)

        self.masked_img_pixmap = QPixmap(masked_img_path)  # Replace with your image file path
        self.segmented_pic.setPixmap(self.masked_img_pixmap)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.stacked_widget = QStackedWidget()

        self.segmentation = Segmentation(self)
        self.detection = Detection(self)

        self.stacked_widget.addWidget(self.segmentation)
        self.stacked_widget.addWidget(self.detection)

        self.btn_to_segmentation = QPushButton("Segmentation")
        self.btn_to_segmentation.clicked.connect(self.switch_to_segmentation_window)

        self.btn_to_detection = QPushButton("Detection")
        self.btn_to_detection.clicked.connect(self.switch_to_detection_window)

        self.btn_to_segmentation.setStyleSheet("""
                                                        background-color: #000000;  /* Green background color */
                                                        color: white;               /* White text color */
                                                        font-size: 30px;            /* Font size */
                                                        font-family: Arial;         /* Font family */
                                                        padding: 10px 20px;         /* Padding */
                                                        border-radius: 25px;        /* Rounded corners */
                                                        
                                                    """)
        self.btn_to_segmentation.setObjectName("hoverButton")

        self.btn_to_detection.setStyleSheet("""
                                                                background-color: #000000;  /* Green background color */
                                                                color: white;               /* White text color */
                                                                font-size: 30px;            /* Font size */
                                                                font-family: Arial;         /* Font family */
                                                                padding: 10px 20px;         /* Padding */
                                                                border-radius: 25px;        /* Rounded corners */
                                                            """)

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_to_segmentation)
        btn_layout.addWidget(self.btn_to_detection)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        layout.addWidget(btn_widget)
        layout.addWidget(self.stacked_widget)

        self.setLayout(layout)

    def switch_to_segmentation_window(self):
        self.stacked_widget.setCurrentWidget(self.segmentation)

    def switch_to_detection_window(self):
        self.stacked_widget.setCurrentWidget(self.detection)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('Main Window with Two Buttons')
    window.show()
    sys.exit(app.exec_())


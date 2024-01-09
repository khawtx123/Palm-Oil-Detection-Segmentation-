import functools
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QHBoxLayout, QFileDialog, QMessageBox, QDesktopWidget, QMainWindow,QLineEdit, QDialog
from UNET import Unet
from ultralytics import YOLO
from roboflow import Roboflow
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import math

DETECTION_MODEL_PATH = r"haarcascasde_files/haarcascade_frontalface_default.xml"
EMOTION_MODEL_PATH = 'model/_mini_XCEPTION.102-0.66.hdf5'
DETECTED_FRAME_PATH = "detection/detected_frame.jpg"
MASK_PATH = "detection/mask.jpg"
MASKED_IMG_PATH = "detection/masked_img.jpg"

DETECTED_RODENT_HOLE_PATH = "detection/detected_rodent_hole.jpg"
DETECTED_RODENT_BITE_MARKS_PATH = "detection/detected_rodent_bite_marks.jpg"

SEGMENT_RODENT_HOLE_MASK_PATH = "detection/segmented_rodent_hole_mask.jpg"
SEGMENT_RODENT_BITE_MARKS_MASK_PATH = "detection/segmented_rodent_bite_mask_marks.jpg"
SEGMENT_RODENT_HOLE_PATH = "detection/segmented_rodent_hole.jpg"
SEGMENT_RODENT_BITE_MARKS_PATH = "detection/segmented_rodent_bite_marks.jpg"

FEEDBACK_IMG_PATH = "ratings/feedback.jpg"
FEEDBACK_PREDS_IMG_PATH = "ratings/feedback_preds.jpg"
PREDS_DATA_PATH = "ratings/predictions.jpg"
ICON_PATH = "icon/HarvestMate.png"
VIDEO_SAD_PATH = "videos/sad.mp4"
DEFAULT_STYLE_SHEET = """
                                                                    background-color: #000000;  /* Green background color */
                                                                    color: white;               /* White text color */
                                                                    font-size: 30px;            /* Font size */
                                                                    font-family: Arial;         /* Font family */
                                                                    padding: 10px 20px;         /* Padding */
                                                                    border-radius: 25px;        /* Rounded corners */
                                                                """
ON_CLICK_STYLE_SHEET = """
                                                                    background-color: #4CAF50;  /* Green background color */
                                                                    color: white;               /* White text color */
                                                                    font-size: 30px;            /* Font size */
                                                                    font-family: Arial;         /* Font family */
                                                                    padding: 10px 20px;         /* Padding */
                                                                    border-radius: 25px;        /* Rounded corners */
                                                                """

DEFAULT_WIDGET_STYLE_SHEET = """
                                                                    background-color: #941010;  /* Green background color */
                                                                    color: white;               /* White text color */
                                                                    font-size: 30px;            /* Font size */
                                                                    font-family: Arial;         /* Font family */
                                                                    padding: 10px 20px;         /* Padding */
                                                                    border-radius: 25px;        /* Rounded corners */
                                                                """

class Segmentation(QWidget):
    def __init__(self, parent=None):
        self.stacked_widget = QStackedWidget()
        global DETECTED_FRAME_PATH, MASK_PATH, MASKED_IMG_PATH
        self.model_path = r"C:\Users\USER\source\repos\Pytorch-UNet\checkpoints\checkpoint_epoch145.pth"
        self.in_files = [DETECTED_FRAME_PATH]
        self.out_files = ["detection/mask.jpg"]
        super(Segmentation, self).__init__(parent)

        self.segmentation_widget = QWidget()
        self.btn_layout = QHBoxLayout()
        self.pic_layout = QHBoxLayout()

        self.btn_rodent = QPushButton("Rodent")
        self.btn_segment_palm_oil = QPushButton("Palm Oil")
        self.btn_rodent.setFixedSize(300, 200)
        self.btn_segment_palm_oil.setFixedSize(300, 200)

        self.btn_segment_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)

        self.btn_rodent.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)

        self.btn_layout.addWidget(self.btn_segment_palm_oil)
        self.btn_layout.addSpacing(70)
        self.btn_layout.addWidget(self.btn_rodent)

        self.btn_rodent.clicked.connect(self.segment_rodent)
        self.btn_segment_palm_oil.clicked.connect(self.segment_palm_oil)

        self.detected_pic = QLabel(self)
        pixmap = QPixmap(DETECTED_FRAME_PATH)  # Replace with your image file path
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.detected_pic.setPixmap(scaled_pixmap)
        self.detected_pic.setStyleSheet('''
                    border: 4px solid black;  
                            padding: 5px;          
                ''')

        self.mask = QLabel(self)
        pixmap = QPixmap(MASK_PATH)  # Replace with your image file path
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.mask.setPixmap(scaled_pixmap)
        self.mask.setStyleSheet('''
                            border: 3px solid black;      
            border-radius: 8px;             
            padding: 5px;                
            background-color: #ecf0f1;      
            box-shadow: 0px 0px 5px #888888; 
                        ''')

        self.segmented_pic = QLabel(self)
        pixmap = QPixmap(MASKED_IMG_PATH)  # Replace with your image file path
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.segmented_pic.setPixmap(scaled_pixmap)
        self.segmented_pic.setStyleSheet('''
                            border: 4px solid black;  /* Add a 2px solid red border */
                            padding: 5px;           /* Add some padding around the image */
                        ''')

        self.pic_layout.addWidget(self.detected_pic)
        detected_label = QLabel("→")
        detected_label.setStyleSheet("""
                font-size: 48px;     /* Set font size */
                color: black;       /* Set text color */
            """)
        self.pic_layout.addWidget(detected_label)
        self.pic_layout.addWidget(self.mask)
        self.pic_layout.addSpacing(70)
        self.pic_layout.addWidget(self.segmented_pic)


        buttons_widget = QWidget()  # Create a QWidget to hold the QHBoxLayout
        buttons_widget.setLayout(self.btn_layout)  # Set the QHBoxLayout as the layout for the QWidget


        pics_widget = QWidget()
        pics_widget.setLayout(self.pic_layout)
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

    def segment_palm_oil(self):
        self.btn_segment_palm_oil.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        global DETECTED_FRAME_PATH
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)',
                                                       options=options)
            QMessageBox.information(self, 'File Selected', f'Selected file path: {file_path}')
            self.performSegmentation(file_path)
            # Load the selected image using QPixmap
            input = cv2.imread(file_path)
            cv2.imwrite(DETECTED_FRAME_PATH, input)
            pixmap = QPixmap(file_path)  # Replace with your image file path
            self.detected_pic.setPixmap(pixmap)
            masked_image = cv2.bitwise_and(cv2.imread(DETECTED_FRAME_PATH), cv2.imread(MASK_PATH))
            cv2.imwrite(MASKED_IMG_PATH, masked_image)
            pixmap = QPixmap(MASKED_IMG_PATH)  # Replace with your image file path
            self.segmented_pic.setPixmap(pixmap)

            pixmap = QPixmap(MASK_PATH)  # Replace with your image file path
            self.mask.setPixmap(pixmap)
            if not file_path:  # Check if no file is selected
                raise Exception("No files selected")  # Raise an exception if no file is selected



        except Exception as e:
            if str(e) == "No files selected":  # Handle the specific exception
                QMessageBox.warning(self, 'Warning', 'No files selected.')
            else:
                QMessageBox.critical(self, 'Error', f'An error occurred: No files selected !')

    def segment_rodent(self):
        self.btn_segment_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(ON_CLICK_STYLE_SHEET)

        rf = Roboflow(api_key="SOTkBQnU3ZURHAp0gJtr")
        project = rf.workspace().project("rodent-palm-oil")
        model = project.version(1).model

        image_path = self.show_file_dialog()
        predictions = model.predict(image_path, confidence=60).json()
        hole_image = cv2.imread(image_path)
        bite_marks_image = cv2.imread(image_path)
        height, width = hole_image.shape[:2]
        hole_mask = np.zeros((height, width))
        bite_marks_mask = np.zeros((height, width))
        for prediction in predictions["predictions"]:
            if prediction['class'] == "holes":
                coordinates = []
                for point in prediction['points']:
                    coordinates.append([int(point['x']), int(point['y'])])
                coordinates = np.array(coordinates)
                contours = [coordinates.reshape((-1, 1, 2)).astype(np.int32)]

                # Draw contours on the empty image
                cv2.drawContours(hole_mask, contours, -1, 255, -1)
                # w = int(prediction['width'])
                # h = int(prediction['height'])
                # x = int(prediction['x'] - w / 2)
                # y = int(prediction['y'] - h / 2)
                # class_name = prediction['class']
                # confidence = prediction['confidence']
                # cv2.rectangle(hole_image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                # cv2.putText(hole_image, f"{class_name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                #             (0, 255, 0), 1)

            else:
                coordinates = []
                for point in prediction['points']:
                    coordinates.append([int(point['x']), int(point['y'])])
                coordinates = np.array(coordinates)
                contours = [coordinates.reshape((-1, 1, 2)).astype(np.int32)]

                cv2.drawContours(bite_marks_mask, contours, -1, 255, -1)
                # w = int(prediction['width'])
                # h = int(prediction['height'])
                # x = int(prediction['x'] - w / 2)
                # y = int(prediction['y'] - h / 2)
                # class_name = prediction['class']
                # confidence = prediction['confidence']
                # cv2.rectangle(hole_image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                # cv2.putText(hole_image, f"{class_name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                #             (0, 255, 0), 1)

        try:
            cv2.imwrite(SEGMENT_RODENT_HOLE_MASK_PATH, hole_mask)
            masked_hole_image = cv2.bitwise_and(hole_image, cv2.imread(SEGMENT_RODENT_HOLE_MASK_PATH))
            cv2.imwrite(SEGMENT_RODENT_HOLE_PATH, masked_hole_image)
        except:
            QMessageBox.critical(self, 'Error', f'An error occurred: No holes detected !')

        try:
            cv2.imwrite(SEGMENT_RODENT_BITE_MARKS_MASK_PATH, bite_marks_mask)
            masked_bite_marks_image = cv2.bitwise_and(bite_marks_image, cv2.imread(SEGMENT_RODENT_BITE_MARKS_MASK_PATH))
            cv2.imwrite(SEGMENT_RODENT_BITE_MARKS_PATH, masked_bite_marks_image)
        except:
            QMessageBox.critical(self, 'Error', f'An error occurred: No bite marks detected !')

        if predictions["predictions"] is not None:
            cv2.destroyAllWindows()

            self.original_img_pixmap = QPixmap(image_path)
            self.detected_pic.setPixmap(self.original_img_pixmap)

            self.masked_img_pixmap = QPixmap(SEGMENT_RODENT_HOLE_PATH)  # Replace with your image file path
            scaled_pixmap = self.masked_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.segmented_pic.setPixmap(scaled_pixmap)

            self.mask_pixmap = QPixmap(SEGMENT_RODENT_BITE_MARKS_PATH)  # Replace with your image file path
            scaled_mask_pixmap = self.mask_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.mask.setPixmap(scaled_mask_pixmap)

    def show_segmentation(self):
        file_path = DETECTED_FRAME_PATH
        self.performSegmentation(file_path)
        # Load the selected image using QPixmap
        input = cv2.imread(file_path)
        cv2.imwrite(DETECTED_FRAME_PATH, input)
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.detected_pic.setPixmap(scaled_pixmap)

        masked_image = cv2.bitwise_and(cv2.imread(DETECTED_FRAME_PATH), cv2.imread(MASK_PATH))
        cv2.imwrite(MASKED_IMG_PATH, masked_image)

        pixmap = QPixmap(MASKED_IMG_PATH)  # Replace with your image file path
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.segmented_pic.setPixmap(scaled_pixmap)

        pixmap = QPixmap(MASK_PATH)  # Replace with your image file path
        scaled_pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.mask.setPixmap(scaled_pixmap)

    def show_file_dialog(self):
        global DETECTED_FRAME_PATH
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)',
                                                       options=options)

            if not file_path:  # Check if no file is selected
                raise Exception("No files selected")  # Raise an exception if no file is selected

            QMessageBox.information(self, 'File Selected', f'Selected file path: {file_path}')
            return file_path

        except Exception as e:
            if str(e) == "No files selected":  # Handle the specific exception
                QMessageBox.warning(self, 'Warning', 'No files selected.')
            else:
                QMessageBox.critical(self, 'Error', f'An error occurred: No files selected !')

class Detection(QWidget):
    video_path = r"videos/Harvesting Palm Oil Using a Machine.mp4"
    model = YOLO(r"C:\Users\USER\source\repos\YOLO\model\7_jan_palm_oil.pt")
    cap = cv2.VideoCapture(video_path)
    continue_detection = True

    def __init__(self, parent=None):
        super(Detection, self).__init__(parent)
        self.stacked_widget = QStackedWidget()
        self.layout = QVBoxLayout()
        self.btn_layout = QHBoxLayout()
        self.btn_segment = QPushButton("Segment")
        self.btn_palm_oil = QPushButton("Palm Oil")
        self.btn_rodent = QPushButton("Rodent")

        self.btn_segment.setFixedSize(300, 200)
        self.btn_palm_oil.setFixedSize(300, 200)
        self.btn_rodent.setFixedSize(300, 200)

        self.btn_segment.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)

        self.btn_palm_oil.clicked.connect(self.palm_oil_detection)
        self.btn_segment.clicked.connect(self.put_detection_frame)
        self.btn_rodent.clicked.connect(self.rodent_detection)

        self.btn_layout.addWidget(self.btn_segment)
        self.btn_layout.addSpacing(70)
        self.btn_layout.addWidget(self.btn_palm_oil)
        self.btn_layout.addSpacing(70)
        self.btn_layout.addWidget(self.btn_rodent)

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
        label = QLabel("→")
        label.setStyleSheet("""
                        font-size: 48px;     /* Set font size */
                        color: black;       /* Set text color */
                    """)
        self.h_layout.addWidget(self.mask)
        self.h_layout.addWidget(self.segmented_pic)

        self.pics_widget = QWidget()
        self.pics_widget.setLayout(self.h_layout)

        self.layout.addWidget(self.pics_widget, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.btn_widget, alignment=Qt.AlignCenter)
        self.setLayout(self.layout)

    def palm_oil_detection(self):
        self.btn_palm_oil.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.btn_segment.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)

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
            cv2.imshow('Original', frame)
            cv2.imwrite(DETECTED_FRAME_PATH, frame)
            cv2.waitKey(0)

    def put_detection_frame(self):
        self.btn_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_segment.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)

        cv2.destroyAllWindows()
        self.continue_detection = False

        segmentation = Segmentation()
        segmentation.performSegmentation(DETECTED_FRAME_PATH)

        self.original_img_pixmap = QPixmap(DETECTED_FRAME_PATH)
        scaled_pixmap = self.original_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.original_pic.setPixmap(scaled_pixmap)

        masked_image = cv2.bitwise_and(cv2.imread(DETECTED_FRAME_PATH), cv2.imread(MASK_PATH))
        cv2.imwrite(MASKED_IMG_PATH, masked_image)

        self.mask_pixmap = QPixmap(MASK_PATH)  # Replace with your image file path
        scaled_mask_pixmap = self.mask_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.mask.setPixmap(scaled_mask_pixmap)

        self.masked_img_pixmap = QPixmap(MASKED_IMG_PATH)  # Replace with your image file path
        scaled_pixmap = self.masked_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
        self.segmented_pic.setPixmap(scaled_pixmap)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)

    def rodent_detection(self):
        self.btn_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_segment.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(ON_CLICK_STYLE_SHEET)

        rf = Roboflow(api_key="SOTkBQnU3ZURHAp0gJtr")
        project = rf.workspace().project("rodent-palm-oil")
        model = project.version(1).model

        image_path = self.show_file_dialog()
        predictions = model.predict(image_path, confidence = 60).json()
        hole_image = cv2.imread(image_path)
        bite_marks_image = cv2.imread(image_path)

        for prediction in predictions["predictions"]:
            if prediction['class'] == "holes":
                w = int(prediction['width'])
                h = int(prediction['height'])
                x = int(prediction['x'] - w/2)
                y = int(prediction['y'] - h/2)
                class_name = prediction['class']
                confidence = prediction['confidence']
                cv2.rectangle(hole_image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                cv2.putText(hole_image, f"{class_name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 255, 0), 1)
            else:
                w = int(prediction['width'])
                h = int(prediction['height'])
                x = int(prediction['x'] - w / 2)
                y = int(prediction['y'] - h / 2)
                class_name = prediction['class']
                confidence = prediction['confidence']
                cv2.rectangle(bite_marks_image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                cv2.putText(bite_marks_image, f"{class_name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 255, 0), 1)

        cv2.imwrite(DETECTED_RODENT_HOLE_PATH, hole_image)
        cv2.imwrite(DETECTED_RODENT_BITE_MARKS_PATH, bite_marks_image)

        if predictions["predictions"] is not None:
            cv2.destroyAllWindows()

            self.original_img_pixmap = QPixmap(image_path)
            self.original_pic.setPixmap(self.original_img_pixmap)

            self.masked_img_pixmap = QPixmap(DETECTED_RODENT_HOLE_PATH)  # Replace with your image file path
            scaled_pixmap = self.masked_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.segmented_pic.setPixmap(scaled_pixmap)

            self.mask_pixmap = QPixmap(DETECTED_RODENT_BITE_MARKS_PATH)  # Replace with your image file path
            scaled_mask_pixmap = self.mask_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.mask.setPixmap(scaled_mask_pixmap)

        else:
            self.original_img_pixmap = QPixmap(DETECTED_FRAME_PATH)
            scaled_pixmap = self.original_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.original_pic.setPixmap(scaled_pixmap)

            masked_image = cv2.bitwise_and(cv2.imread(DETECTED_FRAME_PATH), cv2.imread(MASK_PATH))
            cv2.imwrite(MASKED_IMG_PATH, masked_image)


    def show_file_dialog(self):
        global DETECTED_FRAME_PATH
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)',
                                                       options=options)

            if not file_path:  # Check if no file is selected
                raise Exception("No files selected")  # Raise an exception if no file is selected

            QMessageBox.information(self, 'File Selected', f'Selected file path: {file_path}')
            return file_path

        except Exception as e:
            if str(e) == "No files selected":  # Handle the specific exception
                QMessageBox.warning(self, 'Warning', 'No files selected.')
            else:
                QMessageBox.critical(self, 'Error', f'An error occurred: No files selected !')

class Rate(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)  # Center-align the layout

        self.img_layout = QHBoxLayout()
        self.img_layout.setAlignment(Qt.AlignCenter)

        self.preds_layout = QVBoxLayout()
        self.img_layout.setAlignment(Qt.AlignCenter)

        self.ori_img_label = QLabel(self)
        self.feedback_pred_label = QLabel(self)
        self.feedback_pred_data_label = QLabel(self)
        self.feedback_rating_label = QLabel('Your Rating: ', self)
        self.feedback_rating_label.setStyleSheet("font-size: 24px; color: black;")

        self.preds_layout.addWidget(self.feedback_pred_label, alignment=Qt.AlignCenter)
        self.preds_layout.addWidget(self.feedback_pred_data_label, alignment=Qt.AlignCenter)
        self.preds_layout.addWidget(self.feedback_rating_label, alignment=Qt.AlignCenter)
        self.preds_widget = QWidget()
        self.preds_widget.setLayout(self.preds_layout)

        self.img_layout.addWidget(self.ori_img_label, alignment=Qt.AlignCenter)
        self.img_layout.addWidget(self.preds_widget)

        self.img_widget = QWidget()
        self.img_widget.setLayout(self.img_layout)

        self.layout.addWidget(self.img_widget, alignment=Qt.AlignCenter)  # Center-align the label

        # Create a label for "Video Stopped" text (initially hidden)
        self.stopped_label = QLabel("Video Stopped", self)
        self.stopped_label.setStyleSheet("font-size: 24px; color: red;")
        self.stopped_label.setAlignment(Qt.AlignCenter)  # Center-align the text label
        self.stopped_label.hide()  # Hide the label initially
        self.layout.addWidget(self.stopped_label)
        self.layout.addSpacing(70)

        # Create a button to start/stop video capture
        self.btn_start_stop = QPushButton('Start Video', self)
        self.btn_start_stop.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_start_stop.clicked.connect(self.toggle_video_capture)
        self.layout.addWidget(self.btn_start_stop, alignment=Qt.AlignCenter)  # Center-align the button

        # Set the layout to the main window
        self.setLayout(self.layout)

        # Initialize video capture as stopped
        self.is_capturing = False

        # Create a timer to fetch frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        self.btn_start_stop.setStyleSheet(ON_CLICK_STYLE_SHEET)
        # Capture frame-by-frame
        ret, frame = self.capture.read()

        if ret:
            # Save the frame to display later if capture stops
            self.last_frame = frame.copy()

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Update the QLabel with QPixmap
            pixmap = QPixmap.fromImage(qt_image)
            self.ori_img_label.setPixmap(pixmap)

    def toggle_video_capture(self):
        if not self.is_capturing:
            # Start video capture
            self.capture = cv2.VideoCapture(0)
            self.timer.start(30)  # Update every 30 milliseconds
            self.btn_start_stop.setText('Rate us !')
            self.stopped_label.hide()  # Hide the stopped label
        else:
            # Stop video capture
            self.timer.stop()
            self.capture.release()

            # Clear video feed
            self.ori_img_label.clear()

            # Display the last captured frame
            rgb_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(FEEDBACK_IMG_PATH, self.last_frame)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.ori_img_label.setPixmap(pixmap)

            # Show the label indicating video is stopped
            self.stopped_label.show()

            self.btn_start_stop.setText('Start Video')
            self.emotion_detection()

        # Toggle capturing flag
        self.is_capturing = not self.is_capturing



    def emotion_detection(self):
        face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
        emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
        EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        rating_weights = [1, 1, 2, 10, 4, 8, 7]
        # Read the image
        frame = cv2.imread(FEEDBACK_IMG_PATH)
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                rating_weights[i]*= preds[i]

            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        rating = functools.reduce(lambda x, y: x + y, rating_weights)
        cv2.destroyAllWindows()
        qt_image = QImage(FEEDBACK_PREDS_IMG_PATH)
        pixmap = QPixmap.fromImage(qt_image)
        self.feedback_pred_label.setPixmap(pixmap)

        qt_image = QImage(PREDS_DATA_PATH)
        pixmap = QPixmap.fromImage(qt_image)
        self.feedback_pred_data_label.setPixmap(pixmap)

        self.feedback_rating_label.setText(f"Your rating : {str(math.ceil(rating))} / 10")
        self.feedback_rating_label.setStyleSheet("font-size: 24px; color: black;")

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setStyleSheet("background-color: rgba(255, 255, 255);")
        self.stacked_widget = QStackedWidget()

        self.segmentation = Segmentation(self)
        self.detection = Detection(self.stacked_widget)
        self.rate = Rate(self)

        self.stacked_widget.addWidget(self.segmentation)
        self.stacked_widget.addWidget(self.detection)
        self.stacked_widget.addWidget(self.rate)

        self.btn_to_segmentation = QPushButton("Segmentation")
        self.btn_to_segmentation.clicked.connect(self.switch_to_segmentation_window)

        self.btn_to_detection = QPushButton("Detection")
        self.btn_to_detection.clicked.connect(self.switch_to_detection_window)

        self.btn_to_rate = QPushButton("Rate your experience !")
        self.btn_to_rate.clicked.connect(self.switch_to_rate_window)

        self.btn_to_segmentation.setStyleSheet(DEFAULT_STYLE_SHEET)

        self.btn_to_detection.setStyleSheet(DEFAULT_STYLE_SHEET)

        self.btn_to_rate.setStyleSheet(DEFAULT_STYLE_SHEET)

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_to_segmentation)
        btn_layout.addWidget(self.btn_to_detection)
        btn_layout.addWidget(self.btn_to_rate)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        layout.addWidget(btn_widget)
        layout.addWidget(self.stacked_widget)

        self.setLayout(layout)

    def switch_to_segmentation_window(self):
        self.btn_to_segmentation.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.btn_to_rate.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.btn_to_detection.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.stacked_widget.setCurrentWidget(self.segmentation)

    def switch_to_detection_window(self):
        self.btn_to_segmentation.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.btn_to_rate.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.btn_to_detection.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.stacked_widget.setCurrentWidget(self.detection)

    def switch_to_rate_window(self):
        self.btn_to_segmentation.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.btn_to_rate.setStyleSheet(ON_CLICK_STYLE_SHEET)
        self.btn_to_detection.setStyleSheet(DEFAULT_STYLE_SHEET)
        self.stacked_widget.setCurrentWidget(self.rate)

class RegisterPage(QWidget):
    def __init__(self, credentials):
        super().__init__()

        self.credentials = credentials
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Register Page')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_new_username = QLabel('New Username:')
        self.lbl_new_password = QLabel('New Password:')
        self.txt_new_username = QLineEdit()
        self.txt_new_password = QLineEdit()
        self.btn_register = QPushButton('Register')

        # Password field settings
        self.txt_new_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()
        hbox_password = QHBoxLayout()

        hbox_username.addWidget(self.lbl_new_username)
        hbox_username.addWidget(self.txt_new_username)

        hbox_password.addWidget(self.lbl_new_password)
        hbox_password.addWidget(self.txt_new_password)

        vbox.addLayout(hbox_username)
        vbox.addLayout(hbox_password)
        vbox.addWidget(self.btn_register)

        self.setLayout(vbox)

        # Event Handling
        self.btn_register.clicked.connect(self.register)

        self.show()

    def register(self):
        new_username = self.txt_new_username.text()
        new_password = self.txt_new_password.text()

        # Check if the username already exists
        if new_username in self.credentials:
            QMessageBox.warning(self, 'Registration Error', 'Username already exists. Please choose a different one.')
            return

        # Store the new credentials in the dictionary
        self.credentials[new_username] = new_password

        # Save credentials to a file (for demonstration purposes, you might want to use a more secure storage method)
        with open('credentials.txt', 'a') as file:
            file.write(f'{new_username}:{new_password}\n')

        print('Registration successful!')
        self.close()


class ResetPasswordPage(QDialog):
    def __init__(self, credentials, username):
        super().__init__()

        self.credentials = credentials
        self.username = username
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Reset Password')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_new_password = QLabel('New Password:')
        self.txt_new_password = QLineEdit()
        self.btn_reset_password = QPushButton('Reset Password')

        # Password field settings
        self.txt_new_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_password = QHBoxLayout()

        hbox_password.addWidget(self.lbl_new_password)
        hbox_password.addWidget(self.txt_new_password)

        vbox.addLayout(hbox_password)
        vbox.addWidget(self.btn_reset_password)

        self.setLayout(vbox)

        # Event Handling
        self.btn_reset_password.clicked.connect(self.reset_password)

        self.show()

    def reset_password(self):
        new_password = self.txt_new_password.text()

        # Update the password in the credentials dictionary
        self.credentials[self.username] = new_password

        # Save updated credentials to the file
        with open('credentials.txt', 'w') as file:
            for u, p in self.credentials.items():
                file.write(f'{u}:{p}\n')

        print('Password reset successful!')
        self.accept()


class ForgotPasswordPage(QWidget):
    def __init__(self, credentials):
        super().__init__()

        self.credentials = credentials
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Forgot Password')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_username = QLabel('Username:')
        self.txt_username = QLineEdit()
        self.btn_reset_password = QPushButton('Reset Password')

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()

        hbox_username.addWidget(self.lbl_username)
        hbox_username.addWidget(self.txt_username)

        vbox.addLayout(hbox_username)
        vbox.addWidget(self.btn_reset_password)

        self.setLayout(vbox)

        # Event Handling
        self.btn_reset_password.clicked.connect(self.reset_password)

        self.show()

    def reset_password(self):
        username = self.txt_username.text()

        # Check if the username exists
        if username in self.credentials:
            reset_password_dialog = ResetPasswordPage(self.credentials, username)
            if reset_password_dialog.exec_() == QDialog.Accepted:
                print('Password reset successful!')
            else:
                print('Password reset canceled.')
        else:
            QMessageBox.warning(self, 'Password Reset Error', 'Username not found. Please enter a valid username.')


class PrintRunner(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Print Runner')
        self.setGeometry(400, 400, 300, 100)

        label = QLabel('Hello, World!')
        layout = QVBoxLayout()
        layout.addWidget(label)

        self.setLayout(layout)

        self.show()


class LoginPage(QWidget):
    successful_login = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Load existing credentials from the file or create an empty dictionary if the file doesn't exist
        self.credentials = {}
        try:
            with open('credentials.txt', 'r') as file:
                for line in file:
                    username, password = line.strip().split(':')
                    self.credentials[username] = password
        except FileNotFoundError:
            pass

        self.init_ui()

    def init_ui(self):
        # Widgets
        self.lbl_username = QLabel('Username:')
        self.lbl_password = QLabel('Password:')
        self.txt_username = QLineEdit()
        self.txt_password = QLineEdit()
        self.btn_login = QPushButton('Login')
        self.btn_register = QPushButton('Register')
        self.btn_forgot_password = QPushButton('Forgot Password')

        # Password field settings
        self.txt_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()
        hbox_password = QHBoxLayout()
        hbox_buttons = QHBoxLayout()

        hbox_username.addWidget(self.lbl_username)
        hbox_username.addWidget(self.txt_username)

        hbox_password.addWidget(self.lbl_password)
        hbox_password.addWidget(self.txt_password)

        hbox_buttons.addWidget(self.btn_login)
        hbox_buttons.addWidget(self.btn_register)
        hbox_buttons.addWidget(self.btn_forgot_password)

        vbox.addLayout(hbox_username)
        vbox.addLayout(hbox_password)
        vbox.addLayout(hbox_buttons)

        self.setLayout(vbox)

        # Event Handling
        self.btn_login.clicked.connect(self.login)
        self.btn_register.clicked.connect(self.show_register_page)
        self.btn_forgot_password.clicked.connect(self.show_forgot_password_page)

        self.show()

    def login(self):
        username = self.txt_username.text()
        password = self.txt_password.text()

        # Check credentials
        if username in self.credentials and self.credentials[username] == password:
            print('Login successful!')
            self.successful_login.emit()  # Emit signal on successful login
        else:
            QMessageBox.warning(self, 'Login Error', 'Invalid credentials. Please try again.')

    def show_register_page(self):
        self.register_page = RegisterPage(self.credentials)
        self.register_page.show()

    def show_forgot_password_page(self):
        self.forgot_password_page = ForgotPasswordPage(self.credentials)
        self.forgot_password_page.show()

    def on_successful_login(self):
        window = MainWindow()
        window.setWindowTitle('HarvestMate')
        window.setFixedSize(1080, 720)
        window.setGeometry(0, 0, 800, 500)  # Set initial geometry

        window.show()


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 400, 300)  # Set initial geometry
        self.setWindowIcon(QIcon(ICON_PATH))
        # Center window on the screen
        self.center_window()
        # Create a stacked widget to hold login and main windows
        self.stacked_widget = QStackedWidget(self)

        # Add login page to stacked widget
        self.login_page = LoginPage()
        self.stacked_widget.addWidget(self.login_page)

        # Connect successful_login signal to switch to main window
        self.login_page.successful_login.connect(self.on_successful_login)

        # Set central widget as stacked widget
        self.setCentralWidget(self.stacked_widget)

    def on_successful_login(self):
        # Remove login page from stacked widget
        self.stacked_widget.removeWidget(self.login_page)

        # Create and add main window to stacked widget
        self.main_window = MainWindow()
        self.stacked_widget.addWidget(self.main_window)

        # Switch to main window
        self.stacked_widget.setCurrentWidget(self.main_window)

    def keyPressEvent(self, event):
        """Override keyPressEvent to toggle fullscreen on 'F' key press"""
        if event.key() == 70:  # 'F' key
            if self.isFullScreen():
                self.showNormal()  # Show normal window if in fullscreen
            else:
                self.showFullScreen()  # Show fullscreen window if in normal mode

        # Call base class keyPressEvent for other key events
        super().keyPressEvent(event)

    def center_window(self):
        # Get the screen geometry
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center position for the window
        window_position_x = (screen.width() - self.width()) // 2
        window_position_y = (screen.height() - self.height()) // 2

        # Set the window position
        self.move(window_position_x, window_position_y)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_app = MainApplication()
    main_app.setWindowTitle('HarvestMate')
    main_app.show()

    sys.exit(app.exec_())
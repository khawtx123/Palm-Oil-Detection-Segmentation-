from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QHBoxLayout, QFileDialog, QMessageBox
from UNET import Unet
from ultralytics import YOLO
import cv2
from roboflow import Roboflow
import sys

DETECTED_FRAME_PATH = "detection/detected_frame.jpg"
MASK_PATH = "detection/mask.jpg"
MASKED_IMG_PATH = "detection/masked_img.jpg"
DETECTED_RODENT_FRAME_PATH = "detection/detected_rodent_frame.jpg"
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
        options = QFileDialog.Options()
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)',
                                                       options=options)
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

            QMessageBox.information(self, 'File Selected', f'Selected file path: {file_path}')

        except Exception as e:
            if str(e) == "No files selected":  # Handle the specific exception
                QMessageBox.warning(self, 'Warning', 'No files selected.')
            else:
                QMessageBox.critical(self, 'Error', f'An error occurred: No files selected !')
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                   options=options)

    def segment_rodent(self):
        self.btn_segment_palm_oil.setStyleSheet(DEFAULT_WIDGET_STYLE_SHEET)
        self.btn_rodent.setStyleSheet(ON_CLICK_STYLE_SHEET)
        global DETECTED_FRAME_PATH
        options = QFileDialog.Options()
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)',
                                                       options=options)
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

            QMessageBox.information(self, 'File Selected', f'Selected file path: {file_path}')

        except Exception as e:
            if str(e) == "No files selected":  # Handle the specific exception
                QMessageBox.warning(self, 'Warning', 'No files selected.')
            else:
                QMessageBox.critical(self, 'Error', f'An error occurred: No files selected !')
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                   options=options)

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
            cv2.imshow('frame', frame)
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

        project = rf.workspace().project("rodent_palm_oil")
        model = project.version(1).model

        image_path = self.show_file_dialog()
        predictions = model.predict(image_path, confidence=80, overlap=80).json()
        image = cv2.imread(image_path)

        for prediction in predictions["predictions"]:
            x = int(prediction['x'])
            y = int(prediction['y'])
            w = int(prediction['width'])
            h = int(prediction['height'])
            class_name = prediction['class']
            confidence = prediction['confidence']
            cv2.rectangle(image, (x, y), (x + w+20, y + h+20), (0, 255, 0), 1)  # Draw bounding box
            cv2.putText(image, f"{class_name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 0), 1)

        if predictions["predictions"] is not None:
            cv2.destroyAllWindows()
            cv2.imwrite(DETECTED_RODENT_FRAME_PATH, image)

            masked_image = cv2.bitwise_and(cv2.imread(DETECTED_FRAME_PATH), cv2.imread(MASK_PATH))
            cv2.imwrite(MASKED_IMG_PATH, masked_image)

            self.original_img_pixmap = QPixmap()
            self.original_pic.setPixmap(self.original_img_pixmap)

            self.masked_img_pixmap = QPixmap(DETECTED_RODENT_FRAME_PATH)  # Replace with your image file path
            scaled_pixmap = self.masked_img_pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio)
            self.segmented_pic.setPixmap(scaled_pixmap)

            self.mask_pixmap = QPixmap(image_path)  # Replace with your image file path
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
        self.stacked_widget = QStackedWidget()
        self.layout = QVBoxLayout()
        self.btn_layout = QHBoxLayout()
        self.btn_segment = QPushButton("Segment")
        self.layout.addWidget(self.btn_segment)
        self.setLayout(self.layout)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

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

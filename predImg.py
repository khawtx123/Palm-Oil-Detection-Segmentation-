import cv2
import os
from ultralytics import YOLO
import cv2
import tensorflow as tf

video_path = r"C:\Users\USER\Downloads\Harvesting Palm Oil Using a Machine.mp4"
model = YOLO(r"C:\Users\USER\source\repos\YOLO\model\best_prev.pt")


def process_frame(frame, confidence_threshold=0.5):
    # Predict objects in the frame
    predictions = model.predict(frame)

    print(predictions[0].boxes.conf)
    arr = predictions[0].boxes.conf.numpy()
    if arr.size >0:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

def capture_video():
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no frame is captured


        # Process the frame to detect objects
        process_frame(frame)

    cap.release()  # Release the video capture object

capture_video()
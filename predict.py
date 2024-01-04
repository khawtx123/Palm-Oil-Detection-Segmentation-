import cv2
import os
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, r"palm oil (2).mp4")
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


model_path = os.path.join('.', 'model', 'best.pt')
model = YOLO(model_path)
threshold = 0

model.predict(source=video_path, show=True, conf=0.5)

import cv2
import os
from ultralytics import YOLO
model_path = os.path.join('.', 'model', 'best.pt')
model = YOLO(model_path)
test_img = r"C:\Users\USER\source\repos\YOLO\images\train\ca3_20211007_20211007110915_20211007111052_110915_190940_337.jpg"
image = cv2.imread(test_img)
res = model.predict(image, conf=0.2)
for result in res:
                boxes = result.boxes
                if len(boxes)==0:
                    print("no detections")

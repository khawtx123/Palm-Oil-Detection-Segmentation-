import cv2
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="SOTkBQnU3ZURHAp0gJtr")

# Get the project and model details
project = rf.workspace().project("rodent_palm_oil")
model = project.version(1).model

# Path to the image file
image_path = r"C:\Users\USER\source\repos\Pytorch-UNet\data\imgs\ca3_20211007_20211007110915_20211007111052_110915_190936_449.jpg"

# Perform prediction on the local image
predictions = model.predict(image_path, confidence=70, overlap=30).json()
print(predictions)
# Load the image using OpenCV
image = cv2.imread(image_path)

# Draw bounding boxes on the image based on the predictions
for prediction in predictions["predictions"]:
    x = int(prediction['x'])
    y = int(prediction['y'])
    w = int(prediction['width'])
    h = int(prediction['height'])
    class_name = prediction['class']
    confidence = prediction['confidence']
    # class_name = prediction["class"]
    # confidence = prediction["confidence"]
    # x, y, w, h = prediction["bbox"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
    cv2.putText(image, f"{class_name} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Display class name and confidence

# # Display the image with predictions
cv2.imshow("Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

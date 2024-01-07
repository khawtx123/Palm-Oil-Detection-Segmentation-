import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.9]
    print(detections)
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

inference.Stream(
    source="video.mp4",  # Specify the path to your video file here
    model="taylor-swift-records/3",  # Model from Universe
    output_channel_order="BGR",
    use_main_thread=True,  # For OpenCV display
    on_prediction=on_prediction,
)

import os
import cv2

def resize_video(input_file, output_file, scale_percent):
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get the video details (frame width, frame height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new dimensions
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (new_width, new_height))

    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:
            break  # Break the loop if no frame is read

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Write the resized frame to the output video file
        out.write(resized_frame)

        # Display the frame
        cv2.imshow('Resized Video', resized_frame)

        # Press 'q' to exit the loop and stop resizing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, r"palm oil (2).mp4")
video_path_out = '{}_out.mp4'.format(video_path)

if __name__ == '__main__':
    # Specify the input and output video file paths
    input_file_path = video_path
    output_file_path = video_path

    # Specify the percentage scale for resizing (e.g., 50%)
    scale_percent = 50

    # Call the resize_video function
    resize_video(input_file_path, output_file_path, scale_percent)

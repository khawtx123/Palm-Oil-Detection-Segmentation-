import cv2
import os

def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))
            save_path = os.path.join(root, file)
            x = cv2.imread(save_path)
            cv2.imshow("img", x)
            cv2.waitKey(0)

            y = cv2.resize(x, (256, 256))
            cv2.imshow("img", y)
            cv2.waitKey(0)

            cv2.imwrite(save_path, y)

# Specify the directory path
directory_path = r"C:\Users\USER\Documents\dataset\palm oil\sime darby\All"

# Call the function to list all files in the directory
list_files(directory_path)

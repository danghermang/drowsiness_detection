import os
import random
import tkinter
from tkinter.filedialog import askdirectory

import cv2
import dlib
import progressbar

from utils import image_to_package, common_functions
from configs.config import *
# script that parses images that are not parsed yet

root = tkinter.Tk()
root.update()
root.withdraw()

image_path = askdirectory(initialdir='../')
root.destroy()
IMAGE_DIMS = (200, 200, 3)

if __name__ == "__main__":
    check = int(input("Training or validation? (0 or 1)\n"))
    if check == 0:
        training = True
    else:
        training = False
    if training:
        output_path = PROCESSED_IMAGES_TRAINING
    else:
        output_path = PROCESSED_IMAGES_VALIDATION
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        for element in ['0', '1']:
            os.makedirs(os.path.join(output_path, str(element)))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    l_start, l_end = common_functions.FACIAL_LANDMARKS["left_eye"]
    r_start, r_end = common_functions.FACIAL_LANDMARKS["right_eye"]
    files = image_to_package.get_files(image_path)
    print("\nFiles loaded\n")
    bar = progressbar.ProgressBar(max_value=len(files) + 1)
    for i, image in enumerate(files):
        # print(image)
        if image.endswith(('.jpg', '.jpeg', 'png')):
            gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            faces_detected = detector(gray, 0)
            for rect in faces_detected:
                # print("detected")
                (x, y, w, h) = common_functions.rect_to_bounding_box(rect)
                shape = predictor(gray, rect)
                shape = common_functions.shape_to_np(shape)
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = common_functions.eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = common_functions.eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
                # facial_shapes.append(np.array(left_eye+right_eye))
                output_folder = os.path.join(output_path, str(
                    image_to_package.number_to_sleep_state(
                        eye_aspect_ratio_value)))
                fata = common_functions.resize(gray[max(y, 0):y + h, max(x, 0):x + w], IMAGE_DIMS[0], IMAGE_DIMS[1])
                cv2.imwrite(os.path.join(output_folder, str(round(eye_aspect_ratio_value, 4)) + "000000" +
                                         str(random.randint(1000, 9999)) + ".jpg"), fata)
            bar.update(i)

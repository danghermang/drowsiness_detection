import os
from threading import Thread

import cv2
import dlib
import numpy as np
import progressbar

import image_to_package
import utils

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"


def process_image():
    check = int(input("Training or validation? (0 or 1)\n"))
    if check == 0:
        training = True
    else:
        training = False
    if training:
        mask_path = os.path.join('../data/masks', 'training')
        image_path = os.path.join('../data/processed_images', 'training')
    else:
        mask_path = os.path.join('../data/masks', 'validation')
        image_path = os.path.join('../data/processed_images', 'validation')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    facial_shapes = []
    facial_shapes_tag = []
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        l_start, l_end = utils.FACIAL_LANDMARKS["left_eye"]
        r_start, r_end = utils.FACIAL_LANDMARKS["right_eye"]
        files = image_to_package.get_files(image_path)
        bar = progressbar.ProgressBar(max_value=len(files) + 1)
        for i, image in enumerate(files):
            gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            faces_detected = detector(gray, 0)
            if faces_detected:
                rect = faces_detected[0]
                # (x, y, w, h) = utils.rect_to_bounding_box(rect)
                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = utils.eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = utils.eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
                # facial_shapes.append(np.array(left_eye+right_eye))
                facial_shapes.append(np.array(shape))
                facial_shapes_tag.append(image_to_package.number_to_one_hot(eye_aspect_ratio_value))
            bar.update(i)
    finally:
        np.savez_compressed(os.path.join(mask_path, "mask_export"), facial_shapes=np.array(facial_shapes),
                            facial_shapes_tag=np.array(facial_shapes_tag))


if __name__ == "__main__":
    processing = Thread(target=process_image)
    processing.deamon = True
    processing.start()

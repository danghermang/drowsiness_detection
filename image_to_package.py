import os
from random import shuffle

import cv2
import dlib
import numpy as np
import progressbar
from keras.preprocessing.image import ImageDataGenerator

import utils

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")


def pre_load_images(training=True, width=128, max_img=10000, crossentropy=True):
    if training:
        image_path = os.path.join('../data/processed_images', 'training')
    else:
        image_path = os.path.join('../data/processed_images', 'validation')
    facial_shapes = []
    facial_shapes_tag = []
    files = get_files(image_path)
    shuffle(files)
    bar = progressbar.ProgressBar(max_value=min(len(files) + 1, max_img))
    for i, image in enumerate(files):
        if i == max_img:
            bar.update(i)
            break
        gray = cv2.imread(image)
        if gray is False or gray is None:
            continue
        gray = utils.resize(gray, width)
        gray = gray.astype('float32')
        gray /= 255
        facial_shapes.append(gray)
        base = round(float(os.path.splitext(os.path.basename(image))[0]), 3)
        if crossentropy:
            facial_shapes_tag.append(number_to_one_hot(base))
        else:
            facial_shapes_tag.append(number_to_sleep_state(base))
        bar.update(i)
    return np.array(facial_shapes), np.array(facial_shapes_tag)


def gen_load_images(training=True, width=128, batch_size=32, max_type_img=10000, augumented=False, crossentropy=True):
    if training:
        image_path = os.path.join('../data/processed_images', 'training')
    else:
        image_path = os.path.join('../data/processed_images', 'validation')
    files = []
    for folder in ['0', '1']:
        temp_files = get_files(os.path.join(image_path, folder))
        shuffle(temp_files)
        temp_files = temp_files[:min(max_type_img, len(temp_files))]
        files.extend(temp_files)
    while True:
        shuffle(files)
        facial_shapes = []
        facial_shapes_tag = []
        for i, image in enumerate(files):
            if len(facial_shapes) == batch_size:
                yield np.array(facial_shapes).astype('float32'), np.array(facial_shapes_tag).astype('float32')
                facial_shapes = []
                facial_shapes_tag = []
            gray = cv2.imread(image)
            if gray is False or gray is None:
                continue
            gray = utils.resize(gray, width)
            if augumented:
                gray = aug.random_transform(gray)
            gray = gray.astype('float32')
            gray /= 255
            facial_shapes.append(gray)
            base = round(float(os.path.splitext(os.path.basename(image))[0]), 3)
            if crossentropy:
                facial_shapes_tag.append(number_to_one_hot(base))
            else:
                facial_shapes_tag.append(number_to_sleep_state(base))


def pre_load_mask(training=True, width=128, max_img=10000, crossentropy=True):
    if training:
        image_path = os.path.join('../data/processed_images', 'training')
    else:
        image_path = os.path.join('../data/processed_images', 'validation')
    facial_shapes = []
    facial_shapes_tag = []
    files = get_files(image_path)
    shuffle(files)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    bar = progressbar.ProgressBar(max_value=min(len(files) + 1, max_img))
    for i, image in enumerate(files):
        if i == max_img:
            bar.update(i)
            break
        gray = cv2.imread(image)
        if gray is False or gray is None:
            continue
        gray = utils.resize(gray, width)
        # gray = gray.astype('float32')
        # gray /= 255
        faces_detected = detector(gray, 0)
        if faces_detected:
            rect = faces_detected[0]
            shape = predictor(gray, rect)
            shape = utils.shape_to_np(shape)
            eyes = shape[36:48]
            facial_shapes.append(eyes)
            base = round(float(os.path.splitext(os.path.basename(image))[0]), 3)
            if crossentropy:
                facial_shapes_tag.append(number_to_one_hot(base))
            else:
                facial_shapes_tag.append(number_to_sleep_state(base))
            bar.update(i)
    facial_shapes = np.array(facial_shapes)
    facial_shapes_tag = np.array(facial_shapes_tag)
    facial_shapes = facial_shapes.reshape(facial_shapes.shape[0], facial_shapes.shape[1] * facial_shapes.shape[2])
    facial_shapes = facial_shapes.astype('float32')
    facial_shapes_tag = facial_shapes_tag.astype('float32')
    facial_shapes /= width
    return facial_shapes, facial_shapes_tag


def gen_load_mask(training=True, width=128, batch_size=32, max_type_img=10000, augumented=False, crossentropy=True):
    if training:
        image_path = os.path.join('../data/processed_images', 'training')
    else:
        image_path = os.path.join('../data/processed_images', 'validation')

    files = []
    for folder in ['0', '1']:
        temp_files = get_files(os.path.join(image_path, folder))
        shuffle(temp_files)
        temp_files = temp_files[:min(max_type_img, len(temp_files))]
        files.extend(temp_files)
    while True:
        shuffle(files)
        facial_shapes = []
        facial_shapes_tag = []
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        for i, image in enumerate(files):
            if len(facial_shapes) == batch_size:
                facial_shapes = np.array(facial_shapes)
                facial_shapes_tag = np.array(facial_shapes_tag)
                facial_shapes = facial_shapes.reshape(facial_shapes.shape[0],
                                                      facial_shapes.shape[1] * facial_shapes.shape[2])
                facial_shapes = facial_shapes.astype('float32')
                facial_shapes_tag = facial_shapes_tag.astype('float32')
                facial_shapes /= width
                yield facial_shapes, facial_shapes_tag
                facial_shapes = []
                facial_shapes_tag = []
            gray = cv2.imread(image)
            if gray is False or gray is None:
                continue
            gray = utils.resize(gray, width)
            if augumented:
                gray = utils.resize(aug.random_transform(gray), width)
            faces_detected = detector(gray, 0)
            if faces_detected:
                rect = faces_detected[0]
                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
                eyes = shape[36:48]
                facial_shapes.append(eyes)
                base = round(float(os.path.splitext(os.path.basename(image))[0]), 3)
                if crossentropy:
                    facial_shapes_tag.append(number_to_one_hot(base))
                else:
                    facial_shapes_tag.append(number_to_sleep_state(base))


def get_prediction_image(model, image, normalize=True):
    image = image.astype('float32')
    if normalize:
        image /= 255
    prediction = model.predict_classes(np.array([image, ]))
    if prediction[0] == 0:
        return False
    return True


def get_prediction_mask(model, mask, normalize=True):
    mask = mask.astype('float32')
    mask = np.reshape(mask, (1, 24))
    if normalize:
        mask = mask / 200
    prediction = model.predict_classes(mask)
    if prediction[0] == 0:
        return False
    return True


def get_files(path):
    f = []
    for root, dirs, files in os.walk(path):
        f.extend([os.path.join(root, file) for file in files])
    return f


def number_to_sleep_state(number):
    if round(number, 2) < 0.5:
        return 0
    else:
        return 1


def number_to_one_hot(number):
    output = np.zeros(2, dtype=np.uint8)
    number = round(number, 3)
    if number <= 0.5:
        output[0] = 1
    else:
        output[1] = 1
    return output


def number_to_approximation(number):
    number = round(number, 3)
    if number <= 0.5:
        return 0
    else:
        return 1


def one_hot_to_number(arr):
    index = np.argwhere(arr == 1)[0]
    if index == 0:
        return 0
    else:
        return 1


def result_to_state(arr):
    return np.argmax(arr)


def image_to_array(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


if __name__ == "__main__":
    for element in gen_load_mask():
        print(element)

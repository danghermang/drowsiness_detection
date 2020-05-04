import math
from collections import OrderedDict

import cv2
import numpy as np

# library that provides generic functions
FACIAL_LANDMARKS = OrderedDict([("jaw", (0, 17)),
                                ("right_eyebrow", (17, 22)),
                                ("left_eyebrow", (22, 27)),
                                ("nose", (27, 36)),
                                ("right_eye", (36, 42)),
                                ("left_eye", (42, 48)),
                                ("mouth", (48, 68))])


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def eye_aspect_ratio(eye):
    a = euclidean_distance(eye[1], eye[5])
    b = euclidean_distance(eye[2], eye[4])
    c = euclidean_distance(eye[0], eye[3])
    return (a + b) / c


def resize(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dimension = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dimension = (width, int(h * r))
    else:
        dimension = (width, height)
    resized = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
    return resized


def shape_to_np(shape, dtype="int"):
    coordinates = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates


def get_face_approximate(frame, rect, percentage=0):
    if percentage == 0:
        return frame[rect.left():rect.right(), rect.top():rect.bottom()]
    left = int(rect.left() - percentage * rect.left() / 100)
    right = int(rect.right() + percentage * rect.right() / 100)
    top = int(rect.top() - percentage * rect.top() / 100)
    bottom = int(rect.bottom() + percentage * rect.bottom() / 100)
    return frame[left:right, top:bottom]


def rect_to_bounding_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

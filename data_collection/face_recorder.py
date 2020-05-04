import os
import queue
import random

import cv2
import dlib

from utils import video_stream, fps, common_functions
from configs.config import *
# script that records a person using web cam

messageQueue = queue.Queue()
name = ''
number_of_frames = 4000


if __name__ == "__main__":
    nume = input("Introduceti numele persoanei\n")
    check = int(input("Training or validation? (0 or 1)\n"))
    number_of_frames = int(input("Number of frames:\n"))
    if check == 0:
        training = True
    else:
        training = False
    if training:
        path = os.path.join(ORIGINAL_IMAGES_TRAINING, nume.lower())
    else:
        path = os.path.join(ORIGINAL_IMAGES_VALIDATION, nume.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    dimensions = None
    vs = video_stream.VideoStream(src=DEFAULT_WEBCAM, dimensions=dimensions).start()
    FPS = fps.FPS()
    try:
        cv2.namedWindow("Cadru", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cadru", 720, 480)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        l_start, l_end = common_functions.FACIAL_LANDMARKS["left_eye"]
        r_start, r_end = common_functions.FACIAL_LANDMARKS["right_eye"]
        FPS.start()
        frame = vs.read()
        print("Frame count =", frame.shape)
        counter = 1
        while True and counter < number_of_frames:
            if vs.frame_read():
                FPS.frame_skip()
                continue
            FPS.update()
            frame = vs.read()
            frame = cv2.flip(frame, 1)
            frame = frame[100:400, 200:600]
            if FPS.get_total_frames() > 100:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_detected = detector(gray, 0)
                for rect in faces_detected:
                    (x, y, w, h) = common_functions.rect_to_bounding_box(rect)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 2)
                    shape = predictor(gray, rect)
                    shape = common_functions.shape_to_np(shape)
                    left_eye = shape[l_start:l_end]
                    right_eye = shape[r_start:r_end]
                    left_eye_aspect_ratio = common_functions.eye_aspect_ratio(left_eye)
                    right_eye_aspect_ratio = common_functions.eye_aspect_ratio(right_eye)
                    eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [left_eye_hull], -1, RED, 1)
                    cv2.drawContours(frame, [right_eye_hull], -1, RED, 1)

                    cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                    cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (243, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                    cv2.putText(frame, "FPS {}".format(round(FPS.fps(), 3)), (10, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                    cv2.putText(frame, "Frames taken {}".format(counter), (200, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                    fata = common_functions.resize(gray[max(y, 0):y + h, max(x, 0):x + w], IMAGE_DIMS[0], IMAGE_DIMS[1])
                    cv2.imwrite(os.path.join(path, str(round(eye_aspect_ratio_value, 4)) + "000000" +
                                             str(random.randint(1000, 9999)) + ".jpg"), fata)
                    cv2.imshow("Cadru", frame)
                    counter += 1
            else:
                cv2.putText(frame, "CENTER YOUR FACE IN THE SCREEN", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 1)
                cv2.imshow("Cadru", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == ord("Q"):
                break
    finally:
        print("Done")
        print("FPS average:", FPS.fps(2))
        print("Frames skipped per second:", FPS.get_frame_skip(2))
        cv2.destroyAllWindows()
        vs.stop()
        vs.stream.release()
        print("Done")

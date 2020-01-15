import os
import random
import tkinter
from tkinter.filedialog import askopenfilename

import cv2
import dlib
import imageio
import progressbar

import fps
import utils

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
root = tkinter.Tk()
root.update()
root.withdraw()
default_webcam = askopenfilename(initialdir='../')
root.destroy()

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
name = ''
IMAGE_DIMS = (200, 200, 3)

if __name__ == "__main__":
    print(default_webcam)
    nume = input("Introduceti numele persoanei\n")
    check = int(input("Training or validation? (0 or 1)\n"))
    if check == 0:
        training = True
    else:
        training = False
    if training:
        path = os.path.join('../data/images/training', nume.lower())
    else:
        path = os.path.join('../data/images/validation', nume.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    # dimensions = None
    # vs = video_stream.VideoStream(src=default_webcam, dimensions=dimensions).start()
    reader = imageio.get_reader(default_webcam)

    FPS = fps.FPS()
    try:
        # cv2.namedWindow("Cadru", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Cadru", 720, 480)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        l_start, l_end = utils.FACIAL_LANDMARKS["left_eye"]
        r_start, r_end = utils.FACIAL_LANDMARKS["right_eye"]
        FPS.start()
        # frame = vs.read()
        # print("Frame count =", frame.shape)
        counter = 1
        bar = progressbar.ProgressBar(max_value=len(reader))
        for index, frame in enumerate(reader):
            FPS.update()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
            if frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = detector(gray, 0)
            for rect in faces_detected:
                (x, y, w, h) = utils.rect_to_bounding_box(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 2)
                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = utils.eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = utils.eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
                # left_eye_hull = cv2.convexHull(left_eye)
                # right_eye_hull = cv2.convexHull(right_eye)
                # cv2.drawContours(frame, [left_eye_hull], -1, RED, 1)
                # cv2.drawContours(frame, [right_eye_hull], -1, RED, 1)
                #
                # cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (250, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                # cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (243, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                # cv2.putText(frame, "FPS {}".format(round(FPS.fps(), 3)), (10, 250),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                # cv2.putText(frame, "Frames taken {}".format(counter), (200, 250),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
                fata = utils.resize(gray[max(y, 0):y + h, max(x, 0):x + w], IMAGE_DIMS[0], IMAGE_DIMS[1])
                cv2.imwrite(os.path.join(path, str(round(eye_aspect_ratio_value, 4)) + "000000" +
                                         str(random.randint(1000, 9999)) + ".jpg"), fata)

                # cv2.imshow("Fata", fata)
                counter += 1
            bar.update(index)
            # cv2.imshow("Cadru", frame)
            # key = cv2.waitKey(1) & 0xFF
            # # # if the `q` key was pressed, break from the loop
            # if key == ord("q") or key == ord("Q"):
            #     break
    finally:
        print("Done")
        print("FPS average:", FPS.fps(2))
        print("Frames skipped per second:", FPS.get_frame_skip(2))
        # cv2.destroyAllWindows()
        print("Done")

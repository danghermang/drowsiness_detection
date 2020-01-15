import datetime
import queue
import time
import winsound
from threading import Thread

import cv2
import dlib
import keras
from keras import backend as K

import fps
import image_to_package
import utils
import video_stream

alarm_path = r"../data/alarm3.wav"
predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
default_webcam = 0
messageQueue = queue.Queue()
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
IMAGE_DIMS = (200, 200, 3)
SECONDS = 2
SHOW_MASK = True
SHOW_FPS = True


def switch_condition(current):
    if current >= 2:
        current = 0
    else:
        current += 1
    return current


def show_time():
    check = messageQueue.get()
    while check:
        print("Getting sleepy at", datetime.datetime.now().strftime("%H:%M:%S"))
        time.sleep(1)
        if not messageQueue.empty():
            messageQueue.get()
            break


count_thread = Thread(target=show_time, args=(True,))

if __name__ == "__main__":
    current_condition = 0
    json_file = open('../data/model.json', 'r')
    image_model_json = json_file.read()
    json_file.close()
    image_model = keras.models.model_from_json(image_model_json)
    image_model.load_weights("../data/model_best.h5")

    json_file = open('../data/model_mask.json', 'r')
    mask_model_json = json_file.read()
    json_file.close()
    mask_model = keras.models.model_from_json(mask_model_json)
    mask_model.load_weights("../data/model_mask_best.h5")

    dimensions = (320, 180)
    dimensions = None
    vs = video_stream.VideoStream(src=default_webcam, dimensions=dimensions).start()
    FPS = fps.FPS()

    try:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 720, 480)

        last_alarm = None
        consecutive_alarms = False
        eye_aspect_ratio_threshold = 0.56
        eye_consecutive_frames = 50
        frame_counter = 0
        alarm_on = False
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        # print(utils.FACIAL_LANDMARKS)
        l_start, l_end = utils.FACIAL_LANDMARKS["left_eye"]
        r_start, r_end = utils.FACIAL_LANDMARKS["right_eye"]
        FPS.start()

        left_eye_aspect_ratio = right_eye_aspect_ratio = 0
        frame = vs.read()
        print("Frame count =", frame.shape)
        mask_prediction = True
        image_prediction = True
        mask_color = GREEN
        image_color = GREEN
        threshold_color = GREEN
        while True:
            if vs.frame_read():
                FPS.frame_skip()
                continue
            FPS.update()
            eye_consecutive_frames = int(FPS.fps(0) * SECONDS)
            frame = vs.read()
            frame = frame[150:400, 150:450]
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = detector(gray, 0)
            if faces_detected:
                rect = faces_detected[0]
                (x, y, w, h) = utils.rect_to_bounding_box(rect)
                fata = utils.resize(gray[max(y, 0):y + h, max(x, 0):x + w], IMAGE_DIMS[0], IMAGE_DIMS[1])
                fata = cv2.cvtColor(fata, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 1)
                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
                eyes = shape[36:48]
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = utils.eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = utils.eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, RED, 1)
                cv2.drawContours(frame, [right_eye_hull], -1, RED, 1)
                if SHOW_MASK:
                    for shape_start, shape_end in utils.FACIAL_LANDMARKS.values():
                        landmark = shape[shape_start:shape_end]
                        for point in landmark:
                            cv2.circle(frame, tuple(point), 1, GREEN, 1)
                mask_prediction = image_to_package.get_prediction_mask(mask_model, eyes)
                if mask_prediction:
                    mask_color = GREEN
                else:
                    mask_color = RED

                image_prediction = image_to_package.get_prediction_image(image_model, fata)

                if image_prediction:
                    image_color = GREEN
                else:
                    image_color = RED

                if eye_aspect_ratio_value >= eye_aspect_ratio_threshold:
                    threshold_color = GREEN
                else:
                    threshold_color = RED
                if current_condition == 0:
                    condition = eye_aspect_ratio_value >= eye_aspect_ratio_threshold
                elif current_condition == 1:
                    condition = mask_prediction
                else:
                    condition = image_prediction

                if not condition:
                    frame_counter += 1
                    if frame_counter >= eye_consecutive_frames:
                        # if the alarm is not on, turn it on
                        if not alarm_on:
                            alarm_on = True
                            if last_alarm is not None:
                                if (time.time() - last_alarm) // 60 < 1:
                                    consecutive_alarms = True
                                else:
                                    consecutive_alarms = False
                            messageQueue.put(True)
                            count_thread = Thread(target=show_time)
                            count_thread.daemon = True
                            count_thread.start()
                            if alarm_path != "":
                                winsound.PlaySound(alarm_path, winsound.SND_FILENAME |
                                                   winsound.SND_LOOP | winsound.SND_ASYNC)
                else:
                    if alarm_on:
                        last_alarm = time.time()
                        winsound.PlaySound(None, winsound.SND_FILENAME)
                        messageQueue.put(False)
                        print("ALARM STOPPED")
                    frame_counter = 0
                    alarm_on = False

            if alarm_on:
                cv2.putText(frame, "GETTING SLEEPY", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 2)
                if consecutive_alarms:
                    cv2.putText(frame, "STOP CAR", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 2)
            cv2.putText(frame, "Eye ratios: " + str(round(left_eye_aspect_ratio, 2)) + " " +
                        str(round(right_eye_aspect_ratio, 2)), (150, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, threshold_color, 1)

            cv2.putText(frame, "Mask prediction awake: " + str(mask_prediction),
                        (100, 220), cv2.FONT_HERSHEY_DUPLEX, 0.4, mask_color, 1)
            cv2.putText(frame, "Image prediction awake: " + str(image_prediction),
                        (100, 235), cv2.FONT_HERSHEY_DUPLEX, 0.4, image_color, 1)
            cv2.putText(frame, "Sleep threshold: {} seconds".format(SECONDS), (10, 205),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, BLUE, 1)
            if SHOW_FPS:
                cv2.putText(frame, "FPS {}".format(int(FPS.fps())), (10, 235),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, BLUE, 1)
            if current_condition == 0:
                cv2.putText(frame, "*",
                            (135, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, BLUE, 1)
            elif current_condition == 1:
                cv2.putText(frame, "*",
                            (85, 230), cv2.FONT_HERSHEY_TRIPLEX, 1, BLUE, 1)
            else:
                cv2.putText(frame, "*",
                            (85, 245), cv2.FONT_HERSHEY_TRIPLEX, 1, BLUE, 1)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == ord("Q"):
                break
            if key == ord("m") or key == ord("M"):
                SHOW_MASK = not SHOW_MASK
            if key == ord("f") or key == ord("F"):
                SHOW_FPS = not SHOW_FPS
            if key == ord("s") or key == ord("S"):
                current_condition = switch_condition(current_condition)
            if key == ord("a") or key == ord("A"):
                SECONDS = max(SECONDS - 0.5, 2)
            if key == ord("d") or key == ord("D"):
                SECONDS += 0.5
            elif key == ord("c") or key == ord("C"):
                calibration_start = time.time()
                values = []
                process = True
                while time.time() - calibration_start < 10:
                    if vs.frame_read():
                        FPS.frame_skip()
                        continue
                    FPS.update()
                    frame = vs.read()
                    frame = frame[150:400, 150:450]
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces_detected = detector(gray, 0)
                    if faces_detected:
                        # for rect in faces_detected:
                        rect = faces_detected[0]
                        shape = predictor(gray, rect)
                        shape = utils.shape_to_np(shape)
                        left_eye = shape[l_start:l_end]
                        right_eye = shape[r_start:r_end]
                        left_eye_aspect_ratio = utils.eye_aspect_ratio(left_eye)
                        right_eye_aspect_ratio = utils.eye_aspect_ratio(right_eye)
                        left_eye_hull = cv2.convexHull(left_eye)
                        right_eye_hull = cv2.convexHull(right_eye)
                        cv2.drawContours(frame, [left_eye_hull], -1, YELLOW, 1)
                        cv2.drawContours(frame, [right_eye_hull], -1, YELLOW, 1)
                        values.append((left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0)

                    if left_eye_aspect_ratio > eye_aspect_ratio_threshold:
                        cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (200, 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.4, GREEN, 1)
                    else:
                        cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (200, 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.4, RED, 1)
                    if right_eye_aspect_ratio > eye_aspect_ratio_threshold:
                        cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.4, GREEN, 1)
                    else:
                        cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.4, RED, 1)
                    cv2.putText(frame, "CALIBRATING {}".format(int(time.time() - calibration_start)), (10, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 2)
                    cv2.putText(frame, "Stay in a natural position.".format(int(time.time() - calibration_start)),
                                (10, 200),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, BLUE, 2)
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        process = False
                        break
                if process is False:
                    break
                eye_aspect_ratio_threshold = (sum(values) / len(values) * 3 + 2 * eye_aspect_ratio_threshold) / 5
                print("New eye threshold", eye_aspect_ratio_threshold)

    finally:
        print("Done")
        print("FPS average:", FPS.fps(2))
        # print("Frames skipped per second:", FPS.get_frame_skip(2))
        cv2.destroyAllWindows()
        vs.stop()
        vs.stream.release()
        K.clear_session()

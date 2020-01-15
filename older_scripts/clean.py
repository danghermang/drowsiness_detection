import datetime
import queue
import time
from threading import Thread

import cv2
import dlib

import fps
import utils
import video_stream

alarm_path = r"../data/alarm.wav"
predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
default_webcam = 0
messageQueue = queue.Queue()
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)


def do_nothing():
    pass


def show_time():
    check = messageQueue.get()
    while check:
        print("Getting sleepy at", datetime.datetime.now().strftime("%H:%M:%S"))
        time.sleep(1)
        if not messageQueue.empty():
            messageQueue.get()
            break


count_thread = Thread(target=show_time, args=(True,))


def process_image():
    dimensions = (320, 180)
    dimensions = None
    vs = video_stream.VideoStream(src=default_webcam, dimensions=dimensions).start()
    rect = None
    face_approximate = None
    FPS = fps.FPS()
    try:

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 720, 480)
        # cv2.namedWindow("Original frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Original frame", 720, 480)
        # cv2.namedWindow("Fata",cv2.WINDOW_AUTOSIZE)

        last_alarm = None
        consecutive_alarms = False
        eye_aspect_ratio_threshold = 0.56
        eye_consecutive_frames = 50
        frame_counter = 0
        previous_frame_counter = 0
        alarm_on = False
        previous_face = None
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        # print(utils.FACIAL_LANDMARKS)
        l_start, l_end = utils.FACIAL_LANDMARKS["left_eye"]
        r_start, r_end = utils.FACIAL_LANDMARKS["right_eye"]
        FPS.start()
        # START = datetime.datetime.now()
        left_eye_aspect_ratio = right_eye_aspect_ratio = 0
        frame = vs.read()
        print("Frame count =", frame.shape)
        while True:
            if vs.frame_read():
                # cv2.imshow("Frame", frame)
                # cv2.imshow("Original frame", gray)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q") or key == ord("Q"):
                #     break
                FPS.frame_skip()
                continue
            FPS.update()
            # original_frame = cv2.flip(vs.read(), 1)
            frame = vs.read()
            frame = frame[150:400, 150:450]
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_face and previous_frame_counter < 30 and False:
                face_approximate = utils.get_face_approximate(gray, previous_face, 10)
                faces_detected = detector(face_approximate, 0)
                if len(faces_detected) != 0:
                    found_in_previous = True
                else:
                    found_in_previous = False
                    faces_detected = detector(gray, 0)
            else:
                previous_frame_counter = 0
                found_in_previous = False
                faces_detected = detector(gray, 0)
            if faces_detected:
                if not found_in_previous:
                    rect = faces_detected[0]
                    previous_face = rect
                else:
                    previous_frame_counter += 1
                    # cv2.imshow("Fata", face_approximate)
                    print("FOUND IN PREVIOUS FRAME", previous_frame_counter)
                # rect = faces_detected[0]
                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = utils.eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = utils.eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, YELLOW, 1)
                cv2.drawContours(frame, [right_eye_hull], -1, YELLOW, 1)
                # for shape_start, shape_end in utils.FACIAL_LANDMARKS.values():
                #     landmark = shape[shape_start:shape_end]
                #     for point in landmark:
                #         cv2.circle(frame, tuple(point), 1, GREEN, 1)

                if eye_aspect_ratio_value < eye_aspect_ratio_threshold:
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
                            # if alarm_path != "":
                            #     winsound.PlaySound(alarm_path, winsound.SND_FILENAME |
                            #                        winsound.SND_LOOP | winsound.SND_ASYNC)
                else:
                    if alarm_on:
                        last_alarm = time.time()
                        # winsound.PlaySound(None, winsound.SND_FILENAME)
                        messageQueue.put(False)
                        print("ALARM STOPPED")
                    frame_counter = 0
                    alarm_on = False

            if alarm_on:
                cv2.putText(frame, "GETTING SLEEPY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                if consecutive_alarms:
                    cv2.putText(frame, "STOP CAR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            if left_eye_aspect_ratio > eye_aspect_ratio_threshold:
                cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
            else:
                cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
            if right_eye_aspect_ratio > eye_aspect_ratio_threshold:
                cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
            else:
                cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
            cv2.putText(frame, "FPS {}".format(round(FPS.fps(), 3)), (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)

            cv2.imshow("Frame", frame)
            # cv2.imshow("Original frame", gray)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("c") or key == ord("C"):
                calibration_start = time.time()
                values = []
                process = True
                while time.time() - calibration_start < 10:

                    if vs.frame_read():
                        FPS.frame_skip()
                        continue
                    # original_frame = cv2.flip(vs.read(), 1)
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
                    else:
                        cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
                    if right_eye_aspect_ratio > eye_aspect_ratio_threshold:
                        cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
                    else:
                        cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (193, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
                    cv2.putText(frame, "CALIBRATING {}".format(int(time.time() - calibration_start)), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                    cv2.putText(frame, "Stay in a natural position.".format(int(time.time() - calibration_start)),
                                (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
                    cv2.imshow("Frame", frame)
                    # cv2.imshow("Original frame", gray)

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
        print("Frames skipped per second:", FPS.get_frame_skip(2))
        cv2.destroyAllWindows()
        vs.stop()
        vs.stream.release()


if __name__ == "__main__":
    processing = Thread(target=process_image)
    processing.deamon = True
    processing.start()

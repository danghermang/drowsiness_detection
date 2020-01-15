import datetime
import math
import queue
import time
from threading import Thread

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream

alarm_path = r"../data/alarm.wav"
predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
default_webcam = 0
messageQueue = queue.Queue()


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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


def eye_aspect_ratio(eye):
    a = euclidean_distance(eye[1], eye[5])
    b = euclidean_distance(eye[2], eye[4])
    c = euclidean_distance(eye[0], eye[3])
    return (a + b) / c


def process_image():
    process_start = 0
    total_frames = 0
    print("[INFO] starting video stream thread...")
    vs = VideoStream(default_webcam).start()
    try:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original frame", cv2.WINDOW_NORMAL)
        last_alarm = None
        consecutive_alarms = False
        eye_aspect_ratio_threshold = 0.60
        eye_consecutive_frames = 50
        frame_counter = 0
        alarm_on = False
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        print(face_utils.FACIAL_LANDMARKS_IDXS)
        l_start, l_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        r_start, r_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        time.sleep(1.0)
        process_start = time.time()
        while True:
            total_frames += 1
            time_for_frame = time.time()
            original_frame = vs.read()
            original_frame = cv2.flip(original_frame, 1)
            frame = imutils.resize(original_frame, width=360)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = detector(gray, 0)
            left_eye_aspect_ratio = right_eye_aspect_ratio = 0
            for rect in faces_detected:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = eye_aspect_ratio(right_eye)
                eye_aspect_ratio_value = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (100, 50, 255), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (100, 50, 255), 1)
                for shape_start, shape_end in face_utils.FACIAL_LANDMARKS_IDXS.values():
                    landmark = shape[shape_start:shape_end]
                    for point in landmark:
                        cv2.circle(frame, tuple(point), 1, (0, 255, 0), 1)
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
                        #     if alarm_path != "":
                        #         winsound.PlaySound(alarm_path, winsound.SND_FILENAME |
                        #                            winsound.SND_LOOP | winsound.SND_ASYNC)
                        cv2.putText(frame, "GETTING SLEEPY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if consecutive_alarms:
                            cv2.putText(frame, "STOP CAR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    if alarm_on:
                        last_alarm = time.time()
                        # winsound.PlaySound(None, winsound.SND_FILENAME)
                        messageQueue.put(False)
                        print("ALARM STOPPED")
                    frame_counter = 0
                    alarm_on = False
            frame = imutils.resize(frame, width=720)
            cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (520, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (510, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Miliseconds for frame {}".format(round(time.time() - time_for_frame, 5)), (370, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            cv2.imshow("Original frame", gray)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("c") or key == ord("C"):
                calibration_start = time.time()
                values = []
                process = True
                while time.time() - calibration_start < 10:
                    total_frames += 1
                    original_frame = vs.read()
                    original_frame = cv2.flip(original_frame, 1)
                    frame = imutils.resize(original_frame, width=360)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces_detected = detector(gray, 0)
                    left_eye_aspect_ratio = right_eye_aspect_ratio = 0
                    for rect in faces_detected:
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        left_eye = shape[l_start:l_end]
                        right_eye = shape[r_start:r_end]
                        left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
                        right_eye_aspect_ratio = eye_aspect_ratio(right_eye)
                        values.append((left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0)
                    frame = imutils.resize(frame, width=720)
                    cv2.putText(frame, "Left ratio: {:.2f}".format(left_eye_aspect_ratio), (520, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Right ratio: {:.2f}".format(right_eye_aspect_ratio), (510, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "CALIBRATING {}".format(int(time.time() - calibration_start)), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Original frame", gray)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        process = False
                if process is False:
                    break
                eye_aspect_ratio_threshold = sum(values) / len(values)
                print("New eye threshold", eye_aspect_ratio_threshold)
    finally:
        print("Done")
        print("FPS average:", round(total_frames / (time.time() - process_start)))
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    processing = Thread(target=process_image)
    processing.deamon = True
    processing.start()

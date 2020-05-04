import math
import time
from collections import Counter

import cv2
import dlib
import numpy as np

from utils import common_functions


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def calculateFingers(res, drawing):
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, YELLOW, -1)
                    cv2.circle(drawing, end, 8, BLUE, -1)
                    cv2.circle(drawing, start, 8, BLUE, -1)
            return True, cnt
    return False, 0


RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

predictor_path = "shape_predictor_68_Face_landmarks.dat"
cap = cv2.VideoCapture(0)
# bgModel = cv2.createBackgroundSubtractorMOG2()
# try:
#     while True:
#         frame = cv2.flip(cap.read()[1],1)
#         frame = frame[250:,350:]
#         fg = bgModel.apply(frame)
#         cv2.imshow('imagine', fg)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q") or key == ord("Q"):
#             break
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
time.sleep(1)
skinRGB = np.array(([234, 173, 96], [255, 224, 189]), dtype=np.uint8)
skinHSV = np.array(([0, 48, 80], [20, 225, 255]), dtype=np.uint8)
# skinHSV = np.array(([0, 48, 80], [50, 225, 255]), dtype=np.uint8)
skinYCR = np.array(([0, 133, 77], [255, 173, 127]), dtype=np.uint8)
_, frame = cap.read()
print(len(frame), len(frame[0]))
value = [0] * 20
try:
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[200:, 300:]
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        skinMask = cv2.inRange(converted, skinHSV[0], skinHSV[1])
        # skinMask = cv2.inRange(converted, skinRGB[0], skinRGB[1])
        # skinMask = cv2.inRange(converted, skinYCR[0], skinYCR[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        # skinMask = cv2.erode(skinMask, kernel, iterations=1)
        # skinMask = cv2.dilate(skinMask, kernel, iterations=1)

        contour_image, contours, hierachy = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_index = -1
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > max_area and area > 4000:
                max_area = area
                max_index = i
        drawing = np.zeros(frame.shape, np.uint8)

        if max_index != -1:
            x, y = 0, 0
            # for element in contours[max_index]:
            #     x+=element[0][0]
            #     y+=element[0][1]
            # x,y=int(x/len(contours[max_index])),int(y/len(contours[max_index]))+30

            hull = cv2.convexHull(contours[max_index])
            cv2.drawContours(frame, contours, max_index, (0, 255, 0), 3)
            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)
            cv2.drawContours(drawing, contours, max_index, (0, 255, 0), 3)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            isFinishCal, cnt = calculateFingers(contours[max_index], drawing)
            if isFinishCal is True:
                value.pop(0)
                value.append(cnt)
                approx_value = Most_Common(value)
                if approx_value == 0:
                    cv2.putText(frame, "NO GESTURE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                else:
                    cv2.putText(frame, "{} FINGERS".format(min(approx_value + 1, 5)), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            # cv2.putText(frame, str(max_area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        else:
            cv2.putText(frame, "NO HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        final = common_functions.resize(np.hstack((frame, drawing)), height=700)
        cv2.imshow('output', final)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

import cv2

roi = cv2.imread('D:\opencv\drowsiness-detection\WIN_20180228_15_13_14_Pro.jpg')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

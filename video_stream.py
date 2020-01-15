from threading import Thread

import cv2


class VideoStream:
    def __init__(self, src=0, dimensions=None):
        self.stream = cv2.VideoCapture(src)
        if dimensions:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, dimensions[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, dimensions[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.frame_check = False

    def start(self):
        t = Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def isOpened(self):
        return self.stream.isOpened()

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()
            self.frame_check = False

    def read(self):
        self.frame_check = True
        return self.frame

    def stop(self):
        self.stopped = True

    def get_dimensions(self):
        return self.stream.get(cv2.CAP_PROP_FRAME_WIDTH), self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def set_dimensions(self, width, height):
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frame_read(self):
        return self.frame_check

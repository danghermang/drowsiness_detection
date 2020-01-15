import datetime


class FPS:
    def __init__(self):
        self._start = None
        self._stop = None
        self._current = None
        self._total_frames = 0
        self._frames = 0
        self._frames_skipped = 0
        self._fps = 20

    def start(self):
        self._start = datetime.datetime.now()
        self._stop = datetime.datetime.now()
        self._current = datetime.datetime.now()
        return self

    def stop(self):
        self._stop = datetime.datetime.now()

    def update(self):
        self._total_frames += 1
        self._current = datetime.datetime.now()
        difference = (self._current - self._stop).total_seconds()
        if difference >= 1:
            self._fps = self._frames / difference
            self._frames = 1
            self._stop = datetime.datetime.now()
        else:
            self._frames += 1

    def frame_skip(self):
        self._frames_skipped += 1

    def elapsed(self):
        return (self._stop - self._start).total_seconds()

    def fps(self, rounding=-1):
        if rounding > -1:
            return round(self._fps, rounding)
        else:
            return self._fps

    def average_fps(self, rounding=-1):
        self.stop()
        fps = self._total_frames / (self._stop - self._start).total_seconds()
        if rounding > -1:
            return round(fps, rounding)
        else:
            return fps

    def get_frame_skip(self, rounding=-1):
        if rounding > -1:
            return round(self._frames_skipped / self.elapsed(), rounding)
        else:
            return self._frames_skipped / self.elapsed()

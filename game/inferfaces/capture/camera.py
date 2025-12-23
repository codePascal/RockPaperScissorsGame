#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Camera capture

Module implements the capture element for camera inputs via OpenCV.
"""
import threading
import queue

import cv2
import numpy as np

from .base import Capture


class Camera(Capture):

    def __init__(self, index: int = 0, mirror: bool = False):
        self._mirror = mirror
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._reader_th = None
        self._buffer = queue.Queue(maxsize=5)
        self._is_grabbing = False

    def _read(self):
        # Read continuously from camera.
        while self._is_grabbing:
            ret, frame = self._cap.read()
            if not ret:
                continue
            self._buffer.put(frame)

    def set_resolution(self, width: int, height: int) -> None:
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_resolution(self) -> tuple[int, int]:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def start_grabbing(self):
        self._reader_th = threading.Thread(target=self._read)
        self._reader_th.daemon = True
        self._is_grabbing = True
        self._reader_th.start()

    def stop_grabbing(self):
        self._is_grabbing = False
        self._reader_th.join()

    def setup(self) -> None:
        self.start_grabbing()

    def read(self) -> np.ndarray:
        frame = self._buffer.get()
        if self._mirror:
            return cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        self.stop_grabbing()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Camera module

Module implements the image input for the game.
"""
import logging
import threading
import time

import cv2
import numpy as np
import queue

logger = logging.getLogger(__name__)


class Camera:
    """Camera

    Class implements a OpenCV VideoCapture element to read images from camera
    input. Implements threaded reading, using a buffer to drop frames into and
    hence improve frame-rate.
    """

    def __init__(self, index: int = 0):
        """Create an instance of Camera class.

        Args:
             index: The camera index, defaults to 0.
        """
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            logger.info(f'Opened camera at index {index}')

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
        """Set resolution of camera."""
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        w, h = self.get_resolution()
        logger.info(f'Camera resolution set to {w}x{h}')

    def get_resolution(self) -> tuple[int, int]:
        """Get resolution (width, height) from camera."""
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def start_grabbing(self):
        """Start grabbing thread."""
        self._reader_th = threading.Thread(target=self._read)
        self._reader_th.daemon = True
        self._is_grabbing = True
        self._reader_th.start()
        logger.info('Camera started grabbing')

    def stop_grabbing(self):
        """Stop grabbing thread."""
        self._is_grabbing = False
        self._reader_th.join()
        logger.info('Camera stopped grabbing')

    def read(self) -> np.ndarray:
        """Get image from buffer."""
        return self._buffer.get()

    def fps(self, num_frames: int = 10):
        """Measure the frame rate over num_frames."""
        self.start_grabbing()
        start_time = time.time()
        for _ in range(num_frames):
            _ = self.read()
        end_time = time.time()
        self.stop_grabbing()
        diff_time = end_time - start_time
        if diff_time > 0:
            return num_frames / diff_time
        return np.nan

    def stream(self):
        """Stream the camera output to a window."""
        win = 'Preview (Press Q to quit)'
        cv2.namedWindow(win)
        self.start_grabbing()
        while not cv2.waitKey(1) & 0xFF == ord('q'):
            frame = self.read()
            cv2.imshow(win, frame)
        self.stop_grabbing()
        cv2.destroyAllWindows()

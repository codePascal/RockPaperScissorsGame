#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference utils

Module implements utils for inference such as ROI selection, debounce filter.
"""
from dataclasses import dataclass
import cv2
import time

import numpy as np


@dataclass
class ImgRoi:
    x0: int
    y0: int
    w: int
    h: int
    color: tuple = None
    thickness: int = 2
    x1: int = None
    y2: int = None

    def __post_init__(self):
        if self.color is None:
            self.color = (0, 0, 0)
        self.x1 = self.x0 + self.w
        self.y1 = self.y0 + self.h

    def top_left_corner(self):
        return self.x0, self.y0

    def bottom_right_corner(self):
        return self.x1, self.y1

    def get(self, frame: np.ndarray):
        return frame[self.y0:self.y1, self.x0:self.x1]

    def draw(self, frame: np.ndarray):
        cv2.rectangle(
            frame,
            self.top_left_corner(),
            self.bottom_right_corner(),
            color=self.color,
            thickness=self.thickness
        )


@dataclass
class DebounceConfig:
    min_confidence: float = 0.7
    stable_frames: int = 8
    cooldown: float = 0.6


class InferDebouncer:

    def __init__(self, cfg: DebounceConfig):
        self._cfg = cfg

        self._candidate = None
        self._count = 0
        self._cooldown_timeout = 0.0

    def cooldown(self):
        return time.time() < self._cooldown_timeout

    def update(self, pred: str = None, confidence: float = None) -> str | None:
        now = time.time()

        if self.cooldown():
            return None

        if pred is None or confidence < self._cfg.min_confidence:
            self._candidate = None
            self._count = 0
            return None

        if pred != self._candidate:
            self._candidate = pred
            self._count = 1
            return None

        self._count += 1
        if self._count >= self._cfg.stable_frames:
            accepted = self._candidate
            self._cooldown_timeout = now + self._cfg.cooldown
            self._candidate = None
            self._count = 0
            return accepted

        return None

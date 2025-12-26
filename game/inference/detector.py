#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPS Landmark Detector

Module implements an RPS landmark detector using mediapipe Hands featuring
21 landmarks per hand.
"""
import cv2
import numpy as np
import mediapipe as mp

from .base import Inference

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
Landmarks = mp_hands.HandLandmark


class Detector(Inference):

    def __init__(self):
        self._extractor = HandsExtractor()

        self._extend_th = 160
        self._confidence = 0.9

    def predict(self, frame_bgr: np.ndarray) -> tuple[str | None, float]:
        hand = self._extractor.extract(frame_bgr)
        if hand is None:
            return None, self._confidence
        hand_array = self.lm_to_array(hand.landmark)
        pred = self.classify_rps(hand_array)
        return pred, self._confidence

    @staticmethod
    def lm_to_array(landmarks):
        return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    def classify_rps(self, landmarks) -> str | None:
        idx = self.extended_finger(
            landmarks,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.INDEX_FINGER_DIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        )
        mid = self.extended_finger(
            landmarks,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        )
        ring = self.extended_finger(
            landmarks,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_DIP,
            mp_hands.HandLandmark.RING_FINGER_TIP
        )
        pinky = self.extended_finger(
            landmarks,
            mp_hands.HandLandmark.PINKY_MCP,
            mp_hands.HandLandmark.PINKY_PIP,
            mp_hands.HandLandmark.PINKY_DIP,
            mp_hands.HandLandmark.PINKY_TIP
        )

        if idx and mid and ring and pinky:
            return 'PAPER'
        if idx and mid and (not ring) and (not pinky):
            return 'SCISSORS'
        if (not idx) and (not mid) and (not ring) and (not pinky):
            return 'ROCK'
        return None

    def extended_finger(
            self,
            landmarks: np.ndarray,
            mcp: int,
            pip: int,
            dip: int,
            tip: int
    ) -> bool:
        a1 = angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        a2 = angle(landmarks[pip], landmarks[dip], landmarks[tip])
        return (a1 > self._extend_th) and (a2 > self._extend_th)


class HandsExtractor:

    def __init__(self, min_det_conf: float = 0.5, min_track_conf: float = 0.5):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(frame_rgb)
        if not res.multi_hand_landmarks:
            return None
        return res.multi_hand_landmarks[0]

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks):
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    ba /= (np.linalg.norm(ba) + 1e-8)
    bc /= (np.linalg.norm(bc) + 1e-8)
    cos = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

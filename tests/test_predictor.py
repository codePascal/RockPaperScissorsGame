#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import cv2

from game import inference

from tests.context import test_images


class TestRPSPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = inference.Predictor()

    def test_paper(self):
        img_path = test_images().joinpath('testpaper01-00.png')
        frame_bgr = cv2.imread(str(img_path))
        pred, _ = self.model.predict(frame_bgr)
        self.assertEqual('PAPER', pred)

    def test_rock(self):
        img_path = test_images().joinpath('testrock01-00.png')
        frame_bgr = cv2.imread(str(img_path))
        pred, _ = self.model.predict(frame_bgr)
        self.assertEqual('ROCK', pred)

    def test_scissors(self):
        img_path = test_images().joinpath('testscissors01-00.png')
        frame_bgr = cv2.imread(str(img_path))
        pred, _ = self.model.predict(frame_bgr)
        self.assertEqual('SCISSORS', pred)

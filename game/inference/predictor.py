#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPS Model Predictor

Module implements gateway to inference pipeline for the rps-model.
"""
import numpy as np

from model import infer
from .base import Inference


class Predictor(Inference):

    def __init__(self):
        self._model = infer.load_model()

    def predict(self, frame_bgr: np.ndarray) -> tuple[str | None, float]:
        frame_rgb = infer.frame_to_batch(frame_bgr)
        pred, pred_pct = infer.infer(self._model, frame_rgb)
        return pred.upper(), pred_pct

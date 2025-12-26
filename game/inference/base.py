#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base Inference Protocol

Module implements the basic architecture for inference.
"""
from typing import Protocol

import numpy as np


class Inference(Protocol):

    def predict(self, frame_bgr: np.ndarray) -> tuple[str | None, float]: ...

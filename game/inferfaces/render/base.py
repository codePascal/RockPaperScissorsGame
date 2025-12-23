#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base Render Protocol

Module implements the basic architecture for render elements.
"""
from typing import Protocol
from dataclasses import dataclass

import numpy as np


@dataclass
class RenderContext:
    frame_bgr: np.ndarray
    roi_bgr: np.ndarray
    state: dict
    prediction: float = None
    player_move: str = None
    opponent_move: str = None
    outcome: str = None


class Renderer(Protocol):

    def setup(self) -> None: ...

    def render(self, ctx: RenderContext) -> bool: ...

    def close(self) -> None: ...

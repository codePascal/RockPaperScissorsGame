#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenCV Overlay

Module implements the render overlay using OpenCV.
"""
from pathlib import Path

import cv2
import numpy as np

from .base import Renderer, RenderContext


class OpenCVOverlay(Renderer):
    window_name: str = 'Rock-Paper-Scissors-Game'

    def __init__(self):
        res = Path(__file__).parent.parent.parent.parent.joinpath('res')
        self._opponent_images = {
            'rock': res.joinpath('rock.png'),
            'paper': res.joinpath('paper.png'),
            'scissors': res.joinpath('scissors.png')
        }

        self._size = 224
        self._width = 3 * self._size
        self._border = 2
        self._size_border = self._size - 2 * self._border

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_size = 0.8
        self._thickness = 2
        self._color = (0, 255, 0)

        self._text_cfg = {
            'fontFace': self._font,
            'fontScale': self._font_size,
            'color': self._color,
            'thickness': self._thickness,
            'lineType': cv2.LINE_AA
        }

    def _draw_border(self, frame: np.ndarray, color: tuple):
        return cv2.copyMakeBorder(
            frame,
            top=self._border,
            bottom=self._border,
            right=self._border,
            left=self._border,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )

    def _add_text(self, frame: np.ndarray, text: str, y: int):
        text_w, text_h = cv2.getTextSize(
            text,
            self._font,
            self._font_size,
            self._thickness
        )[0]
        x_pos = frame.shape[1] // 2 - text_w // 2
        cv2.putText(frame, text, (x_pos, y), **self._text_cfg)

    def _opponent_frame(
            self,
            *,
            opponent_move: str = None,
            outcome: str = None
    ) -> np.ndarray:
        if opponent_move is None:
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)

        frame = cv2.imread(self._opponent_images[opponent_move.lower()])
        frame = cv2.resize(frame, (self._size_border, self._size_border))

        if outcome == 'OPPONENT_WINS':
            frame = self._draw_border(frame, (0, 255, 0))
        elif outcome == 'PLAYER_WINS':
            frame = self._draw_border(frame, (0, 0, 255))
        elif outcome == 'DRAW':
            frame = self._draw_border(frame, (255, 0, 0))
        else:
            frame = self._draw_border(frame, (0, 0, 0))

        return frame

    def _player_frame(
            self,
            roi_bgr: np.ndarray,
            *,
            player_move: str = None,
            confidence: float = None,
            outcome: str = None
    ) -> np.ndarray:
        if player_move is None:
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)

        frame = cv2.resize(roi_bgr, (self._size_border, self._size_border))

        text = f'{player_move} ({confidence:.2%})'
        self._add_text(frame, text, y=25)

        if outcome == 'OPPONENT_WINS':
            frame = self._draw_border(frame, (0, 0, 255))
        elif outcome == 'PLAYER_WINS':
            frame = self._draw_border(frame, (0, 255, 0))
        elif outcome == 'DRAW':
            frame = self._draw_border(frame, (255, 0, 0))
        else:
            frame = self._draw_border(frame, (0, 0, 0))

        return frame

    def _state_frame(
            self,
            round_key: int,
            player_score: int,
            opponent_score: int
    ) -> np.ndarray:
        frame = np.zeros((self._size, self._size, 3), dtype=np.uint8)

        text = f'Round {round_key}'
        self._add_text(frame, text, y=25)

        text = f'{player_score} : {opponent_score}'
        self._add_text(frame, text, y=100)

        return frame

    def _capture_frame(self, frame: np.ndarray) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        if w_frame != self._width:
            scale = self._width / w_frame
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
        return frame

    def setup(self) -> None:
        cv2.namedWindow(self.window_name)

    def render(self, ctx: RenderContext) -> bool:
        frame = self._capture_frame(ctx.frame_bgr.copy())
        player_frame = self._player_frame(
            ctx.roi_bgr,
            player_move=ctx.player_move,
            confidence=ctx.prediction,
            outcome=ctx.outcome
        )
        opponent_frame = self._opponent_frame(
            opponent_move=ctx.opponent_move,
            outcome=ctx.outcome
        )
        state_frame = self._state_frame(
            ctx.state['round'],
            ctx.state['player_score'],
            ctx.state['opponent_score']
        )

        h_frame, w_frame = frame.shape[:2]
        final = np.zeros((h_frame + self._size, w_frame, 3), dtype=np.uint8)
        final[self._size:, :] = frame
        final[:self._size, :self._size] = player_frame
        final[:self._size, self._size:2 * self._size] = state_frame
        final[:self._size, w_frame - self._size:] = opponent_frame

        cv2.imshow(self.window_name, final)
        key = cv2.waitKey(1) & 0xFF
        return not (key == 27 or key == ord('q'))

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

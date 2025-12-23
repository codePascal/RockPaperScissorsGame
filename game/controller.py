#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPS Game Controller

Module implements the controller to connect inputs, inference and outputs.
"""
from . import engine, inference
from .inference import utils
from .inferfaces import capture, render


class Controller:

    def __init__(
            self,
            cap: capture.Capture,
            predictor: inference.Inference,
            overlay: render.Renderer,
            game: engine.Game,
            *,
            roi: utils.ImgRoi = None,
            confidence_threshold: float = 0.7,
            cooldown: float = 0.6,
            stable_frames: int = 8
    ):
        self._cap = cap
        self._predictor = predictor
        self._overlay = overlay
        self._game = game

        self._roi = roi

        debounce_config = utils.DebounceConfig(
            min_confidence=confidence_threshold,
            stable_frames=stable_frames,
            cooldown=cooldown
        )
        self._debouncer = utils.InferDebouncer(debounce_config)

        self._last_ctx = None

    def run(self) -> None:
        self._cap.setup()
        self._overlay.setup()

        while True:
            frame = self._cap.read()
            if frame is None:
                break

            if self._roi is None:
                frame_roi = frame
            else:
                frame_roi = self._roi.get(frame).copy()
                self._roi.draw(frame)

            pred, pred_pct = self._predictor.predict(frame_roi)
            pred_debounce = self._debouncer.update(pred, pred_pct)

            if pred_debounce:
                # Valid new prediction --> display it
                player_move = engine.Move[pred]
                opponent_move, outcome = self._game.advance(player_move)
                ctx = render.RenderContext(
                    frame_bgr=frame,
                    roi_bgr=frame_roi,
                    state=self._game.current_state().to_dict(),
                    prediction=pred_pct,
                    player_move=player_move.name,
                    opponent_move=opponent_move.name,
                    outcome=outcome.name
                )
                self._last_ctx = ctx
            elif self._debouncer.cooldown():
                # Show last result during cooldown, but update frame
                ctx = self._last_ctx
                ctx.frame_bgr = frame
            else:
                # No prediction, display frame only
                ctx = render.RenderContext(
                    frame_bgr=frame,
                    roi_bgr=frame_roi,
                    state=self._game.current_state().to_dict()
                )

            if not self._overlay.render(ctx):
                break

        self._cap.close()
        self._overlay.close()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from game import engine, inference, controller
from game.inferfaces import capture, render
from game.inference import utils


def main():
    app = controller.Controller(
        cap=capture.Camera(index=0, mirror=True),
        predictor=inference.Predictor(),
        overlay=render.OpenCVOverlay(),
        roi=utils.ImgRoi(0, 0, 200, 200),
        game=engine.Game(),
        confidence_threshold=0.2,
        cooldown=4
    )
    app.run()


if __name__ == '__main__':
    main()

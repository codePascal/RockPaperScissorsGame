#!/usr/bin/env python
# -*- coding: utf-8 -*-
from game import engine, inference, controller
from game.inferfaces import capture, render
from game.inference import utils


def main():
    app = controller.Controller(
        cap=capture.Camera(index=0, mirror=True),
        predictor=inference.Detector(),
        overlay=render.OpenCVOverlay(),
        roi=utils.ImgRoi(x0=0, y0=0, w=200, h=200),
        game=engine.Game(),
        confidence_threshold=0.6,
        cooldown=4
    )
    app.run()


if __name__ == '__main__':
    main()

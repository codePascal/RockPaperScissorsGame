#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path


def res() -> Path:
    return Path(__file__).parent.parent.joinpath('res')


def test_images() -> Path:
    return res().joinpath('images')

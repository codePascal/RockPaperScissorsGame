#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base Capture Protocol

Module implements the basic architecture for capture elements.
"""
from typing import Protocol

import numpy as np


class Capture(Protocol):

    def setup(self) -> None: ...

    def read(self) -> np.ndarray: ...

    def close(self) -> None: ...

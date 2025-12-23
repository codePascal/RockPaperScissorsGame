#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Game rules

Module implements the basic game setup and the rules.
"""
from enum import Enum, auto


class Move(Enum):
    ROCK = auto()
    PAPER = auto()
    SCISSORS = auto()


class Outcome(Enum):
    PLAYER_WINS = auto()
    OPPONENT_WINS = auto()
    DRAW = auto()


def decide(player: Move, opponent: Move) -> Outcome:
    if player == opponent:
        return Outcome.DRAW
    if (
            player == Move.ROCK and opponent == Move.SCISSORS or
            player == Move.PAPER and opponent == Move.ROCK or
            player == Move.SCISSORS and opponent == Move.PAPER
    ):
        return Outcome.PLAYER_WINS
    return Outcome.OPPONENT_WINS

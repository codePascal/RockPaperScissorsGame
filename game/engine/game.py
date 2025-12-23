#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Game logic

Module implements the logic to handle opponent, classify outcomes and
maintain a game state.
"""
import random
from dataclasses import dataclass, asdict

from .rules import Move, Outcome, decide


@dataclass
class GameState:
    round: int = 0
    player_score: int = 0
    opponent_score: int = 0
    draws: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Game:

    def __init__(self):
        self._opponent = Opponent()
        self._state = GameState()

    def advance(self, player_move: Move) -> tuple[Move, Outcome]:
        opponent_move = self._opponent.choose_move()
        outcome = decide(player=player_move, opponent=opponent_move)

        self._state.round += 1
        if outcome == Outcome.OPPONENT_WINS:
            self._state.opponent_score += 1
        elif outcome == Outcome.PLAYER_WINS:
            self._state.player_score += 1
        else:
            self._state.draws += 1

        return opponent_move, outcome

    def current_state(self) -> GameState:
        return self._state


class Opponent:

    @staticmethod
    def choose_move() -> Move:
        return random.choice([Move.ROCK, Move.PAPER, Move.SCISSORS])

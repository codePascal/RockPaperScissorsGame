#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import random

from game import engine

random.seed(42)


class TestRulesModule(unittest.TestCase):

    def test_decide_player_wins_with_rock(self):
        player_move = engine.Move.ROCK
        opponent_move = engine.Move.SCISSORS
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.PLAYER_WINS, outcome)

    def test_decide_player_wins_with_paper(self):
        player_move = engine.Move.PAPER
        opponent_move = engine.Move.ROCK
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.PLAYER_WINS, outcome)

    def test_decide_player_wins_with_scissors(self):
        player_move = engine.Move.SCISSORS
        opponent_move = engine.Move.PAPER
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.PLAYER_WINS, outcome)

    def test_decide_opponent_wins_with_rock(self):
        player_move = engine.Move.SCISSORS
        opponent_move = engine.Move.ROCK
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.OPPONENT_WINS, outcome)

    def test_decide_opponent_wins_with_paper(self):
        player_move = engine.Move.ROCK
        opponent_move = engine.Move.PAPER
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.OPPONENT_WINS, outcome)

    def test_decide_opponent_wins_with_scissors(self):
        player_move = engine.Move.PAPER
        opponent_move = engine.Move.SCISSORS
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.OPPONENT_WINS, outcome)

    def test_decide_draw_with_rock(self):
        player_move = engine.Move.ROCK
        opponent_move = engine.Move.ROCK
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)

    def test_decide_draw_with_paper(self):
        player_move = engine.Move.PAPER
        opponent_move = engine.Move.PAPER
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)

    def test_decide_draw_with_scissors(self):
        player_move = engine.Move.SCISSORS
        opponent_move = engine.Move.SCISSORS
        outcome = engine.decide(player_move, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)


class TestGameModule(unittest.TestCase):

    def test_advance_current_state(self):
        gm = engine.Game()

        opponent_move, outcome = gm.advance(engine.Move.PAPER)
        self.assertEqual(engine.Move.SCISSORS, opponent_move)
        self.assertEqual(engine.Outcome.OPPONENT_WINS, outcome)

        opponent_move, outcome = gm.advance(engine.Move.ROCK)
        self.assertEqual(engine.Move.ROCK, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)

        opponent_move, outcome = gm.advance(engine.Move.SCISSORS)
        self.assertEqual(engine.Move.ROCK, opponent_move)
        self.assertEqual(engine.Outcome.OPPONENT_WINS, outcome)

        opponent_move, outcome = gm.advance(engine.Move.ROCK)
        self.assertEqual(engine.Move.SCISSORS, opponent_move)
        self.assertEqual(engine.Outcome.PLAYER_WINS, outcome)

        opponent_move, outcome = gm.advance(engine.Move.PAPER)
        self.assertEqual(engine.Move.PAPER, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)

        opponent_move, outcome = gm.advance(engine.Move.ROCK)
        self.assertEqual(engine.Move.ROCK, opponent_move)
        self.assertEqual(engine.Outcome.DRAW, outcome)

        state = gm.current_state()
        self.assertEqual(6, state.round)
        self.assertEqual(1, state.player_score)
        self.assertEqual(2, state.opponent_score)
        self.assertEqual(3, state.draws)




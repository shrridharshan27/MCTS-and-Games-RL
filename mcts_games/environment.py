"""
Fast Tic-Tac-Toe Environment for Multi-Agent RL & MCTS.
"""
from typing import List, Optional, Tuple
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0=empty, 1=X, -1=O
        self.player = 1                      # 1 (X) starts first

    def clone(self) -> 'TicTacToe':
        g = TicTacToe()
        g.board = self.board.copy()
        g.player = self.player
        return g

    def legal_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]

    def make_move(self, pos: int) -> None:
        self.board[pos] = self.player
        self.player = -self.player

    def check_winner(self) -> Optional[int]:
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        b = self.board
        for l in lines:
            s = b[l[0]] + b[l[1]] + b[l[2]]
            if s == 3:  return 1
            if s == -3: return -1
        return None

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or len(self.legal_moves()) == 0

    def result(self) -> int:
        w = self.check_winner()
        return w if w is not None else 0

    def state_key(self) -> Tuple[int, ...]:
        return tuple(self.board)

    def display(self) -> None:
        sym = {0: '.', 1: 'X', -1: 'O'}
        for i in range(3):
            print(' '.join(sym[self.board[i * 3 + j]] for j in range(3)))
        print()

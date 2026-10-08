"""
Tournament Engine for Competitive RL Agents.
"""
from typing import Tuple
from .environment import TicTacToe

def play_game(agent1, agent2, verbose: bool = False) -> int:
    game = TicTacToe()
    agents = {1: agent1, -1: agent2}
    while not game.is_terminal():
        mv = agents[game.player].select_move(game)
        game.make_move(mv)
    if verbose:
        game.display()
    return game.result()

def run_tournament(agent_a, agent_b, n_games: int = 100, name_a: str = 'A', name_b: str = 'B') -> Tuple[int, int, int]:
    wins_a = draws = wins_b = 0
    for i in range(n_games):
        if i % 2 == 0:
            r = play_game(agent_a, agent_b)
            if r == 1:   wins_a += 1
            elif r == -1: wins_b += 1
            else:        draws += 1
        else:
            r = play_game(agent_b, agent_a)
            if r == 1:   wins_b += 1
            elif r == -1: wins_a += 1
            else:        draws += 1
    return wins_a, draws, wins_b

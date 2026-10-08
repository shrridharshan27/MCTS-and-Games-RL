"""
Game Playing Agents: Random, Q-Learning, and Monte Carlo Tree Search (MCTS).
"""
import math, random
from collections import defaultdict
from typing import Dict, List, Optional
from .environment import TicTacToe

class RandomAgent:
    def select_move(self, game: TicTacToe) -> int:
        return random.choice(game.legal_moves())

class QLearningAgent:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.9, epsilon: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.training = True

    def select_move(self, game: TicTacToe) -> int:
        moves = game.legal_moves()
        if self.training and random.random() < self.epsilon:
            return random.choice(moves)
        state = game.state_key()
        return max(moves, key=lambda m: self.Q[state][m])

    def update(self, state, action, reward, next_state, next_moves) -> None:
        max_next = max((self.Q[next_state][m] for m in next_moves), default=0.0)
        td_target = reward + self.gamma * max_next
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

def train_qlearning(agent: QLearningAgent, n_episodes: int = 50000) -> None:
    for ep in range(n_episodes):
        game = TicTacToe()
        history = []
        while not game.is_terminal():
            state = game.state_key()
            action = agent.select_move(game)
            p = game.player
            history.append((state, action, p))
            game.make_move(action)

        final_result = game.result()
        for i, (s, a, p) in enumerate(reversed(history)):
            reward = final_result * p
            agent.update(s, a, reward, game.state_key(), [])

class MCTSNode:
    def __init__(self, state: TicTacToe, parent: Optional['MCTSNode'] = None, move: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: Dict[int, 'MCTSNode'] = {}
        self.wins = 0.0
        self.visits = 0
        self.untried = state.legal_moves()

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1())

class MCTSAgent:
    def __init__(self, n_simulations: int = 300):
        self.n_simulations = n_simulations

    def select_move(self, game: TicTacToe) -> int:
        root = MCTSNode(game.clone())
        for _ in range(self.n_simulations):
            node = root
            # 1. Selection
            while not node.state.is_terminal() and not node.untried:
                node = node.best_child()

            # 2. Expansion
            if node.untried and not node.state.is_terminal():
                move = random.choice(node.untried)
                node.untried.remove(move)
                child_state = node.state.clone()
                child_state.make_move(move)
                child = MCTSNode(child_state, parent=node, move=move)
                node.children[move] = child
                node = child

            # 3. Simulation (random rollout)
            sim = node.state.clone()
            while not sim.is_terminal():
                sim.make_move(random.choice(sim.legal_moves()))
            res = sim.result()

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.wins += res * (-curr.state.player)
                curr = curr.parent

        return max(root.children, key=lambda m: root.children[m].visits)

# Monte Carlo Tree Search (MCTS) & Multi-Agent Game Playing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-11557C.svg?logo=python&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author
- **Shrri Dharshan D R** — [@shrridharshan27](https://github.com/shrridharshan27)
- **Registration Number:** 23BPS1090
- **Course:** Reinforcement Learning Lab (BCSE432E)
- **Lab Slot:** L31+L32

---

## Overview
This repository implements and benchmarks three paradigms of game-playing intelligence in **Tic-Tac-Toe**:
1. **Random Agent:** Uniform random baseline.
2. **Q-Learning Agent:** Model-free tabular temporal difference control trained via 50,000 self-play episodes.
3. **Monte Carlo Tree Search (MCTS):** Online decision-time search algorithm using Upper Confidence Bound for Trees (UCT / UCB1).

The agents are evaluated through head-to-head tournament matchups and across varying MCTS simulation search budgets.

---

## Algorithmic Architecture

### 1. Monte Carlo Tree Search (MCTS) 4-Stage Lifecycle
```text
  [ Selection ] ----> [ Expansion ] ----> [ Simulation ] ----> [ Backpropagation ]
   Walk down tree       Add legal child     Random rollout       Propagate wins
    using UCB1            node to tree        to terminal        up to root
```

#### Upper Confidence Bound for Trees (UCB1) Formula
At each internal tree node, child selection balances exploitation and exploration:
$$\text{UCB1}(i) = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}$$
Where:
- $w_i$: Number of simulated wins for child node $i$.
- $n_i$: Visit count of child node $i$.
- $N$: Total visit count of parent node.
- $c = \sqrt{2} \approx 1.414$: Exploration constant.

### 2. Tabular Q-Learning with Self-Play
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
Trained via 50,000 self-play episodes with $\alpha = 0.3, \gamma = 0.9, \epsilon = 0.2$.

---

## Experimental Results

![Tournament Results](assets/tournament_results.png)
![MCTS Simulation Budget](assets/mcts_budget_scaling.png)

### Tournament Performance Summary
| Matchup | Player 1 Win Rate | Draw Rate | Player 2 Win Rate | Key Takeaway |
|---|---|---|---|---|
| **Random vs Random** | ~58% | ~13% | ~29% | First-player bias in random play |
| **Q-Learning vs Random** | **~85%** | ~11% | ~4% | Q-learning severely punishes random mistakes |
| **MCTS vs Random** | **~90%+** | ~8% | ~2% | MCTS simulations uncover winning paths |
| **MCTS vs Q-Learning** | **~60%** | ~35% | ~5% | MCTS outplays tabular offline policy |

### MCTS Simulation Budget Analysis
As the search budget increases from 10 to 500 simulations per move, MCTS win rate steadily approaches near-perfect play ($>90\%$) against stochastic opponents.

---

## Project Structure
```text
MCTS-and-Games-RL/
├── mcts_games/
│   ├── __init__.py
│   ├── environment.py   # TicTacToe board state engine
│   ├── agents.py        # RandomAgent, QLearningAgent, MCTSAgent
│   └── tournament.py    # Match simulator & win tracking
├── assets/
│   ├── tournament_results.png
│   └── mcts_budget_scaling.png
├── 23BPS1090_ShrriDharshan_RL_Lab8.ipynb
├── main.py              # Tournament benchmark & interactive human game
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart & Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/shrridharshan27/MCTS-and-Games-RL.git
   cd MCTS-and-Games-RL
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tournament benchmark:**
   ```bash
   python main.py
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook 23BPS1090_ShrriDharshan_RL_Lab8.ipynb
   ```

---

## License
Distributed under the [MIT License](LICENSE).

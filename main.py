"""
Main Execution Script for MCTS & Multi-Agent Game Playing
Shrri Dharshan D R - 23BPS1090
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mcts_games.environment import TicTacToe
from mcts_games.agents import RandomAgent, QLearningAgent, MCTSAgent, train_qlearning
from mcts_games.tournament import play_game, run_tournament

def human_vs_mcts(n_sims: int = 300):
    mcts = MCTSAgent(n_simulations=n_sims)
    game = TicTacToe()
    print("You are X (player 1). Enter position 0-8:")
    print("0|1|2\n3|4|5\n6|7|8\n")
    while not game.is_terminal():
        game.display()
        if game.player == 1:
            moves = game.legal_moves()
            while True:
                try:
                    mv = int(input(f"Your move {moves}: "))
                    if mv in moves: break
                    print("Invalid move.")
                except ValueError: pass
        else:
            print("MCTS thinking...")
            mv = mcts.select_move(game)
            print(f"MCTS plays: {mv}")
        game.make_move(mv)

    game.display()
    r = game.result()
    if r == 1:   print("You WIN! Excellent play.")
    elif r == -1: print("MCTS wins!")
    else:        print("Draw! Perfect defense.")

def main():
    print("=" * 60)
    print("   REINFORCEMENT LEARNING LAB 8: MCTS & GAME PLAYING")
    print("=" * 60)

    # 1. Train Q-Learning Agent
    print("\nTraining Q-Learning agent (50k self-play games)...")
    t0 = time.time()
    ql_agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon=0.2)
    train_qlearning(ql_agent, n_episodes=50000)
    ql_agent.training = False
    print(f"Done in {time.time()-t0:.1f}s | Unique states learned: {len(ql_agent.Q)}")

    # 2. MCTS Agent
    mcts_agent = MCTSAgent(n_simulations=250)
    random_agent = RandomAgent()

    # 3. Tournaments
    print("\nRunning Tournament Matchups...")
    r_vs_r = run_tournament(random_agent, random_agent, n_games=500, name_a='Random', name_b='Random')
    print(f"  Random vs Random (500): Random={r_vs_r[0]} | Draw={r_vs_r[1]} | Random={r_vs_r[2]}")

    ql_vs_r = run_tournament(ql_agent, random_agent, n_games=500, name_a='QLearning', name_b='Random')
    print(f"  QL vs Random (500): QL={ql_vs_r[0]} | Draw={ql_vs_r[1]} | Random={ql_vs_r[2]}")

    mc_vs_r = run_tournament(mcts_agent, random_agent, n_games=100, name_a='MCTS', name_b='Random')
    print(f"  MCTS vs Random (100): MCTS={mc_vs_r[0]} | Draw={mc_vs_r[1]} | Random={mc_vs_r[2]}")

    mc_vs_q = run_tournament(mcts_agent, ql_agent, n_games=100, name_a='MCTS', name_b='QLearning')
    print(f"  MCTS vs QL (100): MCTS={mc_vs_q[0]} | Draw={mc_vs_q[1]} | QL={mc_vs_q[2]}")

    # Plot 1: Tournament Pie Charts
    matchups = ['Random vs Random', 'QL vs Random', 'MCTS vs Random', 'MCTS vs QL']
    data = [r_vs_r, ql_vs_r, mc_vs_r, mc_vs_q]
    colors = ['#e63946', '#a8dadc', '#457b9d']

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    fig.suptitle('Tic-Tac-Toe Multi-Agent Tournament Results', fontsize=13, fontweight='bold')

    for ax, (title, (w, d, l)) in zip(axes, zip(matchups, data)):
        total = w + d + l
        labels = [f'P1 Win\n{w/total:.0%}', f'Draw\n{d/total:.0%}', f'P2 Win\n{l/total:.0%}']
        ax.pie([w, d, l], colors=colors, startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(labels, loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.22))

    plt.tight_layout()
    p1 = os.path.join("assets", "tournament_results.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"\nTournament plot saved to {p1}")

    # Plot 2: MCTS Simulation Budget Sweep
    print("\nEvaluating MCTS Simulation Budget scaling...")
    budgets = [10, 50, 100, 200, 350]
    win_rates = []
    for b in budgets:
        agent = MCTSAgent(n_simulations=b)
        wins, draws, losses = run_tournament(agent, random_agent, n_games=40, name_a='MCTS', name_b='Random')
        wr = wins / 40.0
        win_rates.append(wr)
        print(f"  MCTS Budget {b:>4}: {wr*100:.1f}% win rate vs Random")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(budgets, win_rates, 'o-', color='#2d6a4f', lw=2.2, markersize=7)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Optimal Play Ceiling (100%)')
    ax.set_title('MCTS Win Rate vs Simulation Budget (vs Random Agent)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Simulations per Move', fontsize=11)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join("assets", "mcts_budget_scaling.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Budget curve saved to {p2}")

if __name__ == "__main__":
    main()

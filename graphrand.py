# %% reset board
import torch
import torch.nn as nn
import random
import chess
from time import sleep
import matplotlib.pyplot as plt
import statistics

# Create an initial board instance for display
board = chess.Board()
print(board)

# %% random moves lmao
def randmv(time_delay=0):
    """
    Plays a random chess game until it's over and returns the total number of moves.
    Optionally, you can add a delay between moves with the time_delay parameter.
    """
    n = 0
    board = chess.Board()  # reset board for a new game
    while not board.is_game_over():
        n += 1
        print(f"{board}\n n = {n}\n")
        sleep(time_delay)
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
    return n

# A simple test printout of one random game simulation.
print(randmv(0))

# %% avg mv
def avg_n_randmv(count: int):
    """
    Runs 'count' number of random chess games and calculates the average number of moves.
    """
    countrn = 0
    ntotal = 0
    while countrn < count:
        countrn += 1
        n = randmv(0)
        ntotal += n
    nfinal = ntotal / count
    return nfinal

# Calculate and print the average number of moves over 100 games.
print(avg_n_randmv(100))

# %% collect multiple game stats
def collect_games_stats(num_games, time_delay=0):
    """
    Simulates 'num_games' random chess games while collecting the number of moves
    for each game. Returns a list with the move counts.
    """
    moves_counts = []
    for i in range(num_games):
        n_moves = randmv(time_delay)
        moves_counts.append(n_moves)
        print(f"Game {i + 1}/{num_games}: {n_moves} moves")
    return moves_counts

# %% plot stats
def plot_stats(data):
    """
    Plots a histogram of the move counts from the data provided.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Moves in Random Chess Games")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def display_stats(data):
    """
    Calculates and prints statistical metrics such as average, median,
    maximum, minimum, and standard deviation from the provided data.
    """
    avg = sum(data) / len(data)
    med = statistics.median(data)
    maximum = max(data)
    minimum = min(data)
    std_dev = statistics.stdev(data) if len(data) > 1 else 0

    print(f"\nStatistics over {len(data)} games:")
    print(f"Average number of moves: {avg:.2f}")
    print(f"Median number of moves: {med}")
    print(f"Maximum number of moves: {maximum}")
    print(f"Minimum number of moves: {minimum}")
    print(f"Standard Deviation: {std_dev:.2f}")

# %% run simulation and display results
if __name__ == "__main__":
    num_games = 1000  # You can adjust the number of games to simulate
    moves_data = collect_games_stats(num_games, time_delay=0)
    plot_stats(moves_data)
    display_stats(moves_data)

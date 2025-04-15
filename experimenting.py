# %% reset board
import torch
import torch.nn as nn
import random
import chess
from time import sleep
board = chess.Board()
# %%
print(board)
# %% random moves lmao
def randmv(time):
    n = 0
    board = chess.Board()
    while not board.is_game_over():
        n+=1
        print(f"{board}\n n = {n} \n")
        sleep(time)
        legal = list(board.legal_moves)
        move = random.choice(legal)
        board.push(move)
    return(int(n))
# %%
print(randmv(0))

# %% avg mv
def avg_n_randmv(count: int):
    countrn = 0
    ntotal=0
    while countrn < count:
        countrn+=1
        n = randmv(0)
        ntotal = ntotal+n
    nfinal = ntotal / count
    return(nfinal)

avg_n_randmv(100)

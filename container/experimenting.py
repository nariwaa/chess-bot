# %% reset board
import torch
import torch.nn as nn
import random
import chess
import numpy
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

# %%
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)      # raw logits or value

# Example instantiation:
net = MLP(in_dim=768, hidden_dim=256, out_dim=4672)

# %%
import torch
import chess

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess.Board into a 12×8×8 binary tensor:
      6 piece‑types × 2 colors = 12 planes.
    Flattened to 768.
    """
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_to_plane = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            base_plane = piece_to_plane[piece.piece_type]
            color_offset = 6 if piece.color == chess.BLACK else 0
            plane = base_plane + color_offset
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            planes[plane, rank, file] = 1.0
    return planes.view(-1)  # → [768]

t = board_to_tensor(board)         # shape [768]
planes = t.view(12, 8, 8)          # shape [12, 8, 8]
print("White pawn plane:\n", planes[0])
print("Black pawn plane:\n", planes[6])


# ── 1) Prepare the full move list ─────────────────────────────────────────

# Generate every UCI string once (≈4672 total)
all_moves = [move.uci() for move in chess.Board().legal_moves]  # start with initial legal
# But we need the superset: generate moves from every possible from‐square to every to‐square + promos
files = 'abcdefgh'
ranks = '12345678'
promo_pieces = ['q','r','b','n']
all_moves = set(all_moves)
for f1 in files:
    for r1 in ranks:
        for f2 in files:
            for r2 in ranks:
                u = f1 + r1 + f2 + r2
                all_moves.add(u)
                # add promotions
                if r1 == '7' and r2 == '8':
                    for p in promo_pieces:
                        all_moves.add(u + p)
                if r1 == '2' and r2 == '1':
                    for p in promo_pieces:
                        all_moves.add(u + p)
all_moves = sorted(all_moves)
idx_to_move = all_moves
move_to_idx = {u:i for i,u in enumerate(all_moves)}
out_dim = len(all_moves)
print(f"Total movespace = {out_dim}")

# ── 2) Instantiate your MLP ────────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1  = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2  = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

model = MLP(in_dim=768, hidden_dim=256, out_dim=out_dim)

# ── 3) Helper: encode board → tensor ───────────────────────────────────────

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_map = {
        chess.PAWN:   0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK:   3, chess.QUEEN:  4, chess.KING:   5,
    }
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_map[p.piece_type] + (6 if p.color==chess.BLACK else 0)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[idx, r, f] = 1.0
    return planes.view(-1)  # [768]

# ── 4) Select & play one move ──────────────────────────────────────────────

board = chess.Board()
print("Before:\n", board)

# forward pass → logits
state = board_to_tensor(board)
logits = model(state.unsqueeze(0)).squeeze(0)  # [out_dim]

# mask illegal moves
legal = list(board.legal_moves)
mask = torch.zeros_like(logits)
for mv in legal:
    u = mv.uci()
    mask[move_to_idx[u]] = 1.0

# very‐small value for illegal logits
illegal_val = -1e9
masked_logits = logits * mask + (1-mask)*illegal_val

# convert to probabilities
probs = F.softmax(masked_logits, dim=0)

# sample (or argmax)
dist = torch.distributions.Categorical(probs)
chosen_idx = dist.sample().item()
# chosen_idx = torch.argmax(probs).item()   # ← deterministic alternative

chosen_move = chess.Move.from_uci(idx_to_move[chosen_idx])
board.push(chosen_move)

print(f"\nModel chose: {chosen_move}\n")
print("After:\n", board)

import torch
import torch.nn.functional as F
import torch.optim as optim
import chess

# ── Hyperparameters ──────────────────────────────────────────────────────────
learning_rate = 1e-3
gamma         = 0.99        # discount factor
num_episodes  = 1000

# ── 1) Instantiate model & optimizer ─────────────────────────────────────────
model     = MLP(in_dim=768, hidden_dim=256, out_dim=len(idx_to_move))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ── 2) Helper: play one full game and collect (logp, reward) pairs ──────────
def play_episode(model):
    board = chess.Board()
    transitions = []  # list of (logp, reward) tuples

    # play until game over
    while not board.is_game_over():
        state = board_to_tensor(board)
        logits = model(state.unsqueeze(0)).squeeze(0)  # [out_dim]

        # mask illegal moves
        mask = torch.zeros_like(logits)
        for mv in board.legal_moves:
            mask[move_to_idx[mv.uci()]] = 1.0
        illegal_val = -1e9
        masked_logits = logits * mask + (1-mask) * illegal_val

        # sample action
        probs = F.softmax(masked_logits, dim=0)
        dist  = torch.distributions.Categorical(probs)
        idx   = dist.sample()
        logp  = dist.log_prob(idx)

        # step
        move = chess.Move.from_uci(idx_to_move[idx.item()])
        board.push(move)

        # we’ll assign reward only at the end; placeholder=0
        transitions.append([logp, 0.0])

    # terminal reward: +1 win, –1 loss, 0 draw
    result = board.result()
    if   result == "1-0": final_r =  1.0
    elif result == "0-1": final_r = -1.0
    else:                  final_r =  0.0

    # backfill reward for *every* step
    for tr in transitions:
        tr[1] = final_r

    return transitions

# ── 3) Training loop ─────────────────────────────────────────────────────────
for ep in range(1, num_episodes+1):
    transitions = play_episode(model)

    # compute discounted returns G_t
    returns = []
    G = 0.0
    for _, reward in reversed(transitions):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # standardize returns to reduce variance (optional but helpful)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # compute policy loss: –∑ logp_t * G_t
    logps = torch.stack([tr[0] for tr in transitions])
    loss  = -(logps * returns).sum()

    # gradient step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging every 50 episodes
    if ep % 50 == 0:
        avg_return = returns.mean().item()
        print(f"Episode {ep:4d}  Loss {loss.item():.3f}  Avg G {avg_return:.3f}")

# ── 4) Save your trained policy ───────────────────────────────────────────────
torch.save(model.state_dict(), "chess_policy.pt")

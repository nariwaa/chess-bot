# train.py (FIXED!)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import math
import random
from torch.utils.tensorboard import SummaryWriter

# --- 1. Setup TensorBoard ---
writer = SummaryWriter()

# --- 2. Model and Move Logic (same as before) ---
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_map[p.piece_type] + (6 if p.color==chess.BLACK else 0)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[idx, r, f] = 1.0
    return planes.view(-1)

# (move mapping logic is the same, so I'll skip pasting it here for brevity)
files = 'abcdefgh'
ranks = '12345678'
promo_pieces = ['q','r','b','n']
all_moves = set()
for f1 in files:
    for r1 in ranks:
        for f2 in files:
            for r2 in ranks:
                u = f1 + r1 + f2 + r2
                all_moves.add(u)
                if (r1 == '7' and r2 == '8') or (r1 == '2' and r2 == '1'):
                    for p in promo_pieces:
                        all_moves.add(u + p)
all_moves = sorted(list(all_moves))
idx_to_move = all_moves
move_to_idx = {u:i for i,u in enumerate(all_moves)}


# --- 3. Benchmark and Evaluation (same as before) ---
class RandomPlayer:
    def get_move(self, board):
        return random.choice(list(board.legal_moves))

def play_game(player1, player2):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = player1.get_move(board)
        else:
            move = player2.get_move(board)
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0": return 1
    if result == "0-1": return -1
    return 0

def calculate_elo(win_rate):
    if win_rate <= 0: return -400
    if win_rate >= 1: return 400
    return -400 * math.log10(1 / win_rate - 1)

def evaluate_model(model, benchmark_player, num_games=50):
    """Evaluates the model's Elo by playing games."""
    model.eval() # set model to evaluation mode
    
    class ModelPlayer:
        def get_move(self, board):
            with torch.no_grad():
                state = board_to_tensor(board).unsqueeze(0)
                logits = model(state).squeeze(0)
                mask = torch.zeros_like(logits)
                for mv in board.legal_moves:
                    if mv.uci() in move_to_idx:
                        mask[move_to_idx[mv.uci()]] = 1.0
                masked_logits = logits * mask + (1 - mask) * -1e9
                probs = F.softmax(masked_logits, dim=0)
                
                # --- THIS IS THE FIX! ---
                # instead of sampling, we deterministically pick the best move!
                move_idx = torch.argmax(probs).item() 
                
                return chess.Move.from_uci(idx_to_move[move_idx])

    # ... the rest of the function is exactly the same ...
    model_player = ModelPlayer()
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games // 2):
        result = play_game(model_player, benchmark_player)
        if result == 1: wins += 1
        elif result == -1: losses += 1
        else: draws += 1
        result = play_game(benchmark_player, model_player)
        if result == -1: wins += 1
        elif result == 1: losses += 1
        else: draws += 1
    score = wins + 0.5 * draws
    win_rate = score / num_games
    elo = calculate_elo(win_rate)
    model.train()
    return elo, win_rate

# --- 4. The NEW Training Loop ---

# Hyperparameters
learning_rate = 1e-4  # <-- CHANGED: lowered for stability
gamma = 0.99
num_episodes = 3000 # maybe train for longer now!
eval_every = 50

model = MLP(in_dim=768, hidden_dim=256, out_dim=len(idx_to_move))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
benchmark_player = RandomPlayer()

print("starting training! run `tensorboard --logdir=runs` to see progress.")

for ep in range(1, num_episodes + 1):
    model.train()
    board = chess.Board()
    transitions = []

    while not board.is_game_over(claim_draw=True):
        # *** CHANGE 1: Store the color of the current player ***
        whos_turn = board.turn # True for White, False for Black

        state = board_to_tensor(board)
        logits = model(state.unsqueeze(0)).squeeze(0)
        mask = torch.zeros_like(logits)
        for mv in board.legal_moves:
            if mv.uci() in move_to_idx:
                mask[move_to_idx[mv.uci()]] = 1.0
        masked_logits = logits * mask + (1 - mask) * -1e9
        
        probs = F.softmax(masked_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        logp = dist.log_prob(idx)
        
        move = chess.Move.from_uci(idx_to_move[idx.item()])
        board.push(move)
        
        # Store the log probability and the color of the player who made the move
        transitions.append({'logp': logp, 'color': whos_turn})

    # assign rewards
    result = board.result(claim_draw=True)
    final_r = 0.0
    if result == "1-0": final_r = 1.0   # White won
    elif result == "0-1": final_r = -1.0  # Black won (which is a loss for White)
    
    # *** CHANGE 2: Correctly assign rewards based on stored color ***
    rewards = []
    for trans in transitions:
        # if white won (final_r=1), moves by white get +1, moves by black get -1
        # if black won (final_r=-1), moves by white get -1, moves by black get +1
        reward = final_r if trans['color'] == chess.WHITE else -final_r
        rewards.append(reward)

    # compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # compute policy loss
    logps = torch.stack([t['logp'] for t in transitions])
    loss = -(logps * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    writer.add_scalar('Loss/train', loss.item(), ep)
    writer.add_scalar('Game/length', len(transitions), ep)

    if ep % eval_every == 0:
        elo, win_rate = evaluate_model(model, benchmark_player)
        print(f"ep {ep:5d} | loss {loss.item():.3f} | elo vs random: {elo:.1f} (win rate: {win_rate:.2f})")
        writer.add_scalar('Elo/vs_random', elo, ep)
        writer.add_scalar('WinRate/vs_random', win_rate, ep)
        torch.save(model.state_dict(), f"chess_policy_ep{ep}.pt")

torch.save(model.state_dict(), "chess_policy.pt")
writer.close()
print("training finished!")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import math
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Setup TensorBoard with dynamic naming
timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f"runs/cnn_run_{timestamp}")

# (CNN Model definition is the same)
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessCNN(nn.Module):
    def __init__(self, num_channels=64, num_blocks=5, out_dim=4672):
        super().__init__()
        self.conv_in = nn.Conv2d(12, num_channels, kernel_size=3, padding='same')
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy_head = nn.Linear(num_channels * 8 * 8, out_dim)
    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        policy = self.policy_head(x)
        return policy

# (board_to_tensor and move mapping are the same)
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_map = { chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5 }
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_map[p.piece_type] + (6 if p.color==chess.BLACK else 0)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[idx, r, f] = 1.0
    return planes

files, ranks, promo_pieces = 'abcdefgh', '12345678', ['q','r','b','n']
all_moves = set()
for f1 in files:
    for r1 in ranks:
        for f2 in files:
            for r2 in ranks:
                u = f1 + r1 + f2 + r2
                all_moves.add(u)
                if (r1 == '7' and r2 == '8') or (r1 == '2' and r2 == '1'):
                    for p in promo_pieces: all_moves.add(u + p)
all_moves = sorted(list(all_moves))
idx_to_move = all_moves
move_to_idx = {u:i for i,u in enumerate(all_moves)}

# --- Evaluation Logic (Updated) ---
class RandomPlayer:
    def get_move(self, board):
        return random.choice(list(board.legal_moves))

def calculate_elo(win_rate):
    if win_rate <= 0: return -400
    if win_rate >= 1: return 400
    return -400 * math.log10(1 / win_rate - 1)

### --- CHANGE 1: play_game now also returns the game length --- ###
def play_game(player1, player2):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        move = player1.get_move(board) if board.turn == chess.WHITE else player2.get_move(board)
        board.push(move)
    result = board.result(claim_draw=True)
    length = board.fullmove_number
    if result == "1-0": return 1, length
    if result == "0-1": return -1, length
    return 0, length

def safe_avg(l):
    """calculates the average of a list, returning 0 if the list is empty."""
    return sum(l) / len(l) if l else 0

### --- CHANGE 2: evaluate_model now tracks and logs the new stats --- ###
def evaluate_model(model, benchmark_player, num_games=50):
    model.eval()
    class ModelPlayer:
        def get_move(self, board):
            with torch.no_grad():
                state = board_to_tensor(board).unsqueeze(0)
                logits = model(state).squeeze(0)
                mask = torch.zeros_like(logits)
                for mv in board.legal_moves:
                    if mv.uci() in move_to_idx: mask[move_to_idx[mv.uci()]] = 1.0
                masked_logits = logits * mask + (1 - mask) * -1e9
                probs = F.softmax(masked_logits, dim=0)
                move_idx = torch.argmax(probs).item()
                return chess.Move.from_uci(idx_to_move[move_idx])

    model_player = ModelPlayer()
    wins, losses, draws = 0, 0, 0
    win_lengths, loss_lengths, draw_lengths = [], [], []

    for i in range(num_games):
        # alternate who plays white
        if i % 2 == 0:
            result, length = play_game(model_player, benchmark_player)
        else:
            result, length = play_game(benchmark_player, model_player)
            result *= -1 # flip result since model was player2

        if result == 1:
            wins += 1
            win_lengths.append(length)
        elif result == -1:
            losses += 1
            loss_lengths.append(length)
        else:
            draws += 1
            draw_lengths.append(length)

    score = wins + 0.5 * draws
    win_rate = score / num_games
    elo = calculate_elo(win_rate)
    
    # log all our new stats!
    writer.add_scalar('Elo/vs_random', elo, ep)
    writer.add_scalar('WinRate/vs_random', win_rate, ep)
    writer.add_scalar('Eval/Avg_Win_Length', safe_avg(win_lengths), ep)
    writer.add_scalar('Eval/Avg_Loss_Length', safe_avg(loss_lengths), ep)
    writer.add_scalar('Eval/Avg_Draw_Length', safe_avg(draw_lengths), ep)
    writer.add_scalar('Eval/Win_Count', wins, ep)
    writer.add_scalar('Eval/Loss_Count', losses, ep)
    writer.add_scalar('Eval/Draw_Count', draws, ep)
    
    model.train()
    return elo, win_rate

# --- Training Loop (Updated) ---
learning_rate = 1e-4
gamma = 0.99
num_episodes = 20000
eval_every = 100

model = ChessCNN(out_dim=len(idx_to_move))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
benchmark_player = RandomPlayer()

print(f"starting cnn training! logging to {writer.log_dir}")
print("run `tensorboard --logdir=runs` to see progress.")

for ep in range(1, num_episodes + 1):
    model.train()
    board = chess.Board()
    transitions = []
    entropies = []

    while not board.is_game_over(claim_draw=True):
        whos_turn = board.turn
        state = board_to_tensor(board).unsqueeze(0)
        logits = model(state).squeeze(0)
        mask = torch.zeros_like(logits)
        for mv in board.legal_moves:
            if mv.uci() in move_to_idx: mask[move_to_idx[mv.uci()]] = 1.0
        masked_logits = logits * mask + (1 - mask) * -1e9
        
        probs = F.softmax(masked_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        
        ### --- CHANGE 3: track policy entropy --- ###
        entropies.append(dist.entropy().item())
        
        idx = dist.sample()
        logp = dist.log_prob(idx)
        move = chess.Move.from_uci(idx_to_move[idx.item()])
        board.push(move)
        transitions.append({'logp': logp, 'color': whos_turn})

    # (reward assignment and backprop are the same)
    result = board.result(claim_draw=True)
    final_r = 0.0
    if result == "1-0": final_r = 1.0
    elif result == "0-1": final_r = -1.0
    rewards = [final_r if t['color'] == chess.WHITE else -final_r for t in transitions]
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    logps = torch.stack([t['logp'] for t in transitions])
    loss = -(logps * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    writer.add_scalar('Loss/train', loss.item(), ep)
    writer.add_scalar('Game/length', len(transitions), ep)
    ### --- CHANGE 4: log the new training stats --- ###
    writer.add_scalar('Game/Avg_Return', returns.mean().item(), ep)
    writer.add_scalar('Policy/Entropy', safe_avg(entropies), ep)

    if ep % eval_every == 0:
        elo, win_rate = evaluate_model(model, benchmark_player)
        print(f"ep {ep:5d} | loss {loss.item():.3f} | elo vs random: {elo:.1f} (win rate: {win_rate:.2f})")
        torch.save(model.state_dict(), f"chess_policy_cnn_ep{ep}.pt")

torch.save(model.state_dict(), "chess_policy_cnn.pt")
writer.close()
print("training finished!")

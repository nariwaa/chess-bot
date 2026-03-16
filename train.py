import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import math
import random
import copy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f"runs/cnn_run_{timestamp}")


# to solve the vanishing gradient problem during backprop
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


# the model
# board state → CNN understands spatial patterns → "this move looks best"
class ChessCNN(nn.Module):
    def __init__(self, num_channels=64, num_blocks=5, out_dim=4672): # 64 × 73 = 4672 possible moves
        super().__init__()
        self.conv_in = nn.Conv2d(13, num_channels, kernel_size=3, padding='same')  # 13 input planes
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy_head = nn.Linear(num_channels * 8 * 8, out_dim)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        return self.policy_head(x)


# convert the board into smth inputable into the CNN
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(13, 8, 8, dtype=torch.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_map[p.piece_type] + (6 if p.color == chess.BLACK else 0)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[idx, r, f] = 1.0
    # Plane 12: whose turn it is (1 = white to move, 0 = black to move)
    if board.turn == chess.WHITE:
        planes[12] = 1.0
    return planes


# build move vocabulary
files, ranks, promo_pieces = 'abcdefgh', '12345678', ['q', 'r', 'b', 'n']
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
move_to_idx = {u: i for i, u in enumerate(all_moves)}


class RandomPlayer:
    def get_move(self, board):
        return random.choice(list(board.legal_moves))


class ModelPlayer:
    """Frozen model used as opponent (samples moves for variety, no grad)."""
    def __init__(self, model):
        self.model = model

    def get_move(self, board):
        with torch.no_grad():
            state = board_to_tensor(board).unsqueeze(0)
            logits = self.model(state).squeeze(0)
            mask = torch.zeros_like(logits)
            for mv in board.legal_moves:
                if mv.uci() in move_to_idx:
                    mask[move_to_idx[mv.uci()]] = 1.0
            masked_logits = logits * mask + (1 - mask) * -1e9
            probs = F.softmax(masked_logits, dim=0)
            idx = torch.distributions.Categorical(probs).sample()
            return chess.Move.from_uci(idx_to_move[idx.item()])


def calculate_elo(win_rate):
    if win_rate <= 0: return -400
    if win_rate >= 1: return 400
    return -400 * math.log10(1 / win_rate - 1)


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
    return sum(l) / len(l) if l else 0


def evaluate_model(model, benchmark_player, ep, num_games=50):
    model.eval()

    class EvalPlayer:
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
                return chess.Move.from_uci(idx_to_move[torch.argmax(probs).item()])

    eval_player = EvalPlayer()
    wins, losses, draws = 0, 0, 0
    win_lengths, loss_lengths, draw_lengths = [], [], []

    for i in range(num_games):
        if i % 2 == 0:
            result, length = play_game(eval_player, benchmark_player)
        else:
            result, length = play_game(benchmark_player, eval_player)
            result *= -1

        if result == 1:
            wins += 1; win_lengths.append(length)
        elif result == -1:
            losses += 1; loss_lengths.append(length)
        else:
            draws += 1; draw_lengths.append(length)

    score = wins + 0.5 * draws
    win_rate = score / num_games
    elo = calculate_elo(win_rate)

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


def run_episode(model, model_color, opponent, gamma):
    """Play one full game. Only collects gradients for the model's own moves."""
    board = chess.Board()
    logps = []
    entropies = []

    while not board.is_game_over(claim_draw=True):
        if board.turn == model_color:
            state = board_to_tensor(board).unsqueeze(0)
            logits = model(state).squeeze(0)
            mask = torch.zeros_like(logits)
            for mv in board.legal_moves:
                if mv.uci() in move_to_idx:
                    mask[move_to_idx[mv.uci()]] = 1.0
            masked_logits = logits * mask + (1 - mask) * -1e9
            probs = F.softmax(masked_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            entropies.append(dist.entropy().item())
            idx = dist.sample()
            logps.append(dist.log_prob(idx))
            move = chess.Move.from_uci(idx_to_move[idx.item()])
        else:
            move = opponent.get_move(board)
        board.push(move)

    # reinforce behaviour if won, derinforce it if lost
    result = board.result(claim_draw=True)
    if result == "1-0":
        final_r = 1.0 if model_color == chess.WHITE else -1.0
    elif result == "0-1":
        final_r = -1.0 if model_color == chess.WHITE else 1.0
    else:
        final_r = 0.0

    return logps, entropies, final_r, board.fullmove_number


# --- Hyperparameters ---
learning_rate = 1e-4
gamma = 0.99
num_episodes = 100000
eval_every = 100
save_every = 500
phase1_threshold = 0.75     # win rate vs random needed to advance to self-play
opponent_update_every = 500  # how often to refresh the frozen opponent in phase 2

model = ChessCNN(out_dim=len(idx_to_move))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
benchmark_player = RandomPlayer()

phase = 1
frozen_opponent = None

print(f"phase 1: training vs random until {phase1_threshold*100:.0f}% win rate.")
print(f"logging to {writer.log_dir} — run `tensorboard --logdir=runs` to see progress.")

for ep in range(1, num_episodes + 1):
    model_color = chess.WHITE if ep % 2 == 0 else chess.BLACK  # alternate sides each episode

    if phase == 1:
        opponent = benchmark_player
    else:
        # Phase 2: play against a periodically-updated frozen copy of the model
        if frozen_opponent is None or ep % opponent_update_every == 0:
            frozen_opponent = ModelPlayer(copy.deepcopy(model).eval())
            print(f"ep {ep}: refreshed self-play opponent")
        opponent = frozen_opponent

    logps, entropies, final_r, game_len = run_episode(model, model_color, opponent, gamma)

    if not logps:
        continue

    # Discounted returns from terminal reward — no normalization so wins/losses stay distinct
    n = len(logps)
    returns = torch.tensor([final_r * (gamma ** (n - 1 - i)) for i in range(n)], dtype=torch.float32)

    loss = -(torch.stack(logps) * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), ep)
    writer.add_scalar('Game/length', game_len, ep)
    writer.add_scalar('Policy/Entropy', safe_avg(entropies), ep)
    writer.add_scalar('Training/phase', phase, ep)
    writer.add_scalar('Training/final_reward', final_r, ep)

    if ep % eval_every == 0:
        elo, win_rate = evaluate_model(model, benchmark_player, ep)
        phase_str = "vs_random" if phase == 1 else "self_play"
        print(f"ep {ep:6d} [{phase_str}] | loss {loss.item():.3f} | elo {elo:.1f} | win rate {win_rate:.2f}")

        if phase == 1 and win_rate >= phase1_threshold:
            phase = 2
            frozen_opponent = ModelPlayer(copy.deepcopy(model).eval())
            torch.save(model.state_dict(), "chess_policy_cnn_phase1_complete.pt")
            print(f"*** Phase 1 complete! Switching to self-play. ***")

    if ep % save_every == 0:
        torch.save(model.state_dict(), f"chess_policy_cnn_ep{ep}.pt")

torch.save(model.state_dict(), "chess_policy_cnn.pt")
writer.close()
print("training done!")

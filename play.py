import torch
import chess
import torch.nn.functional as F
from experimenting.py import MLP, board_to_tensor, move_to_idx, idx_to_move  # adjust import if needed

# Load model
model = MLP(in_dim=768, hidden_dim=256, out_dim=len(idx_to_move))
model.load_state_dict(torch.load("chess_policy.pt"))
model.eval()

def play_self_game(show=True):
    board = chess.Board()
    while not board.is_game_over():
        if show:
            print(board, "\n")

        state = board_to_tensor(board)
        logits = model(state.unsqueeze(0)).squeeze(0)

        # Mask illegal moves
        mask = torch.zeros_like(logits)
        for mv in board.legal_moves:
            mask[move_to_idx[mv.uci()]] = 1.0
        masked_logits = logits * mask + (1 - mask) * -1e9

        # Choose the move with highest probability
        probs = F.softmax(masked_logits, dim=0)
        move_idx = torch.argmax(probs).item()
        move = chess.Move.from_uci(idx_to_move[move_idx])
        board.push(move)

    if show:
        print(board)
        print("Game over. Result:", board.result())


# Run it
play_self_game()

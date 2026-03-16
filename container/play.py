# play_tui.py

import torch
import chess
import torch.nn.functional as F
import curses
import time
import sys

# --- Your AI and Chess Logic (same as before) ---
# I'm including the necessary classes and functions here so it's all in one file!

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, out_dim)
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

# Generate the move mappings
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

# --- Load The Model ---
model = MLP(in_dim=768, hidden_dim=256, out_dim=len(idx_to_move))
try:
    model.load_state_dict(torch.load("chess_policy.pt"))
except FileNotFoundError:
    print("!!! chess_policy.pt not found! The AI will make random moves. !!!")
    # this is a fallback so the program doesn't crash
model.eval()

def get_ai_move(board):
    state = board_to_tensor(board)
    logits = model(state.unsqueeze(0)).squeeze(0)
    mask = torch.zeros_like(logits)
    for mv in board.legal_moves:
        if mv.uci() in move_to_idx:
            mask[move_to_idx[mv.uci()]] = 1.0
    masked_logits = logits * mask + (1 - mask) * -1e9
    probs = F.softmax(masked_logits, dim=0)
    move_idx = torch.argmax(probs).item()
    return chess.Move.from_uci(idx_to_move[move_idx])


# --- Curses TUI Code ---

# unicode characters for the pieces!
PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

def init_colors():
    """Initializes color pairs for curses"""
    curses.start_color()
    curses.use_default_colors()
    # pair 1: light square, 2: dark square, 3: selected square
    curses.init_pair(1, curses.COLOR_BLACK, 248) # light grey bg
    curses.init_pair(2, curses.COLOR_WHITE, 240) # dark grey bg
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN) # green bg

def draw_board(stdscr, board, flipped, cursor_pos, selected_square):
    """Draws the entire TUI interface"""
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    
    # instructions
    stdscr.addstr(0, 22, "Chess AI - TUI")
    stdscr.addstr(2, 22, "use arrow keys to move")
    stdscr.addstr(3, 22, "press <space> to select/move")
    stdscr.addstr(4, 22, "press <q> to quit")

    # draw the board and pieces
    for r in range(8):
        for c in range(8):
            is_light = (r + c) % 2 != 0
            color_pair = curses.color_pair(1 if is_light else 2)
            
            # highlight cursor
            if (r, c) == cursor_pos:
                color_pair = curses.color_pair(3)
            
            # highlight selected piece
            if selected_square is not None:
                s_r, s_c = (7 - chess.square_rank(selected_square), chess.square_file(selected_square))
                if not flipped and (r, c) == (s_r, s_c):
                    color_pair = curses.color_pair(3)
                elif flipped and (r, c) == ((7-s_r), (7-s_c)):
                    color_pair = curses.color_pair(3)

            # board is 2 chars wide per square
            stdscr.addstr(r + 2, c * 2, "  ", color_pair)

            sq_idx = chess.square(c, 7-r) if not flipped else chess.square(7-c, r)
            piece = board.piece_at(sq_idx)
            if piece:
                symbol = PIECE_SYMBOLS[piece.symbol()]
                stdscr.addstr(r + 2, c * 2, f" {symbol}", color_pair)

    # show turn info
    turn_str = "White's Turn" if board.turn == chess.WHITE else "Black's Turn"
    stdscr.addstr(11, 22, turn_str)
    if board.is_check():
        stdscr.addstr(12, 22, "check!", curses.A_BOLD)

    stdscr.refresh()

def main(stdscr, human_color):
    # curses setup
    curses.curs_set(0) # hide the blinking cursor
    init_colors()
    
    board = chess.Board()
    flipped = (human_color == chess.BLACK)
    cursor_pos = (0, 0) # (row, col)
    selected_square = None

    while not board.is_game_over():
        draw_board(stdscr, board, flipped, cursor_pos, selected_square)
        
        is_human_turn = (board.turn == human_color)

        if is_human_turn:
            key = stdscr.getch() # wait for user input

            # --- Input Handling ---
            if (key == curses.KEY_UP or key == ord('k')) and cursor_pos[0] > 0:
                cursor_pos = (cursor_pos[0] - 1, cursor_pos[1])
            elif (key == curses.KEY_DOWN or key == ord('j')) and cursor_pos[0] < 7:
                cursor_pos = (cursor_pos[0] + 1, cursor_pos[1])
            elif (key == curses.KEY_LEFT or key == ord('h')) and cursor_pos[1] > 0:
                cursor_pos = (cursor_pos[0], cursor_pos[1] - 1)
            elif (key == curses.KEY_RIGHT or key == ord('l')) and cursor_pos[1] < 7:
                cursor_pos = (cursor_pos[0], cursor_pos[1] + 1)
            elif key == ord('q'):
                break
            elif key == ord(' '): # spacebar to select
                r, c = cursor_pos
                clicked_sq_idx = chess.square(c, 7-r) if not flipped else chess.square(7-c, r)
                
                if selected_square is None:
                    # first click: select a piece
                    if board.piece_at(clicked_sq_idx) and board.piece_at(clicked_sq_idx).color == human_color:
                        selected_square = clicked_sq_idx
                else:
                    # second click: try to make a move
                    move_uci = chess.square_name(selected_square) + chess.square_name(clicked_sq_idx)
                    if board.piece_at(selected_square).piece_type == chess.PAWN:
                        if chess.square_rank(clicked_sq_idx) in [0, 7]:
                            move_uci += 'q' # auto-promote to queen
                    
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                    
                    selected_square = None # reset selection
        else:
            # AI's turn
            stdscr.addstr(13, 22, "ai is thinking...")
            stdscr.refresh()
            time.sleep(0.5) # make it feel like it's thinking!
            ai_move = get_ai_move(board)
            board.push(ai_move)

    # Game over
    draw_board(stdscr, board, flipped, cursor_pos, selected_square)
    result_str = f"game over! result: {board.result()}"
    stdscr.addstr(14, 22, result_str)
    stdscr.addstr(15, 22, "press any key to exit.")
    stdscr.getch()


if __name__ == "__main__":
    player_color_str = ""
    while player_color_str not in ['w', 'b']:
        player_color_str = input("do you want to play as white (w) or black (b)? ").lower()

    human_color = chess.WHITE if player_color_str == 'w' else chess.BLACK
    
    # curses.wrapper handles all the setup and cleanup for us!
    curses.wrapper(main, human_color)

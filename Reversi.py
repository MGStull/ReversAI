import os
import time

# Constants for tokens
EMPTY = ' '
WHITE = 'w'
BLACK = 'b'

# Initialize the board
def create_board():
    board = [[EMPTY for _ in range(8)] for _ in range(8)]
    board[3][3], board[4][4] = WHITE, WHITE
    board[3][4], board[4][3] = BLACK, BLACK
    return board

# Print the board
def print_board(board):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("   " + " ".join(str(i) for i in range(8)))
    for i, row in enumerate(board):
        print(str(i) + " |" + "|".join(cell if cell != EMPTY else '.' for cell in row) + "|")

# Check directions
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

def is_on_board(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def valid_moves(board, color):
    opponent = BLACK if color == WHITE else WHITE
    valid = []

    for row in range(8):
        for col in range(8):
            if board[row][col] != EMPTY:
                continue

            for dx, dy in DIRECTIONS:
                x, y = row + dx, col + dy
                if is_on_board(x, y) and board[x][y] == opponent:
                    while is_on_board(x, y) and board[x][y] == opponent:
                        x += dx
                        y += dy
                    if is_on_board(x, y) and board[x][y] == color:
                        valid.append((row, col))
                        break
    return valid

def make_move(board, row, col, color):
    opponent = BLACK if color == WHITE else WHITE
    flipped = []

    board[row][col] = color
    for dx, dy in DIRECTIONS:
        x, y = row + dx, col + dy
        line = []

        while is_on_board(x, y) and board[x][y] == opponent:
            line.append((x, y))
            x += dx
            y += dy

        if is_on_board(x, y) and board[x][y] == color:
            for fx, fy in line:
                board[fx][fy] = color
                flipped.extend(line)

    return flipped

def count_tokens(board):
    white = sum(row.count(WHITE) for row in board)
    black = sum(row.count(BLACK) for row in board)
    return white, black

def game_loop():
    board = create_board()
    current_color = BLACK

    while True:
        print_board(board)
        white_count, black_count = count_tokens(board)
        print(f"Score -> White: {white_count} | Black: {black_count}")
        print(f"{'White' if current_color == WHITE else 'Black'}'s turn")

        ##REMEBER TO SEND THE VALID MOVES TO THE AI
        moves = valid_moves(board, current_color)
        if not moves:
            print(f"No valid moves for {'White' if current_color == WHITE else 'Black'}. Skipping turn...")
            time.sleep(2)
            current_color = BLACK if current_color == WHITE else WHITE
            if not valid_moves(board, current_color):
                break
            continue

        while True:
            try:
                ##EDIT THIS TO REQUEST MOVES FROM THE AI
                move = input("Enter your move as row,col (e.g. 3,4): ").strip()
                row, col = map(int, move.split(','))
                if (row, col) in moves:
                    make_move(board, row, col, current_color)
                    current_color = BLACK if current_color == WHITE else WHITE
                    break
                else:
                    print("Invalid move. Try again.")
            except Exception as e:
                print("Error:", e)
                print("Please enter a valid move like 2,3")

    # Game over
    print_board(board)
    white_count, black_count = count_tokens(board)
    print(f"Final Score -> White: {white_count} | Black: {black_count}")
    if white_count > black_count:
        print("White wins!")
    elif black_count > white_count:
        print("Black wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    game_loop()

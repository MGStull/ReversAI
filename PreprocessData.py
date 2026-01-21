import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os

###CONSTANTS
     # Create token mappings - CORRECTED
letters = 'abcdefgh'  # columns: a=0, b=1, ..., h=7
numbers = '87654321'  # rows: 8=0 (top), 7=1, ..., 1=7 (bottom)
token_to_idx = {}
for row_idx, number in enumerate(numbers):
    for col_idx, letter in enumerate(letters):
        token = letter + number
        token_to_idx[token] = (row_idx, col_idx)  # (row, col)
        print(f"{token} = ({row_idx}, {col_idx})")
    
idx_to_token = {}
for row_idx, number in enumerate(numbers):
    for col_idx, letter in enumerate(letters):
        token = letter + number
        idx = (row_idx, col_idx)
        idx_to_token[idx] =  token # (row, col)

numbers = '12345678'
token_to_hot = {}
i = 0
for letter in letters:
    for number in numbers:
        token = letter+number
        token_to_hot[token] = i 
        i = i+1





class OthelloGame:
    def __init__(self):
        self.board = self._init_board()
    
    def _init_board(self):
        board = np.zeros((8, 8), dtype=int)
        board[3, 4] = 1   # white
        board[4, 3] = 1   # white
        board[3, 3] = -1  # black
        board[4, 4] = -1  # black
        return board
    
    def get_legal_moves(self, player):

        legal_moves = []
        
        for row in range(8):
            for col in range(8):
                if self.is_legal_move(row, col, player):
                    legal_moves.append((row, col))
        
        return legal_moves
    
    def is_legal_move(self, row, col, player):
        """Check if a move is legal"""
        if self.board[row, col] != 0:
            return False
        
        # Check all 8 directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            if self._has_flips(row, col, player, dr, dc):
                return True
        
        return False
    
    def _has_flips(self, row, col, player, dr, dc):
        r, c = row + dr, col + dc
        opponent = -player
        found_opponent = False
        
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == opponent:
                found_opponent = True
            elif self.board[r, c] == player:
                return found_opponent
            else:
                return False
            r += dr
            c += dc
        
        return False
    
    def make_move(self, move_token, player, token_to_idx):
        row, col = token_to_idx[move_token]
        
        if not self.is_legal_move(row, col, player):
            raise ValueError(f"Illegal move: {move_token} at ({row}, {col})")
        
        self.board[row, col] = player
        
        # Flip opponent pieces in all directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            self._flip_pieces(row, col, player, dr, dc)
    
    def _flip_pieces(self, row, col, player, dr, dc):
        r, c = row + dr, col + dc
        opponent = -player
        pieces_to_flip = []
        
        while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
            pieces_to_flip.append((r, c))
            r += dr
            c += dc
        
        # Only flip if we found a player piece at the end
        if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
            for flip_r, flip_c in pieces_to_flip:
                self.board[flip_r, flip_c] = player
    def printBoard(self):
        print(self.board)


def chunk_string(s):
    return [s[i:i+2] for i in range(0, len(s), 2)]


def get_illegal_moves_for_sequence(moves_str, token_to_idx, idx_to_token, token_to_hot):

    game = OthelloGame()
    moves = chunk_string(moves_str)
    
    illegal_moves_sequence = []
    game = OthelloGame()
    current_player = -1 
    # For each position in the game
    for i in range(len(moves)):
        # Replay moves up to this point
        # Reset board
        try:
            illegal_moves_sequence.append(legal_to_illegal(game.get_legal_moves(current_player),idx_to_token,token_to_hot))
            game.make_move(moves[i], current_player, token_to_idx)
            current_player = (-1)*current_player
        except ValueError:
            # Invalid game state
            pass
    return illegal_moves_sequence

#IS ILLEGAL IF 1
#ARGS
# VARIABLE legal_moves_sequence
# CONST CONVERTERs idx_to_token, token_to_hot
# RETURNS 
# string in one_hot of illegal moves if the move is legal it will be set to 1 n
# if the move is legal it is set to 0
def legal_to_illegal(legal_moves_sequence,idx_to_token, token_to_hot):
    illegal_moves = ['1']*64
    for legal_move in  legal_moves_sequence:
        illegal_moves[token_to_hot[idx_to_token[legal_move]]] = '0'
    return ''.join(illegal_moves)
            


def preprocess_dataset(input_csv_path, output_csv_path, use_pickle=False):

    print(f"Loading dataset from {input_csv_path}...")
    df = pd.read_csv(input_csv_path,names=['game_id', 'winner', 'moves','illegal_moves'])
    
    print(f"Dataset has {len(df)} games")
    print(f"Processing illegal moves for each game...")
    
    # Store illegal moves as a list of lists
    all_illegal_moves = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        game_id = row.iloc[0]
        winner = row.iloc[1]
        moves_str = row.iloc[2]
        
        # Calculate illegal moves for each position
        illegal_moves_seq = get_illegal_moves_for_sequence(moves_str, token_to_idx,idx_to_token,token_to_hot)
        
        # Store as a pipe-separated string of binary masks
        illegal_moves_str = '|'.join(illegal_moves_seq)
        
        all_illegal_moves.append(illegal_moves_str)
    # Add to dataframe
    df['illegal_moves'] = all_illegal_moves
    # Save to CSV
    print(f"\nSaving processed dataset to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    
    print(f"✓ Saved CSV with {len(df)} games")
    
    # Optionally save as pickle for faster loading during training
    if use_pickle:
        pickle_path = output_csv_path.replace('.csv', '_illegal_moves.pkl')
        print(f"Saving illegal moves pickle to {pickle_path}...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_illegal_moves, f)
        print(f"✓ Saved pickle file")
    
    return df


def load_preprocessed_dataset(csv_path):
    print(f"Loading preprocessed dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} games")
    return df


def decode_illegal_moves(illegal_moves_str, move_idx):
    illegal_masks = illegal_moves_str.split('|')
    
    if move_idx >= len(illegal_masks):
        # If asking for move beyond sequence, return all legal
        return torch.ones(64, dtype=torch.float32)
    
    mask_str = illegal_masks[move_idx]
    illegal_mask = torch.tensor(
        [int(bit) for bit in mask_str],
        dtype=torch.float32
    )
    
    return illegal_mask




def RUN():
    # Paths to your datasets
    train_csv_path = r"C:\Users\chick\Documents\Code\ReversAI\Data\othello_dataset_train.csv"
    test_csv_path = r"C:\Users\chick\Documents\Code\ReversAI\Data\othello_dataset_test.csv"
    
    # Output paths
    train_output_path = r"C:\Users\chick\Documents\Code\ReversAI\Data\othello_dataset_train_with_illegal_moves.csv"
    test_output_path = r"C:\Users\chick\Documents\Code\ReversAI\Data\othello_dataset_test_with_illegal_moves.csv"
    preprocess_dataset(test_csv_path, test_output_path, use_pickle=True)
    preprocess_dataset(train_csv_path, train_output_path, use_pickle=True)
    
def Testing():
    game = OthelloGame()
    print("\nLegal moves for Black:", game.get_legal_moves(-1))
    game.printBoard()

    
    print('\nMaking move e6 for Black...')
    game.make_move('e6', -1, token_to_idx)
    game.printBoard()
    
    print("\nLegal moves for White:", game.get_legal_moves(1))
    game.printBoard()

    print('\nMaking move e7 for White...')
    game.make_move('f6', 1, token_to_idx)
    game.printBoard()
    

    illegal_turn1 = 'c4d3e6f5'
    illegal_moves = ['-1']*64
    for token in chunk_string(illegal_turn1):
        print(token)
        illegal_moves[token_to_hot[token]] = '0'
    illegal_moves=''.join(illegal_moves)
    print(illegal_moves)

    illegal_sequence = get_illegal_moves_for_sequence('f5d6c3g5c6c5c4b5b6b4f6b3e6e3d3c7a4a3a5d7f4f3e7f8e2g3e8a6g4c2h3h4h6d2h5d8c1d1e1g2b1f1f2a1b8g1a2c8g8g7b7h2f7g6h1b2h7h8a7a8', token_to_idx, idx_to_token, token_to_hot)
    print(illegal_sequence[0])
    assert illegal_moves == illegal_sequence[0]

RUN()
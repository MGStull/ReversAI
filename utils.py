import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



class OthelloDataset(Dataset):
    def __init__(self, csv_path, token_to_hot):
        self.df = pd.read_csv(csv_path)
        #for naming convention retain the token to hot instead of token to idx
        self.token_to_hot = token_to_hot
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        game_id = row[0]
        winner = torch.tensor(int(row[1]), dtype=torch.long)
        moves_str = row[2]
        illegal_masks = row[3].split('|')
        illegal_masks_tensor = torch.tensor([[int(bit) for bit in mask] for mask in illegal_masks], dtype=torch.float32)
                
        # Convert moves string to tokens
        moves = self._chunk_string(moves_str)
        
        # Convert tokens to indices
        move_indices = torch.tensor(
            [self.token_to_hot[move] for move in moves],
            dtype=torch.long
        )
        
        # Generate turns: -1 for black (even indices), 1 for white (odd indices)
        turns = torch.tensor(
            [(-1) ** (i+1) for i in range(len(moves))],
            dtype=torch.long
        )
        
        return {
            'moves': move_indices,
            'winner': winner,
            'turns': turns,
            'illegal_moves':illegal_masks_tensor
        }
    
    @staticmethod
    def _chunk_string(s):
        """Split string into 2-character chunks"""
        return [s[i:i+2] for i in range(0, len(s), 2)]

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    """Pad sequences to same length within a batch"""
    moves = [item['moves'] for item in batch]
    winners = torch.stack([item['winner'] for item in batch])
    turns = [item['turns'] for item in batch]  # Get turns sequences
    
    # Pad sequences
    padded_moves = pad_sequence(moves, batch_first=True, padding_value=0)
    padded_turns = pad_sequence(turns, batch_first=True, padding_value=0)
    max_seq_len = padded_moves.shape[1]  # Already computed from move padding
    batch_size = len(batch)
    padded_illegal_moves = torch.zeros(batch_size, max_seq_len, 64, dtype=torch.float32)

    for i, item in enumerate(batch):
        if item['illegal_moves'] is not None:
            illegal = item['illegal_moves']
            padded_illegal_moves[i, :illegal.shape[0], :] = illegal
        
    return {
        'moves': padded_moves,
        'winners': winners,
        'turns': padded_turns,
        'illegal_moves': padded_illegal_moves
    }


def Testing():
    #Create dictionaries for encoding/decoding
    ###CONSTANTS
        # Create token mappings - CORRECTED
    letters = 'abcdefgh'
    numbers = '12345678'
    token_to_hot = {}
    i = 0
    for letter in letters:
        for number in numbers:
            token = letter+number
            token_to_hot[token] = i 
            i = i+1

    letters = 'abcdefgh'
    numbers = '12345678'
    hot_to_token = {}
    i = 0
    for letter in letters:
        for number in numbers:
            token = letter+number
            hot_to_token[i] = token 
            i = i+1

    # Create dataset and dataloader
    train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train_with_illegal_moves.csv"
    test_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_test_with_illegal_moves.csv"

    try:
        train_dataset = OthelloDataset(train_csv_path, token_to_hot)
        test_dataset = OthelloDataset(test_csv_path, token_to_hot)
        
        batch_size = 32
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        dataloader_test = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
            
        # Test a batch
        batch = next(iter(dataloader_train))
        print(f"\nBatch shapes:")
        print(f"  moves: {batch['moves'].shape}")
        print(f"  winners: {batch['winners'].shape}")
        print(f"  turns: {batch['turns'].shape}")
        print(f"  illegal_moves: {batch['illegal_moves'].shape}")
        print(f"\nFirst game illegal moves (first 3 moves):\n{batch['illegal_moves'][0, :3, :]}")
            
    except FileNotFoundError:
        print("Dataset files not found. Update paths and ensure PreprocessData.py has been run.")



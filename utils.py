import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class OthelloDataset(Dataset):
    def __init__(self, csv_path, token_to_idx):
        self.df = pd.read_csv(csv_path)
        self.token_to_idx = token_to_idx
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        game_id = row[0]
        winner = torch.tensor(int(row[1]), dtype=torch.long)
        moves_str = row[2]
        
        # Convert moves string to tokens
        moves = self._chunk_string(moves_str)
        
        # Convert tokens to indices
        move_indices = torch.tensor(
            [self.token_to_idx[move] for move in moves],
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
            'turns': turns
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
    
    return {
        'moves': padded_moves,
        'winners': winners,
        'turns': padded_turns
    }

#Just manually split into test and train set for ease
csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"

# Create dictionaries for encoding/decoding
numbers = '12345678'
letters = 'abcdefgh'
tokens = []
for a in letters:
    for b in numbers:
        tokens.append(a+b)
token_to_idx = {token: idx for idx, token in enumerate(tokens)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

print(f"Total unique tokens: {len(tokens)}")
print(f"Token examples: {tokens[:5]}")


# Create dataset and dataloader
train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"
train_dataset = OthelloDataset(train_csv_path, token_to_idx)
test_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_test.csv"
test_dataset = OthelloDataset(test_csv_path, token_to_idx)
# Create DataLoader with custom collate function
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
# Example: iterate through one batch
print(f"\nDataset size: {len(train_dataset)}")
print(f"DataLoader created with batch size: {batch_size}\n")
print(f"\nDataset size: {len(test_dataset)}")
print(f"DataLoader created with batch size: {batch_size}\n")
for batch in dataloader_train:
    print("Batch example:")
    print(f"  Moves shape: {batch['moves'].shape}")  # (batch_size, max_sequence_length)
    print(f"  Winners shape: {batch['winners'].shape}")  # (batch_size,)
    print(f"  Turns shape: {batch['turns'].shape}")  # (batch_size,)
    print(f"  First game moves: {batch['moves'][0][:10]}")  # First 10 move indices
    print(f"  Winners in batch: {batch['winners'][:5]}")
    print(f"  Turns in batch: {batch['turns'][:5]}")  # -1 (black), 0 (no moves), 1 (white)
    break  # Just show first batch
for batch in dataloader_test:
    print("Batch example:")
    print(f"  Moves shape: {batch['moves'].shape}")  # (batch_size, max_sequence_length)
    print(f"  Winners shape: {batch['winners'].shape}")  # (batch_size,)
    print(f"  Turns shape: {batch['turns'].shape}")  # (batch_size,)
    print(f"  First game moves: {batch['moves'][0][:10]}")  # First 10 move indices
    print(f"  Winners in batch: {batch['winners'][:5]}")
    print(f"  Turns in batch: {batch['turns'][:5]}")  # -1 (black), 0 (no moves), 1 (white)
    break  # Just show first batch

print("\nToken to Index mapping (first 5):")
for token, idx in list(token_to_idx.items())[:5]:
    print(f"  {token} -> {idx}")
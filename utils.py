import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



class OthelloDataset(Dataset):
    def __init__(self, csv_path, token_to_hot,min_moves=4):
        self.df = pd.read_csv(csv_path)
        self.min_moves = min_moves
        self.examples = []
        self.token_to_hot = token_to_hot

        for idx, row in self.df.iterrows():
            moves_str = row['moves']
            winner =torch.tensor(
                {-1: 0, 0: 2, 1: 1}[int(row['winner'])],
                dtype = torch.long
            )
            partial_examples = generate_partial_games(moves_str, winner, token_to_hot, self.min_moves)
            self.examples.extend(partial_examples)
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        moves = example['moves']
        winner = example['winner']
        turns = example['turns']

        #encoded moves
        encoded_moves = torch.tensor(
            [self.token_to_hot[move] for move in moves],
            dtype = torch.long
        )
        
        return {
            'moves': encoded_moves,
            'winner': winner,
            'turns': turns
        }
    
def chunk_string(s):
    """Split string into 2-character chunks"""
    return [s[i:i+2] for i in range(0, len(s), 2)]

def generate_partial_games(moves_str, winner, token_to_hot, min_moves = 4):
    moves = chunk_string(moves_str)
    examples = []
    total_moves = len(moves)

    for num_moves in range(min_moves, total_moves + 1):
        partial_moves = moves[:num_moves]
        turns = torch.tensor(
            [(-1) ** (i+1) for i in range(num_moves)],
            dtype=torch.long
            )
        
        examples.append({
            'moves': partial_moves,
            'winner': winner,
            'turns': turns,
            'move_count': num_moves,
        })
    
    return examples



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

    return {
        'moves': padded_moves,
        'winners': winners,
        'turns': padded_turns,
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
    train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"
    test_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_test.csv"

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

            
    except FileNotFoundError:
        print("Dataset files not found. Update paths and ensure PreprocessData.py has been run.")



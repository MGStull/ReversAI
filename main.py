import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import your modules
from Transformer import ReversiBotDecoder
from utils import OthelloDataset, collate_fn
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        moves = batch['moves'].to(device)
        illegal_moves_per_timestamp = batch['illegal_moves'].to(device)
        turns = batch['turns'].to(device)

        logits = model(moves, turns=turns,illegal_moves = None)

        input_moves = moves[:,:-1]
        target_moves = moves[:,1:]

        logits_for_training = logits[:,:-1,:]
        
        batch_size,seqlen,vocab_size = logits_for_training.shape
        logits_flat = logits_for_training.reshape(-1,vocab_size)
        target_flat = target_moves.reshape(-1)

        illegal_moves_training = illegal_moves_per_timestamp[:, :-1, :]  # Match sequence length
        illegal_moves_flat = illegal_moves_training.reshape(-1,64)  # (batch*seq_len, 64)

        logits_flat = logits_flat.clone()
        logits_flat[illegal_moves_flat == 0] =float("-1e20")

        loss = loss_fn(logits_flat, target_flat)

        #backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches +=1
        pbar.set_postfix({'loss':loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss
    
def evaluate(model,dataloader,loss_fn,device):
    model.eval()
    total_loss=0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            moves = batch['moves'].to(device)
            illegal_moves_per_timestamp = batch['illegal_moves'].to(device)
            turns = batch['turns'].to(device)
            logits = model(moves,turns=turns, illegal_moves = None)

            input_moves = moves[:,:-1]
            target_moves = moves[:,1:]

            logits_for_training = logits[:,:-1,:]
            
            batch_size,seqlen,vocab_size = logits_for_training.shape
            logits_flat = logits_for_training.reshape(-1,vocab_size)
            target_flat = target_moves.reshape(-1)

            illegal_moves_training = illegal_moves_per_timestamp[:, :-1, :]  # Match sequence length
            illegal_moves_flat = illegal_moves_training.reshape(-1, 64)  # (batch*seq_len, 64)

            logits_flat = logits_flat.clone()
            logits_flat[illegal_moves_flat == 0] =float("-1e20")

            loss = loss_fn(logits_flat, target_flat)
            total_loss += loss.item()
            predictions = torch.argmax(logits_flat,dim=1)
            mask = illegal_moves_flat.gather(1, target_flat.unsqueeze(1)).squeeze()
            correct = (predictions == target_flat).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
    avg_loss = total_loss/len(dataloader)
    accurracy = total_correct/total_samples
    return avg_loss, accurracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    letters = 'abcdefgh'
    numbers = '12345678'
    token_to_hot = {}
    i = 0
    for letter in letters:
        for number in numbers:
            token = letter+number
            token_to_hot[token] = i 
            i = i+1

    train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train_with_illegal_moves.csv"
    
    try:
        train_dataset = OthelloDataset(train_csv_path, token_to_hot)
        batch_size = 128
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f"Loaded {len(train_dataset)} training games")
    except FileNotFoundError:
        print("File not Found or Loaded")
    model = ReversiBotDecoder(
        vocab_size=64,
        embed_size=512,
        num_layers=8,
        heads=16,
        dropout=0.05,
        device=device,
        max_length=60,
        forward_expansion=8
    ).to(device)

    optimizer = optim.Adam(model.parameters(),lr = 5e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    num_epochs = 30
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    train_record = []
    val_record = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model,dataloader_train,optimizer, loss_fn, device)
        train_record.append(train_loss)

        val_loss, val_accurracy = evaluate(model, dataloader_train, loss_fn, device)
        print(f"Validation Accurracy Current: {val_accurracy*100}%")
        val_record.append(val_loss)

        if val_loss< best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_reversi_model.pth')
            print("NEW BEST MODEL!!")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Learning has stabalized")
                break
                
    print("Training Complete!")
    plt.plot(range(len(train_record)),train_record, label = 'Training Loss', color='red')
    plt.plot(range(len(val_record)),val_record, label = 'Validation Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.show()
    plt.savefig('training_loss.png', dpi=100)    

if __name__ == "__main__":
    main()



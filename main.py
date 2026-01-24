import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import your modules
from Transformerv2 import ReversiBotDecoder
from utils import OthelloDataset, collate_fn
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        moves = batch['moves'].to(device)
        turns = batch['turns'].to(device)
        winners = batch['winners'].to(device).long()

        logits = model( moves, turns=turns, illegal_moves=None)

        loss = loss_fn(logits, winners)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches +=1
        pbar.set_postfix({'loss':loss.item()})

    return total_loss/num_batches
    
def evaluate(model,dataloader,loss_fn,device):
    model.eval()
    total_loss=0
    total_correct = 0
    num_batches = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            moves = batch['moves'].to(device)
            turns = batch['turns'].to(device)
            winners = batch['winners'].to(device).long()

            logits = model(moves,turns=turns, illegal_moves = None)

            loss = loss_fn(logits, winners)

            total_loss += loss.item()

            predictions = torch.argmax(logits,dim=1)
            correct = ((predictions == winners)).sum().item()
            total_correct += correct
            num_batches +=1
            for class_id in range(3):
                class_mask = (winners == class_id)
                class_correct[class_id] += (predictions[class_mask] == winners[class_mask]).sum().item()
                class_total[class_id] += class_mask.sum().item()
                         
    avg_loss = total_loss/num_batches
    accurracy = total_correct/(num_batches*logits.shape[0])
    for class_id in range(3):
            if class_total[class_id] > 0:
                class_acc = class_correct[class_id] / class_total[class_id]
                print(f"  Class {class_id}: {class_acc*100:.1f}%")              
    return avg_loss, accurracy


def get_class_weights(dataset):
    class_counts = [0, 0, 0]
    for idx in range(len(dataset)):
        winner = int(dataset.df.iloc[idx]['winner'])
        winner_remapped = {-1: 0, 0: 2, 1: 1}[winner]
        class_counts[winner_remapped] += 1

    total = sum(class_counts)
    class_weights = torch.tensor([total/count for count in class_counts], dtype=torch.float32)
    return class_weights



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
    validate_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_validate_with_illegal_moves.csv"
    try:
        train_dataset = OthelloDataset(train_csv_path, token_to_hot)
        batch_size = 256
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        validate_dataset = OthelloDataset(validate_csv_path, token_to_hot)
        batch_size = 128
        dataloader_validate = DataLoader(
            validate_dataset,
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
        embed_size=256,
        num_layers=6,
        heads=16,
        dropout=0.2,
        device=device,
        max_length=60,
        forward_expansion=4,
        num_classes=3
    ).to(device)

    optimizer = optim.Adam(model.parameters(),lr = 2e-4)

    class_weights = get_class_weights(train_dataset).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    num_epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 100

    train_record = []
    val_record = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model,dataloader_train,optimizer, loss_fn, device)
        train_record.append(train_loss)

        val_loss, val_accurracy= evaluate(model, dataloader_validate, loss_fn, device)
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



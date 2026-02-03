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

        logits = model( moves, turns=turns)

        loss = loss_fn(logits, winners)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
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

            logits = model(moves,turns=turns)

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
    total_samples = sum(class_total)
    accurracy = total_correct/(total_samples)
    print(f"Validation Accurracy Current: {accurracy*100:.1f}% loss: {avg_loss}")
    classes = {0:'black',1:'white',2:'draw'}
    for class_id in range(3):
            if class_total[class_id] > 0:
                class_acc = class_correct[class_id] / class_total[class_id]
                print(f"  {classes[class_id]}: {class_acc*100:.1f}%")              
    return avg_loss


def get_class_weights(dataset):

    class_counts = [0, 0, 0]
    
    # Count from the expanded examples in the dataset
    for example in dataset.examples:
        winner = example['winner'].item()  # Get the tensor value
        class_counts[winner] += 1

    total = sum(class_counts)
    class_weights = torch.tensor([total / count for count in class_counts], dtype=torch.float32)
    print(f"Class weights: {class_weights}")
    print(f"Class counts: black={class_counts[0]}, white={class_counts[1]}, draw={class_counts[2]}")
    return class_weights


def load_model(path, device='cuda'):
    """Load the saved model from disk"""
    # Initialize model with the same hyperparameters used in training
    model = ReversiBotDecoder(
        vocab_size=64,
        embed_size=512,
        num_layers=8,
        heads=16,
        dropout=0.05,
        device=device,
        max_length=60,
        forward_expansion=4,
        num_classes=3
    ).to(device)
    
    # Load the state dict
    state_dict = torch.load(path, map_location=device)
    
    # Load weights into model
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    
    return model



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    load = False
    if input("Load Model y or (n): ") == 'y': 
        load = True 

    #Neccessary Conversion DICT init
    letters = 'abcdefgh'
    numbers = '12345678'
    token_to_hot = {}
    i = 0
    for letter in letters:
        for number in numbers:
            token = letter+number
            token_to_hot[token] = i 
            i = i+1


    #Data Initialization and Data parameters
    train_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"
    validate_csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_validate.csv"

    batch_size = 128
        #Train set Init
    train_dataset = OthelloDataset(train_csv_path, token_to_hot)
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
        #Validation Init
    validate_dataset = OthelloDataset(validate_csv_path, token_to_hot)
    dataloader_validate = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    #Model Initialization and optimization
    model_pth = os.path.join('ReversAI','best_reversi_model.pth')
    
    if load == True:
        model = load_model(model_pth)
    else: 
        model = ReversiBotDecoder(
        vocab_size=64,
        embed_size=256,
        num_layers=5,
        heads=4,
        dropout=0.15,
        device=device,
        max_length=60,
        forward_expansion=3,
        num_classes=3
        ).to(device)

    optimizer = optim.AdamW(model.parameters(),lr = 1e-4, weight_decay=5e-5)

    class_weights = get_class_weights(train_dataset).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    #Training Length and Loop initialization
    num_epochs = 20
    best_val_loss = float('inf')
    train_record = []
    val_record = []


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        #Train
        train_loss = train_epoch(model,dataloader_train,optimizer, loss_fn, device)
        #Validate
        val_loss= evaluate(model, dataloader_validate, loss_fn, device)

        
        #Record Keeping
        train_record.append(train_loss)
        val_record.append(val_loss)

        #Save Model Conditional
        if val_loss< best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),model_pth)
            print("NEW BEST MODEL!!")

    #Performance Review
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



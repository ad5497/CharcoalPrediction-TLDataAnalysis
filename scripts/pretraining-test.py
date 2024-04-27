#!/bin/env python

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import os
import time

from PIL import Image

import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam

import torch.nn as nn
from torch.nn import CrossEntropyLoss

from torchvision import datasets, transforms

import timm


# ----------------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------------

def train_model(epochs, model, criterion, optimizer, train_loader, valid_loader, device, save_dir, save_every):
    hist_train_loss = []
    hist_valid_loss = []
    hist_train_accs = []
    hist_valid_accs = []

    best_accuracy = 0.0

    time_per_iter = [] # for estimating session length
    iters_per_epoch = len(train_loader)
    
    for epoch in range(epochs):
        train_corr = 0
        valid_corr = 0
        batch_corr = 0

        for i, (X_train, y_train) in enumerate(train_loader):
            start = time.time()
            
            X_train, y_train = X_train.to(device), y_train.to(device)

            train_pred = model(X_train)
            train_loss = criterion(train_pred, y_train)

            train_predicted = torch.max(train_pred.data, 1)[1]
            batch_corr = (train_predicted == y_train).sum()
            train_corr += batch_corr

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            end = time.time()
            time_per_iter.append(end - start)
            
            # print progress
            print(f'Iteration {i + 1}/{total_iters} complete; ',
                  f'Loss: {train_loss.item()}; ',
                  f'Time used (in sec): {time_per_iter[i]:.4f}')

            # checkpoint saving based on iterations
            if (i + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'epoch-{epoch}_iter-{i + 1}_checkpoint.pth')
                torch.save({
                    'epoch': epoch, 
                    'iteration': i + 1, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': train_loss.item(), 
                    'time_per_iter': time_per_iter
                }, checkpoint_path)
                print(f'Checkpoint saved at {checkpoint_path} after {i + 1}/{total_iters} iterations of epoch {epoch}')

        train_accuracy = train_corr.item() / len(train_loader.dataset)

        hist_train_loss.append(train_loss.item())
        hist_train_accs.append(train_accuracy)

        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)

                valid_pred = model(X_valid)

                valid_predicted = torch.max(valid_pred.data, 1)[1]
                valid_corr += (valid_predicted == y_valid).sum()

        valid_accuracy = valid_corr.item() / len(valid_loader.dataset)
        valid_loss = criterion(valid_pred, y_valid)

        hist_valid_loss.append(valid_loss.item())
        hist_valid_accs.append(valid_accuracy)

        print(
            f'[epoch: {epoch}]\n', 
            f'- train loss: {train_loss.item():}, train accuracy: {train_accuracy}\n',
            f'- valid loss: {valid_loss.item()}, valid accuracy: {valid_accuracy}'
        )

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hist_train_loss': hist_train_loss, 
                'hist_train_accs': hist_train_accs,
                'hist_valid_loss': hist_valid_loss, 
                'hist_valid_accs': hist_valid_accs
            }, os.path.join(save_dir, f'best_model.pth'))
            print(f'- New best model saved with accuracy {valid_accuracy:.2f}%')


# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------

def main():

    # Set device and random seed
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(42) if device.type == 'cuda' else torch.manual_seed(42)
    print(f'Using device: {device}')

    # Prepare dataset
    
    data_path = '/scratch/ad5497/data/ftp.ebi.ac.uk/pub/databases/IDR/idr0016-wawer-bioactivecompoundprofiling/2016-01-19-screens-bbbc022'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    train_size = int(0.2 * len(dataset))
    valid_size = int(0.05 * len(dataset))
    test_size = int(0.05 * len(dataset))
    rest = len(dataset) - train_size - valid_size - test_size
    _, train_dataset, valid_dataset, test_dataset = random_split(dataset, [rest, train_size, valid_size, test_size])
    
    bs = 128
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Create and train ViT model
        
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(dataset.classes))
    model = model.to(device)
    
    epochs = 2
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    
    save_dir = 'models-test'
    os.makedirs(save_dir, exist_ok=True)
    save_every = 60
    
    train_model(epochs, model, criterion, optimizer, train_loader, valid_loader, device, save_dir, save_every)

if __name__ == '__main__':
    main()
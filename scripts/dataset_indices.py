import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

def split_n_save_indices(data_path, save_path, transform, splits=[.70, .15]):
    """
    Function to split a torchvision dataset and save the indices of the splits
    so that they can be reused.

    ensures recreation of the splits
    """
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    train_size = int(splits[0] * len(dataset))
    valid_size = int(splits[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    torch.save({
        'train_indices': train_dataset.indices, 
        'valid_indices': valid_dataset.indices, 
        'test_indices': test_dataset.indices
    }, save_path)

def main():
    data_dir = '../data/pretraining-data'
    save_dir = '../data/pretraining-dataset-indices.pth'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    split_n_save_indices(data_dir, save_dir, transform)

if __name__ == "__main__":
    main()
import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    # Define a transform to normalize the data
    
    train = []
    for i in range(5):
        with np.load(f'../../../data/corruptmnist/train_{i}.npz') as data:
            
            train.append([torch.Tensor(data['images']).view(-1,1,28,28), torch.from_numpy(data['labels'])])
            
    test = None
    with np.load(f'../../../data/corruptmnist/test.npz') as data:
        test = [torch.Tensor(data['images']).view(-1,1,28,28), torch.from_numpy(data['labels'])]
    
    return train, test

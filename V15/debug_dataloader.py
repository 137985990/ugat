import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = [(torch.randn(10), torch.tensor(0))]
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data[idx]

def my_collate_fn(batch):
    print("🚨 MY COLLATE FUNCTION IS BEING CALLED!")
    return torch.utils.data.dataloader.default_collate(batch)

# Test
dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)

print("Testing DataLoader...")
try:
    batch = next(iter(loader))
    print(f"Success! Batch: {batch}")
except Exception as e:
    print(f"Error: {e}")

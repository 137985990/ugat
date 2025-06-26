#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, random_split
from data import create_multimodal_dataset_from_config, load_config

def debug_collate_fn(batch):
    print("🔥 DEBUG: Custom collate function is being called!")
    print(f"Batch size: {len(batch)}")
    
    tensors = []
    labels = []
    indices_lists = []
    is_real_masks = []
    source_datasets = []
    
    for i, item in enumerate(batch):
        tensor, label, indices_list, is_real_mask, source_dataset = item
        tensors.append(tensor)
        labels.append(label)
        indices_lists.append(indices_list)
        is_real_masks.append(is_real_mask)
        source_datasets.append(source_dataset)
    
    # Stack tensors
    batched_tensors = torch.stack(tensors)
    batched_labels = torch.stack(labels)
    batched_is_real_masks = torch.stack(is_real_masks)
    
    return batched_tensors, batched_labels, indices_lists, batched_is_real_masks, source_datasets

# Load dataset
config = load_config('config.yaml')
dataset = create_multimodal_dataset_from_config(config, phase='encode')

# Create splits
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Creating DataLoader with custom collate function...")

# Create DataLoader with debug collate function
train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=debug_collate_fn
)

print("Testing DataLoader...")
try:
    batch = next(iter(train_loader))
    print("✅ SUCCESS: Custom collate function worked!")
    print(f"Batch type: {type(batch)}")
    print(f"Batch length: {len(batch)}")
    print(f"Tensor shape: {batch[0].shape}")
except Exception as e:
    print(f"❌ ERROR: {e}")

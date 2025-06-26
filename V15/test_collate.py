#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from data import create_multimodal_dataset_from_config, load_config

def simple_collate_fn(batch):
    """Simple collate function that handles lists properly"""
    print(f"Custom collate called with batch size: {len(batch)}")
    
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
    
    print(f"Successfully created batch with shapes:")
    print(f"  tensors: {batched_tensors.shape}")
    print(f"  labels: {batched_labels.shape}")
    print(f"  is_real_masks: {batched_is_real_masks.shape}")
    
    return batched_tensors, batched_labels, indices_lists, batched_is_real_masks, source_datasets

# Test
config = load_config('config.yaml')
dataset = create_multimodal_dataset_from_config(config, phase='encode')

# Create a small subset for testing
from torch.utils.data import Subset
test_dataset = Subset(dataset, range(10))

# Test with custom collate function
print("Testing with custom collate function...")
loader = DataLoader(test_dataset, batch_size=2, collate_fn=simple_collate_fn)
batch = next(iter(loader))
print("Success!")
print(f"Batch type: {type(batch)}")
print(f"Batch length: {len(batch)}")

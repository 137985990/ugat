#!/usr/bin/env python3

from data import create_multimodal_dataset_from_config, load_config

config = load_config('config.yaml')
dataset = create_multimodal_dataset_from_config(config, phase='encode')
print('Dataset created')
print('First item:')
item = dataset[0]
print(f'Type: {type(item)}')
print(f'Length: {len(item)}')
for i, x in enumerate(item):
    print(f'[{i}]: type={type(x)}, shape={getattr(x, "shape", "N/A")}')

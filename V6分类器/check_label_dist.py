import pandas as pd
from collections import Counter

files = [
    '../Data/FM_original.csv',
    '../Data/OD_original.csv',
    '../Data/MEFAR_original.csv',
]

for file in files:
    try:
        df = pd.read_csv(file)
        if 'F' in df.columns:
            label_counts = Counter(df['F'])
            print(f"{file} 标签分布: {label_counts}")
        else:
            print(f"{file} 没有F这一列")
    except Exception as e:
        print(f"读取{file}出错: {e}")

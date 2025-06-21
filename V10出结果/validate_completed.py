import yaml
import numpy as np
import pandas as pd
import os
import logging
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from data import create_dataset_from_config
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import TGATUNet

# 设置随机种子，保证数据集划分一致
torch.manual_seed(42)

class CNN1DClassifier(torch.nn.Module):
    def __init__(self, input_size, seq_len, num_classes, hidden_size=128, num_layers=4, dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        layers = []
        in_channels = input_size
        out_channels = hidden_size
        kernel_size = 3
        for i in range(num_layers):
            layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            # 残差连接
            if i > 0:
                layers.append(ResidualBlock1D(out_channels, kernel_size, dropout))
            in_channels = out_channels
        self.conv = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x
        else:
            x = x.permute(0, 2, 1)
        out = self.conv(x)
        out = self.pool(out).squeeze(-1)
        out = self.fc(out)
        return out

class ResidualBlock1D(torch.nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out + x

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        in_dim = input_size
        for _ in range(4):  # 4层
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(torch.nn.Linear(hidden_size, num_classes))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.mlp(x)

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, 256, 4, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(256 * (2 if bidirectional else 1), 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.gru = torch.nn.GRU(input_size, 256, 4, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(256 * (2 if bidirectional else 1), 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 融合模型：CNN+LSTM
class CNNLSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, seq_len, num_classes, cnn_hidden=64, cnn_layers=2, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = input_size
        for i in range(cnn_layers):
            layers.append(torch.nn.Conv1d(in_channels, cnn_hidden, 3, padding=1))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_channels = cnn_hidden
        self.cnn = torch.nn.Sequential(*layers)
        self.lstm = torch.nn.LSTM(cnn_hidden, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(lstm_hidden*2, num_classes)
    def forward(self, x):
        # x: [batch, C, T] or [batch, T, C] -> [batch, C, T]
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x
        else:
            x = x.permute(0, 2, 1)
        out = self.cnn(x)  # [B, C, T]
        out = out.permute(0, 2, 1)  # [B, T, C]
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# GAT 分类器（需 torch_geometric）
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data as GeoData
    has_gat = True
except ImportError:
    has_gat = False

class GATClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, heads=2, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(input_size, hidden_size, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_size*heads, hidden_size, heads=1, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

# GAT+Transformer 分类器
class GATTransformerClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, heads=2, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(input_size, hidden_size, heads=heads, dropout=dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size*heads, nhead=nhead, dropout=dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size*heads, num_classes)
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = torch.relu(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)
        return x

def run_compare(config_path):
    print("\n===== 先评估原始数据 =====")
    orig_results = run_all_models(config_path, compare_tag='original')
    print("\n===== 再评估补全数据 =====")
    comp_results = run_all_models(config_path, compare_tag='completed')
    # 汇总对比
    print("\n===== 结果对比汇总 =====")
    print(f"{'模型':<10}{'原始Acc':<10}{'原始F1':<10}{'原始AUC':<10}{'补全Acc':<10}{'补全F1':<10}{'补全AUC':<10}")
    for m in orig_results:
        o = orig_results[m]
        c = comp_results.get(m, {'acc': '-', 'f1': '-', 'auc': '-'})
        print(f"{m:<10}{o['acc']:<10}{o['f1']:<10}{o['auc']:<10}{c['acc']:<10}{c['f1']:<10}{c['auc']:<10}")

def run_all_models(config_path, compare_tag=None):
    model_types = ['lstm', 'gru', 'mlp', 'cnn1d', 'cnnlstm']
    if has_gat:
        model_types.append('gat')
        model_types.append('gat_transformer')
    results = {}
    for mtype in model_types:
        print(f"\n===== 开始训练和评估模型: {mtype.upper()} =====")
        acc, f1, auc = main(config_path, model_type=mtype, compare_tag=compare_tag, return_metrics=True)
        results[mtype] = {'acc': acc, 'f1': f1, 'auc': auc}
    return results

def main(config_path, model_type=None, compare_tag=None, return_metrics=False):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 1e-3)
    num_epochs = config.get('epochs', 50)
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    num_classes = config.get('num_classes', 2)
    model_type = config.get('model_type', 'lstm').lower()  # 新增模型类型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type is not None:
        config['model_type'] = model_type

    # 数据集读取和划分
    if compare_tag == 'completed':
        completed_paths = [
            os.path.join(os.path.dirname(config_path), '../Data/FM_completed.csv'),
            os.path.join(os.path.dirname(config_path), '../Data/OD_completed.csv'),
            os.path.join(os.path.dirname(config_path), '../Data/MEFAR_completed.csv')
        ]
        for p in completed_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"未找到补全后的csv: {p}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        config_data['data_files'] = completed_paths
        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml', encoding='utf-8') as tmpf:
            yaml.safe_dump(config_data, tmpf, allow_unicode=True)
            tmp_config_path = tmpf.name
        dataset = create_dataset_from_config(tmp_config_path)
    else:
        # 原始数据
        dataset = create_dataset_from_config(config_path)

    # 划分数据集，只在测试集上评估
    total = len(dataset)
    train_split = config.get('train_split', 0.6)
    val_split = config.get('val_split', 0.2)
    train_len = round(train_split * total)
    val_len = round(val_split * total)
    test_len = total - train_len - val_len
    if test_len < 0:
        test_len = 0
    while train_len + val_len + test_len < total:
        test_len += 1
    while train_len + val_len + test_len > total:
        if test_len > 0:
            test_len -= 1
        elif val_len > 0:
            val_len -= 1
        else:
            train_len -= 1
    assert train_len + val_len + test_len == total
    from torch.utils.data import random_split
    _, _, test_ds = random_split(dataset, [train_len, val_len, test_len])
    loader = DataLoader(test_ds, batch_size=batch_size)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, feats, y, _, _ in loader:
            for i in range(x.size(0)):
                window = x[i].to(device).T
                _, logits = model(window, phase="encode")
                pred = logits.argmax(-1).item()
                all_preds.append(pred)
                all_targets.append(y[i].item())
    acc = sum([p==t for p,t in zip(all_preds, all_targets)]) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='binary') if len(set(all_targets)) > 1 else 0.0
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.0
    print(f"[TGATUNet-{dataset_tag}] Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'TGATUNet {dataset_tag} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    save_dir = 'png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'TGATUNet_{dataset_tag}_confusion_matrix.png'))
    plt.close()
    return acc, f1, auc

def eval_with_tgatunet_classifier(config_path, dataset_tag='original'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    batch_size = config.get('batch_size', 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集读取
    if dataset_tag == 'completed':
        completed_paths = [
            os.path.join(os.path.dirname(config_path), '../Data/FM_completed.csv'),
            os.path.join(os.path.dirname(config_path), '../Data/OD_completed.csv'),
            os.path.join(os.path.dirname(config_path), '../Data/MEFAR_completed.csv')
        ]
        for p in completed_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"未找到补全后的csv: {p}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        config_data['data_files'] = completed_paths
        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml', encoding='utf-8') as tmpf:
            yaml.safe_dump(config_data, tmpf, allow_unicode=True)
            tmp_config_path = tmpf.name
        dataset = create_dataset_from_config(tmp_config_path)
    else:
        # 原始数据
        dataset = create_dataset_from_config(config_path)

    # 取特征维度
    sample_x, sample_feats, _, _, _ = dataset[0]
    input_size = sample_x.shape[0] if sample_x.dim() == 2 else len(sample_x)
    hidden_channels = config.get('hidden_channels', 64)
    out_channels = input_size
    num_classes = config.get('num_classes', 2)

    # 加载模型
    model = TGATUNet(
        in_channels=input_size,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_classes=num_classes
    ).to(device)
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(config_path), 'Checkpoints/best_model.pth'))
    assert os.path.exists(ckpt_path), f"未找到模型权重: {ckpt_path}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 只在测试集上评估
    total = len(dataset)
    train_split = config.get('train_split', 0.6)
    val_split = config.get('val_split', 0.2)
    train_len = round(train_split * total)
    val_len = round(val_split * total)
    test_len = total - train_len - val_len
    if test_len < 0:
        test_len = 0
    while train_len + val_len + test_len < total:
        test_len += 1
    while train_len + val_len + test_len > total:
        if test_len > 0:
            test_len -= 1
        elif val_len > 0:
            val_len -= 1
        else:
            train_len -= 1
    assert train_len + val_len + test_len == total
    from torch.utils.data import random_split
    _, _, test_ds = random_split(dataset, [train_len, val_len, test_len])
    loader = DataLoader(test_ds, batch_size=batch_size)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, feats, y, _, _ in loader:
            for i in range(x.size(0)):
                window = x[i].to(device).T
                _, logits = model(window, phase="encode")
                pred = logits.argmax(-1).item()
                all_preds.append(pred)
                all_targets.append(y[i].item())
    acc = sum([p==t for p,t in zip(all_preds, all_targets)]) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='binary') if len(set(all_targets)) > 1 else 0.0
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.0
    print(f"[TGATUNet-{dataset_tag}] Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'TGATUNet {dataset_tag} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    save_dir = 'png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'TGATUNet_{dataset_tag}_confusion_matrix.png'))
    plt.close()
    return acc, f1, auc

def eval_with_tgatunet_complete_then_classify(config_path, dataset_tag):
    """
    先用TGATUNet补全原始数据，再用同一模型的分类分支对补全后的数据做分类推理和评估。
    只在测试集上评估。
    """
    import torch
    import yaml
    import os
    from data import create_dataset_from_config
    from model import TGATUNet
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    num_classes = config.get('num_classes', 2)
    batch_size = config.get('batch_size', 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = create_dataset_from_config(config_path)
    # 划分测试集
    total = len(dataset)
    train_split = config.get('train_split', 0.6)
    val_split = config.get('val_split', 0.2)
    train_len = round(train_split * total)
    val_len = round(val_split * total)
    test_len = total - train_len - val_len
    if test_len < 0:
        test_len = 0
    while train_len + val_len + test_len < total:
        test_len += 1
    while train_len + val_len + test_len > total:
        if test_len > 0:
            test_len -= 1
        elif val_len > 0:
            val_len -= 1
        else:
            train_len -= 1
    assert train_len + val_len + test_len == total
    _, _, test_ds = random_split(dataset, [train_len, val_len, test_len])
    loader = DataLoader(test_ds, batch_size=1)
    # 加载模型
    sample_x, sample_feats, _, _, _ = dataset[0]
    input_size = sample_x.shape[0] if sample_x.dim() == 2 else len(sample_x)
    hidden_channels = config.get('hidden_channels', 64)
    out_channels = input_size
    model = TGATUNet(
        in_channels=input_size,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_classes=num_classes
    ).to(device)
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(config_path), 'Checkpoints/best_model.pth'))
    assert os.path.exists(ckpt_path), f"未找到模型权重: {ckpt_path}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    # 先补全再分类
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, feats, y, _, _ in loader:
            # x: [1, C, T]
            x = x[0].to(device)  # [C, T]
            # 1. 先补全（假设补全所有通道，实际可按mask逻辑调整）
            # 这里演示：将所有通道置零再补全
            masked = x.clone()
            masked[:, :] = 0
            out, logits = model(masked.T, phase="encode")  # out: [C, T]
            # 2. 用补全结果再分类
            _, logits2 = model(out.T, phase="encode")  # 修正：补全结果转置为 [T, C]
            pred = logits2.argmax(-1).item()
            all_preds.append(pred)
            all_targets.append(y.item())
    acc = sum([p==t for p,t in zip(all_preds, all_targets)]) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='binary') if len(set(all_targets)) > 1 else 0.0
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.0
    print(f"[TGATUNet-补全后分类-{dataset_tag}] Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'TGATUNet Complete-Then-Classify {dataset_tag} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    save_dir = 'png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'TGATUNet_complete_then_classify_{dataset_tag}_confusion_matrix.png'))
    plt.close()
    return acc, f1, auc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--all', action='store_true', help='一次性训练和评估所有模型')
    parser.add_argument('--compare', action='store_true', help='原始与补全数据对比')
    parser.add_argument('--tgatunet', action='store_true', help='仅用TGATUNet分类分支评估')
    parser.add_argument('--tgatunet_complete_then_classify', action='store_true', help='先补全再分类评估')
    args = parser.parse_args()
    if args.tgatunet:
        print("\n===== TGATUNet 分类分支评估：原始数据 =====")
        eval_with_tgatunet_classifier(args.config, 'original')
        print("\n===== TGATUNet 分类分支评估：补全数据 =====")
        eval_with_tgatunet_classifier(args.config, 'completed')
    elif args.tgatunet_complete_then_classify:
        print("\n===== TGATUNet 先补全再分类评估：原始数据 =====")
        eval_with_tgatunet_complete_then_classify(args.config, 'original')
    elif args.compare:
        run_compare(args.config)
    elif args.all:
        run_all_models(args.config)
    else:
        main(args.config)

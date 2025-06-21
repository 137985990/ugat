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
from sklearn.metrics import f1_score, roc_auc_score

# 设置随机种子，保证数据集划分一致
torch.manual_seed(42)

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(torch.nn.Linear(hidden_size, num_classes))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, C, T] or [batch, T, C], flatten except batch
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.mlp(x)

def main(config_path):
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

    # 数据集读取和划分
    dataset = create_dataset_from_config(config_path)
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
    print(f"[数据集分配] 总数: {total}, 训练: {train_len}, 验证: {val_len}, 测试: {test_len}")
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 检查标签分布和block分布
    def print_label_and_block_distribution(loader, name):
        labels = []
        blocks = []
        for _, y, _, _ in loader.dataset:
            labels.append(int(y))
            if hasattr(loader.dataset, 'blocks'):
                # 获取block索引
                blocks.append(loader.dataset.blocks[_])
        from collections import Counter
        print(f"[{name}] 标签分布: {Counter(labels)}")
        # print(f"[{name}] block分布: {Counter(blocks)}")

    # （已注释）统计滑窗后训练、验证、测试集的标签分布
    # def print_window_label_distribution(ds, name, num=10000):
    #     labels = []
    #     for i in range(min(len(ds), num)):
    #         _, label, _, _ = ds[i]
    #         labels.append(label.item())
    #     from collections import Counter
    #     print(f"[{name}] 滑窗后前{num}个样本标签分布: {Counter(labels)}")
    # print_window_label_distribution(train_ds, '训练集')
    # print_window_label_distribution(val_ds, '验证集')
    # print_window_label_distribution(test_ds, '测试集')

    # 注释掉导致报错的窗口F值打印代码
    # print_window_F_values(train_ds, '训练集')
    # print_window_F_values(val_ds, '验证集')
    # print_window_F_values(test_ds, '测试集')

    # 检查原始dataset每个block的F列分布，排查标签单一问题
    def print_block_F_distribution(dataset, num_blocks=5):
        print(f"原始dataset前{num_blocks}个block的F列分布:")
        for i, block in enumerate(dataset.blocks[:num_blocks]):
            f_vals = block[dataset.label_col].values
            from collections import Counter
            print(f"block {i}: {Counter(f_vals)}")
    print_block_F_distribution(dataset)

    # 假设dataset[0][0]是特征，dataset[0][1]是标签
    sample_x, _, _, _ = dataset[0]
    # sample_x: [C, T]，LSTM需要 [T, C] 作为 seq_len, input_size
    input_size = sample_x.shape[0] if sample_x.dim() == 2 else len(sample_x)
    seq_len = sample_x.shape[1] if sample_x.dim() == 2 else 1

    # 根据模型类型选择模型
    if model_type == 'lstm':
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
        model_save_path = 'best_lstm_classifier.pth'
    elif model_type == 'gru':
        model = GRUClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
        model_save_path = 'best_gru_classifier.pth'
    elif model_type == 'mlp':
        model = MLPClassifier(input_size * seq_len, hidden_size, num_layers, num_classes).to(device)
        model_save_path = 'best_mlp_classifier.pth'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    # 计算类别权重，缓解不平衡
    from collections import Counter
    label_counter = Counter([int(dataset[i][1]) for i in range(len(dataset))])
    class_weights = torch.tensor([1.0 / (label_counter.get(i, 1)) for i in range(num_classes)], dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0
    epochs_no_improve = 0
    early_stop_patience = config.get('early_stop_patience', 10)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y, _, _ in train_loader:
            # x: [batch, C, T]，需要转为 [batch, T, C]（RNN类），MLP直接展平
            if model_type in ['lstm', 'gru']:
                x = x.permute(0, 2, 1)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                if model_type in ['lstm', 'gru']:
                    x = x.permute(0, 2, 1)
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                val_targets.extend(y.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total > 0 else 0
        val_f1 = f1_score(val_targets, val_preds, average='binary') if len(set(val_targets)) > 1 else 0.0
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except:
            val_auc = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        scheduler.step(avg_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"早停触发，连续{early_stop_patience}轮未提升，停止训练。")
            break
    print("训练完成，最佳验证集准确率：", best_val_acc)

    # 测试集评估
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    correct = 0
    total = 0
    test_targets = []
    test_preds = []
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            if model_type in ['lstm', 'gru']:
                x = x.permute(0, 2, 1)
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            test_targets.extend(y.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
            correct += (predicted == y).sum().item()
            total += y.size(0)
    test_acc = correct / total if total > 0 else 0
    test_f1 = f1_score(test_targets, test_preds, average='binary') if len(set(test_targets)) > 1 else 0.0
    try:
        test_auc = roc_auc_score(test_targets, test_preds)
    except:
        test_auc = 0.0
    print(f"测试集准确率: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

def run_all_models(config_path):
    model_types = ['lstm', 'gru', 'mlp']
    import yaml
    for mtype in model_types:
        print(f"\n===== 开始训练和评估模型: {mtype.upper()} =====")
        # 读取配置，修改model_type
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config['model_type'] = mtype
        # 临时写入一个新的配置文件
        tmp_config_path = f"tmp_{mtype}_config.yaml"
        with open(tmp_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        main(tmp_config_path)
        # 删除临时文件
        os.remove(tmp_config_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--all', action='store_true', help='一次性训练和评估所有模型')
    args = parser.parse_args()
    if args.all:
        run_all_models(args.config)
    else:
        main(args.config)

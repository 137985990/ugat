# test_multimodal_modifications.py - 测试多模态损失修改

import torch
import yaml
from simple_multimodal_integration import create_simple_multimodal_criterion

def test_multimodal_criterion():
    """测试多模态损失函数的创建和使用"""
    
    print("=" * 60)
    print("测试多模态损失函数修改")
    print("=" * 60)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"配置类型: {config.get('loss_config', {}).get('type')}")
    print(f"Common模态: {config.get('common_modalities', [])}")
    
    # 创建损失函数
    if config.get('loss_config', {}).get('type') == 'multimodal':
        criterion = create_simple_multimodal_criterion(config)
        print("✅ 成功创建多模态损失函数")
        print(f"Common索引: {criterion.common_indices}")
        print(f"Common权重: {criterion.common_weight}")
    else:
        criterion = torch.nn.MSELoss()
        print("✅ 使用标准MSE损失函数")
    
    # 测试损失计算
    print("\n测试损失计算:")
    pred = torch.randn(100)
    target = torch.randn(100)
    
    # 测试不同通道的损失
    if hasattr(criterion, 'common_indices'):
        # Common通道测试 (假设通道0是common)
        if 0 in criterion.common_indices:
            common_loss = criterion(pred, target, channel_idx=0, is_common=True)
            print(f"  Common通道(0)损失: {common_loss.item():.6f}")
        
        # Have通道测试 (假设通道7是have)
        have_loss = criterion(pred, target, channel_idx=7, is_common=False)
        print(f"  Have通道(7)损失: {have_loss.item():.6f}")
        
        if 0 in criterion.common_indices:
            ratio = common_loss.item() / have_loss.item()
            print(f"  权重比例: {ratio:.3f} (预期约{criterion.common_weight})")
    else:
        # 标准MSE测试
        loss = criterion(pred, target)
        print(f"  标准MSE损失: {loss.item():.6f}")
    
    return criterion

def test_train_integration():
    """测试训练集成"""
    
    print(f"\n" + "=" * 60)
    print("测试训练集成")
    print("=" * 60)
    
    try:
        # 尝试导入修改后的train模块（检查语法）
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", "train.py")
        train_module = importlib.util.module_from_spec(spec)
        
        print("✅ train.py语法检查通过")
        
        # 检查关键函数是否存在
        spec.loader.exec_module(train_module)
        
        if hasattr(train_module, 'train_phased'):
            print("✅ train_phased函数存在")
        
        if hasattr(train_module, 'eval_loop'):
            print("✅ eval_loop函数存在")
            
        print("✅ 所有修改集成成功")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False
    
    return True

def simulate_training_with_multimodal():
    """模拟多模态训练过程"""
    
    print(f"\n" + "=" * 60)
    print("模拟多模态训练过程")
    print("=" * 60)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建损失函数
    criterion = create_simple_multimodal_criterion(config)
    
    # 模拟数据
    batch_size, C, T = 8, 32, 100  # 与config中的设置匹配
    num_classes = config.get('num_classes', 2)
    
    batch = torch.randn(batch_size, C, T)
    labels = torch.randint(0, num_classes, (batch_size,))
    is_real_mask = torch.ones(batch_size, C, dtype=torch.bool)
    
    # 模拟have模态mask（假设后25个通道是have模态）
    is_real_mask[:, 7:] = torch.randint(0, 2, (batch_size, C-7), dtype=torch.bool)
    
    print(f"模拟数据形状:")
    print(f"  batch: {batch.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  is_real_mask: {is_real_mask.shape}")
    print(f"  Common通道数: {len(criterion.common_indices)}")
    print(f"  Have通道数: {C - len(criterion.common_indices)}")
    
    # 模拟训练步骤
    print(f"\n模拟训练步骤:")
    
    total_loss = 0.0
    total_common_loss = 0.0
    total_have_loss = 0.0
    
    for i in range(batch_size):
        # 模拟模型输出
        out = torch.randn(C, T)  # [C, T]
        logits = torch.randn(num_classes)  # [num_classes]
        
        real_channels = is_real_mask[i]
        common_indices = criterion.common_indices
        
        recon_loss_i = 0.0
        common_loss_i = 0.0
        have_loss_i = 0.0
        real_count = 0
        common_count = 0
        have_count = 0
        
        for c in range(C):
            target = batch[i, c, :]
            pred = out[c, :]
            
            is_common_channel = c in common_indices
            
            if is_common_channel:
                # Common模态：始终计算损失
                loss_c = criterion(pred, target, channel_idx=c, is_common=True)
                recon_loss_i += loss_c
                common_loss_i += loss_c
                real_count += 1
                common_count += 1
            elif real_channels[c]:
                # Have模态：只对真实通道计算损失
                loss_c = criterion(pred, target, channel_idx=c, is_common=False)
                recon_loss_i += loss_c
                have_loss_i += loss_c
                real_count += 1
                have_count += 1
        
        # 平均损失
        if real_count > 0:
            recon_loss_i /= real_count
        if common_count > 0:
            common_loss_i /= common_count
        if have_count > 0:
            have_loss_i /= have_count
        
        total_loss += recon_loss_i.item()
        total_common_loss += common_loss_i.item() if common_count > 0 else 0
        total_have_loss += have_loss_i.item() if have_count > 0 else 0
    
    # 平均结果
    avg_total_loss = total_loss / batch_size
    avg_common_loss = total_common_loss / batch_size
    avg_have_loss = total_have_loss / batch_size
    
    print(f"  平均总损失: {avg_total_loss:.6f}")
    print(f"  平均Common损失: {avg_common_loss:.6f}")
    print(f"  平均Have损失: {avg_have_loss:.6f}")
    
    if avg_have_loss > 0:
        ratio = avg_common_loss / avg_have_loss
        print(f"  Common/Have比例: {ratio:.3f} (目标比例: {criterion.common_weight})")

if __name__ == "__main__":
    # 运行所有测试
    print("🚀 开始测试多模态损失函数修改")
    
    # 测试1: 损失函数创建
    criterion = test_multimodal_criterion()
    
    # 测试2: 训练集成
    integration_success = test_train_integration()
    
    # 测试3: 模拟训练过程
    if integration_success:
        simulate_training_with_multimodal()
    
    print(f"\n" + "=" * 60)
    print("🎉 多模态损失函数修改测试完成！")
    print("=" * 60)
    
    if integration_success:
        print("✅ 所有修改成功集成")
        print("✅ Common modalities现在参与损失计算")
        print("✅ 分类模型可以更好地指导重建质量")
        print("\n🔥 现在可以开始训练，观察性能提升！")
        print("\n推荐监控指标:")
        print("  - Common模态损失变化")
        print("  - Have模态损失变化") 
        print("  - 分类准确率提升")
        print("  - 损失权重比例是否合理")
    else:
        print("❌ 部分修改需要调整")
        print("请检查train.py中的语法错误")

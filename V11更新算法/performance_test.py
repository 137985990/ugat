# performance_test.py - 性能测试和基准测试脚本

import time
import torch
import psutil
import numpy as np
from contextlib import contextmanager
from typing import Dict, List
from visualization import chart_generator, safe_plot_with_fallback

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文管理器"""
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024**3
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            else:
                end_gpu_memory = 0
            
            # 计算指标
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            self.metrics[name] = {
                'duration': duration,
                'memory_delta': memory_delta,
                'gpu_memory_delta': gpu_memory_delta,
                'peak_memory': end_memory,
                'peak_gpu_memory': end_gpu_memory
            }
            
            print(f"[{name}] 耗时: {duration:.3f}s, 内存增量: {memory_delta:.2f}GB, GPU内存增量: {gpu_memory_delta:.2f}GB")

def benchmark_models():
    """对比标准模型和优化模型的性能"""
    import yaml
    from model import TGATUNet
    from model_optimized import OptimizedTGATUNet
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8  # 小批量测试
    seq_len = 320
    in_channels = 32
    
    # 创建测试数据
    test_input = torch.randn(seq_len, in_channels).to(device)
    
    profiler = PerformanceProfiler()
    
    # 测试标准模型
    print("🔍 测试标准 TGATUNet 模型...")
    model_std = TGATUNet(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=in_channels,
        num_classes=2
    ).to(device)
    
    model_std.eval()
    with torch.no_grad():
        with profiler.profile("TGATUNet_inference"):
            for _ in range(10):  # 多次推理取平均
                output, logits = model_std(test_input)
    
    # 测试优化模型
    print("🚀 测试优化 OptimizedTGATUNet 模型...")
    model_opt = OptimizedTGATUNet(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=in_channels,
        encoder_layers=2,
        decoder_layers=2,
        heads=2,
        trans_layers=1,
        num_classes=2,
        use_checkpoint=False  # 推理时关闭checkpoint
    ).to(device)
    
    model_opt.eval()
    with torch.no_grad():
        with profiler.profile("OptimizedTGATUNet_inference"):
            for _ in range(10):  # 多次推理取平均
                output, logits = model_opt(test_input)
    
    # 计算参数量
    std_params = sum(p.numel() for p in model_std.parameters())
    opt_params = sum(p.numel() for p in model_opt.parameters())
    
    print(f"\n📊 模型对比结果:")
    print(f"标准模型参数量: {std_params:,}")
    print(f"优化模型参数量: {opt_params:,}")
    print(f"参数减少: {(std_params - opt_params) / std_params * 100:.1f}%")
    
    # 推理时间对比
    std_time = profiler.metrics['TGATUNet_inference']['duration'] / 10
    opt_time = profiler.metrics['OptimizedTGATUNet_inference']['duration'] / 10
    print(f"标准模型推理时间: {std_time:.4f}s")
    print(f"优化模型推理时间: {opt_time:.4f}s")
    print(f"推理加速: {std_time / opt_time:.2f}x")
    
    return profiler.metrics

def benchmark_graph_building():
    """图构建性能测试"""
    from graph import build_graph
    from graph_cache import build_graph_cached
    
    seq_lengths = [64, 128, 256, 320, 512]
    channels = 32
    time_k = 1
    
    profiler = PerformanceProfiler()
    
    results = {'standard': [], 'cached': []}
    
    for seq_len in seq_lengths:
        print(f"\n📏 测试序列长度: {seq_len}")
        test_data = torch.randn(seq_len, channels)
        
        # 测试标准图构建
        with profiler.profile(f"standard_graph_{seq_len}"):
            for _ in range(100):  # 多次构建
                graph = build_graph(test_data, time_k=time_k)
        
        # 测试缓存图构建
        with profiler.profile(f"cached_graph_{seq_len}"):
            for _ in range(100):  # 多次构建
                graph = build_graph_cached(test_data, time_k=time_k)
        
        std_time = profiler.metrics[f"standard_graph_{seq_len}"]['duration'] / 100
        cached_time = profiler.metrics[f"cached_graph_{seq_len}"]['duration'] / 100
        
        results['standard'].append(std_time)
        results['cached'].append(cached_time)
        print(f"标准构建: {std_time:.6f}s, 缓存构建: {cached_time:.6f}s, 加速: {std_time/cached_time:.2f}x")
    
    # 使用新的可视化模块生成图表
    try:
        chart_generator.plot_performance_comparison(
            seq_lengths=seq_lengths, 
            results=results, 
            save_path='graph_benchmark',
            use_chinese=True
        )
    except Exception as e:
        print(f"⚠️ 中文图表生成失败: {e}")
        print("🔄 使用英文重新生成...")
        try:
            chart_generator.plot_performance_comparison(
                seq_lengths=seq_lengths, 
                results=results, 
                save_path='graph_benchmark',
                use_chinese=False
            )
        except Exception as e2:
            print(f"❌ 图表生成完全失败: {e2}")
            print("💡 将保存数据到文件...")
            # 保存原始数据
            import json
            with open('benchmark_data.json', 'w') as f:
                json.dump({
                    'seq_lengths': seq_lengths,
                    'results': results
                }, f, indent=2)
            print("📊 基准数据已保存到 benchmark_data.json")
    
    return results

def benchmark_memory_usage():
    """内存使用基准测试"""
    from memory_optimizer import MemoryEfficientDataLoader, MemoryMonitor
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建测试数据集
    size = 1000
    seq_len = 320
    channels = 32
    
    x = torch.randn(size, channels, seq_len)
    y = torch.randint(0, 2, (size,))
    mask_idx = torch.randint(0, channels, (size,))
    is_real = torch.ones(size, channels, dtype=torch.bool)
    
    dataset = TensorDataset(x, y, mask_idx, is_real)
    
    monitor = MemoryMonitor()
    profiler = PerformanceProfiler()
    
    batch_sizes = [8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\n🧪 测试批大小: {batch_size}")
        
        # 标准数据加载器
        monitor.check_and_warn("标准加载器前")
        with profiler.profile(f"standard_loader_{batch_size}"):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(loader):
                if i >= 10:  # 只测试前10个批次
                    break
        monitor.check_and_warn("标准加载器后")
        
        # 优化数据加载器
        monitor.check_and_warn("优化加载器前")
        with profiler.profile(f"memory_loader_{batch_size}"):
            loader = MemoryEfficientDataLoader(dataset, batch_size=batch_size)
            for i, batch in enumerate(loader):
                if i >= 10:  # 只测试前10个批次
                    break
        monitor.check_and_warn("优化加载器后")
    
    return profiler.metrics

if __name__ == "__main__":
    print("🚀 开始性能基准测试...")
    
    print("\n" + "="*50)
    print("1. 模型推理性能测试")
    print("="*50)
    model_results = benchmark_models()
    
    print("\n" + "="*50)
    print("2. 图构建性能测试")
    print("="*50)
    graph_results = benchmark_graph_building()
    
    print("\n" + "="*50)
    print("3. 内存使用测试")
    print("="*50)
    memory_results = benchmark_memory_usage()
    
    print("\n🎉 所有测试完成！")
    
    # 生成综合报告
    try:
        chart_generator.plot_memory_usage(memory_results, 'memory_analysis')
        chart_generator.plot_model_comparison(model_results, 'model_analysis')
        print("📊 所有图表已生成！")
    except Exception as e:
        print(f"⚠️ 图表生成过程中出现问题: {e}")
        print("💡 建议检查依赖库安装: pip install matplotlib seaborn")
    
    print("📄 详细结果文件:")
    print("  - graph_benchmark.png/.pdf/.svg - 图构建性能对比")
    print("  - memory_analysis.png/.pdf - 内存使用分析") 
    print("  - model_analysis.png/.pdf - 模型性能对比")
    print("  - benchmark_data.json - 原始数据备份")

# test_visualization.py - 测试图表生成功能

import sys
import warnings
warnings.filterwarnings('ignore')

def test_font_support():
    """测试中文字体支持"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        print("🔍 检查可用字体...")
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        found_fonts = []
        
        for font in chinese_fonts:
            if font in available_fonts:
                found_fonts.append(font)
        
        if found_fonts:
            print(f"✅ 找到中文字体: {', '.join(found_fonts)}")
            return found_fonts[0]
        else:
            print("⚠️ 未找到中文字体，将使用英文")
            return None
            
    except Exception as e:
        print(f"❌ 字体检测失败: {e}")
        return None

def test_simple_chart():
    """测试简单图表生成"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置字体
        font = test_font_support()
        if font:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
        
        # 生成测试数据
        x = [64, 128, 256, 320, 512]
        y1 = [0.001, 0.004, 0.015, 0.024, 0.045]  # 标准方法
        y2 = [0.0005, 0.001, 0.003, 0.005, 0.008]  # 优化方法
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, 'r-o', label='Standard' if not font else '标准方法', linewidth=2)
        plt.plot(x, y2, 'b-o', label='Optimized' if not font else '优化方法', linewidth=2)
        
        plt.xlabel('Sequence Length' if not font else '序列长度')
        plt.ylabel('Time (seconds)' if not font else '时间(秒)')
        plt.title('Performance Test' if not font else '性能测试')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig('test_chart.png', dpi=150, bbox_inches='tight')
        plt.savefig('test_chart.pdf', bbox_inches='tight')
        
        print("✅ 测试图表生成成功！")
        print("📊 文件已保存: test_chart.png, test_chart.pdf")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")
        return False

def test_visualization_module():
    """测试可视化模块"""
    try:
        from visualization import chart_generator
        
        # 测试数据
        seq_lengths = [64, 128, 256, 320, 512]
        results = {
            'standard': [0.001, 0.004, 0.015, 0.024, 0.045],
            'cached': [0.0005, 0.001, 0.003, 0.005, 0.008]
        }
        
        print("🔍 测试可视化模块...")
        
        # 首先尝试中文
        try:
            chart_generator.plot_performance_comparison(
                seq_lengths=seq_lengths,
                results=results,
                save_path='test_module_chinese',
                use_chinese=True
            )
            print("✅ 中文图表生成成功！")
            
        except Exception as e:
            print(f"⚠️ 中文图表失败: {e}")
            print("🔄 尝试英文图表...")
            
            chart_generator.plot_performance_comparison(
                seq_lengths=seq_lengths,
                results=results,
                save_path='test_module_english',
                use_chinese=False
            )
            print("✅ 英文图表生成成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化模块测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖库"""
    required_packages = [
        'matplotlib',
        'seaborn', 
        'numpy',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    print("🧪 开始图表生成测试...")
    print("="*50)
    
    # 1. 检查依赖
    print("1. 检查依赖库...")
    if not check_dependencies():
        print("❌ 依赖检查失败，请先安装所需库")
        return
    
    # 2. 测试简单图表
    print("\n2. 测试基础图表生成...")
    if not test_simple_chart():
        print("❌ 基础图表测试失败")
        return
    
    # 3. 测试可视化模块
    print("\n3. 测试可视化模块...")
    if not test_visualization_module():
        print("❌ 可视化模块测试失败")
        return
    
    print("\n🎉 所有测试通过！")
    print("📊 图表应该已经正常生成，请检查生成的文件:")
    print("  - test_chart.png/.pdf")
    print("  - test_module_chinese_*.png/.pdf 或 test_module_english_*.png/.pdf")

if __name__ == "__main__":
    main()

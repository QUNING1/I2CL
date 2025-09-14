#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import traceback
import signal
import time

def signal_handler(signum, frame):
    print(f"\n收到信号 {signum}，程序被中断")
    traceback.print_stack(frame)
    sys.exit(1)

def test_with_error_handling():
    """带错误处理的测试"""
    try:
        print("开始调试测试...")
        sys.path.append('.')
        import utils
        import argparse
        
        # 加载配置
        config = utils.load_config('configs/config_i2cl.py')
        
        # 设置参数 - 使用dbpedia数据集，2个聚类
        config['datasets'] = ['dbpedia']
        config['models'] = ['models/llama-2-7b-hf/']
        config['gpus'] = ['3']
        config['run_num'] = 1
        config['run_baseline'] = False
        config['post_fuse_method'] = 'kmeans'
        config['kmeans_n_clusters'] = 2
        config['inject_method'] = 'static_add'
        config['shot_per_class'] = 5
        config['tok_pos'] = 'label'
        config['test_data_num'] = 10  # 减少测试数据量
        
        print(f"实验配置:")
        print(f"  数据集: {config['datasets'][0]}")
        print(f"  后融合方法: {config['post_fuse_method']}")
        print(f"  聚类数: {config['kmeans_n_clusters']}")
        print(f"  注入方法: {config['inject_method']}")
        print(f"  每类样本数: {config['shot_per_class']}")
        print(f"  token位置: {config['tok_pos']}")
        print(f"  测试数据量: {config['test_data_num']}")
        
        # 创建参数对象
        input_args = argparse.Namespace()
        input_args.model_name = config['models'][0]
        input_args.dataset_name = config['datasets'][0]
        input_args.gpu = config['gpus'][0]
        input_args.config = config
        
        # 导入主函数
        from run_i2cl import main
        
        print("开始运行实验...")
        main(input_args)
        print("✅ 实验完成")
        return True
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
        return False
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 50)
    print("I2CL 崩溃调试脚本")
    print("=" * 50)
    
    # 测试单个实验
    if not test_with_error_handling():
        print("实验测试失败")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("测试通过！")
    print("=" * 50)

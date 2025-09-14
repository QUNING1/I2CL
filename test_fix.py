#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import traceback

def test_single_experiment():
    """测试单个实验"""
    try:
        print("测试单个实验...")
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
        
        print(f"实验配置:")
        print(f"  数据集: {config['datasets'][0]}")
        print(f"  后融合方法: {config['post_fuse_method']}")
        print(f"  聚类数: {config['kmeans_n_clusters']}")
        print(f"  注入方法: {config['inject_method']}")
        print(f"  每类样本数: {config['shot_per_class']}")
        print(f"  token位置: {config['tok_pos']}")
        
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
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("I2CL 修复测试脚本")
    print("=" * 50)
    
    # 测试单个实验
    if not test_single_experiment():
        print("实验测试失败")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("测试通过！")
    print("=" * 50)

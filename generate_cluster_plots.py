#!/usr/bin/env python3
"""
生成聚类数vs准确率图表
从实验结果中提取数据并生成可视化图表
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_results_from_experiments():
    """从实验结果中提取聚类数和准确率数据"""
    
    # 定义数据集和参数
    datasets = ["dbpedia", "mr", "subj", "sst5", "hate_speech18", "trec", "sst2", "agnews"]
    shot_per_class = 5
    inject_methods = ["static_add"]
    post_fuse_methods = ["hier"]  # 根据你的脚本调整
    tok_pos_list = ["last", "label"]
    cluster_range = range(2, 11)  # 2到10
    
    results = defaultdict(lambda: defaultdict(list))  # dataset -> tok_pos -> [(clusters, acc)]
    
    # 遍历所有可能的实验目录
    for dataset in datasets:
        for inject_method in inject_methods:
            for post_fuse_method in post_fuse_methods:
                for tok_pos in tok_pos_list:
                    for clusters in cluster_range:
                        # 构建实验名称（与run_experiments.sh中的命名一致）
                        exp_name = f"{dataset}_{post_fuse_method}_{clusters}_{inject_method}_{tok_pos}_{shot_per_class}"
                        
                        # 查找对应的结果文件
                        result_files = glob.glob(f"exps/{exp_name}*/result_dict.json")
                        
                        if result_files:
                            try:
                                with open(result_files[0], 'r') as f:
                                    result_data = json.load(f)
                                
                                # 提取平均准确率
                                if 'mean_results' in result_data and 'acc_mean' in result_data['mean_results']:
                                    acc = result_data['mean_results']['acc_mean']
                                    results[dataset][tok_pos].append((clusters, acc))
                                    print(f"找到结果: {dataset} {tok_pos} {clusters} clusters -> {acc:.4f}")
                                    
                            except Exception as e:
                                print(f"读取文件失败 {result_files[0]}: {e}")
                        else:
                            print(f"未找到结果文件: {exp_name}")
    
    return results

def plot_cluster_vs_accuracy(results):
    """生成聚类数vs准确率图表"""
    
    datasets = list(results.keys())
    tok_pos_list = ["last", "label"]
    
    # 为每个数据集创建图表
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        
        for i, tok_pos in enumerate(tok_pos_list):
            if tok_pos in results[dataset]:
                # 提取数据并排序
                data = results[dataset][tok_pos]
                data.sort(key=lambda x: x[0])  # 按聚类数排序
                
                if data:
                    clusters = [x[0] for x in data]
                    accuracies = [x[1] for x in data]
                    
                    # 绘制线条
                    plt.plot(clusters, accuracies, 
                            marker='o', linewidth=2, markersize=8,
                            label=f'tok_pos={tok_pos}')
                    
                    # 添加数据点标注
                    for x, y in zip(clusters, accuracies):
                        plt.annotate(f'{y:.3f}', (x, y), 
                                   textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=8)
        
        plt.xlabel('聚类数 (Number of Clusters)', fontsize=12)
        plt.ylabel('平均准确率 (Average Accuracy)', fontsize=12)
        plt.title(f'{dataset.upper()} - 聚类数 vs 平均准确率', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xticks(range(2, 11))
        plt.ylim(0, 1)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(f'results/{dataset}_cluster_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{dataset}_cluster_vs_accuracy.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"已生成 {dataset} 的聚类数vs准确率图表")

def main():
    """主函数"""
    print("开始提取实验结果...")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 提取结果
    results = extract_results_from_experiments()
    
    if not results:
        print("未找到任何实验结果，请确保实验已完成")
        return
    
    print(f"找到 {len(results)} 个数据集的结果")
    
    # 生成图表
    print("开始生成图表...")
    plot_cluster_vs_accuracy(results)
    
    # 生成汇总表格
    print("生成汇总表格...")
    generate_summary_table(results)
    
    print("所有图表生成完成！")

def generate_summary_table(results):
    """生成汇总表格"""
    
    datasets = list(results.keys())
    tok_pos_list = ["last", "label"]
    cluster_range = range(2, 11)
    
    # 创建汇总数据
    summary_data = []
    
    for dataset in datasets:
        for tok_pos in tok_pos_list:
            if tok_pos in results[dataset]:
                data = results[dataset][tok_pos]
                data_dict = dict(data)  # 转换为字典便于查找
                
                row = [dataset, tok_pos]
                for clusters in cluster_range:
                    acc = data_dict.get(clusters, None)
                    row.append(f"{acc:.4f}" if acc is not None else "N/A")
                
                summary_data.append(row)
    
    # 保存为CSV
    import csv
    with open('results/cluster_accuracy_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Dataset', 'Token_Position'] + [f'Clusters_{i}' for i in cluster_range]
        writer.writerow(header)
        writer.writerows(summary_data)
    
    print("汇总表格已保存到 results/cluster_accuracy_summary.csv")

if __name__ == "__main__":
    main()

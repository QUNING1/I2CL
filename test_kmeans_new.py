#!/usr/bin/env python3
"""
测试kmeans聚类功能的脚本
"""

import torch
import numpy as np
from wrapper import ModelWrapper

def create_mock_latent_dicts(num_examples=3, num_layers=2, batch_size=1, seq_len=10, hidden_dim=768):
    """
    创建模拟的潜在向量字典用于测试
    """
    all_latent_dicts = []
    
    for example_idx in range(num_examples):
        latent_dict = {}
        for layer in range(num_layers):
            latent_dict[layer] = {
                'attn': torch.randn(batch_size, seq_len, hidden_dim),
                'mlp': torch.randn(batch_size, seq_len, hidden_dim),
                'hidden': torch.randn(batch_size, seq_len, hidden_dim)
            }
        all_latent_dicts.append(latent_dict)
    
    return all_latent_dicts

def test_kmeans_functionality():
    """
    测试kmeans聚类功能
    """
    print("开始测试kmeans聚类功能...")
    
    # 创建模拟数据
    all_latent_dicts = create_mock_latent_dicts(
        num_examples=3, 
        num_layers=2, 
        batch_size=1, 
        seq_len=10, 
        hidden_dim=768
    )
    
    # 创建模拟的ModelWrapper实例
    class MockModelWrapper:
        def _get_kmeans_cluster_result(self, all_latent_dicts, config, ensemble_dict):
            """
            对所有example的token进行kmeans聚类，返回聚类结果
            返回格式: [{layer 0: {module: context_vector}, layer 1: {...},}, 
                       {layer 0: {module:{"cluster": tensor(k,dim), "index": [[1,2,3], ...]}}, ...}]
            """
            from sklearn.cluster import KMeans
            
            # 获取聚类参数
            n_clusters = config.get('kmeans_n_clusters', 3)
            random_state = config.get('kmeans_random_state', 42)
            
            # 第一个字典：正常的context vector（使用现有的ensemble_dict）
            context_vector_dict = {}
            # 第二个字典：聚类结果
            cluster_dict = {}
            
            layers = list(all_latent_dicts[0].keys())
            
            for layer in layers:
                context_vector_dict[layer] = {}
                cluster_dict[layer] = {}
                
                for module in ['attn', 'mlp', 'hidden']:
                    # 收集所有example的所有token的潜在向量
                    all_tokens = []
                    token_indices = []  # 记录每个token来自哪个example
                    
                    for example_idx, latent_dict in enumerate(all_latent_dicts):
                        latent_value = latent_dict[layer][module]  # (b, seq_len, d)
                        batch_size, seq_len, dim = latent_value.shape
                        
                        for b in range(batch_size):
                            for seq in range(seq_len):
                                token_vector = latent_value[b, seq, :].detach().cpu().numpy()
                                all_tokens.append(token_vector)
                                token_indices.append(example_idx)
                    
                    # 转换为numpy数组
                    all_tokens = np.array(all_tokens)  # (total_tokens, dim)
                    
                    # 进行kmeans聚类
                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(all_tokens)
                    cluster_centers = kmeans.cluster_centers_  # (n_clusters, dim)
                    
                    # 使用现有的ensemble_dict中的context vector
                    context_vector = ensemble_dict[module]  # 从现有的ensemble_dict获取
                    context_vector_dict[layer][module] = context_vector[layer].detach().to('cpu')
                    
                    # 组织聚类结果
                    cluster_result = {
                        "cluster": torch.tensor(cluster_centers, dtype=torch.float32),  # (k, dim)
                        "index": []
                    }
                    
                    # 为每个example记录属于每个聚类的token索引
                    for example_idx in range(len(all_latent_dicts)):
                        example_mask = np.array(token_indices) == example_idx
                        example_cluster_labels = cluster_labels[example_mask]
                        
                        # 找到属于每个聚类的token在该example中的索引
                        example_indices = []
                        for cluster_id in range(n_clusters):
                            cluster_mask = example_cluster_labels == cluster_id
                            cluster_token_indices = np.where(cluster_mask)[0].tolist()
                            example_indices.append(cluster_token_indices)
                        
                        cluster_result["index"].append(example_indices)
                    
                    cluster_dict[layer][module] = cluster_result
            
            return [context_vector_dict, cluster_dict]
    
    # 创建配置
    config = {
        'module': ['attn', 'mlp', 'hidden'],
        'kmeans_n_clusters': 3,
        'kmeans_random_state': 42,
        'use_kmeans': True
    }
    
    # 创建模拟的ensemble_dict
    ensemble_dict = {
        'attn': torch.randn(2, 768),  # (layer_num, dim)
        'mlp': torch.randn(2, 768),
        'hidden': torch.randn(2, 768)
    }
    
    # 测试kmeans功能
    mock_wrapper = MockModelWrapper()
    result = mock_wrapper._get_kmeans_cluster_result(all_latent_dicts, config, ensemble_dict)
    
    # 检查结果结构
    context_vector_dict, cluster_dict = result
    
    print(f"结果类型: {type(result)}")
    print(f"结果长度: {len(result)}")
    print(f"Context vector dict keys: {list(context_vector_dict.keys())}")
    print(f"Cluster dict keys: {list(cluster_dict.keys())}")
    
    # 检查具体结构
    for layer in context_vector_dict.keys():
        print(f"\nLayer {layer}:")
        print(f"  Context vector modules: {list(context_vector_dict[layer].keys())}")
        print(f"  Cluster modules: {list(cluster_dict[layer].keys())}")
        
        for module in ['attn', 'mlp', 'hidden']:
            context_vector = context_vector_dict[layer][module]
            cluster_result = cluster_dict[layer][module]
            
            print(f"    {module}:")
            print(f"      Context vector shape: {context_vector.shape}")
            print(f"      Cluster centers shape: {cluster_result['cluster'].shape}")
            print(f"      Number of examples: {len(cluster_result['index'])}")
            print(f"      Example 0 cluster indices: {[len(indices) for indices in cluster_result['index'][0]]}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_kmeans_functionality()



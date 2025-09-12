#!/usr/bin/env python3
"""
測試kmeans聚類功能的腳本
"""

import torch
import numpy as np
from wrapper import ModelWrapper

def create_mock_latent_dicts(num_examples=3, num_layers=2, batch_size=1, seq_len=10, hidden_dim=768):
    """
    創建模擬的潛在向量字典用於測試
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
    測試kmeans聚類功能
    """
    print("開始測試kmeans聚類功能...")
    
    # 創建模擬數據
    all_latent_dicts = create_mock_latent_dicts(
        num_examples=3, 
        num_layers=2, 
        batch_size=1, 
        seq_len=10, 
        hidden_dim=768
    )
    
    # 創建模擬的ModelWrapper實例
    class MockModelWrapper:
        def _get_kmeans_context_vector(self, all_latent_dicts, config):
            """
            對所有example的token進行kmeans聚類，返回聚類結果
            返回格式: [{layer 0: {module: context_vector}, layer 1: {...},}, 
                       {layer 0: {module:{"cluster": tensor(k,dim), "index": [[1,2,3], ...]}}, ...}]
            """
            from sklearn.cluster import KMeans
            
            # 獲取聚類參數
            n_clusters = config.get('kmeans_n_clusters', 3)
            random_state = config.get('kmeans_random_state', 42)
            
            # 第一個字典：正常的context vector
            context_vector_dict = {}
            # 第二個字典：聚類結果
            cluster_dict = {}
            
            layers = list(all_latent_dicts[0].keys())
            
            for layer in layers:
                context_vector_dict[layer] = {}
                cluster_dict[layer] = {}
                
                for module in ['attn', 'mlp', 'hidden']:
                    # 收集所有example的所有token的潛在向量
                    all_tokens = []
                    token_indices = []  # 記錄每個token來自哪個example
                    
                    for example_idx, latent_dict in enumerate(all_latent_dicts):
                        latent_value = latent_dict[layer][module]  # (b, seq_len, d)
                        batch_size, seq_len, dim = latent_value.shape
                        
                        for b in range(batch_size):
                            for seq in range(seq_len):
                                token_vector = latent_value[b, seq, :].detach().cpu().numpy()
                                all_tokens.append(token_vector)
                                token_indices.append(example_idx)
                    
                    # 轉換為numpy數組
                    all_tokens = np.array(all_tokens)  # (total_tokens, dim)
                    
                    # 進行kmeans聚類
                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(all_tokens)
                    cluster_centers = kmeans.cluster_centers_  # (n_clusters, dim)
                    
                    # 為每個example計算context vector（使用聚類中心的加權平均）
                    example_context_vectors = []
                    for example_idx in range(len(all_latent_dicts)):
                        # 找到屬於當前example的所有token
                        example_mask = np.array(token_indices) == example_idx
                        example_cluster_labels = cluster_labels[example_mask]
                        
                        # 計算每個聚類的權重（該example中屬於該聚類的token數量）
                        cluster_weights = np.zeros(n_clusters)
                        for label in example_cluster_labels:
                            cluster_weights[label] += 1
                        
                        # 正規化權重
                        if cluster_weights.sum() > 0:
                            cluster_weights = cluster_weights / cluster_weights.sum()
                        else:
                            cluster_weights = np.ones(n_clusters) / n_clusters
                        
                        # 計算加權平均的context vector
                        context_vector = np.sum(cluster_centers * cluster_weights[:, np.newaxis], axis=0)
                        example_context_vectors.append(context_vector)
                    
                    # 對所有example的context vector進行平均
                    final_context_vector = np.mean(example_context_vectors, axis=0)
                    context_vector_dict[layer][module] = torch.tensor(final_context_vector, dtype=torch.float32)
                    
                    # 組織聚類結果
                    cluster_result = {
                        "cluster": torch.tensor(cluster_centers, dtype=torch.float32),  # (k, dim)
                        "index": []
                    }
                    
                    # 為每個example記錄屬於每個聚類的token索引
                    for example_idx in range(len(all_latent_dicts)):
                        example_mask = np.array(token_indices) == example_idx
                        example_cluster_labels = cluster_labels[example_mask]
                        
                        # 找到屬於每個聚類的token在該example中的索引
                        example_indices = []
                        for cluster_id in range(n_clusters):
                            cluster_mask = example_cluster_labels == cluster_id
                            cluster_token_indices = np.where(cluster_mask)[0].tolist()
                            example_indices.append(cluster_token_indices)
                        
                        cluster_result["index"].append(example_indices)
                    
                    cluster_dict[layer][module] = cluster_result
            
            return [context_vector_dict, cluster_dict]
    
    # 創建配置
    config = {
        'module': ['attn', 'mlp', 'hidden'],
        'kmeans_n_clusters': 3,
        'kmeans_random_state': 42
    }
    
    # 測試kmeans功能
    mock_wrapper = MockModelWrapper()
    result = mock_wrapper._get_kmeans_context_vector(all_latent_dicts, config)
    
    # 檢查結果結構
    context_vector_dict, cluster_dict = result
    
    print(f"結果類型: {type(result)}")
    print(f"結果長度: {len(result)}")
    print(f"Context vector dict keys: {list(context_vector_dict.keys())}")
    print(f"Cluster dict keys: {list(cluster_dict.keys())}")
    
    # 檢查具體結構
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
    
    print("\n測試完成！")

if __name__ == "__main__":
    test_kmeans_functionality()



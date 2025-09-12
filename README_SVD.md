# I2CL with SVD Fusion Method

## 概述

这个版本在原有I2CL (In-Context Learning) 基础上，新增了基于SVD (Singular Value Decomposition) 的向量合并方法。相比简单的平均或PCA方法，SVD能够更好地捕捉向量空间中的主要方向和重要性权重。

## 新增功能

### 1. SVD合并方法

在 `wrapper_svd.py` 中新增了 `_svd_fusion` 方法，支持多种权重计算策略：

- **`singular_value`**: 使用奇异值作为权重
- **`top_k`**: 使用前k个主成分的线性组合  
- **`weighted_combination`**: 加权组合所有主成分
- **`adaptive`**: 自适应权重，根据奇异值大小动态调整

### 2. 配置参数

新增以下配置参数：

```python
config = {
    # 向量合并方法
    'post_fuse_method': 'svd',  # 选择SVD方法
    
    # SVD特定参数
    'svd_rank': 3,  # SVD的秩
    'svd_weight_method': 'singular_value',  # 权重计算方法
}
```

## 使用方法

### 1. 替换原有wrapper

将 `run_i2cl.py` 中的导入语句改为：

```python
# 原来的导入
# import wrapper

# 新的导入
import wrapper_svd as wrapper
```

### 2. 配置SVD参数

在配置文件中设置：

```python
config = {
    'post_fuse_method': 'svd',
    'svd_rank': 3,
    'svd_weight_method': 'singular_value',
}
```

### 3. 运行实验

```bash
python run_i2cl.py --config_path configs/config_svd_example.py
```

## 技术原理

### SVD分解

对于每一层的向量矩阵 $X \in \mathbb{R}^{n \times d}$ (n个示范样本，d维特征)：

$$X = U \Sigma V^T$$

其中：
- $U \in \mathbb{R}^{n \times n}$: 左奇异向量
- $\Sigma \in \mathbb{R}^{n \times d}$: 奇异值矩阵
- $V \in \mathbb{R}^{d \times d}$: 右奇异向量

### 权重计算方法

#### 1. Singular Value Weighting
```python
weights = S / S.sum()  # 归一化奇异值
context_vector = torch.sum(U * weights.unsqueeze(0), dim=1)
context_vector = torch.sum(layer_latents * context_vector.unsqueeze(1), dim=0)
```

#### 2. Top-K Principal Components
```python
context_vector = torch.sum(V * S.unsqueeze(0), dim=1)
```

#### 3. Weighted Combination
```python
normalized_s = S / S.sum()
context_vector = torch.sum(V * normalized_s.unsqueeze(0), dim=1)
```

#### 4. Adaptive Weighting
```python
log_weights = torch.log(S + 1e-8)
normalized_weights = torch.softmax(log_weights, dim=0)
context_vector = torch.sum(V * normalized_weights.unsqueeze(0), dim=1)
```

## 优势分析

### 相比Mean方法
- **信息保留**: 保留主要方向的信息，而不是简单平均
- **噪声抑制**: 通过奇异值筛选，减少噪声影响
- **方向性**: 考虑向量的方向性，而不仅仅是大小

### 相比PCA方法
- **灵活性**: 支持多种权重计算策略
- **可控性**: 通过rank参数控制信息保留程度
- **解释性**: 奇异值直接反映各方向的重要性

## 参数调优建议

### svd_rank 选择
- **小值 (1-3)**: 当示范样本数量少或差异小时
- **中等值 (3-10)**: 平衡信息保留和噪声抑制
- **大值 (>10)**: 当需要保留更多细节信息时

### svd_weight_method 选择
- **`singular_value`**: 推荐用于大多数情况
- **`adaptive`**: 当奇异值分布不均匀时
- **`weighted_combination`**: 当需要保持所有信息时
- **`top_k`**: 当只需要主要方向时

## 实验对比

建议进行以下对比实验：

1. **Baseline**: 原始mean方法
2. **PCA**: 原有PCA方法  
3. **SVD-singular_value**: SVD奇异值加权
4. **SVD-adaptive**: SVD自适应加权
5. **SVD-weighted_combination**: SVD加权组合

## 注意事项

1. **计算复杂度**: SVD比mean方法计算开销更大
2. **内存使用**: 需要额外的内存存储SVD分解结果
3. **数值稳定性**: 奇异值过小时可能影响数值稳定性
4. **超参数调优**: 需要根据具体任务调整svd_rank和svd_weight_method

## 扩展功能

未来可以考虑添加：

1. **动态rank选择**: 根据数据自动选择最优rank
2. **多尺度SVD**: 在不同尺度上进行SVD分解
3. **稀疏SVD**: 使用稀疏SVD提高计算效率
4. **在线SVD**: 支持增量式SVD更新

## 引用

如果使用此方法，请引用相关论文：

```
@article{i2cl_svd,
  title={I2CL with SVD Fusion for Improved Few-Shot Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

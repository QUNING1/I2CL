#!/bin/bash

# KMeans 变体实验脚本
# 测试不同的聚类参数和注入方法

cd /data2/qn/I2CL
mkdir -p logs

echo "开始 KMeans 变体实验..."
echo "开始时间: $(date)"

# 实验1: 基础 KMeans 配置
echo "实验1: 基础 KMeans (5个聚类, clustering注入)"
python run_i2cl.py \
    --config_path configs/config_i2cl.py \
    --datasets trec \
    --shot_per_class 5 \
    --kmeans_n_clusters 5 \
    --inject_method clustering \
    --post_fuse_method kmeans \
    > logs/kmeans_basic_$(date +%Y%m%d_%H%M%S).log 2>&1

# 实验2: 不同聚类数量
echo "实验2: 3个聚类"
python run_i2cl.py \
    --config_path configs/config_i2cl.py \
    --datasets trec \
    --shot_per_class 5 \
    --kmeans_n_clusters 3 \
    --inject_method clustering \
    --post_fuse_method kmeans \
    > logs/kmeans_3clusters_$(date +%Y%m%d_%H%M%S).log 2>&1

# 实验3: 更多聚类
echo "实验3: 8个聚类"
python run_i2cl.py \
    --config_path configs/config_i2cl.py \
    --datasets trec \
    --shot_per_class 5 \
    --kmeans_n_clusters 8 \
    --inject_method clustering \
    --post_fuse_method kmeans \
    > logs/kmeans_8clusters_$(date +%Y%m%d_%H%M%S).log 2>&1

# 实验4: 不同shot数量
echo "实验4: 3 shot per class"
python run_i2cl.py \
    --config_path configs/config_i2cl.py \
    --datasets trec \
    --shot_per_class 3 \
    --kmeans_n_clusters 5 \
    --inject_method clustering \
    --post_fuse_method kmeans \
    > logs/kmeans_3shot_$(date +%Y%m%d_%H%M%S).log 2>&1

# 实验5: 对比传统方法
echo "实验5: 传统mean方法对比"
python run_i2cl.py \
    --config_path configs/config_i2cl.py \
    --datasets trec \
    --shot_per_class 5 \
    --inject_method static_add \
    --post_fuse_method mean \
    > logs/traditional_mean_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "所有实验完成！"
echo "结束时间: $(date)"
echo "日志文件保存在 logs/ 目录"


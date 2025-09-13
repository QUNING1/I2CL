#!/bin/bash

# I2CL 批量实验脚本
# 使用方法: bash run_experiments.sh

# 设置基础路径
BASE_DIR="/data2/qn/I2CL"
cd $BASE_DIR

# 创建日志目录
mkdir -p logs
mkdir -p results
 
# 定义实验参数
DATASETS=( "dbpedia" "mr" "subj" "sst5"  "hate_speech18" "trec" "sst2"  "agnews")
SHOT_PER_CLASS=( 5 )
KMEANS_CLUSTERS=( 2 3 4 5 6 7 8 9 10 )
INJECT_METHODS=("static_add")
POST_FUSE_METHODS=("kmeans" )
TOK_POS=( "label" )

# 计数器
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=0

# 计算总实验数
for dataset in "${DATASETS[@]}"; do
    for shot in "${SHOT_PER_CLASS[@]}"; do
        for clusters in "${KMEANS_CLUSTERS[@]}"; do
            for inject in "${INJECT_METHODS[@]}"; do
                for fuse in "${POST_FUSE_METHODS[@]}"; do
                    for tok_pos in "${TOK_POS[@]}"; do
                        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
                    done
                done
            done
        done
    done
done

echo "总共需要运行 $TOTAL_EXPERIMENTS 个实验"
echo "开始时间: $(date)"
echo "=========================================="

# 开始实验循环
for dataset in "${DATASETS[@]}"; do
    for shot in "${SHOT_PER_CLASS[@]}"; do
        for clusters in "${KMEANS_CLUSTERS[@]}"; do
            for inject in "${INJECT_METHODS[@]}"; do
                for fuse in "${POST_FUSE_METHODS[@]}"; do
                    for tok_pos in "${TOK_POS[@]}"; do
                        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                        
                        # 生成实验名称
                        EXP_NAME="exp_${dataset}_shot${shot}_cls${clusters}_${inject}_${fuse}_${tok_pos}"
                        
                        echo "[$EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS] 开始实验: $EXP_NAME"
                        echo "参数: dataset=$dataset, shot_per_class=$shot, kmeans_clusters=$clusters, inject_method=$inject, post_fuse_method=$fuse, tok_pos=$tok_pos"
                        
                        # 运行实验
                        python run_i2cl.py \
                            --config_path configs/config_i2cl.py \
                            --datasets $dataset \
                            --shot_per_class $shot \
                            --kmeans_n_clusters $clusters \
                            --inject_method $inject \
                            --post_fuse_method $fuse \
                            --tok_pos $tok_pos \
                            --query_pool_method mean \
                            > logs/${EXP_NAME}.log 2>&1
                        
                        # 检查实验是否成功
                        if [ $? -eq 0 ]; then
                            echo "✅ 实验 $EXP_NAME 完成"
                        else
                            echo "❌ 实验 $EXP_NAME 失败，请检查日志: logs/${EXP_NAME}.log"
                        fi
                        
                        echo "----------------------------------------"
                        
                        # 可选：添加延迟避免GPU过载
                        sleep 10
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "所有实验完成！"
echo "结束时间: $(date)"
echo "日志文件保存在 logs/ 目录"
echo "结果文件保存在 results/ 目录"

# 为每个数据集生成聚类数vs准确率图表
echo "开始生成聚类数vs准确率图表..."
python generate_cluster_plots.py

echo "图表生成完成！"

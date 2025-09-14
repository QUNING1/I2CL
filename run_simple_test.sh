#!/bin/bash

# 简单测试脚本
echo "开始简单测试..."

# 设置基础路径
BASE_DIR="/data2/qn/I2CL"
cd $BASE_DIR

# 创建日志目录
mkdir -p logs

# 测试单个实验
echo "运行单个实验测试..."
python test_simple.py > logs/simple_test.log 2>&1

# 检查结果
if [ $? -eq 0 ]; then
    echo "✅ 简单测试成功"
    echo "日志文件: logs/simple_test.log"
else
    echo "❌ 简单测试失败"
    echo "请检查日志: logs/simple_test.log"
fi

# 显示日志内容
echo ""
echo "日志内容:"
echo "----------------------------------------"
cat logs/simple_test.log
echo "----------------------------------------"

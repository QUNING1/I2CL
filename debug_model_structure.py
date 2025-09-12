#!/usr/bin/env python3
"""
调试Llama模型的实际结构
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_model_structure():
    """调试模型结构"""
    print("=== 调试Llama模型结构 ===")
    
    # 尝试加载一个小的Llama模型（如果可用）
    try:
        # 这里使用一个示例模型名，您需要根据实际情况修改
        model_name = "meta-llama/Llama-2-7b-hf"  # 或者您本地的模型路径
        
        print(f"尝试加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"\n模型类型: {type(model)}")
        print(f"模型名称: {model.__class__.__name__}")
        
        print("\n=== 直接属性 ===")
        for attr in dir(model):
            if not attr.startswith('_'):
                try:
                    value = getattr(model, attr)
                    if hasattr(value, '__len__'):
                        print(f"{attr}: {type(value)} (长度: {len(value)})")
                    else:
                        print(f"{attr}: {type(value)}")
                except:
                    print(f"{attr}: 无法访问")
        
        print("\n=== 检查常见结构 ===")
        
        # 检查 model.layers
        if hasattr(model, 'layers'):
            print(f"✓ model.layers 存在，长度: {len(model.layers)}")
        else:
            print("✗ model.layers 不存在")
        
        # 检查 model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            print(f"✓ model.model.layers 存在，长度: {len(model.model.layers)}")
        else:
            print("✗ model.model.layers 不存在")
        
        # 检查 model.transformer.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            print(f"✓ model.transformer.layers 存在，长度: {len(model.transformer.layers)}")
        else:
            print("✗ model.transformer.layers 不存在")
        
        # 检查 model.encoder.layers
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            print(f"✓ model.encoder.layers 存在，长度: {len(model.encoder.layers)}")
        else:
            print("✗ model.encoder.layers 不存在")
        
        print("\n=== 递归检查 model 属性 ===")
        if hasattr(model, 'model'):
            submodel = model.model
            print(f"model.model 类型: {type(submodel)}")
            for attr in dir(submodel):
                if not attr.startswith('_') and 'layer' in attr.lower():
                    try:
                        value = getattr(submodel, attr)
                        if hasattr(value, '__len__'):
                            print(f"  model.{attr}: {type(value)} (长度: {len(value)})")
                        else:
                            print(f"  model.{attr}: {type(value)}")
                    except:
                        print(f"  model.{attr}: 无法访问")
        
        print("\n=== 检查第一层结构 ===")
        # 尝试找到第一层的结构
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            first_layer = model.model.layers[0]
            print(f"第一层类型: {type(first_layer)}")
            print(f"第一层属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            first_layer = model.transformer.layers[0]
            print(f"第一层类型: {type(first_layer)}")
            print(f"第一层属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("\n=== 使用模拟模型进行测试 ===")
        
        # 创建一个模拟的Llama模型结构
        class MockLlamaModel:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'layers': [type('obj', (object,), {
                        'self_attn': type('obj', (object,), {}),
                        'mlp': type('obj', (object,), {})
                    }) for _ in range(32)]
                })
        
        mock_model = MockLlamaModel()
        print(f"模拟模型类型: {type(mock_model)}")
        print(f"模拟模型结构: model.model.layers (长度: {len(mock_model.model.layers)})")

if __name__ == "__main__":
    debug_model_structure()

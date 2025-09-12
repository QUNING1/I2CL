#!/usr/bin/env python3
"""
测试Llama模型结构检测
"""

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import wrapper_svd as wrapper
    print("✓ 成功导入 wrapper_svd")
except ImportError as e:
    print(f"✗ 导入 wrapper_svd 失败: {e}")
    sys.exit(1)

def test_llama_structure():
    """测试Llama模型结构检测"""
    print("\n=== 测试Llama模型结构检测 ===")
    
    # 创建模拟的Llama模型结构
    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟Llama的结构: model.model.layers
            self.model = type('obj', (object,), {
                'layers': nn.ModuleList([
                    type('obj', (object,), {
                        'self_attn': type('obj', (object,), {}),
                        'mlp': type('obj', (object,), {})
                    }) for _ in range(32)
                ])
            })
        
        def forward(self, x):
            return x
    
    try:
        model = MockLlamaModel()
        tokenizer = None
        config = {}
        device = torch.device('cpu')
        
        print("创建模拟Llama模型...")
        wrapper_instance = wrapper.ModelWrapper(model, tokenizer, config, device)
        
        print(f"✓ 成功检测到层数: {wrapper_instance.num_layers}")
        
        # 测试属性路径生成
        print("\n测试属性路径生成...")
        attn_path = wrapper_instance._get_arribute_path(0, 'attn')
        mlp_path = wrapper_instance._get_arribute_path(0, 'mlp')
        hidden_path = wrapper_instance._get_arribute_path(0, 'hidden')
        
        print(f"✓ attn路径: {attn_path}")
        print(f"✓ mlp路径: {mlp_path}")
        print(f"✓ hidden路径: {hidden_path}")
        
        # 测试嵌套属性获取
        print("\n测试嵌套属性获取...")
        try:
            attn_module = wrapper_instance._get_nested_attr(attn_path)
            print(f"✓ 成功获取attn模块: {type(attn_module)}")
        except Exception as e:
            print(f"✗ 获取attn模块失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpt_structure():
    """测试GPT模型结构检测"""
    print("\n=== 测试GPT模型结构检测 ===")
    
    # 创建模拟的GPT模型结构
    class MockGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟GPT的结构: transformer.h
            self.transformer = type('obj', (object,), {
                'h': nn.ModuleList([
                    type('obj', (object,), {
                        'attn': type('obj', (object,), {}),
                        'mlp': type('obj', (object,), {})
                    }) for _ in range(12)
                ])
            })
        
        def forward(self, x):
            return x
    
    try:
        model = MockGPTModel()
        tokenizer = None
        config = {}
        device = torch.device('cpu')
        
        print("创建模拟GPT模型...")
        wrapper_instance = wrapper.ModelWrapper(model, tokenizer, config, device)
        
        print(f"✓ 成功检测到层数: {wrapper_instance.num_layers}")
        
        # 测试属性路径生成
        print("\n测试属性路径生成...")
        attn_path = wrapper_instance._get_arribute_path(0, 'attn')
        mlp_path = wrapper_instance._get_arribute_path(0, 'mlp')
        hidden_path = wrapper_instance._get_arribute_path(0, 'hidden')
        
        print(f"✓ attn路径: {attn_path}")
        print(f"✓ mlp路径: {mlp_path}")
        print(f"✓ hidden路径: {hidden_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试模型结构检测...")
    
    tests = [
        test_llama_structure,
        test_gpt_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！模型结构检测工作正常。")
    else:
        print("❌ 部分测试失败，请检查代码。")

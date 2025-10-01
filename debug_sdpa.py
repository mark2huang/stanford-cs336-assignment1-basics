#!/usr/bin/env python3
"""
调试缩放点积注意力(SDPA)函数的脚本
"""

import torch
import math


def debug_scaled_dot_product_attention():
    """调试SDPA函数"""

    # 创建测试数据
    batch_size = 2
    seq_len = 3
    d_k = 4
    d_v = 5

    print("=== 测试数据 ===")
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"d_k = {d_k}")

    print("\n=== 你的实现 ===")

    # 你的实现（有错误）
    try:
        attentionMatrix = Q @ K.T
        print(f"1. Q @ K.T shape: {attentionMatrix.shape}")

        # 这里会出错
        attentionMatrix = attentionMatrix / torch.sqrt(torch.Tensor.size(K))
        print(f"2. 缩放后 shape: {attentionMatrix.shape}")

        attentionMatrix = torch.softmax(attentionMatrix, dim=-1)
        print(f"3. Softmax后 shape: {attentionMatrix.shape}")

        output = attentionMatrix * V
        print(f"4. 最终输出 shape: {output.shape}")

    except Exception as e:
        print(f"❌ 你的实现出错: {e}")

    print("\n=== 正确实现 ===")

    # 正确实现
    try:
        # 1. 计算注意力分数
        scores = Q @ K.transpose(-2, -1)
        print(f"1. Q @ K^T shape: {scores.shape}")

        # 2. 缩放
        scores = scores / math.sqrt(d_k)
        print(f"2. 缩放后 shape: {scores.shape}")

        # 3. Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        print(f"3. Softmax后 shape: {attention_weights.shape}")

        # 4. 加权求和
        output = attention_weights @ V
        print(f"4. 最终输出 shape: {output.shape}")

        print("✅ 正确实现成功！")

        # 验证形状
        assert output.shape == (batch_size, seq_len,
                                d_v), f"输出形状错误: {output.shape}"
        print("✅ 输出形状正确")

    except Exception as e:
        print(f"❌ 正确实现出错: {e}")

    print("\n=== 详细对比 ===")

    # 对比两种转置方式
    print("转置对比:")
    print(f"K.T shape: {K.T.shape}")
    print(f"K.transpose(-2, -1) shape: {K.transpose(-2, -1).shape}")

    print("\n维度获取对比:")
    print(f"K.size(-1): {K.size(-1)}")
    print(f"d_k: {d_k}")

    print("\n矩阵乘法对比:")
    print(f"attention_weights @ V shape: {(attention_weights @ V).shape}")
    print(f"attention_weights * V shape: {(attention_weights * V).shape}")


if __name__ == "__main__":
    debug_scaled_dot_product_attention()

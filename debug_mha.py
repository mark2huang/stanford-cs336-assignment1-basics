#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from tests.adapters import run_multihead_self_attention


def debug_multihead_attention():
    # 设置与测试相同的参数
    d_model = 64
    num_heads = 4
    batch_size = 4
    seq_len = 12

    # 生成随机权重（与测试类似）
    torch.manual_seed(42)
    q_proj_weight = torch.randn(d_model, d_model)
    k_proj_weight = torch.randn(d_model, d_model)
    v_proj_weight = torch.randn(d_model, d_model)
    o_proj_weight = torch.randn(d_model, d_model)

    # 生成输入特征
    in_features = torch.randn(batch_size, seq_len, d_model)

    print("=== 调试多头自注意力 ===")
    print(f"输入形状: {in_features.shape}")
    print(
        f"权重形状: Q={q_proj_weight.shape}, K={k_proj_weight.shape}, V={v_proj_weight.shape}, O={o_proj_weight.shape}")

    # 运行我们的实现
    output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_features
    )

    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"输出均值: {output.mean():.4f}, 标准差: {output.std():.4f}")

    # 验证基本属性
    assert output.shape == (batch_size, seq_len,
                            d_model), f"输出形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert not torch.isinf(output).any(), "输出包含无穷大值"

    print("✓ 基本验证通过")

    return output


if __name__ == "__main__":
    debug_multihead_attention()

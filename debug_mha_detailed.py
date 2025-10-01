#!/usr/bin/env python3
import torch
import json

# 加载权重和配置
FIXTURES_PATH = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main/tests/fixtures"
state_dict = torch.load(
    FIXTURES_PATH + "/ts_tests/model.pt", map_location="cpu")
config = json.load(open(FIXTURES_PATH + "/ts_tests/model_config.json"))
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# 获取权重
q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]

# 生成测试输入
torch.manual_seed(4)
batch_size = 4
n_queries = 12
d_model = config['d_model']
num_heads = config['num_heads']
in_embeddings = torch.randn(batch_size, n_queries, d_model)

print("=== 参数设置 ===")
print(f"batch_size: {batch_size}")
print(f"n_queries: {n_queries}")
print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}")
print(f"d_k_per_head: {d_model // num_heads}")
print(f"d_v_per_head: {d_model // num_heads}")

# 重新实现run_multihead_self_attention函数进行调试


def debug_multihead_self_attention(in_features, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, d_model, num_heads):
    print("\n=== 步骤1: 投影 ===")
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    batch_size, seq_len, _ = in_features.shape
    d_k_per_head = d_model // num_heads
    d_v_per_head = d_model // num_heads

    print("\n=== 步骤2: 多头重组 ===")
    # 标准的多头重组方式
    Q_multi = Q.view(batch_size, seq_len, num_heads,
                     d_k_per_head).transpose(1, 2)
    K_multi = K.view(batch_size, seq_len, num_heads,
                     d_k_per_head).transpose(1, 2)
    V_multi = V.view(batch_size, seq_len, num_heads,
                     d_v_per_head).transpose(1, 2)

    print(f"Q_multi shape: {Q_multi.shape}")
    print(f"K_multi shape: {K_multi.shape}")
    print(f"V_multi shape: {V_multi.shape}")

    print("\n=== 步骤3: 注意力计算 ===")
    attention_scores = Q_multi @ K_multi.transpose(-2, -1)
    print(f"attention_scores shape: {attention_scores.shape}")

    print("\n=== 步骤4: 缩放 ===")
    attention_scores = attention_scores / \
        torch.sqrt(torch.tensor(d_k_per_head, dtype=attention_scores.dtype))
    print(f"scaled attention_scores shape: {attention_scores.shape}")

    print("\n=== 步骤5: Softmax ===")
    attention_weights = torch.softmax(attention_scores, dim=-1)
    print(f"attention_weights shape: {attention_weights.shape}")

    print("\n=== 步骤6: 加权求和 ===")
    tmp_output = attention_weights @ V_multi
    print(f"tmp_output shape: {tmp_output.shape}")

    print("\n=== 步骤7: 合并多头 ===")
    output = tmp_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, d_model)
    print(f"merged output shape: {output.shape}")

    print("\n=== 步骤8: 输出投影 ===")
    final_output = output @ o_proj_weight.T
    print(f"final_output shape: {final_output.shape}")

    return final_output


# 运行调试
result = debug_multihead_self_attention(
    in_embeddings, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, d_model, num_heads)

print(f"\n=== 最终结果统计 ===")
print(f"结果范围: [{result.min():.4f}, {result.max():.4f}]")
print(f"结果均值: {result.mean():.4f}")
print(f"结果标准差: {result.std():.4f}")

# 检查是否有NaN或无穷大值
print(f"NaN值数量: {torch.isnan(result).sum().item()}")
print(f"无穷大值数量: {torch.isinf(result).sum().item()}")

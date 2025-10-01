#!/usr/bin/env python3
import torch
import json
import numpy as np

# 加载权重和配置
FIXTURES_PATH = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main/tests/fixtures"
state_dict = torch.load(
    FIXTURES_PATH + "/ts_tests/model.pt", map_location="cpu")
config = json.load(open(FIXTURES_PATH + "/ts_tests/model_config.json"))
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

print("=== 配置信息 ===")
print(f"Config keys: {list(config.keys())}")
print(f"Config: {config}")

print("\n=== 权重形状分析 ===")
print(f"d_model: {config['d_model']}")
print(f"num_heads: {config['num_heads']}")
print(f"每个头的维度: {config['d_model'] // config['num_heads']}")

# 检查注意力权重
q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]

print(f"\n投影权重形状:")
print(f"q_proj_weight: {q_proj_weight.shape}")  # 应该是 [d_model, d_model]
print(f"k_proj_weight: {k_proj_weight.shape}")
print(f"v_proj_weight: {v_proj_weight.shape}")
print(f"o_proj_weight: {o_proj_weight.shape}")

print(f"\n权重范围:")
print(f"q_proj_weight: [{q_proj_weight.min():.4f}, {q_proj_weight.max():.4f}]")
print(f"k_proj_weight: [{k_proj_weight.min():.4f}, {k_proj_weight.max():.4f}]")
print(f"v_proj_weight: [{v_proj_weight.min():.4f}, {v_proj_weight.max():.4f}]")
print(f"o_proj_weight: [{o_proj_weight.min():.4f}, {o_proj_weight.max():.4f}]")

# 生成测试输入
torch.manual_seed(4)
batch_size = 4
n_queries = 12
d_model = config['d_model']
in_embeddings = torch.randn(batch_size, n_queries, d_model)

print(f"\n=== 输入数据 ===")
print(f"in_embeddings shape: {in_embeddings.shape}")
print(
    f"in_embeddings range: [{in_embeddings.min():.4f}, {in_embeddings.max():.4f}]")

# 测试投影
Q = in_embeddings @ q_proj_weight.T
print(f"\n=== 投影结果 ===")
print(f"Q shape: {Q.shape}")
print(f"Q range: [{Q.min():.4f}, {Q.max():.4f}]")

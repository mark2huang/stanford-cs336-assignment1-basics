#!/usr/bin/env python3
from tests.adapters import run_multihead_self_attention
import numpy as np
import torch
import json

# 加载快照数据
snapshot_path = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main/tests/_snapshots/test_multihead_self_attention.npz"

# 正确加载快照数据
expected_arrays = dict(np.load(snapshot_path))

print("=== 快照数据分析 ===")
print(f"快照数据键: {list(expected_arrays.keys())}")

for key, value in expected_arrays.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print(f" 范围: [{value.min():.4f}, {value.max():.4f}]")
    print(f" 均值: {value.mean():.4f}, 标准差: {value.std():.4f}")
    print(f" 前5个元素: {value.flatten()[:5]}")

# 重新运行你的实现来比较结果
FIXTURES_PATH = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main/tests/fixtures"
state_dict = torch.load(
    FIXTURES_PATH + "/ts_tests/model.pt", map_location="cpu")
config = json.load(open(FIXTURES_PATH + "/ts_tests/model_config.json"))
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]

torch.manual_seed(4)
batch_size = 4
n_queries = 12
d_model = config['d_model']
num_heads = config['num_heads']
in_embeddings = torch.randn(batch_size, n_queries, d_model)

# 运行你的实现
actual_output = run_multihead_self_attention(
    d_model=d_model,
    num_heads=num_heads,
    q_proj_weight=q_proj_weight,
    k_proj_weight=k_proj_weight,
    v_proj_weight=v_proj_weight,
    o_proj_weight=o_proj_weight,
    in_features=in_embeddings,
)

print(f"\n=== 实际输出统计 ===")
actual_np = actual_output.detach().numpy()
print(f"实际输出形状: {actual_np.shape}")
print(f"实际输出范围: [{actual_np.min():.4f}, {actual_np.max():.4f}]")
print(f"实际输出均值: {actual_np.mean():.4f}")
print(f"实际输出标准差: {actual_np.std():.4f}")

# 比较差异
if 'array' in expected_arrays:
    expected = expected_arrays['array']
    diff = np.abs(actual_np - expected)
    print(f"\n=== 差异分析 ===")
    print(f"最大绝对差异: {diff.max():.6f}")
    print(f"平均绝对差异: {diff.mean():.6f}")
    print(f"差异超过1e-6的元素比例: {(diff > 1e-6).sum() / diff.size:.2%}")

    # 显示差异最大的几个位置
    max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"最大差异位置: {max_diff_indices}")
    print(f"期望值: {expected[max_diff_indices]:.6f}")
    print(f"实际值: {actual_np[max_diff_indices]:.6f}")

    # 检查差异分布
    print(f"\n差异分布:")
    for threshold in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        count = (diff > threshold).sum()
        percentage = count / diff.size * 100
        print(f"  > {threshold:.0e}: {count} 元素 ({percentage:.2f}%)")

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    # 使用torch.matmul执行线性变换: y = xW^T
    # 其中in_features的形状是(..., d_in)，weights.T的形状是(d_in, d_out)
    return torch.matmul(in_features, weights.T)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    # step1:将输入in_features投影到高维度,作为上采样
    value = in_features@w1_weight.T  # 形状: [..., d_ff]
    # step2:将输入in_features投影到高维度，作为门控信号
    gate = in_features@w3_weight.T  # 形状: [..., d_ff]
    # step3:对value应用swish激活函数
    swish_value = value * torch.sigmoid(value)  # 形状: [..., d_ff]
    # step4:门控激活 (Swish(xW₁) ⊙ (xW₃))
    GLU = swish_value * gate  # 形状: [..., d_ff]
    # step5:下采样回原始维度
    Output = GLU@w2_weight.T  # 形状: [..., d_model]
    return Output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    print("\n")
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    # step1: 计算注意力分数矩阵
    attentionMatrix = torch.einsum("... q d,...k d->...q k", Q, K)
    print(f"scores shape: {attentionMatrix.shape}")

    # step2: 缩放
    d_k = Q.shape[-1]
    attentionMatrix = attentionMatrix / \
        (torch.sqrt(torch.tensor(d_k, dtype=attentionMatrix.dtype)))
    print(f"缩放后 shape: {attentionMatrix.shape}")

    # step3: 应用掩码（如果有）
    if mask is not None:
        attentionMatrix = attentionMatrix.masked_fill(~mask, -1e9)
    print(f"掩码后 shape: {attentionMatrix.shape}")

    # step4: Softmax归一化
    attentionMatrix = torch.softmax(attentionMatrix, dim=-1)
    print(f"Softmax后 shape: {attentionMatrix.shape}")

    # step5: 加权求和
    output = torch.einsum("... q k,...k d->... q d", attentionMatrix, V)
    print(f"最终输出 shape: {output.shape}")

    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    # 打印形状信息用于调试
    print(f"\nd_model: {d_model}, num_heads: {num_heads}")
    print(f"q_proj_weight: {q_proj_weight.shape}")
    print(f"k_proj_weight: {k_proj_weight.shape}")
    print(f"v_proj_weight: {v_proj_weight.shape}")
    print(f"o_proj_weight:{o_proj_weight.shape}")
    print(f"in_features:{in_features.shape}")

    """
    tests/test_model.py::test_multihead_self_attention 
    d_model: 64, num_heads: 4
    q_proj_weight: torch.Size([64, 64])
    k_proj_weight: torch.Size([64, 64])
    v_proj_weight: torch.Size([64, 64])
    o_proj_weight:torch.Size([64, 64])
    in_features:torch.Size([4, 12, 64])
    q_proj_weight: torch.Size([64, 64])

    第一个64：d_k - 每个头的查询/键维度
    第二个64：d_in - 输入特征维度
    关键理解：虽然 q_proj_weight 形状是 [64, 64]，但这实际上是所有头的投影权重合并在一起。因为 num_heads = 4，所以：
    每个头的实际维度：d_k_per_head = 64 / 4 = 16
    总投影权重大小：d_k = 16 × 4 = 64

    2. 输入特征形状：[4, 12, 64]
    in_features: torch.Size([4, 12, 64])

    第一个4：batch_size - 批次大小（4个样本）
    第二个12：sequence_length - 序列长度（12个token）
    第三个64：d_in - 输入特征维度（与投影权重的第二个维度匹配）
    """

    # Step 1: 计算Q、K、V投影（所有头一起处理）
    # [..., seq_len, d_model]
    Q = in_features @ q_proj_weight.T
    print(f"Q.shape={Q.shape}")  # [batch=4, seq_len=12, d_k=64]

    # [..., seq_len, d_k * num_heads]
    K = in_features @ k_proj_weight.T
    print(f"K.shape={K.shape}")

    # [..., seq_len, d_v * num_heads]
    V = in_features @ v_proj_weight.T
    print(f"V.shape={V.shape}")

    # 获取输入张量的形状信息
    batch_size, seq_len, _ = in_features.shape

    # 计算每个头的维度
    d_k = d_model  # 所有头的总查询/键维度
    d_v = d_model  # 所有头的总值维度
    d_k_per_head = d_k // num_heads  # 每个头的查询/键维度
    d_v_per_head = d_v // num_heads  # 每个头的值维度

    print(f"d_k (total): {d_k}, d_v (total): {d_v}")
    print(f"d_k_per_head: {d_k_per_head}, d_v_per_head: {d_v_per_head}")

    # step2:将投影后的张量重新组织为多头格式
    # 当前形状：[batch_size, seq_len, d_k] 目标形状：[batch_size, num_heads, seq_len, d_k_per_head]
    # 注意：标准实现中，多头重组应该按照 [batch_size, seq_len, num_heads, d_k_per_head] 然后 transpose(1, 2)
    Q = Q.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_v_per_head).transpose(1, 2)
    # [batch_size, num_heads, seq_len, d_k_per_head/d_v_per_head]
    print(f"Q.shape={Q.shape}")
    print(f"K.shape={K.shape}")
    print(f"V.shape={V.shape}")

    # step3:计算注意力矩阵
    # Q: [batch_size, num_heads, seq_len, d_k_per_head] × K^T: [batch_size, num_heads, d_k_per_head, seq_len]
    # = [batch_size, num_heads, seq_len, seq_len]
    attention_scores = Q @ K.transpose(-2, -1)
    print(f"attention_scores.shape={attention_scores.shape}")

    # step4:缩放
    attention_scores = attention_scores / \
        torch.sqrt(torch.tensor(d_k_per_head, dtype=attention_scores.dtype))
    print(f"缩放之后attention_scores.shape={attention_scores.shape}")

    # step5:softmax
    attention_weights = torch.softmax(attention_scores, dim=-1)
    print(f"softmax之后attention_weights.shape={attention_weights.shape}")

    # step6:乘以V   [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, d_v_per_head]
    # = [batch_size, num_heads, seq_len, d_v_per_head]
    tmp_output = attention_weights @ V
    print(f"tmp_output.shape={tmp_output.shape}")

    # step7: 合并多头输出
    # 当前形状: [batch_size, num_heads, seq_len, d_v_per_head] → 目标形状: [batch_size, seq_len, d_v]
    output = tmp_output.transpose(
        1, 2).contiguous().view(batch_size, seq_len, d_v)
    print(f"合并多头输出之后output.shape={output.shape}")

    # step8: 投影到输出维度
    output = output @ o_proj_weight.T
    print(f"投影到输出维度之后output.shape={output.shape}")
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # 获取输入张量的形状
    batch_size, seq_len, d_in = in_features.shape

    # 计算每个头的维度
    d_k = q_proj_weight.shape[0]  # 查询/键的总维度
    d_v = v_proj_weight.shape[0]  # 值的总维度
    d_k_per_head = d_k // num_heads  # 每个头的查询/键维度
    d_v_per_head = d_v // num_heads  # 每个头的值维度

    # 步骤1: 投影计算（单矩阵乘法）
    Q = in_features @ q_proj_weight.T  # [batch_size, seq_len, d_k]
    K = in_features @ k_proj_weight.T  # [batch_size, seq_len, d_k]
    V = in_features @ v_proj_weight.T  # [batch_size, seq_len, d_v]

    # 步骤2: 将投影后的张量重新组织为多头格式
    # 当前形状：[batch_size, seq_len, d_k] 目标形状：[batch_size, num_heads, seq_len, d_k_per_head]
    Q = Q.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2)
    # [batch_size, num_heads, seq_len, d_k_per_head]

    # V张量单独处理，因为不需要应用RoPE
    V = V.view(batch_size, seq_len, num_heads, d_v_per_head).transpose(1, 2)
    # [batch_size, num_heads, seq_len, d_v_per_head]

    # 步骤3: 应用RoPE到Q和K（如果提供了位置信息）
    if token_positions is not None:
        # 对每个头的Q和K应用RoPE，RoPE维度是每个头的维度
        # 需要将多头张量重塑为 [batch_size * num_heads, seq_len, d_k_per_head]
        Q_reshaped = Q.transpose(1, 2).contiguous().view(
            batch_size * num_heads, seq_len, d_k_per_head)
        K_reshaped = K.transpose(1, 2).contiguous().view(
            batch_size * num_heads, seq_len, d_k_per_head)

        # 扩展位置信息以匹配多头
        # token_positions 的形状可能是 [1, seq_len] 或 [batch_size, seq_len]
        if token_positions.shape[0] == 1:
            # 如果是 [1, seq_len]，扩展到 [batch_size * num_heads, seq_len]
            pos_expanded = token_positions.repeat(batch_size * num_heads, 1)
        else:
            # 如果是 [batch_size, seq_len]，扩展到 [batch_size * num_heads, seq_len]
            pos_expanded = token_positions.unsqueeze(1).repeat(
                1, num_heads, 1).view(batch_size * num_heads, seq_len)

        # 应用RoPE
        Q_rope = run_rope(d_k_per_head, theta, max_seq_len,
                          Q_reshaped, pos_expanded)
        K_rope = run_rope(d_k_per_head, theta, max_seq_len,
                          K_reshaped, pos_expanded)

        # 重塑回多头格式
        Q = Q_rope.view(batch_size, num_heads, seq_len, d_k_per_head)
        K = K_rope.view(batch_size, num_heads, seq_len, d_k_per_head)

    # V已经在步骤2中正确重塑，不需要重复操作
    # [batch_size, num_heads, seq_len, d_v_per_head]

    # 步骤4: 计算注意力分数 QK^T
    # [batch_size, num_heads, seq_len, seq_len]
    attention_scores = Q @ K.transpose(-2, -1)

    # 步骤5: 缩放注意力分数
    # [batch_size, num_heads, seq_len, seq_len]
    attention_scores = attention_scores / (d_k_per_head ** 0.5)

    # 步骤6: 应用Softmax得到注意力权重
    # [batch_size, num_heads, seq_len, seq_len]
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # 步骤7: 加权求和得到输出
    # [batch_size, num_heads, seq_len, d_v_per_head]
    tmp_output = attention_weights @ V

    # 步骤8: 合并多头输出
    # 当前形状：[batch_size, num_heads, seq_len, d_v_per_head] 目标形状：[batch_size, seq_len, d_v]
    output = tmp_output.transpose(
        1, 2).contiguous().view(batch_size, seq_len, d_v)

    # 步骤9: 投影到输出维度
    output = output @ o_proj_weight.T  # [batch_size, seq_len, d_model]

    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """

    #step1:
    batch_size,seq_len=in_query_or_key[:2]

    """
    
    # 获取输入张量的形状
    batch_size, seq_len = in_query_or_key.shape[:2]

    # 确保d_k是偶数，因为RoPE需要将维度分成对
    assert d_k % 2 == 0, "d_k must be even for RoPE"

    # 步骤1: 预计算旋转频率
    # 频率计算公式: theta_i = theta^(-2i/d_k) for i in range(0, d_k//2)
    freqs = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)

    # 步骤2: 创建位置编码矩阵
    # 为每个位置创建旋转角度: position * freqs
    positions = token_positions.unsqueeze(-1)  # [batch_size, seq_len, 1]
    # [batch_size, seq_len, d_k//2]
    angles = positions * freqs.unsqueeze(0).unsqueeze(0)

    # 步骤3: 计算旋转矩阵的cos和sin分量
    cos_vals = torch.cos(angles)  # [batch_size, seq_len, d_k//2]
    sin_vals = torch.sin(angles)  # [batch_size, seq_len, d_k//2]

    # 步骤4: 将输入张量分成实部和虚部（成对处理）
    # 将d_k维分成d_k//2对复数
    # [batch_size, seq_len, d_k//2, 2]
    x = in_query_or_key.view(batch_size, seq_len, d_k//2, 2)

    # 步骤5: 应用旋转操作
    # 旋转公式: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    x_rotated = torch.stack([
        x[..., 0] * cos_vals - x[..., 1] * sin_vals,  # 实部旋转
        x[..., 0] * sin_vals + x[..., 1] * cos_vals   # 虚部旋转
    ], dim=-1)  # [batch_size, seq_len, d_k//2, 2]

    # 步骤6: 将旋转后的张量重新整形为原始形状
    # [batch_size, seq_len, d_k]
    output = x_rotated.view(batch_size, seq_len, d_k)
    """


    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
 # 1. 创建你的 BpeTokenizer 的一个实例
    tokenizer = BpeTokenizer()

    # 2. 配置 tokenizer 的状态
    # 测试传入的 vocab 是 {id: bytes}，我们需要 {bytes: id} 用于编码
    # 和 {id: bytes} 用于解码
    tokenizer.vocab = {token_bytes: token_id for token_id,
                       token_bytes in vocab.items()}
    tokenizer.inverse_vocab = vocab

    # 直接使用传入的合并规则
    tokenizer.merges = merges

    # 3. 处理特殊 token
    if special_tokens:
        special_tokens_bytes = [token.encode(
            'utf-8') for token in special_tokens]
        tokenizer.special_tokens = set(special_tokens_bytes)

        # 将特殊 token 添加到词汇表中（如果它们不存在）
        # 测试代码已经保证了这一点，但双重检查更稳健
        for token_bytes in special_tokens_bytes:
            if token_bytes not in tokenizer.vocab:
                new_id = len(tokenizer.vocab)
                tokenizer.vocab[token_bytes] = new_id
                tokenizer.inverse_vocab[new_id] = token_bytes

    # 4. 返回配置好的、可供使用的 tokenizer 实例
    return tokenizer
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError

from __future__ import annotations

import os
from requests import get
import torch.nn as nn
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
import numpy as np
import time


#cd /Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main && venv/bin/python -m pytest tests/test_model.py::test_linear tests/test_model.py::test_embedding tests/test_model.py::test_swiglu tests/test_model.py::test_rope -v
#git config  user.name=mark.huangcb  

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
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to   #batch_size,seq_len,d_model

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 使用torch.matmul执行线性变换: y = xW^T
    # 其中in_features的形状是(..., d_in),weights.T的形状是(d_in, d_out)
    return in_features @ weights.T


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
    # step2:将输入in_features投影到高维度,作为门控信号
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
    #print(f"Q shape: {Q.shape}")#4,12,64
    #print(f"K shape: {K.shape}")#4,16,64
    #print(f"V shape: {V.shape}")#4,16,64
    #print(f"mask shape: {mask.shape}")#4,12,16

    # step1: calculate attention scores
    attentionScores=Q@K.transpose(-2,-1)

    #step2:normalize
    d_k=Q.shape[-1]
    attentionScores=attentionScores/torch.sqrt(torch.tensor(d_k,dtype=attentionScores.dtype))

    #step3:apply mask
    if mask is not None:
        attentionScores = attentionScores.masked_fill(~mask, -1e9)


    #step4:softmax
    softmaxAttentionScores=torch.softmax(attentionScores,dim=-1)

    #step5:output
    ans=softmaxAttentionScores@V
    return ans


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


    print(f"\nd_model: {d_model}, num_heads: {num_heads}")
    print(f"q_proj_weight: {q_proj_weight.shape}")
    print(f"k_proj_weight: {k_proj_weight.shape}")
    print(f"v_proj_weight: {v_proj_weight.shape}")
    print(f"o_proj_weight:{o_proj_weight.shape}")
    print(f"in_features:{in_features.shape}")

    """
    tests/test_model.py::test_multihead_self_attention 
    num_heads: 4, d_k_per_head: 16，d_model: 64, 
    q_proj_weight: torch.Size([64, 64])
    k_proj_weight: torch.Size([64, 64])
    v_proj_weight: torch.Size([64, 64])
    o_proj_weight:torch.Size([64, 64])
    in_features:torch.Size([4, 12, 64])

    第一个64:d_k - 每个头的查询/键维度
    第二个64:d_in - 输入特征维度
    关键理解:虽然 q_proj_weight 形状是 [64, 64],但这实际上是所有头的投影权重合并在一起。因为 num_heads = 4,所以:
    每个头的实际维度:d_k_per_head = 64 / 4 = 16
    总投影权重大小:d_k = 16 x 4 = 64

    2. 输入特征形状:[4, 12, 64]
    in_features: torch.Size([4, 12, 64])

    第一个4:batch_size - 批次大小(4个样本)
    第二个12:sequence_length - 序列长度(12个token)
    第三个64:d_in - 输入特征维度(与投影权重的第二个维度匹配）


    1. q_proj_weight - 投影权重矩阵
    类型：权重参数（可训练的参数）
    作用：将输入特征映射到查询向量：用于将输入特征转换为查询向量
    形状：[d_k, d_in]（输出维度 × 输入维度）
    2. Q 向量 - 查询向量
    类型：计算得到的张量
    作用：实际的查询向量，用于注意力计算
    形状：[batch_size, seq_len, d_k]
    """
    batch_size, seq_len, d_in = in_features.shape
    d_q=q_proj_weight.shape[0]
    d_k=k_proj_weight.shape[0]
    d_v=v_proj_weight.shape[0]

    d_q_per_head=d_q //num_heads
    d_k_per_head=d_k //num_heads
    d_v_per_head=d_v //num_heads

    #step1:calcluate Q,K,V
    Q=run_linear(d_in=d_in,d_out=d_k,weights=q_proj_weight,in_features=in_features) #4,12,64
    K=run_linear(d_in=d_in,d_out=d_k,weights=k_proj_weight,in_features=in_features)
    V=in_features @ v_proj_weight.T




    #step2:reshape
    Q=Q.view(batch_size,seq_len,num_heads,d_k_per_head)#4,12,4,16
    K=K.view(batch_size,seq_len,num_heads,d_k_per_head)#4,12,4,16
    V=V.view(batch_size,seq_len,num_heads,d_v_per_head)#4,12,4,16


    #step3:Transformer standard format of Q,K,V
    Q=Q.transpose(1,2)#4,4,12,16
    K=K.transpose(1,2)
    V=V.transpose(1,2)


    #step4:attention scores
    attentionScores=Q @ K.transpose(-2,-1)

    #step5:scale
    attentionScores=attentionScores/torch.sqrt(torch.tensor(d_k_per_head, dtype=attentionScores.dtype))

    # step 5.5: mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=attentionScores.device)).bool()
    attentionScores = attentionScores.masked_fill(mask == 0, float('-inf'))

    #step6:softmax
    softmaxAttentionScores=torch.softmax(attentionScores, dim=-1)

    #step7:output
    tmp_output=softmaxAttentionScores @ V

    # step7: 合并多头输出
    # 当前形状: [batch_size, num_heads, seq_len, d_v_per_head] → 目标形状: [batch_size, seq_len, d_v]

    output = tmp_output.transpose(
        1, 2).contiguous().view(batch_size, seq_len, d_v)
    #print(f"合并多头输出之后output.shape={output.shape}")

    # step8: 投影到输出维度
    output = output @ o_proj_weight.T
    #print(f"投影到输出维度之后output.shape={output.shape}")
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
    In this case, tmust be the head embedding dimension (d_model // num_heads).
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
    batch_size, seq_len, d_in = in_features.shape #[4,12,64]

    # 计算每个头的维度
    d_k = q_proj_weight.shape[0]  # 查询/键的总维度 [64,64]
    d_v = v_proj_weight.shape[0]  # 值的总维度 [64,64]
    d_k_per_head = d_k // num_heads  # 每个头的查询/键维度 num_heads=4
    d_v_per_head = d_v // num_heads  # 每个头的值维度


    # 步骤1: 投影计算(单矩阵乘法）
    Q=run_linear(d_in,d_k,q_proj_weight,in_features)# [4,12,64]
    K=run_linear(d_in,d_k,k_proj_weight,in_features)# [4,12,64]
    V=run_linear(d_in,d_k,v_proj_weight,in_features)# [4,12,64]


    # 步骤2: 将投影后的张量重新组织为多头格式
    # 当前形状:[batch_size, seq_len, d_k] 目标形状:[batch_size, num_heads, seq_len, d_k_per_head]
    Q = Q.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2) #[4,4,12,16]
    K = K.view(batch_size, seq_len, num_heads, d_k_per_head).transpose(1, 2)

    # V张量单独处理,因为不需要应用RoPE
    V = V.view(batch_size, seq_len, num_heads, d_v_per_head).transpose(1, 2)


    # 步骤3: 应用RoPE到Q和K(如果提供了位置信息）
    if token_positions is not None:
        Q = run_rope(d_k // num_heads, theta, seq_len, Q, token_positions)
        K = run_rope(d_k // num_heads, theta, seq_len, K, token_positions)


    # 步骤4: 计算注意力分数 QK^T
    # [batch_size, num_heads, seq_len, seq_len]
    attention_scores = Q @ K.transpose(-2, -1)

    # 步骤5: 缩放注意力分数
    # [batch_size, num_heads, seq_len, seq_len]
    attention_scores = attention_scores / (d_k_per_head ** 0.5)

    #步骤5.5 应用因果掩码 ！！！
    # 生成一个下三角因果掩码，确保位置i只能关注到位置j<=i
    mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_scores.device)).bool()
    attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

    # 步骤6: 应用Softmax得到注意力权重
    # [batch_size, num_heads, seq_len, seq_len]
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # 步骤7: 加权求和得到输出
    # [batch_size, num_heads, seq_len, d_v_per_head]
    tmp_output = attention_weights @ V

    # 步骤8: 合并多头输出
    # 当前形状:[batch_size, num_heads, seq_len, d_v_per_head] 目标形状:[batch_size, seq_len, d_v]
    output = tmp_output.transpose(
        1, 2).contiguous().view(batch_size, seq_len, d_v)

    # 步骤9: 投影到输出维度
    output = output @ o_proj_weight.T  # [batch_size, seq_len, d_model]

    return output


def run_rope(
    d_k: int,         #嵌入维度大小(必须是偶数)
    theta: float,     #RoPE参数,控制旋转频率
    max_seq_len: int, #最大序列长度,用于预计算
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],  #输入Q或K张量
    token_positions: Int[Tensor, " ... sequence_length"],        #token_positions 是位置索引张量,它告诉RoPE函数每个token在序列中的具体位置。
) -> Float[Tensor, " ... sequence_length d_k"]:                  #返回应用了RoPE的张量
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

    """
    Rotary Positional Embedding (RoPE): 通过旋转方式将位置信息注入 query 和 key 向量。
    
    公式:
        rope(x) = x[:, 2i]   * cos(pos * θ^(-2i/d_k)) 
                - x[:, 2i+1] * sin(pos * θ^(-2i/d_k))

    支持任意批处理维度,并根据 token_positions 动态索引预计算的 cos/sin 缓存。
    """


    """
    # 假设每个token的4维向量(随机示例)
    # 形状: [batch_size=1, seq_len=3, d_k=4]

    in_query_or_key = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4]],    # "I" 的嵌入向量
    [[0.5, 0.6, 0.7, 0.8]],    # "love" 的嵌入向量  
    [[0.9, 1.0, 1.1, 1.2]]     # "apple" 的嵌入向量
    ])
    RoPE不是直接修改词向量,而是在计算注意力时对查询向量(Q)和键向量(K)进行旋转


    # 将d_k=4分成2对复数 [实部, 虚部]
    # 形状: [1, 3, 2, 2]
    x = in_query_or_key.view(1, 3, 2, 2)

    # 分解后的结构:
    # Token "I" (位置0):
    #   第1对: [0.1, 0.2]  → 复数: 0.1 + 0.2i
    #   第2对: [0.3, 0.4]  → 复数: 0.3 + 0.4i

    # Token "love" (位置1):
    #   第1对: [0.5, 0.6]  → 复数: 0.5 + 0.6i  
    #   第2对: [0.7, 0.8]  → 复数: 0.7 + 0.8i

    # Token "apple" (位置2):
    #   第1对: [0.9, 1.0]  → 复数: 0.9 + 1.0i
    #   第2对: [1.1, 1.2]  → 复数: 1.1 + 1.2i


    # 频率计算: θ^(-2i/d_k) for i=0,1
    freqs = [10000^(-0/4)=1.0, 10000^(-2/4)=0.01]

    # 角度计算: 位置 x 频率
    # Token "I" (位置0): [0x1.0=0°, 0x0.01=0°]
    # Token "love" (位置1): [1x1.0=1°, 1x0.01=0.01°]  
    # Token "apple" (位置2): [2x1.0=2°, 2x0.01=0.02°]

    cos_vals = [[cos(0°)=1.0, cos(0°)=1.0], 
                [cos(1°)=0.9998, cos(0.01°)=0.9999],
                [cos(2°)=0.9994, cos(0.02°)=0.9999]]

    sin_vals = [[sin(0°)=0.0, sin(0°)=0.0],
                [sin(1°)=0.0175, sin(0.01°)=0.0002],
                [sin(2°)=0.0349, sin(0.02°)=0.0003]]

    x_rotated = torch.stack([
        x[..., 0] * cos_vals - x[..., 1] * sin_vals,  # 新的实部
        x[..., 0] * sin_vals + x[..., 1] * cos_vals   # 新的虚部
    ], dim=-1)


    [[0.1, 0.2, 0.3, 0.4]],    # "I" 的嵌入向量
    [[0.5, 0.6, 0.7, 0.8]],    # "love" 的嵌入向量  
    [[0.9, 1.0, 1.1, 1.2]]     # "apple" 的嵌入向量

    # 第1对复数: [0.1, 0.2] → 旋转0°
    新实部 = 0.1x1.0 - 0.2x0.0 = 0.1
    新虚部 = 0.1x0.0 + 0.2x1.0 = 0.2
    # 结果: [0.1, 0.2] (不变,因为位置0不旋转)

    # 第2对复数: [0.3, 0.4] → 旋转0°
    新实部 = 0.3x1.0 - 0.4x0.0 = 0.3  
    新虚部 = 0.3x0.0 + 0.4x1.0 = 0.4
    # 结果: [0.3, 0.4] (不变)


    # 第1对复数: [0.5, 0.6] → 旋转1°
    新实部 = 0.5x0.9998 - 0.6x0.0175 ≈ 0.4999 - 0.0105 = 0.4894
    新虚部 = 0.5x0.0175 + 0.6x0.9998 ≈ 0.0088 + 0.5999 = 0.6087
    # 结果: [0.4894, 0.6087] (轻微旋转)

    # 第2对复数: [0.7, 0.8] → 旋转0.01°  
    新实部 = 0.7x0.9999 - 0.8x0.0002 ≈ 0.6999 - 0.0002 = 0.6997
    新虚部 = 0.7x0.0002 + 0.8x0.9999 ≈ 0.0001 + 0.7999 = 0.8000
    # 结果: [0.6997, 0.8000] (几乎不变)




    # 第1对复数: [0.9, 1.0] → 旋转2°
    新实部 = 0.9x0.9994 - 1.0x0.0349 ≈ 0.8995 - 0.0349 = 0.8646
    新虚部 = 0.9x0.0349 + 1.0x0.9994 ≈ 0.0314 + 0.9994 = 1.0308
    # 结果: [0.8646, 1.0308] (明显旋转)

    # 第2对复数: [1.1, 1.2] → 旋转0.02°
    新实部 = 1.1x0.9999 - 1.2x0.0003 ≈ 1.0999 - 0.0004 = 1.0995
    新虚部 = 1.1x0.0003 + 1.2x0.9999 ≈ 0.0003 + 1.1999 = 1.2002  
    # 结果: [1.0995, 1.2002] (几乎不变)


    新实部 = 原实部xcosθ - 原虚部xsinθ
    新虚部 = 原实部xsinθ + 原虚部xcosθ



    应用RoPE后:

    位置0的token:几乎不旋转(角度小）
    位置越靠后的token:旋转角度越大
    不同频率维度:旋转速度不同(高频维度旋转快,低频维度旋转慢）
    这样,模型就能通过旋转角度来区分不同位置的相同词汇,实现位置编码的效果！
    """
    
    #step1:确认嵌入维度是偶数
    d_k=in_query_or_key.shape[-1]
    seq_len=in_query_or_key.shape[-2]
    if d_k%2!=0:
        raise ValueError("d_k must be even for RoPE")
    
    #step2:确认旋转角度theta, 频率计算: θ^(-2i/d_k) 
    i = torch.arange(0, d_k//2, dtype=torch.float32, device=in_query_or_key.device)
    freqs = theta ** (-2 * i / d_k)  # 形状 [d_k//2]

    # step3:扩展 freqs 以匹配最大序列长度
    freqs = freqs.unsqueeze(0).expand(max_seq_len, -1)  # (max_seq_len, d_k // 2)
    
    # step4:创建位置索引
    positions = torch.arange(max_seq_len).unsqueeze(1)          # (max_seq_len, 1)
    
    # 角度计算: 位置 x 频率
    # Token "I" (位置0): [0x1.0=0°, 0x0.01=0°]
    # Token "love" (位置1): [1x1.0=1°, 1x0.01=0.01°]  
    # Token "apple" (位置2): [2x1.0=2°, 2x0.01=0.02°]


    # 步骤5: 计算角度 (向量化操作)
    angles = positions * freqs  # 广播：形状 [*batch_dims, seq_len, d_k//2]

    #step6:创建旋转矩阵
    cos_emb=torch.cos(angles)
    sin_emb=torch.sin(angles)
    cos_emb, sin_emb = cos_emb[:seq_len], sin_emb[:seq_len]

    #step7:x转换成复数
    x_even=in_query_or_key[...,::2]
    x_odd=in_query_or_key[...,1::2]

    #step8:计算旋转
    rotated_even = x_even * cos_emb - x_odd * sin_emb
    rotated_odd = x_even * sin_emb + x_odd * cos_emb


    # step9:重新组合维度
    result = torch.zeros_like(in_query_or_key)
    result[..., ::2] = rotated_even
    result[..., 1::2] = rotated_odd

    return result
    
    


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

    """
    输入 → RMSNorm → 多头自注意力 → 残差连接 → RMSNorm → 前馈网络 → 残差连接 → 输出
    """

    batch_size,seq_len,d_model=in_features.shape

    #step1:第一个RMSNorm
    in_features_fristRMSNorm=run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights['ln1.weight'], in_features=in_features)

    #step2:多头注意力
    token_positions=torch.arange(seq_len).expand(batch_size,-1)
    MHA=run_multihead_self_attention_with_rope(d_model=d_model,
                                               num_heads=num_heads,
                                               max_seq_len=seq_len,
                                               theta=theta,
                                               q_proj_weight=weights["attn.q_proj.weight"],
                                               k_proj_weight=weights["attn.k_proj.weight"],
                                               v_proj_weight=weights["attn.v_proj.weight"],
                                               o_proj_weight=weights["attn.output_proj.weight"],
                                               in_features=in_features_fristRMSNorm,
                                               token_positions=token_positions)
    
    #step3:第一个残差连接
    MHA_fristResidual=in_features+MHA

    #step4:第二个RMSNorm
    MHA_fristResidual_secondRMSNorm=run_rmsnorm(d_model=d_model,
                                          eps=1e-5,
                                          weights=weights['ln2.weight'],
                                          in_features=MHA_fristResidual)
    
    #step5:前馈网络
    FFN=run_swiglu(d_model=d_model,
                   d_ff=d_ff,
                   w1_weight=weights['ffn.w1.weight'],
                   w2_weight=weights['ffn.w2.weight'],
                   w3_weight=weights['ffn.w3.weight'],
                   in_features=MHA_fristResidual_secondRMSNorm)
    
    #step6:第二个残差连接
    FFN_secondRsidual=MHA_fristResidual+FFN

    #step7:输出
    output=FFN_secondRsidual
    return output
    


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
        rope_theta (float): The RoPE \\Theta parameter.
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
    #step1:获取token_embeddings,即将token转换为embedding之后的向量
    token_embeddings=run_embedding(vocab_size,d_model,weights['token_embeddings.weight'],in_indices) #[batch_size,sequence_length,d_model]=[4,12,64]
    batch_size,seq_len,d_model=token_embeddings.shape
    #step2:构建多层transformer block，将embedding之后的token输入transformer block
    for layer_idx in range(num_layers):
        #构建当前层的权重字典
        layer_weights={
        'attn.q_proj.weight':weights[f'layers.{layer_idx}.attn.q_proj.weight'],
        'attn.k_proj.weight':weights[f'layers.{layer_idx}.attn.k_proj.weight'],
        'attn.v_proj.weight':weights[f'layers.{layer_idx}.attn.v_proj.weight'],
        'attn.output_proj.weight':weights[f'layers.{layer_idx}.attn.output_proj.weight'],
        'ln1.weight':weights[f'layers.{layer_idx}.ln1.weight'],
        'ffn.w1.weight':weights[f'layers.{layer_idx}.ffn.w1.weight'],
        'ffn.w2.weight':weights[f'layers.{layer_idx}.ffn.w2.weight'],
        'ffn.w3.weight':weights[f'layers.{layer_idx}.ffn.w3.weight'],
        'ln2.weight':weights[f'layers.{layer_idx}.ln2.weight'],
        }
        # 注意这里将当前层的输出赋值给token_embeddings，作为下一层的输入，因为左边只能是token_embeddings
        token_embeddings=run_transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,layer_weights,token_embeddings)

    #step3:将多层transformer输出通过RMSNorm
    final_output=run_rmsnorm(d_model=d_model,eps=1e-5,weights=weights['ln_final.weight'],in_features=token_embeddings)
    logits=run_linear(d_in=d_model,d_out=vocab_size,weights=weights['lm_head.weight'],in_features=final_output) #[batch_size, sequence_length, vocab_size]=[4,12,10000]


    return logits
    
    


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

    """
    标准LayerNorm:
        计算均值和方差：对每个样本的所有特征维度
        (x-mean)/sqrt(variance+eps)
    
    RMSNorm:
        只计算均方根(RMS),不计算均值
        output = (x / RMS(x)) * weights

        其中：

        RMS(x) = sqrt(mean(x²) + eps)
        weights 是可学习的缩放参数

    """

    output=in_features.clone()

    #每一个batch
    for i in range(in_features.shape[0]):
        #每个token
        for j in range(in_features.shape[1]):
            #step1:计算均方根rms
            token_feature=in_features[i][j][:]
            token_feature_squared=token_feature**2
            mean_token_feature_squared=torch.mean(token_feature_squared)
            rms=torch.sqrt(mean_token_feature_squared+eps)
            #step2:归一化
            output[i][j][:]=(in_features[i][j][:]/rms)*weights
    return output



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    output=in_features*torch.sigmoid(in_features)
    return output


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

    #step1：从数据集中随机选择起始位置
    max_start=len(dataset)-context_length
    starts=np.random.randint(low=0,high=max_start,size=batch_size,dtype=int)
    
    #step2: 根据起始位置获取输入序列和对应的标签
    inputs=[]
    labels=[] 
    for start in starts:
        input_seq=dataset[start:start+context_length]
        labels_seq=dataset[start+1:start+1+context_length]
        inputs.append(input_seq)
        labels.append(labels_seq)

    #step3:转换成pytorch张量并放到指定设备
    inputs_tensor=torch.tensor(data=inputs,dtype=torch.long,device=device)
    labels_tensor=torch.tensor(data=labels,dtype=torch.long,device=device)

    
    return inputs_tensor,labels_tensor
    



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
    
    print(f"\nin_features.shape={in_features.shape}") #[3,5]
    #safe_softmax
    #step1:计算每行最大值
    max_values=torch.max(in_features,dim=dim,keepdim=True)[0]
    print(f"max_values.shape={max_values.shape}") #[3,1]
    #step2:减去最大值
    exp_vals=torch.exp(in_features-max_values)
    #step3:计算softmax
    softmax_vals=exp_vals/torch.sum(exp_vals,dim=dim,keepdim=True)
    
    return softmax_vals
    #return torch.softmax(in_features,dim=dim)



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

    """
    inputs: 形状为 [batch_size, vocab_size] 的未归一化logits
    targets: 形状为 [batch_size] 的真实标签索引
    """
    #step1:softmax得到概率分布
    probs = torch.log_softmax(inputs, dim=1)  #[8,5]

    #step2:计算交叉熵
    batch_size=inputs.shape[0]
    """
    # 假设批量大小为3,词汇表大小为4
    batch_size = 3
    vocab_size = 4

    # 概率矩阵:3个样本,每个样本有4个类别的概率
    probs = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],  # 样本0的概率分布
        [0.4, 0.3, 0.2, 0.1],  # 样本1的概率分布  
        [0.2, 0.2, 0.3, 0.3]   # 样本2的概率分布
    ])

    # 真实标签:样本0的真实类别是2,样本1的真实类别是0,样本2的真实类别是3
    targets = torch.tensor([2, 0, 3])

    # 高级索引操作
    target_probs = probs[torch.arange(batch_size), targets]
    print("获取的真实类别概率:", target_probs)
    # 输出: tensor([0.3000, 0.4000, 0.3000])
    """
    target_probs=probs[torch.arange(batch_size),targets]
    return -torch.mean(target_probs)



    
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    #step1:计算所有梯度的总L2范数=sqrt(∑∑(grad_ij)²)  # 对所有参数的所有梯度元素求和
    total_norm_squared=0
    for parameter in parameters:
        if parameter.grad is not None:
            total_norm_squared+=torch.sum(parameter.grad**2)
    total_norm=torch.sqrt(total_norm_squared)

    #step2:如果总范数超过最大值，按比列缩放所有梯度
    if total_norm.item()>max_l2_norm:
        scale_factor=max_l2_norm / total_norm.item()

    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad=parameter.grad*scale_factor




def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW

import math
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
    #stage1:线性预热（0 ≤ it < warmup_iters）
    #stage2:余弦退火（warmup_iters ≤ it ≤ warmup_iters + cosine_cycle_iters）
    """
    进度 = (it - warmup_iters) / cosine_cycle_iters
    余弦衰减 = 0.5 * (1 + cos(π * 进度))
    学习率 = min_lr + (max_lr - min_lr) * 余弦衰减
    """
    #stage3:平稳学习率
    if it<warmup_iters:
        learning_rate = max_learning_rate * it / warmup_iters
    elif it<=cosine_cycle_iters:
        process=( it - warmup_iters)/(cosine_cycle_iters - warmup_iters)
        cos_value = np.cos(np.pi * process)
        learning_rate = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        learning_rate=min_learning_rate

    return learning_rate


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
    # 1. 创建检查点字典，包含所有需要保存的信息
    checkpoint = {
        'iteration': iteration,#记录训练到第几步，用于恢复时知道从哪里继续
        'model_state_dict': model.state_dict(),#保存模型的所有可学习参数（权重、偏置等）
        'optimizer_state_dict': optimizer.state_dict(),#保存优化器的所有状态信息（如动量、学习率等）
        'timestamp': time.time(),  # 记录保存时间，用于版本管理
        
    }
    
    # 2. 根据输出类型选择保存方式
    if isinstance(out, (str, os.PathLike)):
        torch.save(checkpoint, out)
    else:
        torch.save(checkpoint, out)#test

    



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
    # 1. 根据输入类型加载检查点
    if isinstance(src, (str, os.PathLike)):
        # 从文件路径加载
        checkpoint = torch.load(src)
    else:
        # 从文件对象加载
        checkpoint = torch.load(src)
    
    # 2. 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. 返回之前的迭代次数
    return checkpoint['iteration']


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
    

    return BpeTokenizer(vocab,merges,special_tokens)


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer from a given vocabulary, list of merges, and (optionally) special tokens.
        
        Args:
            vocab: A dictionary mapping token IDs to their byte representations.
            merges: A list of tuples representing BPE merge operations.
            special_tokens: Optional list of strings that should be treated as unbreakable tokens.
        """


        """
        # 定义词汇表和合并规则
        vocab = {
            0: b"<|endoftext|>",
            1: b" the",
            2: b"ing",
            3: b" apple",
            # ... 其他词汇
        }

        #合并规则
        merges = [
            (b"a", b"p"),      # 合并 a + p → ap
            (b"ap", b"p"),     # 合并 ap + p → app
            (b"app", b"l"),    # 合并 app + l → appl
            (b"appl", b"e"),   # 合并 appl + e → apple
        ]

        # 特殊标记列表
        special_tokens = [
            "<|endoftext|>",    # 文本结束标记
            "<|pad|>",          # 填充标记（用于对齐序列长度）
            "<|unk|>",          # 未知词汇标记
            "<|bos|>",          # 文本开始标记
            "<|eos|>",          # 文本结束标记
            "<|mask|>",         # 掩码标记(用于BERT等模型)
        ]
        """
        self.vocab=vocab
        self.merges=merges

        #step1:将vocab中{tokenid:token}的字典反转为{token:tokenid}的字典
        self.byte_to_token_id={v:k for k,v in vocab.items()}

        #step2:将特殊token转换为字节表示
        self.special_tokens=special_tokens or []
        self.special_token_bytes=[token.encode("utf-8") for token in self.special_tokens]

       
        #step3:bpe_rank
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        #print(f"bpe_ranks: {self.bpe_ranks}")
        """
        (b' re', b'claimed'): 49969, (b' interesting', b'ly'): 49970, (b'\xd7', b'\xa9'): 49971
        """

        #step4:确认special_token在vocab中，如果不在，就添加入vocab
        for token in self.special_token_bytes:
            if token not in self.byte_to_token_id:
                self.vocab[len(self.vocab)]=token
                self.byte_to_token_id[token]=len(self.vocab)-1

    #step5:BPE核心函数，处理普通文本
    def BPE(self, text: str) -> list[int]:
        """
        Tokenize a normal piece of text (not a special token) into token IDs.
        
        Args:
            text: A string to tokenize.
            
        Returns:
            A list of token IDs representing the tokenized text.
        """
        pre_tokens=[]
        for m in re.finditer(PAT,text):
            word=m.group(0)
            pre_tokens.append(word)
        token_ids=[]
        for word in pre_tokens:
            byte_tuple=to_bytes_tuple(word)
            merged=self._apply_merges(byte_tuple)
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        return token_ids




    #step6:定义encode函数
    def encode1(self, text: str) -> list[int]:
        """
        Encode an input text string into a sequence of token IDs.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A list of integer token IDs representing the encoded text.
        """
        pass
    
    


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text string into a sequence of token IDs.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A list of integer token IDs representing the encoded text.
        """
        tokens = []

        # Sort special tokens by length (longest first) to avoid partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # If it's a special token, add its ID directly
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                # Otherwise, tokenize normally using BPE
                tokens.extend(self.BPE(part))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        """
        Given an iterable of strings (e.g., a file handle), yield token IDs lazily.
        
        Args:
            iterable: An iterable source of text chunks.
            
        Yields:
            Token IDs generated by processing the input iterable.
        """
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        # Concatenate all token bytes
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # Decode bytes to string, replacing invalid sequences
        return full_bytes.decode("utf-8", errors="replace")


    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            byte_tuple: A tuple of single-byte tokens.
            
        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """
        word: list[bytes] = list(byte_tuple)

        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        return word


import regex as re

import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Iterable, Optional
##### Helper functions #####

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(word.encode("utf-8"))
    l = [bytes([x]) for x in l]
    return tuple(l)

##### Helper functions #####


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
    # Step 1: Initialize Vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    # Step 2: Pre-tokenization
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')

    # Step 3: Compute BPE Merges
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all adjacent byte pairs
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    return vocab, merges

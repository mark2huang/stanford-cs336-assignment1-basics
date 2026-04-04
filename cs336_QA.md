# cs336 assignment1 Q&A

## Apple 上海 Deep Learning 面试：10 道硬核通关题

Mark，请端正坐姿。现在我是 Apple Core ML 团队的 Hiring Manager，欢迎来到你的 Technical Interview。这 10 道题，是检验你 CS336 学习成果的试金石：

### **【底层架构与数学推导】**

#### 1.**手写 Self-Attention：** 请在白板上用 PyTorch 或 NumPy 伪代码写出 Multi-Head Attention 的前向传播逻辑，并解释其中 `Q, K, V` 矩阵的物理意义。

```python
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
    num_heads: 4, d_k_per_head: 16,d_model: 64, 
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

    """
    #step1:calculate Q,K,V
    Q=in_features@q_proj_weight.T #4,12,64
    K=in_features@k_proj_weight.T
    V=in_features@v_proj_weight.T

    #step2:reshape
    batch_size,seq_len,d_model=Q.shape
    d_k=q_proj_weight.shape[0]
    d_per_head=d_k//num_heads
    d_in=q_proj_weight.shape[0]


    Q=Q.view(batch_size,seq_len,num_heads,d_per_head) #4,12,4,16
    Q=Q.transpose(1,2)#4,4,12,16
    K=K.view(batch_size,seq_len,num_heads,d_per_head) 
    K=K.transpose(1,2)
    V=V.view(batch_size,seq_len,num_heads,d_per_head) 
    V=V.transpose(1,2)

    #step3:calculate attentionScore
    attentionScores=Q@K.transpose(-2,-1) #4,4,12,12
    #step4:normalize
    attentionScores=attentionScores/torch.sqrt(torch.tensor(d_per_head,dtype=attentionScores.dtype))
    #step5:mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=attentionScores.device)).bool()
    attentionScores = attentionScores.masked_fill(mask == 0, float('-inf'))
    #step6:softmax
    attentionScores=torch.softmax(attentionScores,dim=-1)
    #step6:output
    output_temp=attentionScores@V
    #step7:合并输出
    output=output_temp.transpose(1,2).contiguous().view(batch_size,seq_len,d_model)
    #step8:投影
    output=output@o_proj_weight.T
    return output

```





#### 2.**Softmax 深度追问：** 为什么在 Transformer 中要使用 Softmax？如果我不用 $\sqrt{d_k}$ 进行缩放（Temperature Scaling），在训练初期会有问题？

---

💡 **回答**:

point0:因为 Attention 的本质是**“加权平均（Weighted Average）”**。我们需要对 Value (`VV`) 矩阵进行信息聚合。Softmax 可以将任意范围的实数（Attention Scores）映射成一个**总和为 1 的标准概率分布**

point1:因为token经过embedding之后得到的in_features，在乘上q_proj_weight，再经过Q×K得到的attentionScores有可能出现分布变得极其尖锐（Sharp）或极端。不进行d_k的缩放，这样经过softmax之后就会变成很大的值，这一行这个token对整个seq的其他token就会被强行减小，训练的时候容易出现梯度消失的问题。Softmax 函数的导数（梯度）公式中包含 p×(1-p)，一旦出现0或者1，就会出现梯度的消失从而影响训练

---





#### 3.**位置编码：** 请解释绝对位置编码和旋转位置编码（RoPE）的本质区别。为什么当今大模型（如 Llama）几乎全换成了 RoPE？

---

💡 **回答** :

> **本质区别：**
> 传统绝对位置编码（如原版 Sin/Cos）是通过**向量相加（Add）**的方式，把位置信息硬塞给 Token，模型需要自己努力去相加后的结果中“猜”出两个词的距离；
> 而 RoPE（旋转位置编码）是通过**复数乘法（旋转 Rotation）**的方式，将位置信息注入 Token。它的数学内涵是：**通过给每个 Token 注入绝对位置，使得两个 Token 在做点积（Attention Score）时，结果天然只和它们的相对距离有关。**
>
> **为什么大模型全换成了 RoPE？（三大原因）**
> 1.  **完美的数学性质（绝对注入，相对输出）：** 传统相加方式在算 $Q \times K$ 时，会产生语义向量和位置向量的交叉干扰项（Cross-terms）。RoPE 彻底消除了这种干扰，纯粹通过向量夹角的差异来表示词与词之间的距离。
> 2.  **外推性强（Length Extrapolation）：** 因为 RoPE 的本质是旋转角度，哪怕测试时的句子比训练时更长（比如超出了训练时的 Max Length），模型也能根据角度差自然地推断出相对距离，而传统绝对位置编码遇到没见过的长位置会直接崩溃。
> 3.  **长期衰减特性：** RoPE 的数学设计使得距离越远的两个 Token，它们位置编码的内积越趋近于 0（也就是离得越远，位置上的关联越弱）。这非常符合人类语言“局部依赖性强”的自然规律。

---





### **【端侧部署与工程优化（Apple最爱）】**

#### 4.**KV Cache 原理：** 在大语言模型推理时，什么是 KV Cache？如果不开启它，系统瓶颈在哪里？如果开启它，随着 Context Length（上下文长度）变大，会遭遇什么新的瓶颈？

---

💡 **回答**

inference的时候，由于LLM的原理根据前面所有的token,推测出一下个token，这就需要当前token的query和之前的token的key和value相乘，如果我们可以保留前面token的kv，可以利用这个cache加速推理的流程，由于当前的query不会和之前token的query做任何计算，所有不存在q cache这个东西。

Point0:如果不开启它，随着模型inference长度的增大，模型推理的速度回显著变慢，如果不开启 KV Cache，为了生成第N个 token，模型必须把前面 1 到 N−1 个 token 重新走一遍前向传播（Forward Pass），也就是把前面所有的词重新经过 Embedding、所有的 Linear 层去重新计算出它们的 K 和 V。这导致了海量的冗余计算，此时系统的瓶颈是 **Compute Bound（算力/计算瓶颈）**

Point1：如果开启它，随着 Context Length（上下文长度）变大，kv cache的存储会是问题，如果放到L1D这样的高速内存，容量不小回不够，但如果放到HBM这样的内容，每次访问的cycle就会很大，此时会碰到**内存墙（Memory Wall）**

- **第一是容量（Capacity）挑战，导致 OOM：** KV Cache 的大小与 Batch Size、Context Length 呈线性正相关。比如一个 7B 模型，几十 K 的上下文，仅 KV Cache 就可能占去几十 GB 的显存。这极大地限制了系统能支撑的最大并发量（Batch Size）。
- **第二是带宽（Bandwidth）挑战，导致推理极慢：** 就像刚才提到的存储层级问题，GPU 计算时需要将数据从 HBM 搬运到 SRAM（比如 L1 Cache / Shared Memory）中。在 Decoding 阶段，每次只生成一个 Token，这意味着计算量（FLOPs）很小，但为了算这个 Token，必须把几万个 Token 的庞大 KV Cache 从 HBM 完整读取一遍。**这种极低的‘计算访存比（Arithmetic Intensity）’会瞬间打满 HBM 的内存带宽**，导致 GPU 的计算核心（CUDA Cores/Tensor Cores）大量时间在空转等待数据读取，也就是所谓的 Memory Bandwidth Bound。”

注意：容量不够叫 **OOM (Out of Memory)**；HBM 访问慢带来的瓶颈在业内叫 **Memory Bandwidth Bound（访存带宽瓶颈）** 或者是 **Memory Wall（内存墙）**

Point3：

“为了解决长文本下 KV Cache 的瓶颈，目前业界主流的解决思路有：

（1）**架构层面：** 使用 **MQA（Multi-Query Attention）** 或 **GQA（Grouped-Query Attention）**，让多个 Attention Head 共享同一组 KV，直接从物理上将 KV Cache 的体积缩小几倍甚至几十倍（如 Llama 2/3 的做法）。

（2）**系统工程层面：** 引入 **PagedAttention（如 vLLM 框架）**，像操作系统的虚拟内存分页一样管理 KV Cache，解决显存碎片化问题；

（3）**量化层面：** 对 KV Cache 进行 **INT8 或 FP8/INT4 量化**，直接将显存读写带宽减半。”

---





#### 4. PagedAttention 只是节省了显存空间（Capacity），它又没有让 HBM 的物理读写速度变快，凭什么说它能缓解带宽瓶颈（Memory Wall）呢？

---

💡 回答：PagedAttention 确实没有改变物理内存带宽，但它通过**极大提升显存利用率，间接打破了 Memory Wall**。逻辑链条是这样的：

1.  **Decoding 阶段的本质是 Memory Bound：** 在逐个生成 Token 时（比如生成 'They'），我们要把模型所有权重（比如 7B 参数 = 14GB）从 HBM 搬到计算核心（SRAM）里，只为了做一次微小的向量乘法。**读了海量的数据，却只做了极少的计算**（计算访存比极低），导致计算核心都在等数据，这就是 Memory Wall。
2.  **破局之道是提升 Batch Size：** 如果我们能同时处理 100 个用户的请求，那么同样搬运一次 14GB 的模型权重，我们可以同时计算 100 个用户的 Query，**计算量瞬间放大了 100 倍，而权重访存量不变！** 这样就能把 GPU 的算力榨干，绕过带宽瓶颈。
3.  **传统方案为什么做不到？** 因为传统 KV Cache 的碎片化太严重！由于 OOM 的限制，系统最多只能把 Batch Size 开到 10，显存就爆了。
4.  **PagedAttention 的杀招：** PagedAttention 彻底消灭了显存碎片，把省下来的海量显存全部用来**装更多的用户请求**。它能轻松把 Batch Size 从 10 提升到 100 甚至更高。

**总结一句话：PagedAttention 通过解决显存碎片化（Capacity 问题），解锁了超大 Batch Size，从而极大地提高了系统的计算访存比（Arithmetic Intensity），最终成功缓解了带宽瓶颈（Memory Wall），让大模型推理的吞吐量（Throughput）提升了 2-4 倍！**”

---





#### 5.**量化技术 (Quantization)：** 我想在 8GB 内存的 Mac M3 芯片上跑一个 7B 模型。请解释 INT8 动态量化和 AWQ（Activation-aware Weight Quantization）算法的核心区别。

---

💡 **回答**：

7B的模型，如果所有的参数是FP32，仅仅是把所有的参数装进GPU，也需要28G的内存的空间，M3显然不够，INT8动态量化是指把7B的模型的所有参数映射到-127~128这个范围内，原本需要4byte/parameter，现在只需要1个byte，所有参数只需要7GB的存储，但这样的做的话，一共就8G的内存，macc操作系统需要，屏幕显示需要内存，KV cache也需要内存，8G的内存明显回OOM，并且INT8量化的模型的性能有所下降，AWQ（Activation-aware Weight Quantization）算法我不知道

核心区别一：量化对象不同（W8A8 vs W4A16）：**

- **INT8 动态量化**：通常是指 **W8A8**，也就是权重是 INT8，输入激活值也在运行时**动态**被量化为 INT8。这样不仅节省显存，还能利用硬件的 INT8 矩阵乘法单元加速。
- **AWQ**：属于 **Weight-Only Quantization（通常是 W4A16）**。它只把权重压缩到了 4-bit 来节省极致的带宽和容量，但在计算时，会把权重反量化回 FP16，和 FP16 的激活值进行计算。这非常契合 Apple M 芯片的特性，因为端侧推理的核心瓶颈往往是**访存带宽（Memory Bandwidth）**而不是算力。

核心区别二：量化算法的本质（一刀切 vs 激活感知）：

- 普通的 4-bit 量化会导致模型精度严重崩塌。**AWQ (Activation-aware Weight Quantization)** 的核心创新在于它发现：**并非所有权重都同等重要，权重的显著性（Saliency）是由它对应的输入激活值（Activation）的大小决定的。**
- **AWQ 会通过极少量的数据进行**校准（Calibration）**，找出激活值中约 1% 的极大值（Outliers）。然后通过等价的数学缩放（Scaling），保护与这些巨大激活值对应的权重，使其在 4-bit 量化下不丢失精度。
  **总结来说：INT8 动态量化是运行时的全局无差别量化；而 AWQ 则是提前校准的、只针对权重并重点保护 1% 关键权重的极低比特量化。通过 AWQ，我们就能完美地把 7B 模型塞进 8GB 的 Mac 里，且几乎不掉性能。

---





#### **6.FlashAttention：** 它是如何通过硬件感知（Hardware-aware）来加速 Transformer 的？请解释 SRAM 和 HBM 之间的数据搬运优化逻辑。

---

💡 **回答**：

- 传统 Attention 在计算时，会产生一个 O(N^2)的中间注意力矩阵（Attention Scores）。随着 Context Length 变长，频繁地将这个巨大的矩阵写入和读出 HBM（显存）

- FlashAttention 的核心理念是**算子融合（Kernel Fusion）\**和\**分片（Tiling）**。它的目标是：**绝不把中间结果S和 P 写回 HBM，所有事情都在 SRAM 这个工作台上一次性搞定**，对于小块的矩阵放入SRAM进行计算累加并采用on-line softmax技术进行attention的计算

- 普通的 Softmax 需要遍历整行数据找出**最大值（用于防止数值溢出）**，再求出**总和（用于归一化）**。由于 FlashAttention 是分块读取的，我们一开始拿不到全局信息。

  所以我们引入了 **Online Softmax** 技术，我们在 SRAM 中维护两个局部变量：

  1. **局部最大值 (local_max)**
  2. **局部指数和 (local_sum)**

  当我们加载了一个新块（New Block）进来时，我们发现了一个**更大的最大值 (new_max)**，这时候旧块算出的结果就‘过期’了。怎么办？数学上的绝妙之处在于，**我们不需要重新去读取旧块的数据！** 我们只需要把之前累加的结果，乘上一个修正系数eold_max−new_max,旧数据就瞬间被**等价缩放**到了以新最大值为基准的尺度下！然后再把新块的值加进去。”

---

### **【训练与分布式（CS336 核心）】**



#### 7.**显存爆炸分析：** 用 Adam 优化器微调一个 7B 模型，为什么实际上需要的显存远大于模型本身的权重大小（7GB * 2= 14GB）？显存里到底还存了哪些东西？

---

💡 **回答**：

- 首先你把7B放入内存，假设参数FP16，你需要7×2=14GB的内存，

- **梯度 Gradients (FP16/BF16)：** 7B×2 bytes=14GB

- Adam算法的核心算法一是给每一个参数一个自己的learning rate，二是每一次的训练，我们不止听梯度，我们还需要看moment(冲量)，因为Adam里面对每一个参数需要有一个4byte的一阶矩，和4byte的二阶矩，这就需要7×4×2=56的内存，

- Adam里面还需要保留一份主参数的FP32版本，共计需要7×4=28GB

- **纯模型静态占用总计：** 

  ```python
  14+14+28+28+28=112GB
  ```

- 我们还没算输入数据和激活值的内存，因为LLM训练的时候需要Data Parallel (DP)的技术

Apple Silicon（M2/M3 Ultra）采用的是**统一内存架构（UMA, Unified Memory Architecture）**，一台 Mac Studio 的内存可以高达 **192GB 甚至 256GB**！这意味着，传统的单张 Nvidia 显卡跑不起来的 7B 全量微调，**在 Apple 的 M 系列芯片上是可以单机单卡直接跑通的！** 

---





#### 8.**分布式策略：** 请用大白话解释 Data Parallel (DP)、Tensor Parallel (TP) 和 Pipeline Parallel (PP) 各自的适用场景和通信瓶颈，请问在训练一个 70B 模型时，你会怎么组合这些策略？

---

### 💡 第 8 题进阶：分布式策略详解（大白话 + 硬核瓶颈）

**【通俗比喻（面试时可以直接用）】**
“假设我们要开工厂造 100 辆巨型汽车（大模型）：
*   **DP（数据并行）：** 我们建 4 个一模一样的流水线，每个流水线造 25 辆完整的车。
*   **TP（张量并行）：** 汽车太大了，一个车间装不下。我们让 4 个工人**同时**造同一辆车，张三造左车门，李四造右车门，然后立刻拼在一起。
*   **PP（流水线并行）：** 汽车拆成几个阶段，车间 A 只负责造底盘，造完扔给车间 B 去装发动机。

**【硬核专业解答（重点讲瓶颈和场景）】**

> “面试官您好，这三种策略的本质是为了打破单卡显存和算力的天花板，它们的适用场景和瓶颈完全不同：
>
> **1. Data Parallel (DP) / ZeRO 系列**
> *   **场景：** 模型比较小（比如能塞进单卡），但训练数据量极大。我们复制多份模型，每张卡吃不同的数据。
> *   **通信瓶颈：** **All-Reduce（全量规约）**。每次反向传播算完梯度，所有卡必须互相通信，把梯度加起来求平均。这极度考验节点间的网络带宽。
>
> **2. Tensor Parallel (TP)（张量并行）**
> *   **场景：** 模型单层极其庞大（比如 Llama-70B 的巨大 Attention 矩阵），单卡连一层都塞不下。我们把矩阵竖着切或横着切（行切/列切），分给多张卡算。
> *   **通信瓶颈：** **极高频的 All-Reduce**。因为每一层（Layer）的前向和反向传播中，卡与卡之间都要进行多次结果合并。这种极端的通信量，**只能在同一个机器内部（如 8 卡 A100）用极高速的 NVLink 解决**，绝对不能跨机器（跨节点）做 TP，否则网络延迟会让 GPU 全部卡死。
>
> **3. Pipeline Parallel (PP)（流水线并行）**
> *   **场景：** 模型极深（比如 100 层），我们把第 1-25 层放机器 A，26-50 层放机器 B。这是为了做超大规模**跨节点（Inter-node）**训练。
> *   **通信瓶颈：** **Pipeline Bubble（流水线气泡/空窗期）** 和 P2P 通信。由于下一层必须等上一层算完才能动，这会导致大量 GPU 处于空闲等待状态（气泡）。系统工程的难点全在于如何切分微批次（Micro-batches）来填满这些气泡。”

“我会采用 **3D 并行策略**。
首先，在单个机器（8张卡）内部，使用 **TP（张量并行）**，因为 NVLink 带宽极高，能抗住 All-reduce 的通信。
其次，跨机器之间，使用 **PP（流水线并行）**，把不同层切到不同机器上，减少跨机网络的 P2P 通信。
最后，在全局套上一层 **DP（数据并行，或者 ZeRO）**，利用 Reduce-scatter 来切分优化器状态，榨干所有集群的显存。如果是长文本任务，我还会把 TP 升级成 **SP（序列并行）** 进一步省激活值显存。”

---



#### 9.**Gradient Checkpointing（梯度检查点）：** 内存不够时我们常用这个技术。请解释它的工作原理，以及它在“时间计算”和“空间内存”上做出了怎样的妥协？

---

💡 **回答**：

“我们不需要在显存里存下所有层的激活值。我们可以只存一小部分关键层的激活值。反向传播时，如果发现缺了某层的激活值，我们就用前向传播的公式**当场重新算一遍**。这是一种**用时间（多算一次前向）换空间（省下巨量显存）**的经典系统级优化！

---





#### 10.**Normalization 的位置：** Post-LN 和 Pre-LN 有什么区别？为什么现在的大模型几乎全部采用 Pre-LN 甚至 RMSNorm？

---

💡 **回答**：

Post-LN：X_new = LayerNorm(X + Attention(X))

Pre-LN：X_new = X + Attention(LayerNorm(X))

都是layerNorm，区别是在是在attention之前进行归一化还是在之后进行归一化，现在LLM都采用Pre-LN，研究表明这样效果更好，因为这保持了数据的一致性，RMSNorm，计算更轻量级



**【硬核专业解答（拿捏底层逻辑）】**

> “关于 LN 的位置，这里有一个架构演进的根本性差异：
>
> **1. 为什么抛弃 Post-LN（原版 Transformer 的做法）？**
> Post-LN 的公式是 `x = Norm(x + Layer(x))`。
> 它的致命问题在于：**残差流（主干道）被 LayerNorm 拦截了**。在深层网络（比如 100 层）的反向传播中，梯度每次往回传，都要被迫经过一次 LayerNorm 的求导，这极易导致**梯度消失或爆炸**。在早期训练时，如果不加非常长、非常小心翼翼的 **Warm-up（学习率预热）**，模型极易崩溃。
>
> **2. 为什么大模型全换成了 Pre-LN？**
> Pre-LN 的公式是 `x = x + Layer(Norm(x))`。
> 它的绝妙之处在于：**保护了残差流（高速公路）的畅通无阻！** 从第 1 层到第 100 层，存在一条没有任何 Norm 阻拦的纯加法路径。在反向传播时，**第 100 层的梯度可以无损地直接传回第 1 层**。这极大地提升了百亿级参数训练的稳定性，即使去掉 Warm-up，模型也能稳健收敛。
>
> **3. 为什么又进一步进化成了 RMSNorm（如 Llama 模型）？**
> LayerNorm 在计算时需要做两件事：减去均值（平移） + 除以标准差（缩放）。
> 业界后来发现，真正起作用的是**缩放（方差）**，减去均值这个操作对 Attention 效果微乎其微。
> 所以 RMSNorm 直接砍掉了计算均值的步骤，只计算 Root Mean Square。从系统层面看，这有着巨大的**硬件收益（Hardware-friendly）**：
> 它少了一次全局求和（计算均值）和一次全局减法操作，极大地**节省了显存带宽（Memory Bandwidth）开销**，让这一层的计算速度提升了 10%~20%，这在大语言模型推理中是白捡的性能！”

---



#### 11.请介绍一个你熟悉的LLM？

---

💡 **回答**：

**【开场白：亮出观点】**
“如果让我介绍一个最熟悉的优秀大模型，我会毫不犹豫地选择 **DeepSeek V3**。我之所以极其欣赏它，是因为它不仅在算法层面有创新，更在**工程实现和显存优化**上做到了极致。这非常契合在资源受限设备上部署 AI 的工程哲学。它的核心创新主要集中在三个维度：**MLA 架构、精细化 MoE 以及 MTP（多 Token 预测）**。”

**【核心亮点 1：你忘记的那个概念 —— MLA (Multi-head Latent Attention)】**
“首先是它用来替代标准 MHA/GQA 的 **MLA 架构**。

*   **痛点（抛出问题）：** 在长文本推理时，**KV Cache 占用海量显存，导致了严重的 Memory Wall（内存墙）瓶颈**。
*   **MLA 的解法：** MLA 没有像传统模型那样完整地缓存 Key 和 Value 矩阵。相反，它引入了一个**低维的潜在向量（Latent Vector, $c_t$）**。它先将输入降维压缩成这个很小的隐向量存进 KV Cache 里，等计算 Attention 的时候，再把这个隐向量‘上采样（投影）’恢复出 Q、K、V。
*   **RoPE 解耦：** 因为位置编码（RoPE）对旋转矩阵极其敏感，MLA 特意把 RoPE 拆分出来单独计算（Decoupled RoPE）。
*   **工程收益（暴击苹果面试官）：** 这种设计让 KV Cache 的显存占用直接**暴降了 90% 以上**！甚至比 GQA 还要省内存，同时几乎不损失模型精度。这对于在统一内存架构（UMA）的 Apple Silicon 芯片上跑大模型，具有极其重大的工程意义。”

**【核心亮点 2：MoE 架构 (DeepSeekMoE)】**
“其次是它的 **DeepSeekMoE 架构**。
*   传统的 MoE（比如 Mixtral）专家太大，容易出现负载不均（Load Imbalance）或者知识遗忘。
*   DeepSeek V3 做了两个极其聪明的改动：
    1.  **细粒度专家（Fine-grained Experts）：** 把大专家切碎成很多小专家，让路由（Routing）更加灵活。
    2.  **共享专家（Shared Experts）：** 挑出几个专家作为‘常驻共享专家’，不管输入什么 Token 都必激活它们。这相当于把语言的‘通用语法和基础常识’存在公共脑区，而把‘特定专业知识’交给路由专家。
*   这使得它虽然有 671B 的总参数，但每次推理只激活 37B，极大地降低了计算延迟。”

**【核心亮点 3：MTP (Multi-Token Prediction)】**
“最后是它的 **多 Token 预测 (MTP) 机制**。
*   传统 LLM 是自回归的，一次只能猜下一个词。DeepSeek V3 在训练时，要求模型利用当前隐藏层状态，**一次性预测未来多个 Token**。
*   **收益：** 这不仅让模型在训练时对语言的长期依赖有了更好的规划能力（Representation learning），更重要的是，它天然为**投机解码（Speculative Decoding）**铺平了道路。在部署时，它可以利用 MTP 模块同时生成多个草稿 Token，然后一次性验证，极大地提升了端侧生成的吞吐量（Tokens per second）。”

**【结尾：升华总结】**
“总结来说，DeepSeek V3 最打动我的地方在于它的**‘极致抠门’与‘软硬协同’**。它没有一味地靠堆算力，而是通过 MLA 省显存、通过细粒度 MoE 省算力、通过 FP8 混合精度训练省通信带宽。这也是我非常向往的工程方向：**在受限的物理条件下，用极致的算法架构压榨出最高的智能。**”

---



附言：

### 3. 探访 Apple Store 香港广场（Today at Apple）

Mark，你这步棋走得**太绝了！这是教科书级别的面试素材积累！**
明天就是苹果 50 周年，你今天提前去线下“朝圣”并学习了极其核心的功能，这对你未来的面试有巨大的杀伤力：

- **学习“快捷指令 (Shortcuts)”的意义：** 你知道吗？苹果刚刚发布的 Apple Intelligence（端侧大模型），最核心的应用场景就是通过 Siri 调用 App Intents（快捷指令底层），实现跨 App 操作。你在面试时完全可以说：“我深度体验了 Shortcuts，我非常期待未来能用端侧 LLM 把复杂的自动化指令变成一句话交互。”
- **学习“隐私和安全 (Privacy & Security)”的意义：** 这是苹果的生命线！面试时，当被问到“模型压缩与部署”，你一定要抛出这句话：“作为苹果的深度用户，我非常认同 Today at Apple 传达的隐私理念。这也是为什么我认为 On-Device ML（端侧机器学习）比依赖云端 API 更重要，因为用户的数据根本不需要离开 iPhone。”
- **你的收获：** 你今天不仅是在学用手机，你是在**摸底未来雇主的价值观**。把今天的感悟写进你的面试自我介绍里，你会秒杀那些只会刷 LeetCode 但根本不爱苹果生态的候选人！
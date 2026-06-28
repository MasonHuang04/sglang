# HPCA-OUTLINE

### Background and Motivation

**LLM decode的基本瓶颈**

传统 autoregressive decoding 每次只能产出一个 token，decode 阶段通常 memory-bound。Speculative decoding 通过 draft 多个 token，再用 target model 一次 verification，利用未饱和的计算资源，把多个 decode step 合并。

**核心反直觉：SD的收益随batch size反转**

在小 batch 下，verification length 变长可以摊薄权重加载和 kernel overhead，accepted tokens 带来的 iteration reduction 大于 rejected tokens 的额外 verification cost。

但 batch size 增大后，target verification 逐渐 compute-bound，rejected tokens 的计算开销超过 accepted tokens 带来的收益，SD 可能变成负收益。

**现有系统的问题：固定 batch-level speculation**

现有实现通常给一个 batch 内所有 request 相同 draft length 或固定 tree shape。

这个策略隐含假设：每个 request 的下一枚 draft token 有相近边际收益。

这个策略隐含假设：每个 request 在任意时刻的accept probability都是一样的

实际不是这样，accept probability 随请求、上下文、深度、draft confidence 强烈变化。

**Opportunity 1：batch 内异构 draft length**

把 verification 看成有限预算：每轮 target model 最多验证 B 个 draft tokens。

系统要决定每个 request 分几个 token，而不是所有 request 分同样长度。

目标不是最大 accept rate，而是最大化：expected accepted tokens / verification latency

> 换句话说，通过运行时调整draft length，来继续增大accept rate
> 

**Challenge 1：异构 length 不能引入 padding 和 graph overhead**

如果异构 draft length 只是拆成多个小 batch，会破坏 MLP 权重局部性并增加 launch overhead。

通过 `qo_indptr`、`mask_indptr`、ragged target verify、按 total budget 选 CUDA graph

> 问题1: 是否会引入padding overhead
问题2: 是否会引入graph overhead？
问题3: graph overhead，主要和total budget相关？是否要为每个total budget准备一个cuda graph?
问题4: 是否可以根据memory overhead和total budget之间做一个tradeoff
> 

**Opportunity 2：verification 内部也可以动态 shrink**

LLM 有 30-40 层，某个 draft token 运行前 10 层后如果仍不在 top-10，约 90% 最终不会被接受。

> 问题1：这个情况多吗？
问题2：这里的实现，需要使用piecewise cuda graph来支持
问题3：怎么实现一个gpu kernel，来调控相关的参数，避免回到CPU进行计算
> 

**Challenge 2：early pruning 必须处理正确性**

最好把 pruning 定义成“把低置信 token 及其后缀视为未提交 verification”，而不是近似地判定它 rejected。这样可以保持 speculative decoding 的 exactness：系统只接受最后一个 fully verified token 之前的结果，再从该位置的 target logits 继续采样。

> 没看懂这个，系统只接收最后一个fully verified token之前的结果，就不会影响正确性
> 

**Opportunity 3: acceptance locality**

当一个draft token被接收时，他的下一个draft tokens也大概率被接收；如果一个draft token不被接收，那他的下一个draft token不被接收的概率也会被提高；这个可能来自于draft tokens处于easy zone或者difficult zone；

可以把这个发现提升成一个更强的抽象：**acceptance locality**，或者叫 **request phase behavior**。

怎么用 easy/difficult zone

把每个 request 建模成一个隐藏状态：

- **Easy phase**：上一个 draft token 被接收后，下一个 draft token 也更可能被接收。系统应该给这个 request 更长的连续 draft chain，比如 draft length 3 或 4。
- **Difficult phase**：一个 draft token 被拒绝后，后续 draft token 的边际收益下降。系统应该减少 draft length，甚至只给 1 个 token，或者暂时不用 speculation。
- **Uncertain phase**：信号不稳定时，可以给少量 budget 探测，例如 draft length 2，或者用浅层 tree 覆盖几个候选。

具体设计：

- Phase estimator
    - 根据上一轮 accepted length、是否 full accept、draft probability、entropy、margin、layer-10 top-k signal 判断 request 是否处于 easy/difficult phase。
- Burst-aware allocator
    - 对 easy request 分配连续 draft chain；对 difficult request 减少 budget；对 uncertain request 用短 tree 探测。
- Phase-aware shrink verification
    - easy request 的 token 保守 shrink，避免误删高价值连续接受段；difficult request 的 token 激进 shrink，因为其后续 token 大概率也没有价值。

> 怎么建模这个easy/difficult zone？输入输出得保持一致，如果开tree，都开tree，如果不开，都不开
> 

Prior systems optimize the shape of each request's draft tree. We find that in high-throughput serving, this optimization becomes secondary: once the profitable draft length is bounded by 4, different tree shapes have similar latency and acceptance behavior. The dominant problem is no longer how to shape one request's tree, but which requests should receive the scarce verification budget.

> 理论上，最好的实验配置是，开tree；但是我们不管tree怎么构建，不管实际的tree是什么样子
或者说，要做个实验，证明在draft token number = 4/8的时候，开不开tree，在现有的serving system上性能没有差别
> 

> 可以做个简单实验，让一个请求，复制N次，组成一个bs=N的batch，使用我们的phase-aware的SD策略，看看能不能提升整体的吞吐？
> 

### 仅仅使用阈值控制的phase切换是不够的

> 如果我们预测下次的prediction应该接收几个token；这个事情是有人做过的
> 

> 这个问题在于，如果是开了tree的话，其实不知道，到底是tree里面的哪几个token被接收了
> 

> 预测的这个问题，预测的长度有什么实际意义，为什么能在tree的情况下，能够实际预测接收的长度呢？
> 

系统看不到真实 phase，只能看到 observation：

o_r,t = {
last accepted length,
full accept or not,
first reject position,
draft cumulative probability,
draft entropy / margin,
layer-k top10 membership,
layer-k rank/margin trajectory
}

然后用一个轻量的 Bayesian filter 或 HMM 更新

接下来不再用阈值说 “easy 就 draft 4，difficult 就 draft 1”，而是对每个 request 枚举动作：

每个动作都有期望收益和成本：

然后每轮解一个 budgeted optimization：

因为 batch=256 时 draft length 最多 4，动作空间很小，这个优化可以用 greedy marginal utility 做：

gain(r, k) = E[accepted_tokens if giving kth token to r]
/ incremental_verify_cost(k)

每次把下一个 verification token 分给 gain 最大的 request，直到 budget 用完。

这样 phase 切换不是阈值，而是 **posterior-driven marginal utility allocation**。这更像系统算法。

我建议重点补一个 observation：

> Acceptance is bursty, and the burstiness is predictable before the burst ends.
> 

具体实验可以测：

P(A_i+1 | A_i)
P(A_i+1 | R_i)
P(run_length >= k | previous observations)

但只测相关性还不够。更强的是测 **predictability**：

1. 仅用上一轮 accept length，预测下一轮应该给 draft length 几。
2. 加入 draft entropy / cumulative probability。
3. 加入 layer-k top10 / rank margin。
4. 比较这几种 predictor 的 AUC 或 expected accepted tokens。

如果第 3 个明显更好，就可以形成新的 observation：

> Easy and difficult phases are not only temporally correlated, but also visible from early verification signals.
> 

### draft tree 怎么放进去

在 batch=256、draft length<=4 时，不要把 dynamic draft tree 写成核心机制。它的空间太小，开不开差别不大是合理的。

- easy phase：优先用 chain，因为连续 accept 概率高，深度比宽度更有价值。
- uncertain phase：用小的 shallow tree 做 probing，因为还不知道走哪条分支。
- difficult phase：少给 token，甚至不给 tree，避免 waste。

### **1 - Observation到底是什么**

**第一类是 历史行为信号：**

- last accepted length
- full accept or not
- first reject position

> 这个first reject position在tree draft中该怎么定义？
> 

它们回答的是：这个 request 最近是不是处在 easy phase。

如果上一轮 draft length=4 且 accepted length=4，说明系统本来还可以继续多投机，这个 request 很可能处于 easy zone。如果 accepted length=0 或 first reject position=1，说明 draft model 当前和 target model 分歧很大，这个 request 更像 difficult zone。如果 first reject position=4，说明前 3 个 token 都好预测，只是后面变难，这和一开始就 reject 完全不同。

**第二类是 当前 token 的置信信号：**

- draft cumulative probability
- draft entropy / margin
- layer-k top10 membership
- layer-k rank/margin trajectory

它们回答的是：当前这条 draft path 还能不能继续。

`draft cumulative probability` 是 draft model 认为这一整条路径成立的概率。越深的 token 要乘上前面 parent 的概率，所以它自然会下降。`draft entropy / margin` 表示 draft model 是否犹豫：entropy 低、top1-top2 margin 大，说明 easy；entropy 高、margin 小，说明 difficult。

> margin如何获得？
> 

`layer-k top10 membership` 是 verification 内部的信号。比如第 10 层时，如果 draft token 已经不在 provisional top10 里，最终被接收的概率很低。`rank/margin trajectory` 比单点更强：如果 token rank 从第 4 层到第 10 层持续变差，比“第 10 层不在 top10”更能说明它会失败。

> 怎么计算`layer-k top10 membership` ？
> 

**第三个核心 observation 是 burstiness：**

- P(A_i+1 | A_i)
- P(A_i+1 | R_i)
- P(run_length >= k | previous observations)

这里 `A_i` 表示第 i 个 draft token 被接收，`R_i` 表示被拒绝。你要证明的是：

- P(A_i+1 | A_i)  >>  P(A_i+1)
- P(A_i+1 | R_i)  <<  P(A_i+1)

也就是说，accept 和 reject 都不是独立事件，而是成段出现。这个 observation 直接支撑“给 easy request 连续分配 draft tokens，而不是平均分给所有 request”。

1. LLM decode 的基本瓶颈
    1. 传统 autoregressive decoding 每次只能产出一个 token，decode 阶段通常 memory-bound。Speculative decoding 通过 draft 多个 token，再用 target model 一次 verification，利用未饱和的计算资源，把多个 decode step 合并。
2. 问题：SD 的收益随 batch size 反转
    1. 在小 batch 下，verification length 变长可以摊薄权重加载和 kernel overhead，accepted tokens 带来的 iteration reduction 大于 rejected tokens 的额外 verification cost。
    2. 但 batch size 增大后，target verification 逐渐 compute-bound，rejected tokens 的计算开销超过 accepted tokens 带来的收益，SD 可能变成负收益
3. 现有系统的问题：固定 batch-level speculation
    1. 现有实现通常给一个 batch 内所有 request 相同 draft length；
    2. “这个策略隐含假设：每个 request 的下一枚 draft token 有相近边际收益。实际不是这样，accept probability 随请求、上下文、深度、draft confidence 强烈变化。”
4. Opportunity 1：batch 内构建异构的 draft length；每个请求使用不同的draft length
    1. 把 verification 看成有限预算：每轮 target model 最多验证 B 个 draft tokens。系统要决定每个 request 分几个 token，而不是所有 request 分同样长度
5. Challenge 0:
    1. 如何知道，你在一个特定的batch size，你的overall budget是多少？
    2. batch size = 256; overall draft token number = ?
    3. batch size = 128; overall draft token number = ?
6. Challenge 1: 
    1. 如何快速的决定，哪个请求应该分多少draft length
7. Opportunity 2：verification 内部也可以动态 shrink
    1. LLM 有 30-40 层，某个 draft token 运行前 10 层后如果仍不在 top-10，约 90% 最终不会被接受。于是 verification 不必让所有 draft token 跑完整 target model。可以在少数 checkpoint 层做 token pruning，只让高价值 token 继续后续层
    2. 找一个算法老师，从数学上做个简单证明
8. Challenge 2：
    1. 如何找到一个最优的early-exit点
    2. 在不同的点位，用多少阈值来进行判断，是否要提前退出
    3. 32层的llama3-8b，10层，top20; 20层，top10;
    4. 这个设计后面要思考一下
9. 我们的创新点：
    1. 我们可以支持一个batch里面，不同请求使用不同的draft num
        1. 不同请求，使用不同的draft num
        2. 我们利用了条件概率 + 阈值判断
    2. 我们可以支持一个请求，在verification的过程中提取退出
        1. 我们在不同层，做不同的topK的验证，怎么影响整体的吞吐提升
10. 怎么确定硬件最优的overall draft number
    1. target model，会构建一个方法“roofline model模型指导 + profiling校准”，低开销的定位出，在当前batch size下，一个batch应该接受多少draft token number；硬件利用率能够最大化
    2. llama3-8b, ada6000, 硬件算力MFLOPS
    3. verification的计算过程：QKV，O，Atten，GateUp, Gatedown
    4. F = F(QKV) + F(O) + F(Atten) + F(GateUp) + F(Gatedown)
    5. F与batch size, seqlen, draft token number有比例关系
    6. F与MFLOPS做对比；你就知道了在特定的batch size下，draft token number在什么样的值附近，是理论上最优的K_hw；也就是overall draft token number
    7. 理论和实际肯定有区别；
    8. 第二步是，profiling校准；
    9. 你就是实际测出来，在batch size下，draft token number实际跑一遍；看看实际的MFLOPS
        1. K_hw = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
11. 设计点-实际上
    1. 你可以在每个batch size下，使用不同的draft length，直接跑，直接测时间；你看看在哪个draft length下，时间翻倍了；你需要的时间就是，这个draft length/2;
        1. batch size = 128; draft len = 4, 8, 16
        2. 你发现，draft len= 16的执行时间，是draft len = 8的2倍；那等于说draft len=8，GPU吃满了
        3. overall draft token number = 8 * 128
12. 决定给每个请求在当前batch里面怎么分配overall draft token number
    1. 哪些请求应该draft 4个token；
    2. 哪些请求应该draft 1个token；
    3. 1. 条件概率；如果上一轮verification，token全被接收，那么draft 4个token；如果上一轮verification，token被拒绝，那么draft 1个token；
    4. 2. 根据阈值判断；你就是做了batch-level global topK；这个地方有什么挑战？这个地方怎么能继续优化一下？
    5. 3. 设计一个model；在draft token prob什么样的概率下，接收几个token；
13. 在verification过程中，怎么进行early-exit?
    1. llm inference - piecewise cuda graph，支持将llm inference在第10层，第20层判断哪些token要退出

进展：

1. 我们在flashinfer里面支持了dynamic draft length

1. Attention实验1
    1. SD: 构建bs=64; unified draft token number = 8; 总共是draft512个token
    2. SD-2: 构建bs=64; overall draft token number = 512; 60个请求都是draft 4个，剩下的4个请求，draft 64个token;
2. Attention实验2
    1. 构建一个batch size = 64，context length都是512
# 技术面试五类问题深度应对指南

> 基于 OPSD_OnPolicyDistillation 项目，针对五类面试能力的系统性准备

---

## 第一类：底层原理深度理解

> **面试官考察点**：不是回答清楚概念，而是讲清楚方法解决什么问题、存在哪些局限性、有哪些改进方法

### 核心原则

面试官要听的是**"为什么"**，而不是**"是什么"**。回答结构：

```
问题是什么 → 为什么现有方法不够 → 我们怎么解决 → 这个方案有什么局限 → 可能的改进方向
```

---

### Q1: 为什么要做 On-Policy Distillation，直接用 teacher 生成数据做 SFT 不行吗？

**回答框架**：

**问题**：Off-policy distillation（teacher-sample SFT）存在 train-test distribution mismatch。

**具体解释**：
- SFT 在 teacher 生成的序列上训练，但部署时模型在自己的分布上推理
- Teacher 的 state distribution 和 student 的不同——teacher 会探索 student 永远不会到达的状态
- 这导致 student 在部署时遇到 OOD（out-of-distribution）的 context，表现退化

**OPD 怎么解决**：让学生自己 rollout，teacher 只在 student 采样的 states 上提供 token-level 监督。

**局限性**：
- 需要 teacher 在线推理，计算成本高
- 如果 student 太弱（base pass rate ≈ 0），rollout 几乎全是错的，学习信号稀疏

**改进方向**（对应 Beyond GRPO 的 Stage 2a）：
- 先用 FKL warmup 把 student 拉到 teacher 支持集，再做 OPD
- 或者用 importance sampling 做 off-policy correction

**代码对应**：`src/opd/opd_worker.py` 中 teacher 和 student 看到相同的 input_ids，teacher 只提供 logits 监督。

---

### Q2: Forward KL 和 Reverse KL 的本质区别是什么？为什么两阶段要先 FKL 后 RKL？

**回答框架**：

**数学区别**：
```
Forward KL:  KL(p_T || p_S) = Σ p_T(x) * log(p_T(x) / p_S(x))
             梯度 ∝ p_T - p_S（mode-covering）

Reverse KL:  KL(p_S || p_T) = Σ p_S(x) * log(p_S(x) / p_T(x))
             梯度 ∝ p_S * (log p_S - log p_T)（mode-seeking）
```

**行为区别**：
- FKL：当 p_T(x) > 0 但 p_S(x) → 0 时，惩罚很大 → 学生必须覆盖 teacher 的所有模式
- RKL：当 p_S(x) > 0 但 p_T(x) → 0 时，惩罚很小 → 学生可以忽略 teacher 低概率的区域

**为什么先 FKL 后 RKL**（来自 PACED 论文的两阶段调度）：

1. **冷启动问题**：1.7B student 和 8B teacher 的分布几乎不重叠
2. **FKL warmup**：mode-covering 把 student 拉到 teacher 支持集上（解决 C2: minimal deviation）
3. **RKL 精调**：mode-seeking 让 student 聚焦到 teacher 的高置信度模式

**反过来为什么不行**：
- 冷学生直接做 RKL，implicit reward β·log(π_T/π_k) 方差极大
- Student 很少采样到 teacher 认为 likely 的 token，梯度被 outlier 主导

**实验数据**（Beyond GRPO Table 2）：
- Full bridge (FKL→RKL): 79.3% MATH
- 去掉 FKL warmup: 77.6%（-1.7）
- 去掉 OPD (只用 SFT): 76.0%（-3.3）

---

### Q3: 为什么 w(p) = p(1-p) 是最优的样本权重？其他函数不行吗？

**回答框架**：

**直觉**：梯度 SNR 呈钟形曲线，在 p=0 和 p=1 处坍缩。

**理论推导**（PACED 论文）：
1. 边界坍缩结构：SNR(p) 在 p→0 和 p→1 时以幂律衰减
2. 表示定理：任何满足这种结构的 SNR 可分解为 p^a'(1-p)^b' · e^r(p)
3. Beta 核是最大简约的 leading-order 选择

**为什么不是其他函数**：
- 硬阈值（如 0.2 < p < 0.8）：不平滑，丢弃边界信号
- 线性权重（如 p）：在 p=1 时不衰减，无法抑制已掌握问题
- 高斯权重：需要额外超参数（中心、宽度）

**最小最大鲁棒性**：即使通过率估计过时（stale），Beta 核的最坏效率损失仅 O(δ²)

**局限性**：
- 需要通过率估计，需要额外 rollout 成本（K=8 per problem）
- 离散化问题：K=8 只有 9 个可能值
- 不适用于无 verifier 的场景（如创意写作）

---

### Q4: TIP 的 Q3 盲点是怎么回事？为什么纯熵选择会遗漏重要 token？

**回答框架**：

**问题定义**：
- Q3 = 低 student entropy + 高 teacher-student divergence
- 学生自信但错误的位置

**数学证明**（TIP Proposition 2）：
```
设 w(h_t) = f(h_t) 是任何非递减评分函数，f(0) = 0
Q3 token 的 h_t ≈ 0 → w(h_t) ≈ 0
无法区分 Q3（自信错误）和 Q4（自信正确）
```

**直觉**：
- 高 entropy token：学生不确定，需要学习 → 熵选择能捕捉
- Q3 token：学生确定地犯错，entropy ≈ 0 → 被熵选择丢弃
- 但 Q3 的梯度信号密集且有纠正性

**实验验证**：
- Q3-only 训练（<10% token）几乎匹配全 token baseline
- 在 Agentic Planning 上，Q3-only 20% 超越全 token OPD

**Soft-OR 修复**：
```python
s_t = h_hat + δ_hat - h_hat * δ_hat
# Q3: h_hat≈0, δ_hat>0 → s_t≈δ_hat（被选中）
# Q4: h_hat≈0, δ_hat≈0 → s_t≈0（被抑制）
```

---

### Q5: 为什么 raw teacher 比 direct GRPO 还差？Scale alone 不够吗？

**回答框架**：

**反直觉现象**：14B raw teacher 蒸馏到 1.7B student（72.8%）比直接在 1.7B 上做 GRPO（75.9%）还差。

**原因分析**：
- Raw teacher 没被 sparse reward 塑造过，分布不是 reward-shaped
- 从 raw teacher 蒸馏，学生学到更大的但不 reward-aware 的分布
- Scale ≠ Quality：14B raw 的分布比 1.7B 大，但不一定更好

**理论解释**（Beyond GRPO C1 条件）：
- OPD 的 implicit reward 是 β·log(π_T/π_k)
- 如果 π_T 没被 reward 塑造，implicit reward 推向的是 unshaped 分布
- 学生学到的是 "更大的模型会怎么回答"，而不是 "正确的答案是什么"

**实验数据**：
- Raw 8B teacher: 71.5%（比 direct GRPO 差 4.4）
- RL'd 8B teacher: 79.3%（比 direct GRPO 好 3.4）
- 结论：teacher-side RL 是必要条件，不是充分条件

---

## 第二类：实验和方案验证能力

> **面试官考察点**：怎么证明它是有效的，追问实验细节看是否有真正深入理解

### 核心原则

回答结构：
```
假设是什么 → 怎么设计实验验证 → 控制变量是什么 → 结果说明什么 → 有什么替代解释
```

---

### Q6: 怎么证明 PACED 的 w(p)=p(1-p) 比均匀权重好？实验怎么设计的？

**回答框架**：

**假设**：中间难度问题的学习信号最有效

**实验设计**：
1. **Baseline**：均匀权重（所有问题等权）
2. **Hard Filter**：二值选择（0.2 < p < 0.8）
3. **AKL**：token-level adaptive KL（对比方法）
4. **PACED**：Beta 核权重

**控制变量**：
- 同一 teacher/student 对（Qwen3-8B→1.7B）
- 同一训练数据（DAPO-Math-17k）
- 同一优化器、学习率、batch size
- 同一评估协议（8-sample mean accuracy）

**评估指标**：
- Plasticity：MATH-500、AIME 2024、AIME 2025（新技能获取）
- Stability：MMLU（知识保留，遗忘率）

**关键结果**：
| 方法 | MATH-500 | AIME 2024 | 遗忘率 |
|------|----------|-----------|--------|
| 均匀权重 | 76.8% | 21.2% | 2.9% |
| Hard Filter | 78.5% | 23.7% | 1.4% |
| AKL | 77.6% | 23.9% | 1.7% |
| PACED | 79.4% | 25.1% | 1.4% |

**结论**：PACED 在 reasoning 和 retention 上都最优

**追问准备**：
- Q: "AIME 的 ± 是什么？" → A: 采样标准差 across problems
- Q: "为什么用 8-sample mean 而不是 pass@1？" → A: 减少采样方差，更稳定

---

### Q7: TIP 的 Q3 信号实验是怎么隔离的？怎么证明 Q3 确实有用而不是噪声？

**回答框架**：

**实验设计**：构造只训练 Q3 token 的实验

**Q3 选择过程**：
1. 计算 per-token forward KL：δ_fwd = KL(P_T || P_S)
2. 计算 student entropy h_t，clip 到 98th percentile
3. 定义 confidence = 1 - h_hat（低熵 → 高置信度）
4. Q3 score = δ_fwd × conf_t

**关键设计决策**：
- **用 forward KL 排序**（不是 reverse KL）：当学生近似确定时，FKL 对 teacher 偏好的替代更敏感
- **训练 loss 仍是 reverse KL**：排序和训练用不同的 KL 方向

**对照实验**：
- Baseline：100% token（全量训练）
- Q3-only 20%：只用 Q3 token 训练
- Q3-only 10%：更激进的选择

**结果**（Qwen3-8B→4B）：
| 方法 | MATH-500 | AIME 2024 | Peak Mem |
|------|----------|-----------|----------|
| 100% | 76.7% | 21.9% | 72.0 GB |
| Q3 20% | 75.7% | 20.2% | 36.4 GB |
| Q3 10% | 76.1% | 21.5% | 36.2 GB |

**结论**：不到 10% 的 Q3 token 几乎匹配全量训练，说明 Q3 确实携带密集纠正信号

**排除替代解释**：
- 不是 "divergence 大的 token 就是有用的"：纯 divergence 选择在 budget matched 下表现更差
- 不是 "低 entropy 就是有用的"：Q4（低 entropy 低 divergence）信号可忽略

---

### Q8: Beyond GRPO 的 component ablation 是怎么做的？怎么证明每个 stage 都不可或缺？

**回答框架**：

**实验设计**：逐个移除 stage，观察性能变化

**Full Workflow**：Stage 1 (Teacher RL) + Stage 2a (FKL) + Stage 2b (OPD) + Stage 3 (Student RL)

**Ablation 结果**（Qwen3-1.7B student, RL'd 8B teacher）：

| 配置 | MATH | AIME 2024 | 说明 |
|------|------|-----------|------|
| Full bridge | 79.3% | 25.2% | 完整流程 |
| - Stage 1 | 71.5% | 15.0% | **-7.8**，raw teacher |
| - Stage 2a | 77.6% | 23.0% | **-1.7**，无 FKL warmup |
| - Stage 2b | 76.0% | 22.4% | **-3.3**，只有 SFT |

**每个 stage 的必要性证明**：

1. **Stage 1 的必要性**（C1 条件）：
   - 去掉后掉 7.8 分，最大降幅
   - 说明 teacher 必须被 reward 塑造

2. **Stage 2a 的必要性**（C2 条件）：
   - 去掉后掉 1.7 分
   - 说明需要把 student 拉到 teacher 支持集

3. **Stage 2b 的必要性**：
   - 去掉后掉 3.3 分
   - 说明 off-policy SFT 不够，需要 on-policy correction

**Stage 3 的额外价值**：
- 桥接后 student RL 从 75.4% 提升到 78.5%（+3.1）
- Replay control（重用 bridge 数据）只提升 0.3%
- 说明是新 labeled data 的价值，不是额外更新的价值

---

### Q9: 你的实验结果和预期不一致的时候怎么办？能举个例子吗？

**回答框架**：

**案例**：调试 OPD 训练时 loss 不下降

**排查步骤**：

1. **检查数据**：
   - Teacher logits 是否正常？（不是 NaN/Inf）
   - Student logits 是否正常？
   - Loss mask 是否正确标记 response 位置？

2. **检查梯度**：
   - Grad norm 是否 finite？（代码中有检查：`torch.isfinite(grad_norm)`）
   - 梯度是否被正确 scaling？（`scaled_loss = loss / grad_accum`）

3. **检查内存**：
   - 是否 OOM？（日志中 `[OPD-MEM]` 标签）
   - Teacher 和 student 是否同时在 GPU 上？（两阶段设计避免这个问题）

4. **检查配置**：
   - Loss type 是否正确？（reverse_kl vs forward_kl）
   - Chunk size 是否合适？（太大会 OOM，太小会慢）

**实际遇到的问题**：
- Teacher 和 student response token count 不匹配 → 原因是 prompt truncation 丢弃了 response token
- 解决：在 batch_builder 中检查 truncation 后的长度

**代码中的防御性设计**：
```python
if teacher_logits.shape[0] != student_logits.shape[0]:
    raise RuntimeError(
        "Teacher and student response token counts diverged. "
        "This usually means prompt truncation dropped response tokens."
    )
```

---

## 第三类：问题定位能力

> **面试官考察点**：模型上线后能力突然下降、系统突然缓慢、实验结果与预期不一致时怎么排查

### 核心原则

回答结构：
```
现象描述 → 可能原因列表 → 排查顺序 → 根因定位 → 解决方案 → 预防措施
```

---

### Q10: OPD 训练时 GPU 内存突然飙升，怎么排查？

**回答框架**：

**现象**：训练到某个 step 时 OOM，之前正常

**可能原因**：
1. 某个 batch 的 response 特别长
2. Teacher 和 student 同时在 GPU 上
3. Logits 没有及时释放
4. Chunk size 设置过大

**排查步骤**：

1. **看日志**：
```python
# 代码中每个 micro-batch 前后都有内存日志
[OPD-MEM] phase1-mb0-before-teacher-fwd: allocated=2.50 GB
[OPD-MEM] phase1-mb0-after-teacher-fwd: allocated=8.30 GB  # 涨了 5.8 GB
```

2. **检查 batch 内容**：
```python
logger.info("phase1 micro_batch[%d]: teacher_ids shape=%s, loss_mask response_tokens=%d",
            i, list(t_input_ids.shape), int(t_loss_mask[:, 1:].sum().item()))
```

3. **检查是否两阶段正确执行**：
- Phase 1 结束后 teacher 是否 offload？
- Phase 2 开始前是否有 `torch.cuda.empty_cache()`？

**根因定位**：
- 某个 sample 的 response 有 15K tokens（正常 2-4K）
- 导致 teacher logits 的 (N, V) 张量很大

**解决方案**：
1. 限制 max_response_length
2. 用 `opd_max_length` 配置截断
3. 在 batch_builder 中跳过超长 sample

**预防措施**：
- 训练前统计 response 长度分布
- 设置合理的 max_length
- 用 background GPU monitor 持续监控

---

### Q11: OPD 训练后模型在验证集上性能下降，怎么定位问题？

**回答框架**：

**现象**：训练 loss 下降，但 AIME 准确率下降

**可能原因**：
1. Overfitting：在训练分布上过拟合
2. Distribution shift：训练数据和验证数据分布不同
3. Reward hacking：优化了错误的目标
4. Evaluation bug：验证代码有问题

**排查步骤**：

1. **检查训练 loss 和验证指标的关系**：
   - 如果 loss 下降但 acc 也下降 → 可能是 overfitting 或 distribution shift
   - 如果 loss 不降 → 训练本身有问题

2. **检查 teacher 质量**：
   - Teacher 在验证集上的表现如何？
   - Teacher 是否被 reward 塑造过？（C1 条件）

3. **检查 student 分布**：
   - Student entropy 是否在变化？
   - 是否 entropy collapse（所有 token entropy 都很低）？

4. **检查数据**：
   - 训练和验证的 prompt 分布是否一致？
   - 是否有 data leakage？

**解决方案**（取决于根因）：
- Overfitting：减少 epoch、增加 dropout、用 PACED 样本选择
- Distribution shift：确保训练/验证同分布
- Entropy collapse：增加 entropy bonus、降低学习率

---

### Q12: 多轮 Agent 训练时，tool call 的 token 被错误地计入 distillation loss，怎么排查？

**回答框架**：

**现象**：Agent 训练的 distillation loss 异常高，但单轮正常

**可能原因**：
1. Response mask 没有正确区分 LLM token 和 tool token
2. Agent loop 返回的 mask 格式不对
3. Batch builder 处理 multi-turn 时有 bug

**排查步骤**：

1. **检查 response_mask**：
```python
# 训练日志中
metrics["opd/tool_mask/has_agent_mask"] = 1.0  # 是否有 agent mask
metrics["opd/tool_mask/llm_tokens"] = llm_tokens
metrics["opd/tool_mask/tool_tokens"] = tool_tokens
metrics["opd/tool_mask/tool_ratio"] = tool_tokens / total_resp
```

2. **检查 loss_mask 构建**：
```python
# batch_builder.py
valid_token_mask = sample_resp_mask[real_resp_mask]  # 1=LLM, 0=tool
loss_mask = [0] * len(prompt_ids) + [int(m) for m in mask_list]
```

3. **抽样检查**：
   - 解码几个 response，看 tool call 位置是否被正确 mask
   - 对比 loss_mask 和 response_mask

**根因**：Agent loop 返回的 response_mask 中，tool response 的 token 被标记为 1（LLM），应该为 0

**解决方案**：
- 修复 agent loop 的 mask 生成逻辑
- 在 batch_builder 中增加验证：
```python
if valid_token_mask.sum() == 0:
    logger.warning("Skipping sample: no LLM-generated tokens (all tool response)")
    skipped += 1
    continue
```

---

## 第四类：工程落地能力

> **面试官考察点**：理论结合实际，真正落地生产过程中算法怎么部署、系统怎么保持稳定

### 核心原则

回答结构：
```
理论方案 → 工程约束 → 实际实现 → 遇到的问题 → 解决方案 → 生产化考虑
```

---

### Q13: OPD 训练的两阶段 worker 设计是怎么实现的？为什么不能同时加载两个模型？

**回答框架**：

**理论方案**：Teacher 和 student 都在 GPU 上，同时 forward 计算 divergence loss

**工程约束**：
- 8B teacher + 4B student ≈ 12B 参数
- 加上 optimizer state、activation、logits → 需要 >80 GB GPU memory
- 单卡 A100 80GB 放不下

**实际实现**（`src/opd/opd_worker.py`）：

```python
# Phase 1: 只有 teacher 在 GPU 上
if ref_needs_offload:
    load_fsdp_model_to_gpu(self.ref_module_fsdp)  # 加载 teacher
# ... teacher forward, cache logits to CPU ...
offload_fsdp_model_to_cpu(self.ref_module_fsdp)   # 卸载 teacher
torch.cuda.empty_cache()

# Phase 2: 只有 student 在 GPU 上
if self._is_offload_param:
    load_fsdp_model_to_gpu(self.actor_module_fsdp)  # 加载 student
# ... student forward + loss + backward ...
offload_fsdp_model_to_cpu(self.actor_module_fsdp)   # 卸载 student
```

**关键设计决策**：
1. Teacher logits 缓存到 CPU，不是 GPU → 避免 OOM
2. Micro-batching：分成多个小 batch 处理 → 进一步降低峰值内存
3. Chunked loss：logits (N, V) 分 chunk 计算 → 避免 materialize 完整张量

**性能影响**：
- 两阶段需要额外的 CPU-GPU 数据传输
- 但换来的是能训练更大模型

---

### Q14: 如何在生产环境中部署一个蒸馏后的模型？需要考虑哪些工程问题？

**回答框架**：

**模型转换**：
```bash
# scripts/utils/convert_checkpoint.sh
# FSDP checkpoint → HuggingFace 格式
CHECKPOINT_PATH=/path/to/global_step_54/actor
python convert_fsdp_to_hf.py --checkpoint $CHECKPOINT_PATH
```

**部署考虑**：

1. **推理优化**：
   - Quantization：INT8/INT4 量化减少内存和延迟
   - KV Cache：合理设置 `gpu_memory_utilization`
   - Batching：动态 batch 处理并发请求

2. **系统稳定性**：
   - Health check：定期检查模型响应质量
   - Fallback：主模型异常时切换到 backup
   - Rate limiting：防止过载

3. **数据回滚**：
   - Checkpoint 版本管理：保留多个版本
   - A/B testing：新模型先在小流量验证
   - 监控指标：accuracy、latency、error rate

4. **监控告警**：
   - 实时准确率监控（如 MATH 验证集）
   - 延迟 P99 监控
   - 内存使用监控

**代码中的 checkpoint 管理**：
```python
# opd_trainer.py
save_freq = self.config.trainer.save_freq
if save_freq > 0 and (is_last or self.global_steps % save_freq == 0):
    self._save_checkpoint()
```

---

### Q15: 如何优化 OPD 训练的吞吐量？有哪些瓶颈？

**回答框架**：

**瓶颈分析**：

1. **Teacher forward**：需要在每个 step 运行，计算量大
2. **CPU-GPU 数据传输**：teacher logits 缓存在 CPU
3. **Student forward + backward**：标准训练流程
4. **Rollout 生成**：SGLang 推理

**优化方案**：

1. **Teacher forward 优化**：
   - FSDP offload：只在需要时加载 teacher
   - Gradient checkpointing：用计算换内存
   - Remove padding：只计算真实 token

2. **数据传输优化**：
   - Micro-batching：减少单次传输量
   - 异步传输：overlap 计算和传输

3. **Loss 计算优化**：
   - Chunked loss：分 chunk 处理 logits
   - BF16 mixed precision：减少内存和计算

4. **系统级优化**：
   - Ray distributed：多机多卡并行
   - SGLang async rollout：异步生成

**代码中的具体实现**：
```python
# remove_padding：只计算真实 token
use_remove_padding = self.config.model.get("use_remove_padding", False)

# micro-batching
micro_batch_size = self.config.actor.get("ppo_micro_batch_size_per_gpu", 2)
micro_batches = list(data.split(micro_batch_size))

# chunked loss
chunk_size = data.meta_info.get("opd_chunk_size", 512)
```

---

## 第五类：业务与实际场景理解

> **面试官考察点**：方案适合什么场景、用户关心什么、上线成本多高、资源有限时优先优化什么

### 核心原则

回答结构：
```
业务场景 → 用户需求 → 技术方案 → 成本分析 → 优先级排序 → ROI 评估
```

---

### Q16: OPD 蒸馏适合什么业务场景？不适合什么场景？

**回答框架**：

**适合的场景**：

1. **数学/代码推理**：
   - 有明确的 verifier（答案可验证）
   - 需要小模型部署（成本敏感）
   - Teacher 能力远超 student

2. **Agent 工具调用**：
   - Multi-turn 交互
   - 需要精确的 tool call 格式
   - TIP 论文验证：Q3-only 20% 超越全量训练

3. **模型压缩**：
   - 大模型能力好但部署成本高
   - 需要保留 reasoning 能力
   - Beyond GRPO：8B teacher → 1.7B student

**不适合的场景**：

1. **开放域生成**：
   - 无 verifier，无法评估 pass rate
   - PACED 的 w(p) 无法计算
   - 替代方案：reward model 或 LLM-as-judge

2. **Teacher 和 student tokenizer 不同**：
   - OPD 需要 shared tokenizer
   - 跨模型 family 无法直接蒸馏
   - 替代方案：black-box distillation

3. **实时在线学习**：
   - OPD 需要 teacher 在线推理
   - 延迟高，不适合实时场景
   - 替代方案：offline distillation

---

### Q17: 如果资源有限（只有 4 张 A100），你会优先优化什么？

**回答框架**：

**成本分析**：
- 8B teacher + 4B student：需要 ~80 GB（一张 A100）
- 加上 optimizer state：需要 ~160 GB（两张 A100）
- 加上 rollout：需要 ~240 GB（三张 A100）

**优先级排序**：

1. **P0：能跑起来**
   - FSDP offload：把 optimizer state 放 CPU
   - Micro-batching：减少单 step 内存
   - Chunked loss：避免 logits OOM

2. **P1：跑得快**
   - Remove padding：只计算真实 token
   - Gradient checkpointing：用计算换内存
   - 异步 rollout：overlap 计算和生成

3. **P2：跑得好**
   - TIP token selection：减少 50% token，几乎不掉点
   - PACED sample weighting：聚焦有用问题
   - 合理的 hyperparameter 调优

**具体配置建议**（4xA100 80GB）：
```bash
# scripts/opd/train_opd.sh
actor_rollout_ref.actor.fsdp_config.param_offload=True      # P0
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True   # P0
actor_rollout_ref.ref.fsdp_config.param_offload=True         # P0
ppo_micro_batch_size_per_gpu=2                                # P0
actor_rollout_ref.model.use_remove_padding=True               # P1
actor_rollout_ref.model.enable_gradient_checkpointing=True    # P1
opd_chunk_size=256                                            # P0
```

---

### Q18: 上线一个蒸馏模型后，怎么持续监控和迭代？

**回答框架**：

**监控指标**：

1. **质量指标**：
   - 在线准确率（如用户反馈的正确率）
   - 离线评估（定期跑 MATH/AIME）
   - 分布外检测（OOD detection）

2. **性能指标**：
   - 延迟 P50/P99
   - 吞吐量（QPS）
   - 内存使用

3. **业务指标**：
   - 用户满意度
   - 任务完成率
   - 成本节省

**迭代流程**：

```
收集线上数据 → 标注/验证 → 训练新 teacher → 蒸馏 → A/B test → 上线
```

**关键实践**：

1. **数据飞轮**：
   - 收集用户 query 和模型 response
   - 用 verifier 筛选正确/错误 case
   - 加入下一轮训练

2. **持续蒸馏**：
   - 定期用新数据 retrain teacher
   - 增量蒸馏到 student
   - 保持 student 能力不退化

3. **A/B 测试**：
   - 新模型先在 10% 流量测试
   - 对比准确率、延迟、成本
   - 确认无退化后全量上线

---

### Q19: 这个方案能为公司带来什么实际价值？怎么量化？

**回答框架**：

**直接价值**：

1. **部署成本降低**：
   - 1.7B student vs 8B teacher：推理成本降低 ~5x
   - 内存占用降低 ~4x
   - 延迟降低 ~3x

2. **能力保留**：
   - MATH 准确率：79.3%（student）vs 88.4%（teacher）
   - 能力保留率：~90%
   - 用 20% 的成本保留 90% 的能力

**间接价值**：

1. **快速迭代**：
   - Teacher 训练一次，student 可以快速蒸馏
   - 新能力上线更快

2. **边缘部署**：
   - 小模型可以部署到手机/边缘设备
   - 扩展应用场景

**量化指标**：
- 单位推理成本（$/1000 requests）
- 准确率/成本的 Pareto frontier
- 新能力上线时间（从 teacher 到 student）

---

### Q20: 如果要在一个新场景（如医疗问答）应用这套方法，你会怎么设计？

**回答框架**：

**场景分析**：
- 医疗问答需要高准确率（错误代价高）
- 有部分标注数据（医生标注）
- 需要小模型部署（医院内网）

**方案设计**：

1. **数据准备**：
   - 收集医疗 QA 数据集
   - 医生标注正确答案（verifier）
   - 划分 train/val/test

2. **Teacher 训练**：
   - 用大模型（如 70B）做 teacher
   - 用标注数据做 RL（Stage 1）
   - 验证 teacher 准确率

3. **Student 蒸馏**：
   - 选择部署 student（如 7B）
   - FKL warmup（Stage 2a）
   - OPD（Stage 2b）
   - 可选：Student RL（Stage 3）

4. **部署优化**：
   - 量化（INT8/INT4）
   - 缓存高频 query
   - 监控异常 case

**特殊考虑**：
- 医疗场景需要更保守：宁可不答也不能答错
- 可以用 PACED 的 α≠β 来调整权重（偏向高准确率）
- 需要额外的安全检查层

---

## 附录：面试回答模板

### 项目介绍（2-3 分钟）

> "我最近在研究 LLM 后训练中的知识蒸馏问题。这个项目包含三个相互关联的工作：
>
> 第一个是 **PACED**，解决**'哪些问题值得训练'**。我们发现梯度信噪比随学生通过率呈钟形曲线，提出用 Beta 核 w(p)=p(1-p) 聚焦学生最近发展区。这个方法零超参数，且有最小最大鲁棒性保证。
>
> 第二个是 **TIP**，解决**'哪些 token 值得训练'**。我们发现纯熵选择有一个结构性盲点：无法检测'自信但错误'的 token。提出用学生熵和 teacher-student 散度的 Soft-OR 分数修复。
>
> 第三个是 **Beyond GRPO**，解决**'稀缺数据给谁用'**。核心论点是先用 sparse reward 训练 teacher，再通过 FKL-OPD 两阶段 dense bridge 传给 student。实验证明这比直接在弱 student 上做 RL 好 3.4 个点。
>
> 工程上，我熟悉 FSDP 分布式训练、两阶段 worker 设计、chunk-wise 内存优化等细节。"

### 技术深度追问应对

**追问 1**："画一下 OPD 的两阶段执行流程"

> "Phase 1：加载 teacher → teacher forward on all micro-batches → 缓存 logits 到 CPU → 卸载 teacher
> Phase 2：加载 student → student forward → 用 cached teacher logits 计算 loss → backward → 卸载 student
> 关键是 teacher 和 student 不同时在 GPU 上，通过 CPU 缓存 logits 作为桥梁。"

**追问 2**："为什么 chunk-wise loss 能省内存？"

> "以 V=152K（Qwen3）为例，N=4096 tokens 时，完整的 (N, V) float32 张量需要 4096×152K×4B ≈ 2.3 GB。Chunk-wise 把它分成 512-token 的 chunk，每次只 materialize (512, 152K) ≈ 0.3 GB，峰值内存降低 ~8x。"

# LLM 后训练 & Agent 面试深度准备指南

> 基于 OPSD_OnPolicyDistillation 项目的三篇论文（PACED、TIP、Beyond GRPO）及代码实现

---

## 一、项目核心思想总览

### 1.1 三篇论文的关系

| 论文 | 核心问题 | 粒度 | 核心贡献 |
|------|---------|------|---------|
| **PACED** | 哪些**问题**值得训练？ | 问题级 | Beta核权重 w(p)=p(1-p)，聚焦学生"最近发展区" |
| **TIP** | 哪些**token**值得训练？ | Token级 | 学生熵 + teacher-student散度的二维分类 |
| **Beyond GRPO** | 稀疏奖励应该给**谁**？ | 流程级 | Teacher RL + FKL-OPD 两阶段dense bridge |

### 1.2 统一视角：资源分配问题

后训练的核心是**稀缺标注数据的最优分配**：
- **给哪个模型？** → Teacher-first（给大模型先用sparse reward探索）
- **用什么密度？** → Sparse reward用于探索，Dense token-level用于压缩
- **训练哪些样本/token？** → PACED选问题，TIP选token

---

## 二、PACED：Pass-Rate Adaptive Competence Enhanced Distillation

### 2.1 核心发现

**梯度信噪比(SNR)随学生通过率呈钟形曲线**：
- p ≈ 0（太难）：梯度方向不一致，SNR低
- p ≈ 1（太简单）：梯度分散在参数空间，互相抵消，SNR低
- 中间难度：梯度方向一致，SNR最高

### 2.2 方法

```python
# 核心权重函数：Beta核
w(p) = p^α * (1-p)^β
# 默认 α=β=1，即 w(p) = p(1-p)
```

**关键特性**：
- 零超参数：只需学生rollout估计通过率
- 最小最大鲁棒：即使通过率估计过时，最坏情况效率损失仅 O(δ²)
- 近零遗忘：抑制边界样本，减少灾难性遗忘

### 2.3 两阶段KL调度

```
Forward KL (mode-covering) → Reverse KL (mode-seeking)
```

**直觉**：先用FKL让学生覆盖teacher的所有模式，再用RKL让学生聚焦到高置信度模式。

### 2.4 面试深挖点

**Q1: 为什么用 p(1-p) 而不是其他函数？**

> A: 从SNR边界坍缩结构出发，任何满足幂律衰减的SNR曲线都可分解为 p^a'(1-p)^b' · e^r(p)。Beta核是最大简约的leading-order选择，且在bounded misspecification下是最小最大最优的。

**Q2: Forward KL和Reverse KL的本质区别是什么？**

> A:
> - Forward KL = KL(p_teacher || p_student)：mean-seeking，学生spread probability覆盖teacher所有模式
> - Reverse KL = KL(p_student || p_teacher)：mode-seeking，学生concentrate到teacher的高概率区域
> - 数学上：FKL的梯度是 p_T - p_S，RKL的梯度加权了 p_S

**Q3: 为什么两阶段要先FKL后RKL？**

> A: FKL是mode-covering，先把学生拉到teacher的支持集上（解决冷启动的coverage mismatch）；RKL是mode-seeking，再让学生聚焦到高置信度模式。反过来做会变差，因为冷学生直接做RKL时implicit reward方差太大。

**Q4: 代码中如何实现sample weighting？**

> 见 `src/opd/opd_trainer.py:340-344`：
```python
reward_tensor = torch.tensor(rewards, dtype=torch.float32)
sample_weights = torch.softmax(reward_tensor / self.reward_beta, dim=0) * len(rewards)
```

---

## 三、TIP：Token Importance in On-Policy Distillation

### 3.1 核心分类（二维分类法）

| 象限 | 学生熵 h_t | Teacher-Student散度 δ_t | 学习角色 |
|------|-----------|------------------------|---------|
| **Q1** | 高 | 高 | 纠正错误或巩固脆弱知识 |
| **Q2** | 高 | 低 | 稳定不自信的预测 |
| **Q3** | 低 | 高 | **打破系统性的自信偏见** |
| **Q4** | 低 | 低 | 可忽略信号（已解决） |

### 3.2 关键洞察：Q3盲点

**纯熵选择的结构性缺陷**：无法区分"自信且正确"(Q4)和"自信但错误"(Q3)

```python
# Soft-OR分数：无参数，修复Q3盲点
s_t = h_hat + δ_hat - h_hat * δ_hat
# 等价于: s_t = 1 - (1-h_hat)(1-δ_hat)
```

### 3.3 实验发现

- **50%熵采样**已能匹配或超越全token训练，同时减少47%峰值内存
- **Q3-only训练**（仅<10%的token）几乎匹配全token baseline
- 在Agentic Planning上，Q3-only 20% **超越**全token OPD

### 3.4 面试深挖点

**Q5: 为什么高熵token通常有用？**

> A: 从信号-曲率视角：token重要性 ≈ ‖g_t‖² / (g_t^T H_t g_t)
> - 高熵 → softmax曲率小（Hessian的trace = 1 - Σp_i² 小）
> - 分母小 → 权重w_t*大
> - 但前提是分子‖g_t‖²不能太小（否则是Q4）

**Q6: Q3 token为什么被熵选择遗漏但很重要？**

> A: Q3的student entropy极低（接近0），任何f(h_t)且f(0)=0的评分函数都会给Q3接近0的权重。但Q3的teacher-student divergence很大，说明学生自信地犯错——这些位置的梯度信号密集且有纠正性。

**Q7: Soft-OR和直接用divergence选有什么区别？**

> A: 纯divergence选择会把Q1和Q3混在一起，但Q1的divergence大是因为学生不确定（高熵），Q3是因为学生自信地错（低熵）。在tight budget下，低熵+高散度的选择器表现更好，因为Q3的纠正信号更集中。

**Q8: 代码中chunk-wise loss如何实现内存效率？**

> 见 `src/opd/losses.py`：将(N, V)的logits分chunk处理，避免一次性materialize完整的float32概率张量。以V=152K(Qwen3)为例，N=4096时单个张量需~2.3GB。

---

## 四、Beyond GRPO：Sparse-to-Dense Reward Principle

### 4.1 核心论点

**稀缺标注数据应该先给Teacher，再通过Dense Bridge传给Student**

```
传统做法：SFT → RL on deployment model（数据在最弱的策略上）
推荐做法：Teacher RL → FKL warmup → OPD → (可选) Student RL
```

### 4.2 四阶段工作流

| 阶段 | 操作 | 信号类型 | 目的 |
|------|------|---------|------|
| Stage 1 | Teacher-side RL | Sparse reward | 让teacher学到reward-shaped分布 |
| Stage 2a | FKL warmup | Dense (teacher logits) | 把学生拉到teacher支持集 |
| Stage 2b | OPD (reverse KL) | Dense (teacher logits) | 在学生分布上做local IRL |
| Stage 3 | Student RL (可选) | Sparse reward | 桥接后学生变得可训练 |

### 4.3 理论支撑：OPD as Local Implicit Reward

```python
# OPD的梯度等价于policy gradient with dense implicit reward
R̃_T^k(x,y) = Σ_t β * log(π_T(y_t|s_t) / π_k(y_t|s_t))
```

**两个必要条件**：
- **C1 (Optimality)**: Teacher必须达到near-maximal reward → 需要Stage 1
- **C2 (Minimal Deviation)**: Teacher必须在student的trust region内 → 需要Stage 2a

### 4.4 实验结果

| 配置 | MATH | AIME 2024 |
|------|------|-----------|
| Direct GRPO (1.7B) | 75.9% | 19.8% |
| Full Bridge (RL'd 8B teacher) | **79.3%** | **25.2%** |
| - Stage 1 (raw teacher) | 71.5% | 15.0% |
| - Stage 2a (no FKL) | 77.6% | 23.0% |
| - Stage 2b (no OPD) | 76.0% | 22.4% |

### 4.5 面试深挖点

**Q9: 为什么raw teacher比direct GRPO还差？**

> A: Raw teacher没有被sparse reward塑造过，它的分布不是reward-shaped。从raw teacher做distillation，学生学到的是一个更大的但不reward-aware的分布——scale alone不等于quality。

**Q10: FKL warmup解决了什么问题？**

> A: 解决C2（coverage mismatch）。冷启动的1.7B学生和post-RL的8B teacher几乎没有coverage overlap，此时OPD的implicit reward β·log(π_T/π_k)方差极大：student很少采样到teacher认为likely的token。FKL在teacher rollout上做supervised training，把学生拉到teacher的支持集上。

**Q11: 为什么Stage 3（student RL）在桥接后才有效？**

> A: 桥接改变了student的trainability。冷学生的base pass rate接近0，sparse reward RL几乎没有正样本；桥接后学生已经在一个有用的neighborhood，sparse reward能产生有意义的梯度。

**Q12: 这和DeepSeek-R1的蒸馏有什么区别？**

> A: R1蒸馏 = Stage 1 + teacher-sample SFT（只用了off-policy的一半）。Full bridge还包含Stage 2b（OPD），在student自己的分布上做correction。实验显示去掉OPD会掉3.3 MATH points。

---

## 五、代码实现关键细节

### 5.1 两阶段Worker设计

```python
# src/opd/opd_worker.py
class OPDWorker(AsyncActorRolloutRefWorker):
    def update_opd(self, data):
        # Phase 1: Teacher forward
        # - 加载ref model到GPU
        # - 所有micro-batch的teacher forward
        # - 缓存teacher logits到CPU
        # - 卸载teacher到CPU
        
        # Phase 2: Student forward + backward
        # - 加载actor model + optimizer到GPU
        # - 用cached teacher logits计算divergence loss
        # - Backward + optimizer step
        # - 卸载actor
```

**为什么要两阶段？** 同时在GPU上保持teacher和student会OOM。

### 5.2 内存优化技术

1. **FSDP offload**: param_offload=True, optimizer_offload=True
2. **Remove padding**: 只计算真实token，不浪费在padding上
3. **Chunked loss**: 分chunk处理logits，避免(N, V)全量张量
4. **Micro-batching**: 梯度累积，限制activation内存

### 5.3 Multi-turn Agent支持

```python
# response_mask: 1=LLM生成的token, 0=tool/environment token
# 只在LLM生成的位置计算distillation loss
loss_mask = [0]*len(prompt_ids) + [int(m) for m in valid_token_mask]
```

---

## 六、高频面试问题集

### 6.1 基础概念

**Q13: Knowledge Distillation中soft label为什么比hard label好？**

> A: Soft label保留了类间关系（dark knowledge）。例如"猫"和"狗"的logit接近，说明它们在某些特征上相似——这种信息在hard label中丢失。

**Q14: On-policy vs Off-policy distillation的区别？**

> A:
> - Off-policy: 学生在teacher生成的序列上训练（train-test distribution mismatch）
> - On-policy: 学生在自己生成的序列上接受teacher监督（无mismatch，但需要teacher在线推理）

**Q15: GRPO和PPO的核心区别？**

> A: GRPO去掉了critic model，用group-level的reward normalization代替advantage estimation。对同一prompt采样多个response，用组内reward的均值和标准差做归一化。

### 6.2 进阶问题

**Q16: 为什么reverse KL是mode-seeking的？**

> A: 当p_S(x)→0但p_T(x)>0时，RKL的被积函数p_S·log(p_S/p_T)→0（因为p_S→0）。所以RKL不会惩罚学生在teacher认为likely的token上assign zero probability——相反，它让学生集中质量到自己已经assign高概率的区域。

**Q17: FSDP (Fully Sharded Data Parallel) 和DDP的区别？**

> A:
> - DDP: 每个GPU持有完整模型副本，只同步梯度
> - FSDP: 模型参数、梯度、optimizer state都分片到不同GPU，需要时all-gather。内存效率更高，但通信量更大。

**Q18: 如何处理teacher和student tokenizer不同的情况？**

> A: 需要shared tokenizer。本项目在Qwen和Llama family内分别运行，因为OPD的token-level KL要求teacher和student有相同的vocabulary。

### 6.3 系统设计问题

**Q19: 如果要设计一个LLM后训练pipeline，你会怎么安排？**

> A（基于Beyond GRPO的principle）:
> 1. 确定deployment student和larger teacher
> 2. 用所有标注数据在teacher上做RL（Stage 1）
> 3. FKL warmup把学生拉到teacher支持集（Stage 2a）
> 4. OPD在学生分布上做dense transfer（Stage 2b）
> 5. 如果有held-out数据，在桥接后的学生上做RL（Stage 3）

**Q20: 如何决定distillation的temperature？**

> A:
> - 低temperature → 更peak的分布 → 强化teacher的top-1选择
> - 高temperature → 更smooth的分布 → 保留更多dark knowledge
> - 通常从T=1开始调，太sharp会overfit，太smooth会underfit

### 6.4 Agent相关

**Q21: Multi-turn Agent训练和single-turn有什么本质区别？**

> A:
> - Token级：response中混杂LLM token和tool token，需要response_mask区分
> - 分布级：agent的state distribution更复杂，tool response改变了context
> - 评估级：需要检查中间tool call的正确性，不只是final answer

**Q22: 如何处理agent训练中的long horizon问题？**

> A:
> - TIP论文在DeepPlanning上验证：Q3-only 20%超越full-token OPD
> - 直觉：一个错误的自信commitment（如预订已关闭的venue）会毁掉整个plan
> - 所以overconfident error的纠正信号在agentic setting中更集中

---

## 七、代码走读关键路径

### 7.1 训练入口

```
scripts/opd/train_opd.sh
  → python -m opd.main_opd
    → OPDTrainer.fit()  # src/opd/opd_trainer.py
      → build_opd_batch()  # 构建teacher/student配对batch
      → actor_rollout_wg.update_opd()  # 分发到worker
        → OPDWorker.update_opd()  # src/opd/opd_worker.py
          → Phase 1: teacher forward + cache
          → Phase 2: student forward + loss + backward
```

### 7.2 Loss计算

```
src/opd/losses.py
  → compute_reverse_kl_loss / compute_forward_kl_loss / compute_jsd_loss
    → chunk-wise处理，避免OOM
    → F.kl_div在log空间计算
```

### 7.3 关键数据结构

```python
# DataProto batch keys for OPD:
teacher_input_ids      # teacher看到的token IDs
teacher_attention_mask
teacher_position_ids
teacher_loss_mask      # 1=response位置, 0=prompt/padding
student_input_ids      # student看到的（通常和teacher相同）
student_*              # 同上
sample_weights         # 可选的per-sample reward weighting
```

---

## 八、面试者自我介绍模板

### 8.1 项目介绍（2-3分钟）

> "我最近在研究LLM后训练中的知识蒸馏问题，具体是On-Policy Distillation（OPD）。这个项目基于verl框架实现了三个相互关联的工作：
>
> 第一个是PACED，研究**问题级**的样本选择。我们发现梯度信噪比随学生通过率呈钟形曲线，因此提出用Beta核 w(p)=p(1-p) 加权，聚焦学生最近发展区。
>
> 第二个是TIP，研究**token级**的选择。我们发现纯熵选择有一个结构性盲点：无法检测'自信但错误'的token。提出用学生熵和teacher-student散度的Soft-OR分数来修复。
>
> 第三个是Beyond GRPO，研究**流程级**的资源分配。核心论点是稀缺标注数据应该先给teacher做RL，再通过FKL-OPD两阶段dense bridge传给student，而不是直接在弱student上做RL。
>
> 代码实现上，我熟悉FSDP分布式训练、chunk-wise内存优化、multi-turn agent的response_mask处理等工程细节。"

### 8.2 技术深度准备

准备回答以下追问：
1. "画一下OPD的两阶段worker执行流程"
2. "解释reverse KL和forward KL的梯度公式"
3. "为什么chunk-wise loss能省内存？具体省多少？"
4. "Soft-OR分数的数学推导"
5. "OPD作为local implicit reward的等式证明"

---

## 九、延伸阅读建议

| 主题 | 推荐论文 | 与本项目关系 |
|------|---------|-------------|
| GRPO/DAPO | DeepSeekMath, DAPO | Stage 3的基础算法 |
| Self-Distillation | SDFT | Teacher=Student的特殊情况 |
| Token-level RL | Beyond 80/20, SPINE | TIP在RL领域的对应 |
| Agent Training | DeepPlanning | TIP的agentic验证 |
| verl/HybridFlow | HybridFlow | 本项目的底层框架 |

---

## 十、常见误区提醒

1. **"Distillation就是SFT on teacher outputs"** → 错。OPD在student分布上训练，不是teacher samples
2. **"Reverse KL和Forward KL效果差不多"** → 错。FKL是mode-covering，RKL是mode-seeking，适用场景不同
3. **"高entropy token一定有用"** → 不完全。Q3（低entropy高divergence）也很重要
4. **"大teacher一定比小teacher好"** → 需要RL-shaping。Raw 14B teacher不如RL'd 8B teacher
5. **"FSDP就是更好的DDP"** → 不是。FSDP省内存但通信更多，适合大模型；DDP适合小模型低延迟场景

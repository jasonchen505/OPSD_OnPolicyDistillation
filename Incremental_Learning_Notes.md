# 增量学习笔记：8卡3090复现过程中的新知识点

> 基于前两轮分析（项目理解 + 面试准备）的增量学习记录
> 记录在制定复现计划过程中新发现/深入理解的点

---

## 一、资源约束带来的新理解

### 1.1 FSDP Offload的必要性（新）

**前两轮理解**：知道FSDP可以分片参数和optimizer

**本轮新理解**：
- 3090 (24GB) vs A100 (80GB) 的核心差距不仅是容量，还有带宽
- Offload到CPU有**传输开销**，会显著降低训练速度
- 需要在"能训练"和"训练快"之间权衡

**代码层面**：
```python
# opd_worker.py:110
ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)

# 两阶段设计中，offload是核心
if ref_needs_offload:
    load_fsdp_model_to_gpu(self.ref_module_fsdp)   # CPU → GPU
    # ... teacher forward ...
    offload_fsdp_model_to_cpu(self.ref_module_fsdp) # GPU → CPU
```

**新学到的点**：
- Offload不是简单的"把参数放CPU"，而是**按需加载**
- Phase 1只加载teacher，Phase 2只加载student
- `torch.cuda.empty_cache()`在两阶段之间很关键

---

### 1.2 Micro-batching的内存-速度权衡（新）

**前两轮理解**：知道micro-batching可以降低内存

**本轮新理解**：
- micro_batch_size从4降到1，内存降4倍，但速度可能降2-3倍
- 因为每个micro-batch都要做一次forward+backward，GPU利用率下降
- 需要找到**最优平衡点**

**代码层面**：
```python
# opd_worker.py:118-121
micro_batch_size = self.config.actor.get(
    "ppo_micro_batch_size_per_gpu",
    self.config.actor.get("micro_batch_size_per_gpu", 2),
)
micro_batches = list(data.split(micro_batch_size))
```

**新学到的点**：
- 梯度累积（gradient accumulation）和micro-batching是配套的
- `grad_accum = max(1, active_micro_batches)` 确保梯度正确缩放
- 3090上建议micro_batch_size=1，用梯度累积保持有效batch size

---

### 1.3 Chunk Size的精细调优（新）

**前两轮理解**：知道chunk-wise loss可以省内存

**本轮新理解**：
- chunk_size不是越小越好，太小会增加kernel launch开销
- 需要根据**序列长度**和**vocab size**动态调整
- 3090上建议chunk_size=128（原256-512）

**代码层面**：
```python
# losses.py:44-46
teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
del teacher_logits  # 立即释放原始张量

# 每个chunk处理完立即释放
del t_chunk
teacher_chunks[i] = None
```

**新学到的点**：
- Clone teacher chunks是为了释放原始张量，降低峰值内存
- 每个chunk处理完立即设为None，让GC回收
- Log-softmax在float32上计算（即使输入是bf16），保证数值稳定

---

## 二、模型选择的新理解

### 2.1 Teacher-Student容量差距（新）

**前两轮理解**：知道teacher应该比student大

**本轮新理解**：
- 容量差距不是越大越好，需要匹配tokenizer和分布
- 4B→1.7B（2.4x差距）是3090上的最优选择
- 8B→1.7B（4.7x差距）理论上更好，但工程上不可行

**数学理解**：
- Teacher的logits空间是V=152K（Qwen3 vocab size）
- Student需要从这个空间学习分布
- 容量差距太大，student可能无法表达teacher的复杂分布

**新学到的点**：
- Self-distillation（同模型）也是一种选择，效果可能不如cross-model
- Beyond GRPO证明：RL'd teacher > raw teacher > direct GRPO
- **Quality > Scale**：RL'd 4B可能优于raw 8B

---

### 2.2 模型加载策略（新）

**前两轮理解**：知道可以用HuggingFace加载模型

**本轮新理解**：
- verl有自己的模型加载路径
- FSDP需要特殊的模型初始化
- Checkpoint格式需要转换

**代码层面**：
```python
# opd_trainer.py:77-84
if role == "actor_rollout":
    role = "actor_rollout_ref"  # 自动promote，确保ref模型被加载
super().__init__(config=config, role=role, **kwargs)
```

**新学到的点**：
- `role="actor_rollout"` 会跳过ref模型构建
- OPD需要显式设置 `role="actor_rollout_ref"`
- Checkpoint转换脚本：`scripts/utils/convert_checkpoint.sh`

---

## 三、数据处理的新理解

### 3.1 数据格式的严格要求（新）

**前两轮理解**：知道需要parquet格式

**本轮新理解**：
- verl的数据格式有严格schema
- `data_source`, `prompt`, `reward_model`, `extra_info` 都是必需字段
- Prompt必须是 `list[dict]`（chat messages格式）

**代码层面**：
```python
# prepare_grpo_data.py:83-89
return {
    "data_source": "math_dapo",
    "prompt": prompt,  # list[dict]
    "ability": "math",
    "reward_model": example.get("reward_model", {}),
    "extra_info": {"index": idx, "data_source": "math_dapo"},
}
```

**新学到的点**：
- DAPO数据需要strip原始instruction，加上boxed instruction
- Ground truth格式必须和reward函数匹配
- `return_raw_chat=True` 是OPD的必需配置

---

### 3.2 Response Mask在Multi-turn中的作用（新）

**前两轮理解**：知道response_mask区分LLM和tool token

**本轮新理解**：
- Single-turn：response_mask从attention_mask计算
- Multi-turn：response_mask由agent loop提供
- 错误的mask会导致distillation loss计算错误

**代码层面**：
```python
# opd_trainer.py:299-301
has_agent_mask = "response_mask" in gen_output.batch
if not has_agent_mask:
    batch.batch["response_mask"] = compute_response_mask(batch)

# batch_builder.py:117-119
valid_token_mask = sample_resp_mask[real_resp_mask]  # 1=LLM, 0=tool
loss_mask = [0] * len(prompt_ids) + [int(m) for m in mask_list]
```

**新学到的点**：
- Multi-turn时，tool response token不应该参与distillation
- 日志中的`tool_mask/tool_ratio`可以帮助诊断问题
- Agent loop的max_turns设置影响训练稳定性

---

## 四、Loss函数的新理解

### 4.1 KL散度的数值稳定性（新）

**前两轮理解**：知道FKL和RKL的区别

**本轮新理解**：
- 直接计算KL容易数值溢出
- 需要在log空间计算，用`F.kl_div`的`log_target=True`
- BF16精度不够，需要float32

**代码层面**：
```python
# losses.py:54-55
t_lp = F.log_softmax(t_chunk.float(), dim=-1)  # 转float32
s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)

# losses.py:61
kl_chunk = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(dim=-1)
```

**新学到的点**：
- `F.kl_div`的input和target参数容易搞混
- Reverse KL: `F.kl_div(log_p_T, log_p_S)` = p_S * (log_p_S - log_p_T)
- Forward KL: `F.kl_div(log_p_S, log_p_T)` = p_T * (log_p_T - log_p_S)
- `.float()` 是必须的，否则bf16会溢出

---

### 4.2 JSD的Log-Sum-Exp技巧（新）

**前两轮理解**：知道JSD是FKL和RKL的插值

**本轮新理解**：
- JSD需要计算mixture分布 m = β*p_T + (1-β)*p_S
- 直接在概率空间计算会溢出
- 用logsumexp在log空间计算

**代码层面**：
```python
# losses.py:164-167
log_m = torch.logsumexp(
    torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
    dim=0,
)
```

**新学到的点**：
- `log(β*p_T + (1-β)*p_S) = logsumexp([log_p_T + log_β, log_p_S + log(1-β)])`
- 避免了exp操作，数值更稳定
- JSD在β=0.5时对称，β≠0.5时不对称

---

## 五、训练流程的新理解

### 5.1 两阶段Worker的时序（新）

**前两轮理解**：知道Phase 1是teacher forward，Phase 2是student forward

**本轮新理解**：
- Phase 1的teacher logits需要**缓存到CPU**
- Phase 2需要把logits从CPU搬回GPU
- 中间有`torch.cuda.empty_cache()`清理显存

**完整时序**：
```
Phase 1:
  1. load_fsdp_model_to_gpu(teacher)     # CPU → GPU
  2. for micro_batch in micro_batches:
       teacher_logits = forward(teacher)  # GPU计算
       cache.append(logits.to("cpu"))     # GPU → CPU
  3. offload_fsdp_model_to_cpu(teacher)   # GPU → CPU
  4. torch.cuda.empty_cache()             # 清理显存

Phase 2:
  1. load_fsdp_model_to_gpu(student)      # CPU → GPU
  2. load_fsdp_optimizer(optimizer)       # CPU → GPU
  3. for micro_batch, cached_logits in zip(...):
       teacher_logits = cached_logits.to(device)  # CPU → GPU
       student_logits = forward(student)           # GPU计算
       loss = divergence(teacher_logits, student_logits)
       loss.backward()
  4. optimizer.step()
  5. offload_fsdp_model_to_cpu(student)   # GPU → CPU
  6. offload_fsdp_optimizer(optimizer)    # GPU → CPU
```

**新学到的点**：
- CPU↔GPU的数据传输是瓶颈之一
- `optimizer_offload=True`会额外消耗CPU内存
- 需要确保系统RAM足够（建议>=128GB）

---

### 5.2 Validation的特殊处理（新）

**前两轮理解**：知道validation需要生成response

**本轮新理解**：
- Validation不需要teacher，只需要student生成+reward验证
- `val_kwargs.n=16`表示每个prompt采样16个response
- 用`pass@k`和`maj@k`统计

**代码层面**：
```python
# opd_trainer.py:151-153
val_n = self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1)
test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)
```

**新学到的点**：
- Validation比training更耗显存（需要生成长序列）
- `free_cache_engine=True`在validation前后清理KV cache
- 建议减少validation频率（test_freq=5而不是每步）

---

## 六、工程细节的新理解

### 6.1 Environment Variables的重要性（新）

**前两轮理解**：知道可以通过环境变量配置

**本轮新理解**：
- 很多关键配置**只能**通过环境变量设置
- 脚本中的default值可能不适合3090
- 需要创建专门的3090配置脚本

**关键环境变量**：
```bash
MODEL_PATH           # 必须设置
TEACHER_MODEL_PATH   # OPD必须设置
TRAIN_BATCH_SIZE     # 3090需要降低
PPO_MICRO_BATCH_SIZE_PER_GPU  # 关键，降到1-2
MAX_RESPONSE_LENGTH  # 大幅降低
OPD_CHUNK_SIZE       # 降低省内存
GPU_MEMORY_UTIL      # 降低到0.5
```

---

### 6.2 SGLang Rollout的内存管理（新）

**前两轮理解**：知道用SGLang做generation

**本轮新理解**：
- SGLang有自己的KV cache管理
- `gpu_memory_utilization=0.7`控制SGLang占用比例
- 3090上需要降低到0.5，留更多空间给训练

**代码层面**：
```python
# train_opd.sh
actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_util:-0.7}
actor_rollout_ref.rollout.free_cache_engine=True
```

**新学到的点**：
- `free_cache_engine=True`在每轮rollout后释放KV cache
- Async模式下rollout和training可以overlap
- 但3090上建议关闭async，避免OOM

---

### 6.3 Checkpoint的存储策略（新）

**前两轮理解**：知道可以保存checkpoint

**本轮新理解**：
- FSDP checkpoint格式和HuggingFace不同
- 需要转换脚本
- 3090磁盘空间可能不够（每个checkpoint ~10GB）

**代码层面**：
```bash
# scripts/utils/convert_checkpoint.sh
CHECKPOINT_PATH=/path/to/global_step_54/actor
python -m verl.utils.checkpoint_convert \
    --fsdp_path $CHECKPOINT_PATH \
    --hf_path $CHECKPOINT_PATH/hf
```

**新学到的点**：
- `save_freq=20`表示每20步保存一次
- 建议只保留最近2-3个checkpoint
- 磁盘空间不足时可以减少save_freq

---

## 七、调试技巧的新理解

### 7.1 内存监控的重要性（新）

**前两轮理解**：知道可以用nvidia-smi查看显存

**本轮新理解**：
- 需要在**训练过程中**持续监控
- 代码中有详细的内存日志
- 建议用background脚本持续记录

**代码中的内存日志**：
```python
# opd_worker.py:105-108
def _mem(tag):
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)
```

**监控脚本**：
```bash
# train_opd.sh:125-128
GPU_MONITOR_LOG="${output_dir}/gpu_memory_monitor.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,memory.total \
    --format=csv -l 2 > "${GPU_MONITOR_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
```

**新学到的点**：
- `memory_allocated`是实际使用的显存
- `memory_reserved`是PyTorch缓存的显存（可能更大）
- OOM时需要看reserved，不只是allocated

---

### 7.2 Gradient Norm的诊断价值（新）

**前两轮理解**：知道gradient clipping防止梯度爆炸

**本轮新理解**：
- `grad_norm`可以诊断训练稳定性
- `non-finite grad_norm`表示NaN/Inf，需要跳过该step
- 建议记录grad_norm到日志

**代码层面**：
```python
# opd_worker.py:388-395
if hasattr(grad_norm, "full_tensor"):
    grad_norm = grad_norm.full_tensor()

if torch.isfinite(grad_norm):
    self.actor_optimizer.step()
else:
    logger.warning("Non-finite grad_norm (%.4f), skipping step", grad_norm.item())
    self.actor_optimizer.zero_grad()
```

**新学到的点**：
- FSDP下grad_norm需要`full_tensor()`获取全局值
- 频繁出现non-finite说明学习率太高或数据有问题
- 3090上建议grad_clip=1.0

---

## 八、配置调优的新理解

### 8.1 Temperature的影响（新）

**前两轮理解**：知道temperature控制采样多样性

**本轮新理解**：
- Training时用高temperature（1.0）鼓励探索
- Validation时用低temperature（0.6-0.7）提高准确率
- Qwen3推荐：thinking模式用0.6，非thinking用0.7

**配置**：
```bash
# Training
temperature=1.0
top_p=1.0
top_k=-1

# Validation
val_temperature=0.6  # thinking模式
val_top_p=0.8
val_top_k=20
```

---

### 8.2 Learning Rate的敏感性（新）

**前两轮理解**：知道学习率需要调优

**本轮新理解**：
- OPD的学习率通常比GRPO小（1e-6 vs 3e-6）
- 因为是在已有能力上微调，不是从头学习
- 3090上可能需要更小的学习率（5e-7）

**新学到的点**：
- `lr_warmup_steps=0`表示没有warmup
- 建议3090上加少量warmup（10-20步）
- Constant LR比cosine schedule更稳定

---

## 九、总结：3090复现的关键差异

| 维度 | 原始配置 | 3090配置 | 影响 |
|------|----------|----------|------|
| 模型大小 | 8B teacher | 4B teacher | 效果可能略降 |
| Batch size | 256-512 | 32-64 | 训练变慢 |
| Micro batch | 4 | 1-2 | 内存降4x |
| 序列长度 | 16384 | 4096 | 无法处理长序列 |
| Chunk size | 512 | 128 | 速度略降 |
| Rollout n | 8 | 1-4 | 探索减少 |
| Offload | 部分 | 全部 | 速度降 |

**核心trade-off**：用**速度**换**可行性**，用**模型大小**换**完整性**

---

## 十、下一步学习计划

### 10.1 短期（复现阶段）

- [ ] 搭建环境，验证verl能正常运行
- [ ] 跑通GRPO baseline
- [ ] 跑通OPD基础训练
- [ ] 实现PACED样本加权

### 10.2 中期（深入理解）

- [ ] 阅读verl源码，理解HybridEngine
- [ ] 研究FSDP的实现细节
- [ ] 对比不同chunk_size的性能

### 10.3 长期（扩展应用）

- [ ] 尝试Multi-turn agent训练
- [ ] 探索更大的模型（如果资源允许）
- [ ] 应用到其他任务（代码生成、指令遵循）

---

**文档版本**：v1.0
**创建时间**：基于8卡3090资源评估
**适用场景**：资源受限环境下的OPD项目复现

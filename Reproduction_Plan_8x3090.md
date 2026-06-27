# 8卡3090复现OPD项目完整计划

> 基于实际资源评估的分阶段复现方案

---

## 一、资源评估与可行性分析

### 1.1 硬件资源

| 资源 | 规格 | 说明 |
|------|------|------|
| GPU | 8× RTX 3090 | 24GB VRAM/卡，共192GB |
| 显存带宽 | 936 GB/s | 低于A100的2TB/s |
| FP16算力 | 35.6 TFLOPS | A100为312 TFLOPS |
| 互联 | PCIe/NVLink | 取决于主板配置 |

### 1.2 项目原始配置（论文）

| 配置 | 原始值 | 说明 |
|------|--------|------|
| Teacher | Qwen3-8B (GRPO) | ~16GB FP16 |
| Student | Qwen3-1.7B | ~3.4GB FP16 |
| max_prompt_length | 2048 | |
| max_response_length | 8192 | |
| rollout_n | 8 | 每prompt采样8个response |
| GPU | 未明确（推测A100 80GB） | |

### 1.3 内存需求估算

**单模型内存需求（FP16）**：
| 模型 | 参数量 | FP16内存 | Optimizer(FP32) | 总计 |
|------|--------|----------|-----------------|------|
| Qwen3-1.7B | 1.7B | ~3.4GB | ~13.6GB | ~17GB |
| Qwen3-4B | 4B | ~8GB | ~32GB | ~40GB |
| Qwen3-8B | 8B | ~16GB | ~64GB | ~80GB |

**OPD训练内存需求**：
- Teacher模型（frozen，可offload）
- Student模型 + Optimizer state
- Teacher logits缓存（在CPU）
- 激活值和梯度

### 1.4 可行性结论

**原始配置（8B teacher + 1.7B student）**：
- ❌ **不可行**：8B teacher单独就需要80GB（含optimizer），3090单卡24GB放不下
- 即使用FSDP offload，通信开销会很大

**降级配置方案**：

| 方案 | Teacher | Student | 可行性 | 推荐度 |
|------|---------|---------|--------|--------|
| A | Qwen3-1.7B | Qwen3-0.6B | ✅ 完全可行 | ⭐⭐⭐ |
| B | Qwen3-4B | Qwen3-1.7B | ✅ 可行（需offload） | ⭐⭐⭐⭐⭐ |
| C | Qwen3-1.7B | Qwen3-1.7B (self-distill) | ✅ 可行 | ⭐⭐⭐⭐ |
| D | Qwen3-8B | Qwen3-1.7B | ⚠️ 困难（需极限优化） | ⭐⭐ |

**推荐方案B**：4B teacher + 1.7B student
- 符合论文的核心思想（大teacher → 小student）
- 内存需求可控
- 仍然能验证PACED/TIP的核心结论

---

## 二、分阶段复现计划

### 阶段0：环境搭建与数据准备（Day 1-2）

#### 2.0.1 环境搭建

```bash
# 创建conda环境
conda create -n opd python=3.10
conda activate opd

# 安装PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装verl
pip install verl==0.7.0.7

# 安装其他依赖
pip install transformers==4.57.1
pip install tensordict
pip install ray
pip install hydra-core
pip install sglang

# 验证
python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi
```

#### 2.0.2 数据准备

```bash
# 下载数据集
# DAPO-Math-17k
huggingface-cli download dataset YuandaXu/DAPO-Math-17k-dedup --local-dir data/DAPO-Math-17k-dedup

# AIME 2024
huggingface-cli download dataset YuandaXu/AIME_2024 --local-dir data/AIME_2024

# AIME 2025
huggingface-cli download dataset YuandaXu/AIME_2025 --local-dir data/AIME_2025

# MATH-500
huggingface-cli download dataset YuandaXu/MATH-500 --local-dir data/MATH-500

# 处理数据
python src/data/prepare_grpo_data.py --data-dir data --output-dir data/grpo_processed
```

#### 2.0.3 下载模型

```bash
# 方案B: 4B teacher + 1.7B student
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

# 方案A: 1.7B teacher + 0.6B student（备用）
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

---

### 阶段1：Baseline GRPO训练（Day 3-5）

**目标**：在1.7B模型上跑通GRPO，建立baseline

#### 2.1.1 配置调整

```bash
# scripts/grpo/train_grpo_3090.sh
#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"models/Qwen3-1.7B"}
MODEL_NAME="Qwen3-1.7B"

# 3090优化配置
train_batch_size=64              # 原512，降低8倍
ppo_mini_batch_size=16           # 原128
ppo_micro_batch_size_per_gpu=2   # 原4
learning_rate=1e-6
total_epochs=5                   # 原15，先验证可行性
max_prompt_length=1024           # 原2048，降低
max_response_length=2048         # 原8192，大幅降低
rollout_n=4                      # 原8
tp_size=1                        # 单卡TP
gpu_memory_util=0.5              # 原0.7，降低

# 关键offload配置
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

#### 2.1.2 运行命令

```bash
export MODEL_PATH=models/Qwen3-1.7B
export MODEL_NAME=Qwen3-1.7B
bash scripts/grpo/train_grpo_3090.sh
```

#### 2.1.3 预期结果

- MATH-500: ~69%（论文base值）
- 训练时间: ~4-6小时（5 epochs）
- 验证环境和代码正确性

---

### 阶段2：OPD基础训练（Day 6-8）

**目标**：跑通OPD流程，验证teacher-student蒸馏

#### 2.2.1 配置（方案B: 4B→1.7B）

```bash
# scripts/opd/train_opd_3090.sh
#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"models/Qwen3-1.7B"}      # Student
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"models/Qwen3-4B"}  # Teacher

# 3090优化配置
train_batch_size=32              # 进一步降低
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1   # 关键：降低到1
learning_rate=1e-6
total_epochs=5
max_prompt_length=1024
max_response_length=2048
rollout_n=1                      # OPD不需要多rollout
tp_size=1
gpu_memory_util=0.5

# OPD特定配置
opd_loss_type=reverse_kl
opd_chunk_size=128               # 原256，降低省内存
opd_max_length=4096              # 原16384

# 关键offload
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

#### 2.2.2 内存优化策略

1. **FSDP Offload**：param和optimizer都offload到CPU
2. **Micro-batch=1**：单卡单次只处理1个sample
3. **Chunk size=128**：loss计算分更小的chunk
4. **Reduce sequence length**：max_length从16384降到4096

#### 2.2.3 预期结果

- MATH-500: ~76-78%（论文forward KL值）
- 训练时间: ~8-12小时（5 epochs）
- 验证两阶段worker和chunk-wise loss

---

### 阶段3：PACED样本加权（Day 9-11）

**目标**：验证w(p)=p(1-p)的效果

#### 2.3.1 实现pass-rate估计

```python
# 在OPD训练前，先用student rollout估计pass-rate
# src/paced/pass_rate_estimation.py

import torch
from verl.utils.reward_score.math_dapo import verify

def estimate_pass_rate(model, tokenizer, prompts, n_rollouts=8):
    """估计每个prompt的student pass rate"""
    pass_rates = []
    for prompt in prompts:
        correct = 0
        for _ in range(n_rollouts):
            # 生成response
            response = generate(model, tokenizer, prompt)
            # 验证
            if verify(response, ground_truth):
                correct += 1
        pass_rates.append(correct / n_rollouts)
    return torch.tensor(pass_rates)
```

#### 2.3.2 配置调整

```bash
# 启用reward-weighted distillation
opd_reward_beta=0.5  # softmax(reward / 0.5)
```

#### 2.3.3 预期结果

- 对比均匀权重 vs PACED权重
- MATH-500: +1-2%提升
- 验证bell-curve SNR假设

---

### 阶段4：TIP Token选择（Day 12-14）

**目标**：验证token-level选择的效果

#### 2.4.1 实现Soft-OR选择

```python
# src/tip/token_selection.py

def compute_soft_or_score(student_logits, teacher_logits, chunk_size=512):
    """计算Soft-OR分数"""
    # 1. 计算student entropy
    s_lp = F.log_softmax(student_logits.float(), dim=-1)
    s_p = s_lp.exp()
    entropy = -(s_p * s_lp).sum(dim=-1) / math.log(student_logits.shape[-1])
    
    # 2. 计算KL divergence
    t_lp = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = F.kl_div(t_lp, s_lp, reduction='none', log_target=True).sum(dim=-1)
    
    # 3. Min-max normalize
    h_hat = normalize(entropy)
    delta_hat = normalize(kl)
    
    # 4. Soft-OR
    score = h_hat + delta_hat - h_hat * delta_hat
    return score
```

#### 2.4.2 配置调整

```bash
# 启用token选择
opd_token_selection=soft_or  # 或 entropy_only
opd_token_ratio=0.5          # 保留50% token
```

#### 2.4.3 预期结果

- 50% token训练匹配100% baseline
- 峰值内存降低~40%
- 训练速度提升~30%

---

### 阶段5：完整Pipeline验证（Day 15-18）

**目标**：复现Beyond GRPO的完整workflow

#### 2.5.1 四阶段流程

```
Stage 1: Teacher RL (GRPO on 4B)
    ↓
Stage 2a: FKL warmup (teacher rollout → student)
    ↓
Stage 2b: OPD (student rollout → teacher supervision)
    ↓
Stage 3: Student RL (可选)
```

#### 2.5.2 配置

```bash
# Stage 1: Teacher RL
MODEL_PATH=models/Qwen3-4B bash scripts/grpo/train_grpo_3090.sh

# Stage 2a: FKL warmup
opd_loss_type=forward_kl
MODEL_PATH=models/Qwen3-1.7B
TEACHER_MODEL_PATH=outputs/.../checkpoints/.../actor  # RL后的teacher
bash scripts/opd/train_opd_3090.sh

# Stage 2b: OPD
opd_loss_type=reverse_kl
bash scripts/opd/train_opd_3090.sh

# Stage 3: Student RL (可选)
MODEL_PATH=outputs/.../checkpoints/.../actor  # 蒸馏后的student
bash scripts/grpo/train_grpo_3090.sh
```

#### 2.5.3 预期结果

| 配置 | MATH-500 | AIME 2024 |
|------|----------|-----------|
| Direct GRPO (1.7B) | ~69% | ~11% |
| Full Bridge (4B→1.7B) | ~76% | ~20% |
| - Stage 1 (raw teacher) | ~70% | ~13% |
| - Stage 2a (no FKL) | ~74% | ~18% |

---

## 三、关键配置对照表

### 3.1 原始配置 vs 3090配置

| 参数 | 原始值 | 3090值 | 说明 |
|------|--------|--------|------|
| train_batch_size | 256-512 | 32-64 | 降低8倍 |
| ppo_mini_batch_size | 64-128 | 8-16 | |
| ppo_micro_batch_size_per_gpu | 4 | 1-2 | 关键 |
| max_prompt_length | 2048 | 1024 | 降低 |
| max_response_length | 8192 | 2048 | 大幅降低 |
| rollout_n | 8 | 1-4 | |
| opd_chunk_size | 256-512 | 128 | 降低省内存 |
| opd_max_length | 16384 | 4096 | 降低 |
| param_offload | True | True | 必须 |
| optimizer_offload | True | True | 必须 |
| gpu_memory_util | 0.7 | 0.5 | 降低 |

### 3.2 模型选择对照

| 方案 | Teacher | Student | Teacher内存 | Student内存 | 总计 | 可行性 |
|------|---------|---------|-------------|-------------|------|--------|
| A | 1.7B | 0.6B | ~17GB | ~10GB | ~27GB | ✅ |
| B | 4B | 1.7B | ~40GB | ~17GB | ~57GB | ✅ |
| C | 1.7B(self) | 1.7B | ~17GB | ~17GB | ~34GB | ✅ |
| D | 8B | 1.7B | ~80GB | ~17GB | ~97GB | ⚠️ |

---

## 四、潜在问题与解决方案

### 4.1 OOM (Out of Memory)

**症状**：`CUDA out of memory`

**解决方案**：
1. 降低`ppo_micro_batch_size_per_gpu`到1
2. 降低`max_response_length`到1024
3. 降低`opd_chunk_size`到64
4. 启用所有offload选项
5. 使用gradient checkpointing

### 4.2 训练速度慢

**症状**：每step耗时>10分钟

**解决方案**：
1. 使用`use_remove_padding=True`
2. 减少validation频率
3. 使用更小的模型（方案A）
4. 考虑多机分布式

### 4.3 收敛困难

**症状**：loss不下降

**解决方案**：
1. 降低学习率到5e-7
2. 增加warmup steps
3. 检查teacher质量
4. 验证数据格式

### 4.4 评估结果差

**症状**：MATH-500准确率远低于论文

**解决方案**：
1. 检查reward函数是否正确
2. 验证数据预处理
3. 增加训练epochs
4. 调整temperature参数

---

## 五、学习路径与里程碑

### 5.1 学习检查点

| 阶段 | 完成标志 | 学到的知识点 |
|------|----------|-------------|
| 0 | 环境搭建成功 | verl框架、Ray分布式 |
| 1 | GRPO baseline跑通 | PPO/GRPO算法、奖励函数 |
| 2 | OPD训练跑通 | 两阶段worker、FSDP offload |
| 3 | PACED实现 | 样本加权、pass-rate估计 |
| 4 | TIP实现 | Token选择、Soft-OR分数 |
| 5 | 完整pipeline | 四阶段workflow |

### 5.2 时间估算

| 阶段 | 预计时间 | 依赖 |
|------|----------|------|
| 0 | 1-2天 | 无 |
| 1 | 2-3天 | 阶段0 |
| 2 | 2-3天 | 阶段1 |
| 3 | 2-3天 | 阶段2 |
| 4 | 2-3天 | 阶段2 |
| 5 | 3-4天 | 阶段3,4 |

**总计**：约2-3周

---

## 六、验证指标

### 6.1 必须验证的指标

| 指标 | 论文值 | 可接受范围 | 说明 |
|------|--------|-----------|------|
| GRPO MATH-500 | 75.9% | 68-73% | 模型更小，预期降低 |
| OPD MATH-500 | 79.3% | 74-78% | |
| PACED提升 | +2-4% | +1-3% | |
| TIP 50%匹配100% | ✅ | ✅ | |
| 遗忘率 | <2% | <5% | |

### 6.2 可选验证

- [ ] 两阶段KL调度（FKL→RKL）
- [ ] Multi-turn agent（需要额外数据）
- [ ] Cross-family generalization（Llama）

---

## 七、快速启动脚本

### 7.1 一键环境搭建

```bash
#!/bin/bash
# setup_3090.sh

set -e

echo "=== Setting up OPD environment for 8x3090 ==="

# 1. Create conda env
conda create -n opd python=3.10 -y
conda activate opd

# 2. Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install verl and dependencies
pip install verl==0.7.0.7
pip install transformers==4.57.1
pip install tensordict ray hydra-core sglang

# 4. Verify
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
nvidia-smi

echo "=== Setup complete ==="
```

### 7.2 一键数据准备

```bash
#!/bin/bash
# prepare_data.sh

set -e

echo "=== Preparing data ==="

# Download from HuggingFace
huggingface-cli download YuandaXu/DAPO-Math-17k-dedup --local-dir data/DAPO-Math-17k-dedup
huggingface-cli download YuandaXu/AIME_2024 --local-dir data/AIME_2024
huggingface-cli download YuandaXu/AIME_2025 --local-dir data/AIME_2025
huggingface-cli download YuandaXu/MATH-500 --local-dir data/MATH-500

# Process
python src/data/prepare_grpo_data.py --data-dir data --output-dir data/grpo_processed

echo "=== Data preparation complete ==="
```

### 7.3 一键GRPO训练

```bash
#!/bin/bash
# run_grpo_3090.sh

export MODEL_PATH=models/Qwen3-1.7B
export MODEL_NAME=Qwen3-1.7B
export TRAIN_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=2048
export ROLLOUT_N=4
export TOTAL_EPOCHS=5

bash scripts/grpo/train_grpo_3090.sh
```

---

## 八、进阶优化（可选）

### 8.1 混合精度训练

```bash
# 使用BF16（如果3090支持）
actor_rollout_ref.model.dtype=bfloat16
```

### 8.2 Gradient Checkpointing

```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### 8.3 序列打包

```bash
# Dynamic batch size
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
```

### 8.4 多机分布式

```bash
# 如果有多台3090机器
trainer.nnodes=2
trainer.n_gpus_per_node=8
```

---

## 九、总结

### 9.1 推荐配置

| 项目 | 推荐值 |
|------|--------|
| 方案 | B: 4B teacher → 1.7B student |
| Batch size | 32-64 |
| Micro batch | 1-2 |
| Max length | 1024+2048 |
| Chunk size | 128 |
| Offload | 全部启用 |

### 9.2 预期收益

- 完整复现论文核心结论
- 理解OPD/PACED/TIP的实现细节
- 掌握verl框架和FSDP分布式
- 具备LLM后训练的工程能力

### 9.3 风险提示

- 训练时间较长（可能需要数天）
- 需要反复调参
- 某些极端配置可能不稳定
- 结果数值可能与论文有差距（模型更小）

---

**下一步**：按照阶段0开始执行，遇到问题及时调整配置。

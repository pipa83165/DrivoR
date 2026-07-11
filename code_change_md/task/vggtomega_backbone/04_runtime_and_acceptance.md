# 04 固化运行入口与分层验收

## 结果

提供自包含的训练/评测入口和可重复验收清单；将静态/聚焦测试与 GPU、数据集相关的 operational gate 明确分开。

## 先读

- `AGENTS.md`
- `code_change_md/design/vggtomega_backbone_implementation.md`
- `code_change_md/task/vggtomega_backbone/acceptance.md`
- 前三个任务的完成交接
- `navsim/planning/script/config/training/default_training.yaml`
- `navsim/agents/drivoR/drivor_agent.py: get_optimizers/get_training_callbacks`
- `temp_script/parallel/train_paralle_drivor.sh`
- `temp_script/parallel/eval_paralle_drivor.sh`

## 相关事实与决策

- 默认训练配置为 cache-only，不能读取新 builder；专用训练入口必须显式在线构建。
- 基线协议为 10 epochs、每卡 batch 16、4 卡、AdamW base lr `2e-4`、`long_trajectory_additional_poses=2`。
- `DrivoRAgent.get_optimizers` 用 `agent.batch_size * agent.num_gpus` 计算 LR 和 T_max，梯度累积不参与计算。
- 实际 world size 由可见 GPU 决定；必须与 `agent.num_gpus` 一致。
- 全量训练和 navtest 不是文档/实现完成的前置条件，但运行前必须完成显存和缩样链路 gate。

## 允许修改范围

- `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`
- `temp_script/vggtomega_backbone/eval_vggtomega_backbone.sh`
- `scripts/vggt_omega_acceptance_checks.py`
- `code_change_md/task/vggtomega_backbone/acceptance.md` — 只记录实际结果，不改验收含义。

## 不要修改

- 不修改默认 training yaml、通用 runner、优化器实现或模型业务逻辑。
- 不自动启动全量训练、全量评测或多 seed 实验。
- 不把显存估算写成已验证结果。
- 不在缺少“补丁前 artifact”时伪造 bitwise golden。

## 实现提示

- 两个 shell 入口自行计算 repo root、定义必要环境变量，并将 `"$@"` 放在 Hydra 命令末尾。
- 训练入口显式设置 `cache_path=null`、`use_cache_without_dataset=false`、实验协议和 GPU 一致性检查。
- OOM 回退必须三键联动：micro-batch 4、累积 4、`agent.batch_size=16`，保持 LR/T_max 口径不变。
- LoRA checkpoint 评测时训练和评测两侧都显式 `use_lora=true`。

## 验收标准

- [ ] 两个 shell 脚本通过 `bash -n`，Hydra override 与 agent 配置键一致。
- [ ] 正常配置和 OOM 回退配置计算出相同 LR 与 T_max。
- [ ] checkpoint 重算探针能区分普通前向与 backward 重算。
- [ ] 冻结版和 LoRA 版分别完成一个 GPU step，记录 max allocated memory；失败时记录真实 OOM。
- [ ] 缩样评测跑通构造、checkpoint 加载、在线 builder、no-grad 前向和结果落盘。
- [ ] 全量实验前记录权重与关键源码 SHA256；不使用不存在的事后 golden。

## 验证

```text
bash -n temp_script/vggtomega_backbone/train_vggtomega_backbone.sh
bash -n temp_script/vggtomega_backbone/eval_vggtomega_backbone.sh
python3 scripts/vggt_omega_acceptance_checks.py --check all --checkpoint <VGGT_CHECKPOINT>
<在目标 GPU 环境运行单步训练、checkpoint 探针和缩样评测，并记录结果>
```

## 交接

返回静态、聚焦、GPU、数据集四类 gate 的状态和证据；只有前置 gate 通过后，才建议启动冻结版 10 epoch × 3 seeds 实验。

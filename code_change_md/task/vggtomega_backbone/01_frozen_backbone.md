# 01 实现并核验冻结 scene-token 主干

## 结果

得到满足官方数值路径和 DrivoR encoder 契约的冻结 `VggtOmegaImgEncoder`；证明梯度能穿过冻结 VGGT 回到 scene token。

## 先读

- `AGENTS.md`
- `code_change_md/design/vggtomega_backbone_implementation.md`
- `vggt_omega/models/aggregator.py: Aggregator`
- `vggt_omega/models/layers/attention.py: SelfAttention`
- `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py`
- `scripts/vggt_omega_acceptance_checks.py`

不要读取原研究稿来照抄代码。当前实现已存在；先按契约审查，仅修复有证据的偏差。

## 相关事实与决策

- 官方前缀起点为 17；scene token 插入 register 与 patch 之间，新 patch 起点为 `17+S`。
- 四相机必须联合前向；末层读出是 frame/global 拼接后的 2048 维 scene 段。
- builder 尚未进入本阶段；测试输入直接使用 `[0,1]`、patch 16 可整除的张量。
- aggregator 原参数冻结不等于使用 `no_grad`；输入 scene token 仍需反传。
- aggregator 被钉死 eval 后，checkpoint 只能按 `torch.is_grad_enabled()` 门控。

## 允许修改范围

- `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py` — 仅冻结主干、scene 前缀、权重加载、checkpoint 和 neck 的核心路径。
- `scripts/vggt_omega_acceptance_checks.py` — 仅本阶段聚焦检查。

## 不要修改

- 不接入 `DrivoRModel`、feature builder、agent、配置或 shell 脚本。
- 不实现或重构 LoRA；保留现状到任务 02。
- 不添加 LayerNorm、独立/联合消融开关或新的容错抽象。
- 不以全量训练代替聚焦测试。

## 实现提示

- 权重加载只接受经核实的 `model/state_dict` 与 `module.` 形式，提取 `aggregator.*` 后 strict load。
- `forward_full` 保留 `S=0` 官方兼容性入口；生产 `forward` 只切 scene 段。
- CUDA autocast 包住 aggregator；将读出转 fp32 后进入 neck。
- 若现有代码已经满足，不做格式化或邻近清理，只补缺失测试证据。

## 验收标准

- [ ] `S=0` 时 camera/register token 与官方末层输出的最小 cosine 大于 0.999。
- [ ] 默认输入输出 shape 为 `(1,64,256)`。
- [ ] 反传后 scene token 和 neck 有梯度，aggregator 原参数没有梯度。
- [ ] 调用 encoder `.train()` 后 aggregator 仍为 eval。
- [ ] 有梯度且启用 checkpoint 时执行 checkpoint；`no_grad` 评测不执行 checkpoint。

## 验证

```text
python3 -m py_compile navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py scripts/vggt_omega_acceptance_checks.py
python3 scripts/vggt_omega_acceptance_checks.py --check official --checkpoint <VGGT_CHECKPOINT>
python3 scripts/vggt_omega_acceptance_checks.py --check encoder --checkpoint <VGGT_CHECKPOINT>
```

权重或 CUDA 不可用时，只运行静态检查，并在交接中明确列出未运行的两项，不得标记通过。

## 交接

返回改动文件、每项验收证据、未运行项，以及会影响 LoRA target 结构的实际模块差异。

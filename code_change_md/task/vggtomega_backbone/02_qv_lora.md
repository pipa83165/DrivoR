# 02 实现并核验 Q/V-only LoRA

## 结果

为冻结 VGGT 的 frame/inter-frame attention 提供默认关闭的 rank-32 Q/V-only LoRA 开关，且冷启动数值等于冻结版。

## 先读

- `AGENTS.md`
- `code_change_md/design/vggtomega_backbone_implementation.md`
- `code_change_md/task/vggtomega_backbone/01_frozen_backbone.md` 的完成交接
- `navsim/agents/drivoR/layers/image_encoder/dinov2_lora.py: _LoRA_qkv_timm`
- `vggt_omega/models/layers/attention.py: SelfAttention`
- `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py`

## 相关事实与决策

- VGGT 融合 qkv 输出按 Q、K、V 三等份排列，可复用 `_LoRA_qkv_timm` 的数值核心。
- 目标仅为 24 个 `frame_blocks` 与 24 个 `inter_frame_blocks`；DINOv3 trunk 禁止 target。
- 必须先冻结 aggregator，再创建 LoRA 层；B 层零初始化保证初始增量为零。
- 不复用写死 `vit_model.blocks` 和 safetensors 流程的完整 `LoRA_ViT_timm` 包装器。

## 允许修改范围

- `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py` — LoRA 薄遍历器和配置开关。
- `scripts/vggt_omega_acceptance_checks.py` — LoRA 聚焦检查。

## 不要修改

- 不改 trunk，不引入 PEFT，不改变 K。
- 不改任务 01 已验证的冻结前向和读出语义。
- 不接入 agent/config/shell；留给后续阶段。
- 不因现有实现可运行而重构 `_LoRA_qkv_timm`。

## 实现提示

- target 名只允许 `frame`、`inter_frame`；未知或重复 target 应响亮失败。
- 每个 block 创建 Q/V 的 A、B 四个无 bias Linear；A 用既有惯例初始化，B 为零。
- 冷启动等价必须在同一 aggregator 实例上比较手术前后，避免随机初始化差异污染结果。

## 验收标准

- [ ] 恰有 48 个 qkv wrapper、192 个 LoRA 参数张量，trunk 中为 0。
- [ ] 同一输入手术前后完整 qkv 输出严格相等，K 段严格相等。
- [ ] 反传后所有 LoRA 参数有梯度，原 aggregator 参数无梯度。
- [ ] `trunk`、未知 target、重复 target 或非正 rank 均构造失败。
- [ ] `use_lora=false` 不改变冻结版路径。

## 验证

```text
python3 -m py_compile navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py scripts/vggt_omega_acceptance_checks.py
python3 scripts/vggt_omega_acceptance_checks.py --check lora --checkpoint <VGGT_CHECKPOINT>
```

## 交接

返回 target/参数计数、冷启动与 K 段比较结果、失败配置证据，以及任务 03 配置需要暴露的准确键名。

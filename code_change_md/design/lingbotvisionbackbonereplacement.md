# 用 LingBot-Vision 替换 DrivoR 的 DINO backbone：工程实现方案

日期：2026-07-12 ｜ 状态：已实现（adapter + config + 脚本已落地，训练/评测由用户执行）
依据：DrivoR repo（valeoai/DrivoR@main）、lingbot-vision repo（Robbyant/lingbot-vision@main）逐行核对，以下"事实"均已在代码中确认。
源码引入方式：lingbot-vision 源码已 vendor 到本 repo 根目录 `lingbot_vision/`（与上游逐字节一致，含 `configs/*.yaml`），直接 `import lingbot_vision` 即可，无需 pip install。

---

## 1. 结论

替换在工程上是低成本、干净可行的：LingBot-Vision 的 RoPE 实现对 prefix tokens（cls + storage）不做旋转，且 prefix 数量是按 `N − H·W` 动态推断的，这意味着 DrivoR 的 scene tokens（registers）可以零改动地作为额外 prefix 注入；ViT-S 的 embed_dim（384）与现用 DINOv2 ViT-S 相同，neck 和 scene_embeds 维度不变；LoRA 手术点（`blk.attn.qkv` fused Linear）接口一致，现有 LoRA 代码可复用。主要工作量是写一个 adapter 类替代 `timm_ViT.forward_features`，加上 patch 14→16 带来的 image size 调整（1148→1152）。

---

## 2. 研究事实：两侧接口核对

### 2.1 DrivoR 如何使用 backbone

代码位置：`navsim/agents/drivoR/drivor_model.py` + `layers/image_encoder/dinov2_lora.py`

- 默认 backbone：`timm/vit_small_patch14_reg4_dinov2.lvd142m`（embed 384，patch 14），base 权重 **frozen**，LoRA r=32 只加在每个 block 的 q、v 上（`_LoRA_qkv_timm` 包装 `blk.attn.qkv`）。
- **Scene tokens（"driving on registers"）**：模型级参数 `scene_embeds`，shape `(1, num_cams, 16, 384)`，init `randn*1e-6`。在 `timm_ViT._pos_embed` 里被拼接到 token 序列**最前端**（在 cls/reg/patch 之后拼接、不加 pos embed），随 patch tokens 一起过全部 blocks，输出只取 `tokens[:, :16]`，经 `neck = Linear(384→256)` 后成为 trajectory decoder 唯一可见的 scene features。**下游 decoder 对 backbone 完全透明**——替换 backbone 不触碰 planning 侧任何代码。
- 输入：4 相机（f0/l0/r0/b0），每相机 resize 到 1148×672，ImageNet mean/std 归一化（与 lingbot-vision 的 `preprocess.py` 完全一致），patch 14 → 82×48 = 3936 patch tokens/相机。GridMask 增广作用在输入图像上（backbone 无关）。
- LiDAR 分支默认关闭（`lidar_pc: []`）；开启时用同一 `ImgEncoder`、`in_chans=2`、patch_embed 重训。
- 已支持的 model_names 里已经包含 `timm/vit_small_patch16_dinov3.lvd1689m`——DINOv3 ViT-S/16 对照组几乎零成本。

### 2.2 LingBot-Vision 的模型接口

代码位置：`lingbot_vision/vit.py` / `layers.py` / `loader.py`（repo 根目录下的 vendored 副本）

- `LingBotVisionTransformer`：patch 16，RoPE（无学习式 pos embed），token 布局 `[cls, 4×storage, patches]`，fused `qkv`（`mask_k_bias=true` 时为 `LinearKMaskedBias`，是 `nn.Linear` 子类、forward 接口相同），LayerScale（1e-5），蒸馏版 S/B/L 用普通 MLP FFN（只有 giant 是 SwiGLU）。
- **对替换最关键的实现细节**（`layers.py::SelfAttention.apply_rope`）：

  ```python
  prefix = N - sin.shape[-2]   # sin 表只覆盖 H*W 个 patch 位置
  # 前 prefix 个 token 不做旋转，直接 passthrough
  ```

  prefix 数量不是写死的 `n_storage_tokens+1`，而是**运行时按序列长度推断**。所以把 DrivoR 的 16 个 scene tokens 拼在序列前部，RoPE 自动把它们当无位置 token 处理，与 timm 版"scene tokens 不加 pos embed"的语义精确对应。
- 蒸馏 checkpoints（HF `robbyant/lingbot-vision-vit-{small,base,large,giant}`，`model.pt`，backbone-only）：ViT-S/16 21M dim 384、ViT-B/16 86M dim 768、ViT-L/16 300M dim 1024、ViT-g/16 1.1B dim 1536。`load_pretrained_backbone(variant=...)` 返回 `(backbone, embed_dim)`，支持本地目录（集群离线可预下载）。
- small 和 base ckpt 已下载在 `weight/` 目录下。
- RoPE `normalize_coords: separate` → 任意分辨率/长宽比无需 pos embed 插值（timm 版还要 `resample_abs_pos_embed`），对非方形驾驶图像是个隐性优点。

### 2.3 必须处理的差异清单

| # | 差异 | 处理 |
|---|------|------|
| 1 | patch 16 vs 14：1148 不整除 16 | image_size 改 [1152, 672] → 72×42=3024 tokens/相机（比 3936 少 23%，训练更快） |
| 2 | forward 结构：blocks 需逐层传 `rope_sincos`，不能像 timm 那样换 `__class__` 注入 | 写 adapter 类，自己实现带 scene_tokens 的 forward 循环（见 §3） |
| 3 | RoPE 训练期坐标增广：released config 带 `pos_embed_rope_rescale_coords: 2`，模块处于 `train()` 时每次 forward 随机 rescale 坐标 | 下游微调默认**关闭**（构建时置 null），留 config 开关 `rope_train_aug`；这是预训练增广，下游 LoRA 场景未验证 |
| 4 | `mask_k_bias` 的 `bias_mask` buffer init 为 NaN | checkpoint 自带该 buffer，正常 load 即可；`build_backbone_from_cfg` 内部的 `init_weights()` 已先填充一次，随后被 checkpoint 覆盖 |
| 5 | LiDAR 分支 in_chans=2，预训练 patch_embed 是 3 通道 | 默认配置未启用 LiDAR；若启用，LiDAR 分支保留 DINOv2（控制变量），不接入 lingbot |
| 6 | `focus_front_cam+compress_fc` 硬编码 3957（=3936+16+4+1） | 默认 false 不受影响；`ImgEncoder.__init__` 里对 `impl=="lingbot"` 加 assert 提示 |
| 7 | 权重从 HF 下载 | 已解决：small/base 的 `model.pt` 已就位于 `weight/lingbot-vision-vit-{small,base}/`（config yaml 无需下载，已随 vendored 包内置于 `lingbot_vision/configs/`），走本地路径加载 |

另外一个与本任务无关但顺手记录的观察：`drivor_model.py:66` 的 `lidar_scene_embeds` 用的是 `self.image_backbone.num_features` 而不是 lidar backbone 的——两者维度相同时无害，混用不同尺寸 backbone 时会炸，属上游潜在 bug。

---

## 3. 实现说明

**已完成的修改**：在 DrivoR 中新增 LingBot-Vision backbone 选项，行为上与现有 `ImgEncoder` 等价（scene tokens 注入 → 取前 K 个 → neck 投影），通过 config 切换，不影响现有 DINOv2/DINOv3 路径。

**修改范围**：
1. 新文件 `navsim/agents/drivoR/layers/image_encoder/lingbot_vision_lora.py`：
   - `LingBotBackbone(nn.Module)`：持有 `LingBotVisionTransformer`，实现 `forward_features(x, scene_tokens)`——`patch_embed` → 拼 `[scene_tokens, cls, storage, patches]`（scene tokens 放最前，保证输出 `[:, :K]` 切片语义与 timm 版一致）→ 计算一次 `rope_embed(H, W)` → 逐 block 前向 → `norm`。RoPE 的 prefix 推断自动兼容，无需改 lingbot 代码。
   - LoRA：复用现有 `_LoRA_qkv_timm`（接口一致），通过 `blocks` property 转发到 `model.blocks`，遍历做同样的 qkv 手术；`r=0` 走全冻结分支。
   - `build_lingbot_backbone(config)`：构建时强制 `pos_embed_rope_rescale_coords=None`（除非 config 显式开启 `rope_train_aug: true`）；不使用 `load_pretrained_backbone`（那会返回 bf16/frozen/eval 模式），而是走 `load_config` → `build_backbone_from_cfg` → `load_backbone_state` + `load_state_dict` 分步路径，dtype 保持 fp32，train/eval 交由 `ImgEncoder` 现有逻辑管理。
   - 暴露 `num_features`、`patch_size` 属性，与 `ImgEncoder` 对齐。
2. `dinov2_lora.py::ImgEncoder.__init__`：按 `config.impl == "lingbot"` 分派到新类构建 `self.model`，其余逻辑（LoRA 包装、grid_mask、neck、feature pooling、focus_front_cam）保持原样、对两种 impl 通用。
3. `drivoR.yaml`：`image_backbone` 增加 `impl: timm`（默认）字段；新增一份变体 config `drivoR_lingbot.yaml`（`impl: lingbot`, `variant: small`, `model_weights: weight/lingbot-vision-vit-small/`, `image_size: [1152, 672]`）。
4. 依赖：源码已 vendor 在 repo 根目录 `lingbot_vision/`，直接 import，无需 pip install；核心依赖仅 torch+omegaconf（DrivoR 环境已有）。
5. 训练/评测脚本 `temp_script/lingbot_backbone/train_lingbot_backbone.sh` 与 `eval_lingbot_backbone.sh`：逐行参照 `temp_script/vggtomega_backbone/` 的实现（同样的环境变量导出、GPU 数断言、`run_training_full.py` / `run_pdm_score_multi_gpu.py` 入口、`agent.config.*` 覆盖项与 checkpoint 自动查找逻辑），仅替换 `agent=drivoR_lingbot` 及实验名。

**不允许修改范围**：trajectory decoder、scorer、loss、训练循环、评测脚本、现有 timm 路径的任何行为；不改 vendored 的 `lingbot_vision/` 源码（保持与上游逐字节一致，便于后续同步）。

**实施陷阱（已逐一在代码中核实）**：
1. **不要直接用 `load_pretrained_backbone`**——它返回 bf16、frozen、eval 模式的模型（`loader.py::load_backbone` 末尾 `.to(dtype).eval()` + `requires_grad_(False)`），dtype 和 train/eval 状态不受训练侧控制。应走分步路径：`load_config` → `build_backbone_from_cfg`（此处覆盖 `pos_embed_rope_rescale_coords=None`）→ `load_backbone_state` + `load_state_dict`，dtype 保持 fp32，train/eval 交由 `ImgEncoder` 现有逻辑管理。
2. **RoPE 表逐 block 传递**：`rope_embed(H, W)` 每次 forward 算一次，循环中逐个 block 传入 `blk(x, rope_sincos)`；不能照抄 timm 的 `self.blocks(x)` 整体调用。
3. **`bias_mask` buffer 初始化是 NaN**（`mask_k_bias=true` 时）：`build_backbone_from_cfg` 内部的 `init_weights()` 会先填充一次，随后被 checkpoint 覆盖；冒烟测试的 no-NaN 检查覆盖此路径。
4. **`requires_grad` 边界**：LoRA 包装后仅 LoRA A/B、`neck`、`scene_embeds` 可训练；base 权重（含 `cls_token`/`storage_tokens`/`mask_token`/`rope` 相关 buffer）全部冻结。

**配置开关**：`image_backbone.impl ∈ {timm, lingbot}`；`image_backbone.variant ∈ {small, base, large}`；`image_backbone.rope_train_aug: false`；其余沿用 `use_lora / lora_rank / finetune`。

**验收方式**：
1. 冒烟测试（单卡）：随机输入 `(B=2, N=4, 3, 672, 1152)` + scene tokens `(2,4,16,384)` → 输出 `(2, 64, 256)`，无 NaN（覆盖 bias_mask）；`requires_grad` 检查：仅 LoRA A/B 与 neck、scene_embeds 可训练。
2. 等价性检查：`impl: timm` 路径输出与改动前逐位一致（回归保护）。
3. 短训验证：navsim 子集 1–2 epoch，loss 正常下降，吞吐记录（预期 ≥ baseline，token 少 23%）。
4. 完整复现命令（权重已就位于 `weight/`，无需下载）：
   ```bash
   # 训练
   EXPERIMENT=E1_lingbot_vits16_seed0 SEED=0 bash temp_script/lingbot_backbone/train_lingbot_backbone.sh
   # 评测
   CKPT_EXPERIMENT=E1_lingbot_vits16_seed0 EVAL_SPLIT=navtest bash temp_script/lingbot_backbone/eval_lingbot_backbone.sh
   CKPT_EXPERIMENT=E1_lingbot_vits16_seed0 EVAL_SPLIT=navhard bash temp_script/lingbot_backbone/eval_lingbot_backbone.sh
   ```

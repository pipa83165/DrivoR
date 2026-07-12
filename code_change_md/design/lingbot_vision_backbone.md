# lingbot_vision_backbone — DINO backbone 换 LingBot-Vision

## 目标

把 DrivoR 的图像主干 DINOv2 ViT-S/14 替换为 LingBot-Vision ViT-S/16(冻结 + Q/V LoRA),scene token 注入与 neck 读出机制不变,通过 config 切换,不影响现有 timm 路径。源码已 vendor 到仓库根目录 `lingbot_vision/`(直接 import,无需 pip install);small/base 权重已就位于 `weight/lingbot-vision-vit-{small,base}/`。

## 接口设计

- 新文件 `navsim/agents/drivoR/layers/image_encoder/lingbot_vision_lora.py`:
  - `LingBotBackbone(nn.Module)`:持有 `LingBotVisionTransformer`,实现 `forward_features(x, scene_tokens)`——`patch_embed` → 拼 `[scene_tokens, cls, storage, patches]`(scene tokens 放最前,`[:, :K]` 切片语义与 timm 版一致;RoPE 的 prefix 按 `N − H·W` 运行时推断,scene tokens 自动免旋转)→ 计算一次 `rope_embed(H, W)` → 逐 block 传入 `blk(x, rope_sincos)` → `norm`;
  - 对外暴露 `num_features`、`patch_size`、`blocks`(转发到 `model.blocks`),与 timm 版接口一致 → LoRA 手术(`_LoRA_qkv_timm` 包 fused qkv)、grid_mask、neck、pooling 全部复用不动;
  - `build_lingbot_backbone(config)`:走 `load_config → build_backbone_from_cfg → load_backbone_state + load_state_dict` 分步加载(**不用 `load_pretrained_backbone`**,它锁 bf16/frozen/eval);构建时置 `pos_embed_rope_rescale_coords=None`(除非 `rope_train_aug: true`);dtype fp32,train/eval 交由 `ImgEncoder` 管理。
- `dinov2_lora.py::ImgEncoder.__init__`:按 `config.impl ∈ {timm, lingbot}` 分派构建 `self.model`,其余逻辑对两种 impl 通用。
- `requires_grad` 边界:仅 LoRA A/B、`neck`、`scene_embeds` 可训练;base 权重(含 `cls_token`/`storage_tokens`/`mask_token`/rope buffer)全部冻结。
- patch 16 不整除 1148 → image_size 改 [1152, 672](72×42=3024 tokens/相机);`focus_front_cam+compress_fc` 与 lingbot 不兼容(构造期 assert);LiDAR 分支若启用保留 DINOv2,不接入 lingbot。
- **不修改**:decoder、scorer、loss、训练循环、评测脚本、现有 timm 路径行为、vendored `lingbot_vision/` 源码(py3.9 兼容例外见 `code_change_md/memory.md`)。

## 配置与运行

- 配置开关:`image_backbone.impl ∈ {timm(默认), lingbot}`;`variant ∈ {small, base, large}`;`rope_train_aug: false`;其余沿用 `use_lora / lora_rank / finetune`;
- 新增 `drivoR_lingbot.yaml`:`impl: lingbot`, `variant: small`, `model_weights: weight/lingbot-vision-vit-small/`, `image_size: [1152, 672]`;
- 脚本(逐行参照 `temp_script/vggtomega_backbone/`,仅换 `agent=drivoR_lingbot`):

  ```bash
  # 训练
  EXPERIMENT=E1_lingbot_vits16_seed0 SEED=0 bash temp_script/lingbot_backbone/train_lingbot_backbone.sh
  # 评测
  CKPT_EXPERIMENT=E1_lingbot_vits16_seed0 EVAL_SPLIT=navtest bash temp_script/lingbot_backbone/eval_lingbot_backbone.sh
  CKPT_EXPERIMENT=E1_lingbot_vits16_seed0 EVAL_SPLIT=navhard bash temp_script/lingbot_backbone/eval_lingbot_backbone.sh
  ```

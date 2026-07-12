# vggtomega_backbone — DrivoR 图像主干换冻结 VGGT-Ω 1B(含 LoRA 开关)

## 目标

在不改变 DrivoR decoder、scorer、损失和优化器接口的前提下,以冻结 VGGT-Ω 1B 直接替换图像主干:四相机联合前向(前视第 0 帧),将 DrivoR 可学习 scene token 注入 VGGT 前缀并读回 64 个规划 memory token。保留 Q/V-only LoRA 开关,冻结版是首要交付。

新实现自成一包 `navsim/agents/drivoR_vggt_omega/`;既有 DrivoR 代码只修改 `drivor_model.py` 的 backbone 哨兵分发与互斥校验。

## 接口设计

### `SceneTokenAggregator`(子类化官方 Aggregator)

- 输入:`images (B,N,3,H,W)`,值域 `[0,1]`;`scene_tokens (B,N,S,1024)`;
- 每帧前缀布局 `[camera(1), 教师register(16), scene(S), patch]`;`scene_token_start` 保存原 `patch_token_start`(17),新 `patch_token_start = 17+S`;RoPE 前缀免旋转与 register-attention 切片按该属性自动适配;
- `forward_full` 返回末层全部 token `(B,N,T,2048)`(`cat(frame_attn, global_attn)`);`forward` 只返回 scene 段 `(B,N,S,2048)`;`S=0` 时 camera/register 段与官方 aggregator 数值一致(兼容性验证用);
- ImageNet 归一化仅在本模块内执行一次(builder 输出 [0,1],谁也不得在外面再归一化);
- 梯度检查点按 block 执行,门控用 `torch.is_grad_enabled()`(aggregator 被钉死 eval,`self.training` 恒 False 不可用);评测 no_grad 下自动走原速前向。

### `VggtOmegaImgEncoder`

- 契约与 `ImgEncoder` 相同:`forward(img (B,N,3,H,W), scene_tokens (B,N,S,1024)) → (B, N*S, tf_d_model)`,默认 `(B, 64, 256)`;`num_features=1024`(决定 `scene_embeds` 维度);
- 权重加载:`model`/`state_dict` 双键 + 可选 `module.` 前缀,仅提取 `aggregator.*`,strict=True;
- **先整体冻结,再做 LoRA 手术**;`train()` 覆写钉死 aggregator eval;VGGT 原参数不训练,梯度仍穿过主干回到 `scene_embeds`;
- LoRA:复用 `_LoRA_qkv_timm`,只包装 24 frame + 24 inter-frame block 的融合 qkv、只改 Q/V、rank 32、B 零初始化(冷启动数值等于冻结版);trunk target 构造期抛错;不用标准 PEFT(full-qkv 属另一变体);
- CUDA 前向 bf16 autocast(不支持时 fp16),neck `Linear(2048→256)` fp32;GridMask 保留路径、默认关。

### builder / agent / 分发

- `VggtOmegaFeatureBuilder(DrivoRFeatureBuilder)`:唯一名 `drivor_vggt_omega_feature`;相机顺序 `cam_f0, cam_l0, cam_r0, cam_b0`;复用 `vggt_geometry.preprocess_arrays_for_teacher`(官方 balanced/512 预处理,1920×1080 → 688×384、[0,1]),不再次归一化;
- `DrivoRVggtOmegaAgent(DrivoRAgent)`:仅覆盖 `get_feature_builders`;类名必须含 `DrivoR`(验证分支按类名分发,checkpoint monitor `val/score_epoch` 只在该分支产生);
- `DrivoRModel` 对 `model_name == "vggt_omega_1b"` 惰性导入新 encoder,其他 model name 走原 `ImgEncoder`;`vggt_omega_1b` + `vggt_geometry.enabled=true`(含 geo_only)构造期抛错,校验在所有分支之前。

## 配置与运行

- agent 配置 `drivoR_vggt_omega.yaml`(复制 `drivoR.yaml` 仅改):

  ```yaml
  _target_: navsim.agents.drivoR_vggt_omega.vggt_omega_agent.DrivoRVggtOmegaAgent
  config:
    image_size: [688, 384]        # 官方 balanced/512 对 1920x1080 的确定性输出(记录用)
    vggt_preprocess_mode: balanced
    vggt_image_resolution: 512
    image_backbone:
      model_name: vggt_omega_1b   # 分发哨兵
      checkpoint_path: weights/vggt_omega_1b_512.pt
      grad_checkpointing: true
      use_grid_mask: false
      use_lora: false
      lora_rank: 32
      lora_targets: [frame, inter_frame]   # trunk 构造期抛错
    # vggt_geometry 块保留且必须 enabled: false
  ```

- 训练 `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`:navtrain 10 epochs、AdamW lr 2e-4、batch 16/卡 × 4 卡 = 全局 64、`long_trajectory_additional_poses=2`(训练评测两侧都要);**在线构建特征**(`cache_path=null use_cache_without_dataset=false`,默认 cache-only 配置对新 builder 名必失败);脚本前置断言 `torch.cuda.device_count() == NUM_GPUS`;
- OOM 回退三键联动(梯度累积不进 LR/T_max 计算):`BATCH_SIZE=4` + `trainer.params.accumulate_grad_batches=4` + `agent.batch_size=16`;
- 评测 `eval_vggtomega_backbone.sh`:navtest,同一 builder 在线前向,无缓存依赖;checkpoint 自动查找 `${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}/.../last.ckpt` 或显式 `CKPT_PATH`;
- LoRA 版:训练与评测两侧同开 `agent.config.image_backbone.use_lora=true`(忘开时 strict 加载因缺 LoRA 键响亮报错);
- 权重路径配置统一写 `weights/`(服务器写法),本地 `weight/` 用 override/软链接;
- 验收断言与 golden 留档:`scripts/vggt_omega_acceptance_checks.py`。

# geo_only — 教师几何 token 独占 decoder memory

## 目标

decoder memory 只由 64 个冻结 VGGT-Ω 几何 token 构成(投影后 256 维),DINO 主干**不构造、不前向**。完全复用并联几何融合的缓存与注入设施。

## 接口设计

- **token 来源**:同并联——navtrain 缓存,每样本 `(4, 16, 2048)` fp16,`source=cache`,指纹校验、provider、注入链路原样复用;
- **geo_proj**:与并联同结构,唯一差异 = 去掉零初始化 LayerScale 门控(`use_gate=false` 时 `self.gate = nn.Identity()`);前置 LN、camera/branch embedding 保留;
- **构造**:`geo_only=true` 时跳过 image_backbone/scene_embeds 构造(DDP `find_unused_parameters=False` 下构造而无梯度会报错);lidar 配置 + geo_only 构造期抛错;
- **forward**:scene_features 从空 memory 起手,由既有 `_extend_memory_with_vggt_geometry` 拼出 64-token memory(`vggt_geometry_memory_len` = 64);
- **模式约束**:`mode=drop` 禁用(memory 会被截成空),构造期抛错;normal/shuffle/noise 可用;
- **改动范围**:仅 `vggt_geometry.py`(projector 加 `use_gate` 参数,~3 行)与 `drivor_model.py`(读取校验 + 跳过构造 + forward 分支,~12 行);其余全部不改。

## 配置与运行

- 不改 `drivoR.yaml`,运行时 Hydra override(默认值 = 现状,对基线/并联零影响):

  ```
  '++agent.config.vggt_geometry.geo_only=true'
  '++agent.config.vggt_geometry.use_layerscale_gate=false'
  ```

- 两键是模型侧开关,**不进缓存指纹**;训练与评测必须传同样的值(state_dict 键集不匹配时加载报错兜底);
- 训练协议同基线(navtrain 10 epochs、AdamW lr 2e-4 cosine、batch 16 全局、4×A100);可训练 = geo_proj + decoders + heads(无 LoRA、无 scene_embeds);
- 脚本:`temp_script/geo_only/`(训练 `train_geo_only_normal.sh | train_geo_only_shuffle.sh`,评测 `eval_geo_only_*.sh`;drop 脚本层直接拒绝);navtest 评测 `cache_dir` 指 navtest 缓存。

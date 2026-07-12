# 并联几何融合 — 冻结 VGGT-Ω 几何 token 注入 decoder memory

## 目标

把冻结 VGGT-Ω 1B 的 register token 投影后**拼接**到 DrivoR decoder memory(64 DINO + 64 几何 = 128 token)。主干、decoder 结构、损失、数据全部不变;融合点严格限于 decoder memory,不碰主干 ViT。

## 接口设计

**教师前向**(默认走离线缓存):

- 冻结 VGGT-Ω 1B(512 版),eval() + no_grad + bf16,不加载 camera/depth head;
- 4 相机联合前向,前视 = reference frame,相机顺序固定 **前 / 前左 / 前右 / 后**(写入指纹);
- 预处理必须直接 import 官方 `vggt_omega.utils.load_fn.load_and_preprocess_images`(`mode="balanced"`, `image_resolution=512`,禁止手写复刻);教师输入不做颜色增广;
- register 切片:`camera_and_register_tokens[:, :, :1]` = camera token(丢弃),`[:, :, 1:]` = 16 registers;
- 每相机 16 register → **64 个几何 token**,维度 2048(1024 frame + 1024 global feature)。

**融合**:

- 几何 token 经 geo_proj(前置 LN → Linear(2048→256) → +branch/camera embedding → LN → **零初始化 LayerScale 门控 γ=0**)投影到 256 维;
- 与 DrivoR 原生 64 scene tokens 拼接 → **128×256 联合 memory**,进 trajectory 与 scoring 两个 decoder 的 cross-attention。

**代码位置**:`navsim/agents/drivoR/`;新增 `vggt_geometry.py`、`scripts/cache_vggt_geometry_tokens.py`;修改 `drivor_model.py` 等 4 个文件。

## 配置与运行

- 默认关闭:`agent.config.vggt_geometry.enabled=false`(基线零影响);实验用 `++agent.config.vggt_geometry.enabled=true` 打开;
- 训练协议同基线:navtrain 10 epochs,AdamW lr 2e-4 cosine,batch 16,4×A100,损失全 1;可训练 = geo_proj + register/LoRA + decoders;
- 脚本:`temp_script/parallel/`(训练 `train_paralle_*.sh`,评测 `eval_paralle_*.sh`)。

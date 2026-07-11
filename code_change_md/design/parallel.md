# 并联几何融合 — 冻结 VGGT-Ω 几何分支增益探针

> 方案命名与项目定位见 `code_change_md/design/overview.md`。

## 目的与假设
把教师几何信息**以最短路径**送进规划 decoder,测量几何先验增益,直接检验增益存在性。

## 精确配置

### 教师前向
- 冻结 VGGT-Ω 1B(默认 512 版;教师审计可改判 256-Text-Alignment 版),eval() + no_grad + bf16,**不加载 camera/depth head**
- **4 相机联合前向**:前视 = reference frame,相机顺序固定 **前 / 前左 / 前右 / 后**(写入指纹)
- 教师预处理管线与 DrivoR 主干**完全解耦**,且**必须与官方 `vggt_omega.utils.load_fn.load_and_preprocess_images` 保持一致(直接 import,禁止手写复刻)**:
  - 默认 `mode="balanced"`, `image_resolution=512`(= 面积 ≈512²、AR 保持,官方默认与训练分布一致;NAVSIM ~16:9 输入约得 672×384 级别,尺寸自动为 patch16 倍数)
  - `mode="max_size"`(长边 512,~512×336)**不作默认**,仅用于长边约束的省算并联变体
  - **教师输入不做颜色增广**;归一化统计量以 load_fn 内部实现为准(不手抄常量)
  - register 切片按官方约定:`camera_and_register_tokens[:, :, :1]` = camera token(默认丢弃),`[:, :, 1:]` = 16 registers
- 取每相机 16 个 register → **64 个几何 token**,维度 D(2048,1024 frame featrue + 1024 global feature)

### 融合
- 几何 token 经 geo_proj（LN → MLP → LN + **零初始化 LayerScale 门控(γ=0)**）投影到 256 维，加分支 embedding + 相机 embedding
- 与 DrivoR 原生 64 scene tokens 拼接 → **128×256 联合 memory**,进 trajectory 与 scoring 两个 decoder 的 cross-attention
- **融合点严格限于 decoder memory,不碰主干 ViT**(这是"并联几何融合"的定义;任何流向主干的路径都改变方案语义)
- 高范数 register 前置 LN(否则 attention sink)

### 训练
- 同基线协议:navtrain 10 epochs,AdamW lr 2e-4 cosine,batch 16,4×A100,损失全 1,只动 perception 侧(geo_proj + register/LoRA + decoders)


## 与基线的差异点
教师 token 缓存读取 → geo_proj → memory 拼接。主干、decoder 结构、损失、数据全部不变。

## 代码状态
- 位置:`DrivoR/navsim/agents/drivoR/`;新增 `vggt_geometry.py`、`scripts/cache_vggt_geometry_tokens.py`;修改 `drivor_model.py` 等 4 个文件
- 默认关闭:`agent.config.vggt_geometry.enabled=false`
- **已通过代码审查

## 内部对照(必跑,精确定义)

| 对照 | 精确定义 | 回答的问题 |
|---|---|---|
| **跨样本错配几何令牌** | **整样本级**错配:样本 i 的几何 token 换成随机 partner 样本 j≠i 的(整组 64 token 一起换,不做 token 级打乱),partner 在**数据加载层从 split 全体缓存 token 中抽取**,**每次样本访问重新随机配对**;训练与评测**一致错配**(评测也 shuffle),同一 provider 实现,任意 batch size(含 batch=1)可执行。| 增益是"几何内容"还是"多了 64 个统计量正常的 token"(容量/正则效应) |
| **分布匹配噪声令牌** | 高斯噪声 token,**匹配缓存全体的均值/方差**(per-dim 统计),**注入在投影前的 D=2048 维空间**(不是投影后),再走同一 geo_proj | 比跨样本错配更弱的参照:连"真实特征流形"都没有时 decoder 会不会自己找到用法 |
| **评测时移除几何令牌** | 训练带几何 token,**推理时截断 memory 回 64**(禁止置零实现——零 key 仍分走 softmax 质量造成稀释伪影) | 训练后模型对几何 token 的**依赖度读数**:移除后回落幅度 = 依赖度 |

## 开关消融矩阵
- 联合前向 vs 各相机独立前向 —— 联合是默认(多视几何交互是教师价值所在)
- ± camera token(默认**关**,教师 camera token 不取)
- ~~512 版 vs 256-Text-Alignment 版 checkpoint~~ 

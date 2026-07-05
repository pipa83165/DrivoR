# 项目总览:VGGT-Ω 几何先验 × DrivoR 端到端规划

> 叙事主线:**"几何先验对端到端驾驶到底值多少钱——C1 先定价,蒸馏再砍价。"**
> C1 是分母(不计成本的增益上界),B 系 / B1-iso 是分子(零/低开销能拿回多少)。

## 1. 项目背景

在 DrivoR(valeo.ai,arXiv:2601.05083,DINOv2 ViT-S + LoRA + register 压缩 + 双 decoder,navtest 93.7 PDMS,110ms)上验证 VGGT-Ω(arXiv:2605.15195,大规模前馈 3D 重建模型)的几何先验能否提升规划性能。采用 **"DINO 主干 + 几何蒸馏 / 并联分支"的混合方案**,而非直接替换主干。

### 三篇基础论文速览
1. **VGGT** (arXiv:2503.11651):1.2B 前馈 transformer,交替 frame/global attention,DINOv2 tokenize,单次前向输出相机位姿/深度/点图/跟踪特征;不支持大动态场景。
2. **VGGT-Ω** (arXiv:2605.15195):VGGT 升级版。register attention(25% global 层换成 register-only,省 23% FLOPs)、简化 dense head、单深度头 + 点图/匹配辅助损失;基于 **DINOv3**、patch 16;支持动态场景;浅层(~layer 4)运动分割最干净;每帧 16 个 register(scene tokens);冻结 register 已验证可提升 OpenVLA-OFT(LIBERO 97.1→98.5)。**关键约束:官方只公开 1B checkpoint 两个版本(VGGT-Omega-1B-512 / -256-Text-Alignment),200M/500M/10B 无权重,重训不现实。**
3. **DrivoR** (arXiv:2601.05083):DINOv2 ViT-S(22M)+ LoRA rank-32,每相机 16 register → 64 scene tokens(4 相机),双 decoder(trajectory + scoring,stop-grad 解耦),6 个 PDMS sub-score BCE 监督。关键消融:DINOv2 预训练必需(+15 PDMS);预训练 register 初始化反而比随机差;后视相机 token 坍缩;scoring decoder 看四周、trajectory decoder 主要看前视。

## 2. 核心假设(预注册,勿修改)

| 编号 | 假设 | 主要验证实验 |
|---|---|---|
| **H1 增益存在性** | 几何先验提升 NC/TTC/EP(几何组),不伤 TLC/DAC/DDC(语义组) | C1 vs A1,sub-score 分解 |
| **H2 蒸馏可传递性** | 增益可蒸馏进 DINO 主干,零推理开销保留 ≥70% | B 系 vs C1 |
| **H3 增益分布** | 增益集中在 OOD(NAVSIM-v2 stage-2 视角扰动、HUGSIM 闭环),v1 近饱和 | v1/v2/HUGSIM 三评测对比 |
| **H4 时序特权** | 看历史帧的教师蒸馏单帧学生,缓解无历史帧歧义 | B4,动态密集切片 |

## 3. 实验矩阵总表

### A 组:基线
| 实验 | 内容 | 定位 |
|---|---|---|
| A0 | DrivoR 原版复现(DINOv2+LoRA),navval 目标 ≈90.0,3 seeds | 锚点 |
| A1 | 同 A0 但 DINOv3 ViT-S | **关键对照**:排除"增益来自 DINOv3 本身"。若 A1−A0 ≥ 0.5,后续基线全换 A1 |
| A2' | 冻结 VGGT-Ω 1B 直接做主干 + 新 register + decoder | 参考点,预期显著掉点;原 A2 LoRA 版仅在 A2' 意外好时才跑 |

### B 组:蒸馏(教师 = 冻结 VGGT-Ω 1B,离线缓存伪标签,**推理零开销**)
| 实验 | 内容 |
|---|---|
| B1 | register 级蒸馏:可学习 cross-attention reader + cosine,避免无序集合硬匹配 |
| B2 | dense patch 级蒸馏:global attention 后 image tokens,双线性插值对齐(patch 14 vs 16),消融教师层 {浅/中/末} |
| B3 | 输出级:轻量深度头 + 教师深度伪标签(confidence 加权 scale-invariant loss),推理时扔掉。**最简单,Phase 1 先跑** |
| B4 | 时序特权蒸馏(H4):教师看历史帧,学生只看当前帧 |
| B1-iso | DINO 特征上 2–4 层 ~3–5M adapter 旁路 + B1 损失,adapter 入口 stop-grad。容量隔离假设(B1 掉语义分时的出口),科学职能承接被砍的 C5 |

**蒸馏共通消融**:联合训练(λ∈{0.1, 0.5, 1.0})vs 两阶段;监控蒸馏/规划损失梯度余弦,持续为负则切两阶段或 λ 线性衰减。

### C 组:并联分支
| 实验 | 内容 | 定位 |
|---|---|---|
| C1 | 冻结 1B,4 相机联合前向,64 几何 token 拼进 decoder memory(已实现) | **增益上界探针 + Phase 1 试金石,不是部署方案**(端到端 ~250ms) |
| C1 内部对照 | shuffle / noise / drop(必跑,精确定义见 C1.md) | 区分真几何信号 vs 容量效应 |
| C2 | C1 + 教师 LoRA | 仅 C1 有正增益后才跑 |
| C3 | 免训练效率出口 a–d(256 版/时序摊销/减相机/推理加速),承接 C5 的工程职能,可叠加 | 部署路径 |
| C4 | 显式深度图分支(伪标签 + ~2M CNN) | 隐式 latent vs 显式深度对照 |

### 已砍方案(详见 graveyard.md,勿复活)
原 A2(200M 直接做主干)、原 C3(register-attention-only 推理开关)、C5(自蒸馏 22M 独立几何分支)。

## 4. 执行阶段

> 资源决定(2026-07-04):**算力不设限,按全量矩阵执行**,不启用最小可行集裁剪(原备选:A0+A1+C1 含 shuffle+B3+B4+B1-iso)。可并行的实验尽量并行(A0/A1 同时开;缓存生成与 A 组训练并行)。

| 阶段 | 内容 | Gate |
|---|---|---|
| **Phase 0(哨兵)** | 复现 A0;跑 A1;nuScenes LiDAR 真值验证教师深度(AbsRel、δ1.25);定 center_crop vs letterbox、512 vs 256 checkpoint | AbsRel > 0.15 → 教师域外退化成首要嫌疑,触发 confidence 门控预案,B3 降权,重心转 B1/B4 |
| **Phase 1(试金石)** | C1(含 shuffle)+ B3 | C1 ≈ shuffle → 混合方案先验大幅下调 |
| **Phase 2** | B1/B2/B4 层选择与 λ 扫描、B1-iso、C3/C4 | — |
| **Phase 3** | 最优 B vs B1-iso vs C1 vs A0/A1:全测试集 + HUGSIM + 效率,3 seeds | — |

## 5. 训练协议(所有消融统一)

- **消融版**:navtrain,10 epochs,AdamW lr 2e-4 + cosine annealing,batch 16,4×A100,损失权重全 1,只动 perception 侧。
- **正式版**:NAVSIM-v1 → 25 epochs(navtrain,竞赛版可 +navval);NAVSIM-v2 → 10 epochs navhard 方向。
- **种子**:核心对比 3 seeds 报 mean±std。DrivoR 消融差距常仅 0.3–1.0 PDMS,**单 seed 不可靠**。

## 6. 判定标准(摘要,全文见 expectations.md)

- C1 ≈ shuffle > A1 → 容量效应假增益。
- **TLC 红线**:降 >0.5 即语义挤占,总分再高也不采纳 → 触发 B1-iso 与门控融合。
- B 恢复 C1 增益 ≥70% → H2 成立,部署推荐 B 与 B1-iso 中 Pareto 占优者。
- 三种结局都有话可说:正结果 → 部署方案;语义挤占 → 容量隔离结论;负结果 → 最强公开几何先验盖棺,本身有价值。

## 7. 文档索引

`A0.md` `A1.md` `A2prime.md` | `B1.md` `B2.md` `B3.md` `B4.md` `B1-iso.md` | `C1.md`(最详细)`C2.md` `C3.md` `C4.md` | `phase0.md` `expectations.md`(评测前锁定)`graveyard.md`

# 项目总览:VGGT-Ω 几何先验 × DrivoR 端到端规划

> 叙事主线:**"几何先验对端到端驾驶到底值多少钱"**

## 1. 项目背景

在 DrivoR(valeo.ai,arXiv:2601.05083,DINOv2 ViT-S + LoRA + register 压缩 + 双 decoder,navtest 93.7 PDMS,110ms)上验证 VGGT-Ω(arXiv:2605.15195,大规模前馈 3D 重建模型)的几何先验能否提升规划性能。

### 基础论文速览
1. **VGGT-Ω** (arXiv:2605.15195):VGGT 升级版。register attention(25% global 层换成 register-only,省 23% FLOPs)、简化 dense head、单深度头 + 点图/匹配辅助损失;基于 **DINOv3**、patch 16;支持动态场景;浅层(~layer 4)运动分割最干净;每帧 16 个 register(scene tokens);冻结 register 已验证可提升 OpenVLA-OFT(LIBERO 97.1→98.5)。**关键约束:官方只公开 1B checkpoint 两个版本(VGGT-Omega-1B-512 / -256-Text-Alignment),200M/500M/10B 无权重,重训不现实。**
2. **DrivoR** (arXiv:2601.05083):DINOv2 ViT-S(22M)+ LoRA rank-32,每相机 16 register → 64 scene tokens(4 相机),双 decoder(trajectory + scoring,stop-grad 解耦),6 个 PDMS sub-score BCE 监督。关键消融:DINOv2 预训练必需(+15 PDMS);预训练 register 初始化反而比随机差;后视相机 token 坍缩;scoring decoder 看四周、trajectory decoder 主要看前视。

## 2. 核心假设

| 假设名称 | 假设 | 主要验证方式 |
|---|---|---|
| **增益存在性** | 几何先验提升 NC/TTC/EP(几何组),不伤 TLC/DAC/DDC(语义组) | sub-score 分解 |
| **蒸馏可传递性** | 增益可蒸馏进 DINO 主干,零推理开销保留 ≥70% | 几何先验蒸馏方案 |
| **分布外增益** | 增益集中在 OOD(NAVSIM-v2 stage-2 视角扰动、HUGSIM 闭环),v1 近饱和 | v1/v2/HUGSIM 三评测对比 |

## 3. 方案命名与定位


| 形象名称 | 完整名称 | 一句话定义 | 配套文档 |
|---|---|---|---|
| **复现** | DrivoR 官方复现 | DrivoR 原版配置复现,作为实现回归基准 | — |
| **基线** | DrivoR 协议基线 | 按本项目统一协议重训的 DrivoR,作为所有读数的参照点 | — |
| **并联** | 并联几何融合 | 冻结 VGGT-Ω register 并联进 decoder memory,作为慷慨预算下的增益探针;配套样本错配、分布匹配噪声和评测移除对照,另有省算变体 | 并联几何融合设计 |
| **geo_only** | 纯几何令牌规划 | 教师自带 64 register 独占 decoder memory,测量几何 token 独立承载规划的能力 | 纯几何令牌规划设计 |
| **裸替** | 冻结 VGGT 主干替换 | 冻结 VGGT-Ω 1B 直接做主干并注入可学习 scene token,作为替换可行性参考点 | VGGT 主干替换设计 |
| **裸替微调** | VGGT 主干 LoRA 微调 | 在裸替基础上启用 LoRA,仅当裸替意外接近基线时触发 | VGGT 主干替换实现指南 |

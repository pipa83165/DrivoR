# A2 设计思路 — DrivoR backbone 换为 VGGT-Ω 1B

> 本文档回答"为什么这么设计";具体改动清单、代码骨架与验收断言见 `A2_code_changes.md`。
> 实验编号 A2′(冻结版)/ A2-LoRA(原 A2)只用于文档与读数台账;代码实体一律按内容命名(包 `drivoR_vggt_omega`、类 `VggtOmegaImgEncoder` 等)。

---

## 1. 目标与定位

在 DrivoR(DINOv2 ViT-S + LoRA + 16 register/相机 + 双 decoder)上,把图像主干**直接换成**冻结的 VGGT-Ω 1B,回答实验矩阵中 A2′ 的问题:**最强公开几何主干"裸替"到端到端规划里,到底是什么水平?**

定位(来自 `c1/overview.md` 预注册,勿漂移):

- **参考点,预期显著掉点**——它是混合方案(C1 并联 / B 系蒸馏)的对照背景,不是候选部署方案;
- A2-LoRA(教师加 LoRA)**仅在 A2′ 意外接近或超过 A1 时才跑**,因此 LoRA 以"实现好、开关关"的方式一并落地,复活时零代码改动;
- 与 C1 的分工:C1 是"DINO 主干 + 并联几何 token"测增益上界;A2′ 是"没有 DINO、只有几何主干"测下界/替换可行性。

## 2. 总体思路(一句话)

**DrivoR 的 pipeline 一行不变,只把 `ImgEncoder`(timm DINOv2)换成一个满足同一接口契约的 `VggtOmegaImgEncoder`**;DrivoR 的感知压缩机制——可学习 scene token(register)注入主干、穿过网络被"写入"图像信息、出口取回 16×4=64 个 token 进 decoder——原样平移到 VGGT-Ω 上。

对外契约(`DrivoRModel` 无感知):

```
forward(img (B,N,3,H,W), scene_tokens (B,N,16,num_features)) → (B, N*16, tf_d_model)
num_features 属性 → 决定 scene_embeds 维度(=1024, VGGT hidden)
```

decoder、scorer、损失、优化器、训练协议、`drivor_agent.py` 全部复用,零改动。

## 3. 关键设计决策与理由

### 3.1 接入方式:4 行构造分发,不是新模型类

`drivor_model.py:54` 的 backbone 构造点加一个以 `model_name == "vggt_omega_1b"` 为哨兵的分支(惰性 import,防循环导入)。这是对既有代码的**唯一**修改。

- 否掉的方案 A(初版):修改 drivoR 包 5+ 文件 → 与"改动要小"冲突;
- 否掉的方案 B(中间版):drivoR 包零修改,用"空相机配置借道父类构造 + 子类覆盖 `_drivor_model`"的 hack → 能工作但引入一次性丢弃模型、`_config` 复位等隐患,收回"一行不改"约束后不再值得;
- 采纳的方案:**最小哨兵分发**。A0/A1/C1 的 timm 名称恒走原路径,执行序列逐指令不变(零影响论证与 bitwise 验证见 `A2_code_changes.md` §4.1)。

新增部分自成一包 `navsim/agents/drivoR_vggt_omega/`(带 `drivoR_` 前缀,避免与顶层官方包 `vggt_omega/` 同名);agent 只覆盖 `get_feature_builders` 一个方法。

### 3.2 "新 register" 的实现语义:可学习 scene token 注入冻结主干

预注册里 A2′ = "冻结 1B + **新 register** + decoder"。两种解读:

- ✅ **采纳**:新建每相机 16 个可学习 scene token(1024 维),插入冻结 aggregator 的前缀 token 区,梯度穿透冻结的 1B 网络流回 token(参数冻结不阻断对输入的梯度)。这忠实平移 DrivoR 的机制,且 `DrivoRModel` 创建 `scene_embeds` 的那行代码(randn×1e-6,维度由 `num_features` 派生)原样生效;
- ⬇ **降级为 A2′-lite 备选**:直接拿教师自带 16 register 当 scene token——那等价于"C1 减去 DINO 分支",可完全复用 C1 缓存、训练成本 ≈A1,留作快速下界参考,不默认跑。

注入的两个"免费兼容"事实(勘察确认,出处见 `A2_code_changes.md` §1):

- VGGT 的 RoPE 只作用于序列尾部 patch 段(`prefix = N - sin.shape[-2]`),前缀新增 token **自动免 RoPE**,零改动;
- register-attention 层按 `patch_token_start` 切前缀,把该属性 17→33 后,新 scene token **自动参与跨相机交互**。

前缀布局:`[camera(1), 教师register(16), 新scene(16), patch(1032)]`。

### 3.3 联合前向 vs 每相机独立(唯一的 pipeline 内部差异)

DINO 版 `(b n)` 展平 = 每相机独立前向。若照搬,VGGT 的 frame/global 交替 attention 在单帧输入下完全退化(global 作用域 == frame),它就变成一个 72 层的单目 DINOv3 变体——**多视几何交互这个换主干的核心动机被丢掉**。

因此 encoder 内部把 4 相机拼成一个 N=4 帧序列**联合前向**,前视排第 0 帧(VGGT 的 reference frame 有专属 camera/register token)。此差异完全封装在 encoder 内,对外形状不变;"独立 vs 联合"保留 reshape 级开关可做消融。

### 3.4 输入尺寸:真实原图 → 官方预处理 → 688×384

builder 拿 1920×1080 原图,过官方 `load_and_preprocess_images` 等价逻辑(balanced / 512:AR 保持 + 面积归一 ≈512² + patch16 对齐),对 NAVSIM 输入确定性输出 688×384。配置里的 `image_size: [688, 384]` 只是记录值。

- 不能沿用 DrivoR 的 1148×672:1148 不被 16 整除,前向直接报错;
- 不能用原生分辨率:1080/16 不整除;裁齐后每帧 8040 patch、4 相机联合 attention >3.2 万 token(单层成本 ≈60 倍,48 层反传不可行),且远离 checkpoint 的 ≈512² 训练分布,是分布外输入,特征更差而非更好;
- 预处理代码**直接复用 C1 已过三方 cosine>0.999 验收的 `preprocess_arrays_for_teacher`**,禁止复刻数值逻辑。

### 3.5 归一化只做一次,位置随官方

builder 输出 [0,1](官方 load_fn 约定);ImageNet 归一化在 aggregator 内部(官方位置)。DrivoR builder 的 ImageNet 统计量与 VGGT 内部 `_RESNET_MEAN/STD` 数值相同,但绝不能两边都做——双重归一化是静默错误,由验收 [1](与官方前向 cosine>0.999)兜底。

### 3.6 读出:最末层 frame‖global 拼接(2048 维)

读出取最末 block 的 `cat(frame_attn 输出, global_attn 输出)` = 2048 维的 scene 段,与 C1 缓存的语义完全一致(便于跨实验对比"同一种教师特征")。neck `Linear(2048→256)` 在 encoder 内部,`num_features`(注入维度 1024)与读出维度的差异对外不可见。

### 3.7 冻结为主、LoRA 为开关

- 冻结版(A2′):可训练参数 = decoder/scorer/heads(同 A1 量级)+ scene_embeds 65k + neck 0.5M。教师无 dropout/BN,`train()` 覆写把 aggregator 钉死 eval(语义明确);
- LoRA 版(A2-LoRA):VGGT 的 `attn.qkv` 是融合 `Linear(dim, 3*dim)`,unbind 布局与 timm 完全一致 → **直接复用 DrivoR 的 `_LoRA_qkv_timm`**,零复刻。默认打 frame + inter-frame 共 48 层(+6.3M 参数),DINOv3 trunk 24 层作可选目标;只加 q/v、rank 32(DrivoR 惯例);B 零初始化保证冷启动数值上等于冻结版;
- 铁律:先整体冻结、再做 LoRA 手术(顺序反了 LoRA 会被一起冻掉)。

### 3.8 训练可行性:梯度检查点 + 门控规则

scene token(和 LoRA)的梯度要穿透 1B×48 block:不开 checkpoint 单样本激活 >5GB,不可行;开启后 ≈0.4GB/样本,4×A100 各 batch 4 满足消融协议,全程 ≈20–30h。

**门控铁律**(代码审查发现的关键坑):aggregator 被钉死在 eval,`self.training` 恒 False,**checkpoint 门控只能用 `torch.is_grad_enabled()`**——训练 step 梯度开启走 checkpoint,Lightning 验证与 `run_pdm_score` 都在 no_grad 下自动走原速前向。验收用重算计数探针(2×depth)证明 checkpoint 实际生效,防止静默失效导致显存预算作废。

### 3.9 其余固定项

- GridMask 保留代码路径但默认关:对冻结主干是训练分布外输入;LoRA 版可开(有能力适应增广);
- bf16 autocast 包裹 aggregator(镜像官方),读出后 fp32 进 neck;
- feature builder 换唯一名 `drivor_vggt_omega_feature`,与 DrivoR 特征缓存物理隔离;
- 权重加载:过滤 `aggregator.*` 前缀后 **strict=True**(键集精确匹配);
- checkpoint 含冻结 1.2B(≈4.9GB/份),接受,换取断点续训/加载零改动;
- 溯源不用 git,一律文件 sha256(项目约定)。

## 4. 方案演化史(否掉的路,勿复走)

| 版本 | 方案 | 结局 |
|---|---|---|
| v1 | 修改 drivoR 包 5+ 文件(仿 C1 落点) | 否:改动过大 |
| v2 | drivoR 包零修改,空相机配置借道 + 覆盖 `_drivor_model` | 否:hack 换来的收益在约束收回后不成立 |
| v3(定稿) | 4 行哨兵分发 + 独立新包,联合前向,LoRA 一并实现 | ✅ |
| 命名 | A2Model / A2Agent | 否:实验编号不进代码,改 VggtOmega* |

审查修正(详见 `A2_code_question.md` 及回应):checkpoint 门控 `self.training`→`torch.is_grad_enabled()`;`forward` 拆出 `forward_full` 供 S=0 验收;权重加载改 strict=True;`sys.modules` 零回归断言改子进程执行。

## 5. 判定与读数(锁定)

- 训练:navtrain 10 epochs、AdamW、batch 16(全局)、3 seeds,同 A0/A1 消融协议;
- 读数入 overview 实验矩阵:A2′ − A1(mean±std)、TLC 子分;
- 触发条件:A2′ 意外接近或超过 A1 → 命令行开 `use_lora=true` 跑 A2-LoRA;
- 无论结果如何都有话可说:显著掉点 → 佐证"混合方案而非替换主干"的项目主线;意外不掉 → 几何主干可直接承载规划,B/C 系的必要性需重估。

## 6. 文档索引

- `A2_code_changes.md` — 改动清单、代码骨架、零影响论证(§4.1)、验收断言(§9)、实施顺序(§12)
- `A2_code_question.md` — 代码审查 5 问(已全部回应并修正)
- `c1/overview.md` — 实验矩阵与预注册假设;`c1/C1.md` / `c1/C1_code_changes.md` — 被复用的 C1 设施(预处理桥接、sha256 溯源约定)

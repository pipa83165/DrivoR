# A2 代码改动文档 — DrivoR pipeline 不变,backbone 换冻结 VGGT-Ω 1B(含 LoRA)

> 依据 `c1/overview.md` A 组定义:**A2′ = 冻结 VGGT-Ω 1B 直接做主干 + 新 register + decoder**(预期显著掉点的参考点);**A2-LoRA(原 A2)仅在 A2′ 意外好时才跑**——本文档把 LoRA 一并实现,配置开关默认关,复活时零代码改动。
> **命名约定:A2′/A2-LoRA 只是实验编号,仅出现在文档与读数记录;代码实体一律按内容命名为 vggt-omega**(包 `drivoR_vggt_omega`、类 `VggtOmegaImgEncoder/VggtOmegaAgent/...`)。agent 包名带 `drivoR_` 前缀,避免与仓库顶层官方包 `vggt_omega/` 同名混淆。
> **改动最小化原则**:对既有代码的修改只有 `drivor_model.py` 的 **4 行构造分发**(以 `model_name` 为哨兵,原值走原路径,A0/A1/C1 行为零变化);其余全部是新增文件,通过继承/import 复用 DrivoR 的 decoder、scorer、损失、训练协议。
> 版本标识约定同 C1:**不使用 git,一律文件内容 sha256**。

---

## 0. 设计决策(先读)

### 0.1 "pipeline 不变"的精确含义

对外契约全部保持:`DrivoRModel.forward` 一行不动(它只消费 `num_cams / image_backbone / scene_embeds / features["image"]`);backbone 接口仍是 `forward(img (B,N,C,H,W), scene_tokens (B,N,16,num_features)) → (B, N*16, tf_d_model)`;scene token(新 register)仍由 `DrivoRModel` 以 `randn*1e-6` 创建(`drivor_model.py:55` 不动,维度由 `num_features=1024` 自动派生)、注入主干、出口取回;decoder / scorer / 损失 / 优化器 / 训练协议零改动。

**唯一的内部差异**:encoder 内部把 4 相机拼成一个序列做**联合前向**(前视 = 第 0 帧 reference frame),而不是 DINO 版的 `(b n)` 每相机独立前向。理由:独立前向下 VGGT 的 frame/global 交替 attention 完全退化(单帧时 global 作用域 == frame),多视几何交互——换主干的核心动机——被丢掉。联合前向完全封装在 encoder 内,对外形状不变;"独立 vs 联合"留有 reshape 级开关可做消融。

### 0.2 输入尺寸:真实原图 → 官方处理流程 → 688×384(答"为什么不用真实尺寸")

采用的正是"真实尺寸过一遍 VGGT-Ω 官方处理流程":builder 拿 1920×1080 **原图**,走官方 `load_and_preprocess_images` 的等价逻辑(`balanced`, `image_resolution=512`:AR 保持 + 面积归一 ≈512² + 对齐 patch16 倍数),对 NAVSIM 输入**确定性输出 688×384**。配置里的 `image_size: [688, 384]` 只是记录这个确定性结果,不是自选尺寸。

原生分辨率不缩放直接进网络不可行:1080/16 不整除;裁到 1920×1072 后每帧 120×67=8040 patch,4 相机联合 global attention >3.2 万 token,单层成本 ≈688×384 的 60 倍、48 层反传不可行;且 checkpoint 训练分布是 ≈512² 面积,原生高分辨率反而是**分布外**输入,特征更差而非更好。

### 0.3 归一化分工(与官方位置一致)

builder 输出 **[0,1]**(官方 load_fn 约定);ImageNet 归一化在 encoder 内部完成(官方位置,`aggregator.py:108`,统计量用继承来的 `_resnet_mean/std` buffer,不手抄常量)。注意 VGGT 的归一化统计量与 DrivoR builder 用的 ImageNet 值相同,但**归一化只能做一次**——本 builder 不做归一化,谁也别在 encoder 外面再做。

### 0.4 scene token(新 register)注入与读出

- 每帧 token 序列 `[camera(1), 教师register(16), 新scene(16), patch(1032)]`,`patch_token_start` 17→**33**。RoPE 只作用于序列尾部 patch 段(`attention.py:92` 的 `prefix = N - sin.shape[-2]`),前缀自动免 RoPE,零改动兼容;register-attention 层按 `patch_token_start` 切前缀(`aggregator.py:191`),新 scene token 自动参与跨相机交互。
- 注入维度 1024(VGGT hidden)= `num_features`,`DrivoRModel` 的 `scene_embeds` 形状自动派生正确、初始化沿用 `randn*1e-6` 惯例。
- 读出:最末层 `cat(frame_attn, global_attn)` = **2048 维**(与 C1 缓存语义一致,`aggregator.py:150`),取 scene 段 → neck `Linear(2048→256)`(neck 在 encoder 内部,维度差异对外不可见)。

### 0.5 LoRA(本次一并实现,默认关)

- VGGT 的 `attn.qkv` 是融合 `Linear(dim, 3*dim)`,unbind 布局 q 前 1/3、v 后 1/3(`attention.py:78,128-129`),与 timm **完全一致** → **直接 import 复用 DrivoR 的 `_LoRA_qkv_timm`**,零复刻。
- 打点:默认 `frame_blocks + inter_frame_blocks`(48 层),`trunk`(DINOv3 patch_embed 的 24 层,`aggregator.patch_embed.blocks`)作可选目标;只加 q/v、rank 32(DrivoR 惯例)。
- LoRA B 矩阵零初始化 → 冷启动数值上 == 冻结版(验收 §9-[3])。
- LoRA 参数随整体 checkpoint 保存(state_dict 自带),不需要 DrivoR 的 safetensors 单独存取路径。

### 0.6 其余固定决策

- GridMask:保留代码路径(pipeline 不变),`use_grid_mask` 默认 **false**——对冻结主干是训练分布外输入;A2-LoRA 变体可开(LoRA 有能力适应增广)。开启时作用在 [0,1] 图上(归一化之前,与 DrivoR 的"归一化之后"不同,已知差异,记录在案)。
- **bf16 autocast** 包裹 aggregator(镜像 `vggt_omega.py:39-41`),读出后 float32 进 neck。
- **梯度检查点默认开**:scene token(和 LoRA)的梯度都要穿透 1B×48 block,不开则单样本激活 >5GB。
- 相机顺序 `[f0, l0, r0, b0]`(前视第 0 帧;与 DrivoR 原生 builder 的 f0,b0,l0,r0 不同且无需一致)。

---

## 1. 代码勘察结论(事实依据)

| 事项 | 结论 | 出处 |
|---|---|---|
| backbone 构造点 | `self.image_backbone = ImgEncoder(config_image_backbone)`,前三行已把 `image_size/num_scene_tokens/tf_d_model` 注入 backbone 配置 → 分发处新 encoder 免费获得这三个键 | `drivor_model.py:50-54` |
| backbone 接口契约 | `forward(img, scene_tokens) → (B, N*16, tf_d_model)`;`num_features` 决定 `scene_embeds` 维度 | `drivor_model.py:54-55,136-137` |
| scene_embeds 创建 | `randn*1e-6`,维度取 `image_backbone.num_features` → `num_features=1024` 即自动正确,**该行不改** | `drivor_model.py:55` |
| qkv 布局(LoRA 可移植性) | `qkv = Linear(dim, 3*dim)`;`reshape(B,N,3,heads,hd); unbind(2)` → q 前 1/3、v 后 1/3,与 timm 一致;`_LoRA_qkv_timm` 直接复用(它往 `[:, :, :dim]` 加 new_q、`[:, :, -dim:]` 加 new_v) | `attention.py:78,128-129`;`dinov2_lora.py:126-134` |
| LoRA 包装兼容 forward_list | trunk 的 `SelfAttention.forward_list` 也是调 `self.qkv(x_flat)`,模块级替换同样生效 | `attention.py:111-121` |
| trunk block 路径 | DINOv3 trunk = `aggregator.patch_embed.blocks`(nn.ModuleList,同款 SelfAttentionBlock) | `vision_transformer.py:166` |
| RoPE 前缀处理 | `prefix = N - sin.shape[-2]`,前缀 token 不加 RoPE → 新增前缀 token 零改动兼容 | `attention.py:92-99` |
| register-attention 层切片 | 按 `self.patch_token_start` 切,改该属性即自动带上新 token | `aggregator.py:190-203` |
| 2048 维语义 | 末层 `cat([frame_tokens, tokens], -1)`,前 1024 frame-attn、后 1024 global-attn | `aggregator.py:150` |
| reference frame | `slice_expand_and_flatten` 给第 0 帧专属 token → 前视排第 0 | `aggregator.py:246-250` |
| 归一化位置 | ImageNet mean/std 在 `Aggregator.forward` 内部;输入应为 [0,1] | `aggregator.py:108` |
| 优化器 | `AdamW(model.parameters())`,requires_grad=False 参数不被更新,无需过滤;LoRA 新参数自动进入 | `drivor_agent.py:245` |
| checkpoint 加载 | `initialize()` 只做 key 前缀替换,与模型类无关,继承即正确 | `drivor_agent.py:148` |
| feature builder 唯一名 | 缓存按 `get_unique_name()` 区分 → 新 builder **必须换名** | `drivor_features.py:36` |
| 教师输入尺寸 | 1920×1080 → balanced/512/patch16 → **688×384**(每帧 33+1032=1065 token,4 帧联合 4260) | `load_fn.py` 逻辑,同 C1 §0 |
| VGGT 权重加载 | 先过滤 `aggregator.*` 前缀再 **strict=True** 加载(键集精确匹配,missing/unexpected 双向为空);LoRA 手术在加载之后、scene token 在模块外,不影响键集 | 修订自 `c1/C1_code_changes.md` §3.3(该处 strict=False 是因为未过滤前缀) |
| 在线预处理桥接 | `c1_vggt.preprocess_arrays_for_teacher` 已过三方 cosine>0.999 验收 → **直接 import,禁止再写一份** | `navsim/agents/drivoR/c1_vggt.py` |

---

## 2. 文件清单

| 操作 | 文件 | 内容 |
|---|---|---|
| **修改** | `navsim/agents/drivoR/drivor_model.py` | **仅 4 行**:backbone 构造分发(§4) |
| 新增 | `navsim/agents/drivoR_vggt_omega/__init__.py` | 空 |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py` | `SceneTokenAggregator`(联合前向 + scene token 注入 + 梯度检查点)、`apply_lora_to_blocks`、`VggtOmegaImgEncoder` |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py` | `VggtOmegaFeatureBuilder(DrivoRFeatureBuilder)`:原图过官方预处理,唯一名 `drivor_vggt_omega_feature` |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py` | `VggtOmegaAgent(DrivoRAgent)`:仅覆盖 `get_feature_builders` |
| 新增 | `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml` | agent 配置 |
| 新增 | `navsim/planning/script/config/training/vggt_omega_training.yaml` | 训练入口 |
| 新增 | `scripts/vggt_omega_acceptance_checks.py` | 6 条验收断言(§9) |

**不改**:`drivor_agent.py`(`DrivoRModel(config)` 构造经分发自动得到 VGGT backbone)、`drivor_features.py`、decoder、scorer、损失、`dataset.py`、`run_training.py`、C1 全部代码。

---

## 3. 新增 `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py`

### 3.1 `SceneTokenAggregator` — 子类化官方 Aggregator,只改 forward

```python
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten
from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import _LoRA_qkv_timm
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask


class SceneTokenAggregator(Aggregator):
    """官方 Aggregator + 每帧前缀区注入可学习 scene token(新 register)。
    前缀布局: [camera(1), 教师register(16), 新scene(S), patch...]
    patch_token_start 由 17 -> 17+S,register-attention 层与 RoPE 前缀逻辑自动适配。"""

    def __init__(self, num_scene_tokens: int = 16, grad_checkpointing: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_scene_tokens = num_scene_tokens
        self.grad_checkpointing = grad_checkpointing
        self.scene_token_start = self.patch_token_start          # 17: scene 段起点
        self.patch_token_start = self.patch_token_start + num_scene_tokens   # 33

    def forward_full(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """images: (B, N, 3, H, W) in [0,1];scene_tokens: (B, N, S, 1024) 可学习。
        4 相机作为一个 N 帧序列联合前向(前视=第0帧 reference frame)。
        返回最末层 cat(frame, global) 的全 token: (B, N, num_tokens, 2048)。
        生产路径只用 forward();本方法同时服务验收 [1](S=0 时取 camera+register 段
        与官方对比 —— forward() 的 scene 切片在 S=0 下为空,不能用于该验收)。"""
        B, N, C, H, W = images.shape
        images = (images - self._resnet_mean) / self._resnet_std   # 官方位置的唯一一次归一化
        images = images.view(B * N, C, H, W)

        camera_token = slice_expand_and_flatten(self.camera_token, B, N)
        register_token = slice_expand_and_flatten(self.register_token, B, N)
        scene = scene_tokens.reshape(B * N, self.num_scene_tokens, -1)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, scene, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        grid = (H // self.patch_size, W // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (rope_sin.to(patch_tokens.device, torch.float32),
                          rope_cos.to(patch_tokens.device, torch.float32))

        def run_block(tokens, idx):
            tokens, frame_tokens = self._run_frame_block(
                tokens, B, N, num_tokens, embed_dim, idx, frame_rope)
            tokens = self._run_inter_frame_attention_block(
                tokens, B, N, num_tokens, embed_dim, idx,
                self.inter_frame_attention_types[idx])
            return tokens, frame_tokens

        frame_tokens = None
        for idx in range(self.depth):
            # 门控禁止用 self.training:encoder.train() 把本模块钉死在 eval,
            # self.training 恒为 False,checkpoint 会被静默关闭(审查问题 #1)。
            # 训练/评测的区分交给梯度开关:训练 step 梯度开启;Lightning 验证与
            # run_pdm_score(abstract_agent.py:78)都在 no_grad 下 → 自动走原速前向。
            if self.grad_checkpointing and torch.is_grad_enabled():
                tokens, frame_tokens = checkpoint(run_block, tokens, idx, use_reentrant=False)
            else:
                tokens, frame_tokens = run_block(tokens, idx)

        return torch.cat([frame_tokens, tokens], dim=-1)           # (B, N, num_tokens, 2048)

    def forward(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """生产路径:只取 scene 段 (B, N, S, 2048)。"""
        out = self.forward_full(images, scene_tokens)
        return out[:, :, self.scene_token_start:self.patch_token_start]
```

要点:`_run_frame_block` / `_run_inter_frame_attention_block` / `_resnet_mean` 等全部继承官方实现,数值逻辑零复刻;只保留最末层输出;checkpointing 以 `torch.is_grad_enabled()` 为门控(训练开、no_grad 评测自动关),**不依赖 `self.training`**——该标志被 encoder 的 eval 钉死策略置为 False,用它做门控会静默关闭 checkpoint(审查问题 #1/#4);验收 [6] 用重算计数探针证明 checkpoint 实际生效。

### 3.2 `apply_lora_to_blocks` — 复用 DrivoR 的 LoRA 包装

```python
def apply_lora_to_blocks(blocks: nn.ModuleList, r: int) -> list:
    """对一组 SelfAttentionBlock 做 qkv 手术(只加 q/v,同 DrivoR 惯例)。
    返回新建的 LoRA 线性层列表(调用方负责保证它们 requires_grad=True)。
    必须在整体冻结之后调用 —— 新建层默认可训练,顺序错了会把 LoRA 一起冻掉。"""
    lora_layers = []
    for blk in blocks:
        qkv = blk.attn.qkv
        dim = qkv.in_features
        a_q, b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
        a_v, b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
        for a in (a_q, a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (b_q, b_v):
            nn.init.zeros_(b.weight)                 # 冷启动 == 冻结版(验收 §9-[3])
        blk.attn.qkv = _LoRA_qkv_timm(
            qkv, a_q, b_q, a_v, b_v,
            nn.Identity(), nn.Identity(),            # k 路不加(同 DrivoR use_qkv=False)
            nn.Identity(), nn.Identity(), nn.Identity(),   # 无额外 LayerNorm
        )
        lora_layers += [a_q, b_q, a_v, b_v]
    return lora_layers
```

依据:VGGT `attn.qkv` 融合布局与 timm 一致(§1),`_LoRA_qkv_timm` 往前 1/3 加 new_q、后 1/3 加 new_v,语义正确;`LinearKMaskedBias` 的 bias mask 行为被原样保留(原 qkv 模块整体包在里面)。trunk 的 `forward_list` 路径同样走 `self.qkv(...)` 模块调用,替换后自动生效。

### 3.3 `VggtOmegaImgEncoder` — 满足 DrivoRModel 的 backbone 契约

```python
class VggtOmegaImgEncoder(nn.Module):
    """VGGT-Omega 1B 做图像主干(默认冻结 = 实验 A2';use_lora=true = 实验 A2-LoRA)。
    接口契约与 ImgEncoder 相同:
    forward(img (B,N,3,H,W), scene_tokens (B,N,S,num_features)) -> (B, N*S, tf_d_model);
    num_features=1024(注入维度);neck 输入 2048(读出维度),对外不可见。"""

    VGGT_EMBED_DIM = 1024
    READOUT_DIM = 2048          # cat(frame_attn, global_attn),同 C1 缓存语义

    def __init__(self, config):
        super().__init__()
        self.num_features = self.VGGT_EMBED_DIM

        self.aggregator = SceneTokenAggregator(
            num_scene_tokens=config["num_scene_tokens"],
            grad_checkpointing=config.get("grad_checkpointing", True),
        )
        state = torch.load(config["checkpoint_path"], map_location="cpu")
        state = state.get("model", state)
        state = {k[len("aggregator."):]: v for k, v in state.items() if k.startswith("aggregator.")}
        # 已过滤到 aggregator.* 前缀,且 LoRA 手术在加载之后、scene token 参数在模块外
        # → 键集必须精确匹配,直接 strict=True(missing 与 unexpected 双向为空;
        # 审查问题 #3:strict=False + 只查 missing 会静默忽略多余键,不是"全命中")
        self.aggregator.load_state_dict(state, strict=True)

        # 1) 先整体冻结
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        # 2) 再做 LoRA 手术(新建层默认可训练;顺序不可颠倒)
        self.use_lora = config.get("use_lora", False)
        if self.use_lora:
            targets = {
                "frame": self.aggregator.frame_blocks,
                "inter_frame": self.aggregator.inter_frame_blocks,
                "trunk": self.aggregator.patch_embed.blocks,
            }
            for name in config.get("lora_targets", ["frame", "inter_frame"]):
                apply_lora_to_blocks(targets[name], r=config.get("lora_rank", 32))

        self.neck = nn.Linear(self.READOUT_DIM, config["tf_d_model"])

        # pipeline 保留 GridMask 代码路径;冻结版默认关(教师域外输入),LoRA 版可开
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = config.get("use_grid_mask", False)

    def train(self, mode: bool = True):
        # Lightning 每个 epoch 全模型 .train();主干钉死 eval 语义
        # (VGGT/DINOv3 无 dropout/BN,数值本就一致;LoRA 线性层不受 eval 影响,梯度照常)
        super().train(mode)
        self.aggregator.eval()
        return self

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        B, N = img.shape[:2]
        if self.use_grid_mask and self.training:
            img = self.grid_mask(img.flatten(0, 1)).view_as(img)   # [0,1] 图上打洞(归一化前)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            geo = self.aggregator(img, scene_tokens)               # (B, N, S, 2048)
        tokens = self.neck(geo.float())                            # (B, N, S, 256), fp32
        return tokens.reshape(B, -1, tokens.shape[-1])             # (B, N*S, 256)
```

要点:
- **梯度路径**:`scene_embeds`(以及 LoRA A/B)→ 穿过 48 个冻结 block → 读出;参数冻结不阻断对输入的梯度(验收 §9-[2]);
- neck 在 autocast 外 fp32 计算(与 C1 geo_proj 约定一致);
- checkpoint 含冻结 1.2B 权重(≈4.9GB fp32/份)。**接受**:保证 `initialize()`/断点续训零改动;磁盘敏感时可后续加 `on_save_checkpoint` 钩子剔除,不进本次改动。

---

## 4. 修改 `navsim/agents/drivoR/drivor_model.py`(唯一的既有文件改动,4 行)

`__init__` 中 `self.image_backbone = ImgEncoder(config_image_backbone)`(54 行)改为分发:

```python
            if config_image_backbone.get("model_name") == "vggt_omega_1b":
                from navsim.agents.drivoR_vggt_omega.vggt_omega_backbone import VggtOmegaImgEncoder
                self.image_backbone = VggtOmegaImgEncoder(config_image_backbone)
            else:
                self.image_backbone = ImgEncoder(config_image_backbone)
```

- 哨兵是 `model_name`,A0/A1/C1 的 timm 名称走原路径,行为零变化;
- **惰性 import**(放分支内):不用 VGGT 时零导入成本,也天然避免 drivoR ↔ drivoR_vggt_omega 的循环导入;
- 前三行(50-53)已把 `image_size / num_scene_tokens / tf_d_model` 注入 `config_image_backbone`,新 encoder 免费获得;
- 下一行 `scene_embeds`(55 行)**不动**:`num_features=1024` 自动派生正确形状,初始化沿用 `randn*1e-6`;
- `drivor_agent.py` **不动**:`DrivoRModel(config)` 构造经此分发自动得到 VGGT backbone。

### 4.1 对 DrivoR 既有实验(A0/A1/C1)零影响的论证(硬性要求)

逐一核对所有可能的影响途径:

| 途径 | 论证 |
|---|---|
| 代码路径 | 唯一改动是构造分发;A0/A1/C1 的 `model_name` 是 timm 名称 → 恒走 `else` 原路径,执行序列与改动前**逐指令相同** |
| import 副作用 | 新包 import 是**惰性**的(在 vggt 分支内),原实验从不触发;`drivoR_vggt_omega` 包不被加载 |
| 配置 | `drivoR.yaml` / `c1_training.yaml` 零改动;新增 yaml 是独立文件,hydra 不会隐式加载 |
| 特征缓存 | 新 builder 唯一名 `drivor_vggt_omega_feature`,与 `drivor_feature` 缓存目录天然隔离 |
| C1 代码 | 只 import `preprocess_arrays_for_teacher`(纯函数),不修改 c1_vggt.py 任何内容 |
| lidar 分支 | `drivor_model.py:65` 的 lidar `ImgEncoder` 构造不在分发范围内,原样保留 |

**验证(不止靠论证)**:验收 [4] 要求两条证据——(a) 原 `model_name` 下构造出的 backbone 类型仍是 `ImgEncoder`;(b) **强证据**:A0 配置、同种子、假 batch 跑 1 个训练 step,loss 与打分发补丁前**逐位一致**(bitwise equal)。

---

## 5. 新增 `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py`

```python
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.c1_vggt import preprocess_arrays_for_teacher

# 前视必须第 0 帧(VGGT reference frame)。与 DrivoR 原生顺序(f0,b0,l0,r0)不同,无需一致。
CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")


class VggtOmegaFeatureBuilder(DrivoRFeatureBuilder):
    """真实原图(1920x1080)→ 官方 VGGT 预处理(balanced/512, AR 保持, patch16 对齐)
    → (4, 3, 384, 688), [0,1]。不做 ImageNet 归一化(在 encoder 内部做,官方位置)、
    不做颜色增广。ego_status 逻辑继承不变。"""

    def get_unique_name(self) -> str:
        return "drivor_vggt_omega_feature"   # 必须区别于 drivor_feature,防缓存互相污染

    def _get_camera_feature(self, agent_input):
        cameras = agent_input.cameras[-1]
        raw = [getattr(cameras, name).image for name in CAMERA_ORDER]
        images = preprocess_arrays_for_teacher(
            raw,
            mode=self._config.get("vggt_preprocess_mode", "balanced"),
            image_resolution=self._config.get("vggt_image_resolution", 512),
        )
        return {"image": images}
```

- 键仍叫 `image`,`DrivoRModel.forward`(129 行)零改动消费;`cam_K`/`world_2_cam` 不再输出(无人消费,§1);
- 复用 C1 已验收的 `preprocess_arrays_for_teacher`,**禁止复刻数值逻辑**;
- 特征缓存体量 688×384×3×4 fp32 ≈12.7MB/样本(< DrivoR 原生的 ≈37MB)。

---

## 6. 新增 `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py`

```python
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .vggt_omega_features import VggtOmegaFeatureBuilder


class VggtOmegaAgent(DrivoRAgent):
    """DrivoRAgent 全部逻辑继承(模型构造经 drivor_model.py 分发自动得到 VGGT backbone;
    loss、优化器、metric cache、callbacks、checkpoint 加载均复用),仅换 feature builder。"""

    def get_feature_builders(self):
        return [VggtOmegaFeatureBuilder(config=self._config)]
```

---

## 7. 配置

### 7.1 新增 `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml`

复制 `drivoR.yaml` 后仅改:

```yaml
_target_: navsim.agents.drivoR_vggt_omega.vggt_omega_agent.VggtOmegaAgent

config:
  # ...(其余键与 drivoR.yaml 完全相同,略)...

  image_size: [688, 384]        # = 官方 balanced/512 对 1920x1080 的确定性输出(记录用)
  vggt_preprocess_mode: balanced
  vggt_image_resolution: 512

  image_backbone:
    model_name: vggt_omega_1b   # 分发哨兵(drivor_model.py §4)
    checkpoint_path: weight/vggt_omega_1b_512.pt
    grad_checkpointing: true
    use_grid_mask: false        # 冻结版默认关;LoRA 版可开(消融)
    # ---- LoRA(实验 A2-LoRA = 原 A2,仅 A2' 意外好时才跑;开关在此,零代码改动)----
    use_lora: false             # false = 实验 A2'(冻结版)
    lora_rank: 32
    lora_targets: [frame, inter_frame]   # 可选加 trunk(DINOv3 24 层)
    # ImgEncoder 专属键(model_weights/finetune/focus_front_cam/...)全部删除
```

### 7.2 新增 `navsim/planning/script/config/training/vggt_omega_training.yaml`

```yaml
defaults:
  - default_training
  - override /agent: drivoR_vggt_omega
  - _self_

experiment_name: vggt_omega_backbone    # 实验编号 A2'/A2-LoRA 记在实验台账,不进代码命名
```

A2-LoRA 复活时命令行覆盖:`agent.config.image_backbone.use_lora=true`(+ 按需 `use_grid_mask=true`)。

---

## 8. 训练成本与显存核算(预算依据,启动前实测校准)

- 每帧 token:1+16+16+1032=**1065**;4 帧联合 global attention **4260 token**;48 个 aggregator block + 24 层 DINOv3 trunk。
- **梯度检查点必开**:关闭时激活 >5GB/样本;开启后 block 边界激活 ≈0.4GB/样本 + 单 block 重算峰值,A100-80G 单卡 batch 4–8 可行。消融协议 batch 16(全局)= 4×A100 各 4,无需梯度累积。
- 时间粗估:bf16 前向 ~0.1–0.15s/样本,checkpointing 反传 ≈2×前向 → 85k×10 epochs ÷ 4 卡 ≈ **20–30h**。
- 可训练参数:冻结版 = decoders+scorer+heads(同 A1 量级)+ `scene_embeds` 65k + `neck` 0.5M;LoRA 版另加 48 block × (q,v) × (A+B) ≈ **6.3M**(trunk 打点再 +3.1M)。**日志打印 trainable/frozen 计数**(验收 §9-[2] 断言)。
- LoRA 版反传新增权重梯度计算,时间 ≈ 冻结版(输入梯度本来就要算),显存增量可忽略。

---

## 9. 新增 `scripts/vggt_omega_acceptance_checks.py`(训练启动前跑一遍留档)

```python
# [1] 前向正确性:num_scene_tokens=0 时,SceneTokenAggregator 与官方 VGGTOmega 输出一致
#     同一输入 (1,4,3,384,688) [0,1],取 forward_full(...)[:, :, :17](camera+register 段;
#     注意不能用 forward() —— 其 scene 切片在 S=0 下为空,审查问题 #2),
#     与 VGGTOmega(enable_camera=False, enable_depth=False) 的
#     camera_and_register_tokens 逐 token cosine > 0.999
#     —— 验证 forward 复制 + patch_token_start 改写 + 归一化位置都没引入偏差

# [2] 梯度路径(冻结版):DrivoRModel(vggt 配置) 1 次前向+反传后
#     assert model.scene_embeds.grad is not None
#     assert model.image_backbone.neck.weight.grad is not None
#     assert all(p.grad is None for p in model.image_backbone.aggregator.parameters())
#     打印并断言 trainable 参数量 < 20M(防误解冻 1B)

# [3] LoRA 冷启动与梯度:use_lora=true 时
#     a. B 零初始化 → 同输入下输出与 use_lora=false 逐元素 allclose(atol=0)
#     b. 反传后所有 LoRA A/B .grad is not None,教师原始参数 .grad is None
#     c. trainable 计数 = 冻结版 + 6.3M(lora_targets 默认值下)

# [4] 分发零回归(§4.1 硬性要求):
#     a. 必须在**新起子进程**中执行(subprocess 跑 python -c;若在验收脚本主进程里查,
#        脚本自身早已 import 过新包,sys.modules 断言必然假失败 —— 审查问题 #5):
#        子进程内构造 A0 配置的 DrivoRModel,断言 image_backbone 类型是 ImgEncoder,
#        且 "navsim.agents.drivoR_vggt_omega" not in sys.modules(注意查完整模块名,
#        不是裸 "drivoR_vggt_omega")
#     b. 强证据:A0 配置、pl.seed_everything 同种子、假 batch 1 个训练 step,
#        loss 与打分发补丁前逐位一致(bitwise equal)

# [5] 模式钉死:model.train() 之后 assert not model.image_backbone.aggregator.training

# [6] 显存冒烟 + checkpoint 生效探针:batch=4、grad_checkpointing=true,
#     单卡 1 个 step(前向+反传+step),冻结版与 LoRA 版各跑一次:
#     a. 不 OOM,记录 torch.cuda.max_memory_allocated 留档 —— §8 预算的实测校准
#     b. **checkpoint 生效证明**(审查问题 #1/#4 的回归防线):给 run_block 包计数 hook,
#        1 次前向+反传后计数 == 2 × depth(前向 48 + backward 重算 48;
#        若门控失效退化为普通前向,计数只有 48,直接 fail)

# 附带:builder 回归 —— VggtOmegaFeatureBuilder 输出 与 官方 load_and_preprocess_images(jpg 路径)
#     对 8 个样本 cosine > 0.999(桥接函数已过 C1 三方验收,此处只是回归确认)
```

---

## 10. 备选变体 A2′-lite(不默认跑,记录在案)

教师自带 64 register(= C1 缓存内容)直接当 scene token,无注入、无教师在线前向:`geo_proj` 式投影 + 从零 decoder。实现 = C1 代码开 `c1_vggt.enabled=true` 且关掉 DINO 分支(memory 只留 64 几何 token)。价值:训练成本 ≈A1、完全复用 C1 缓存,给出"冻结几何 token 裸奔"的快速下界;若主变体与 lite 差距很小,说明注入 token 从冻结主干里读不出额外信息。仅当 Phase 2 有富余算力时跑。

---

## 11. 已知坑

1. **agent 包名不要叫裸 `vggt_omega`**:仓库顶层已有官方包 `vggt_omega/`,同名会造成人读混淆与部分工具的模块解析歧义 → 用 `drivoR_vggt_omega`。
2. **分发 import 必须惰性**(放分支内):顶层 import 会形成 drivoR → drivoR_vggt_omega → drivoR 的循环导入。
3. **builder 唯一名必须换**(`drivor_vggt_omega_feature`),否则与 DrivoR 特征缓存互相污染——图像尺寸/归一化完全不同,错读缓存静默得到垃圾输入。
4. **归一化恰好一次**:builder 输出 [0,1],归一化在 `SceneTokenAggregator.forward` 内部(官方位置)。任何人在 encoder 外再做 ImageNet 归一化都是双重归一化,静默出错(验收 [1] 兜底)。
5. **冻结与 LoRA 手术的顺序不可颠倒**:先 `requires_grad_(False)` 整体冻结,再 `apply_lora_to_blocks`(新建层默认可训练);反过来 LoRA 会被一起冻掉(验收 [3]b 兜底)。
6. **Lightning 的 `.train()` 波及主干** → `VggtOmegaImgEncoder.train()` 覆写钉死 eval(LoRA 线性层无 dropout,eval 不影响其训练)。**连带铁律**:aggregator 内部任何逻辑都不得以 `self.training` 做分支(它被钉死恒 False)——checkpoint 门控只能用 `torch.is_grad_enabled()`(评测路径在 no_grad 下,`abstract_agent.py:78`),违反即静默关闭 checkpoint、显存预算作废(审查问题 #1/#4,验收 [6]b 兜底)。
7. **DDP**:requires_grad=False 参数不进 reducer;LoRA 参数正常同步;冻结 1.2B 权重初始化时广播一次(~5GB),启动慢属正常。
8. **checkpoint 体积** ≈4.9GB/份(含冻结主干),`save_top_k=1 + save_last` 两份,注意配额。
9. **评测路径**:`run_pdm_score` 走同一 builder + 在线前向,无缓存依赖;延迟预计 ≥C1 的 250ms,本 agent 不是部署方案,仅记录不优化。
10. **对 C1 代码的唯一依赖**是 `preprocess_arrays_for_teacher`(纯函数);c1_vggt.py 若重构,这是本包唯一跟动点。
11. **溯源**(项目约定,不用 git):实验记录写 `weight/vggt_omega_1b_512.pt` 与 `aggregator.py`/`attention.py`/`dinov2_lora.py` 的 sha256;复用 C1 的 `file_sha256` 工具。

---

## 12. 建议实施顺序

1. `vggt_omega_backbone.py`(SceneTokenAggregator + VggtOmegaImgEncoder,先不接 LoRA)→ 验收 [1][2]。
2. `apply_lora_to_blocks` 接入 → 验收 [3](冷启动等价 + 梯度 + 参数计数)。
3. `drivor_model.py` 分发 4 行 → 验收 [4](原 model_name 走原路径,A0/A1/C1 零回归)。
4. `vggt_omega_features.py` + `vggt_omega_agent.py` + 两个 yaml → 假 batch 冒烟 1 个 step + 验收 [5][6](显存留档)。
5. `scripts/vggt_omega_acceptance_checks.py` 全过、留档(含 max_memory 与 checkpoint sha256)。
6. 实验 A2′(冻结版):navtrain 10 epochs × 3 seeds(消融协议),读数记入 overview 实验矩阵:
   (A2′ − A1 = ___ ± ___;TLC = ___;**若 A2′ 意外接近或超过 A1 → 命令行开 `use_lora=true` 跑 A2-LoRA**)

## 状态

☐ 代码实现 → ☐ 验收断言留档 → ☐ A2′ 训练中 → ☐ 完成(A2-LoRA:☐ 未触发 / ☐ 已触发)

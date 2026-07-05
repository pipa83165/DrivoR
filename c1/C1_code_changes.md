# C1 代码改动文档 — 冻结 VGGT-Ω 并联几何分支

> 依据 `C1.md` 与 `overview.md`，基于当前 repo（DrivoR 官方代码，`vggt_omega/` 官方代码已复制入库，checkpoint 在 `weight/vggt_omega_1b_512.pt`）。
> 本文档中所有行号引用以当前 repo 状态为准。
> 版本标识约定：**不使用 git**。指纹中所有"来源版本"字段一律用**文件内容 sha256**（`load_fn.py`、缓存脚本自身、checkpoint），比 commit 更直接反映数值行为是否变化。

---

## 0. 代码勘察结论（写入缓存 metadata 的事实依据）

| 事项 | 结论 | 出处 |
|---|---|---|
| `vggt_dim=2048` 语义 | 每个缓存层输出 = `cat([frame_tokens, tokens], dim=-1)`：**前 1024 维 = frame-attention 输出，后 1024 维 = global/register-attention 输出**（1B hidden=1024 的拼接）。C1.md 问题 3 已闭环 | `vggt_omega/models/aggregator.py:150` |
| register 切片 | `camera_and_register_tokens` 形状 `[B, S, 17, 2048]`；`[:, :, :1]`=camera token（丢弃），`[:, :, 1:]`=16 registers | `vggt_omega/models/vggt_omega.py:48`，`aggregator.py:83` |
| reference frame | `slice_expand_and_flatten` 给**序列第 0 帧**专属 token → 前视 cam_f0 必须排第 0 | `aggregator.py:246-250` |
| 归一化位置 | ImageNet mean/std 归一化在 `Aggregator.forward` **内部**；`load_and_preprocess_images` 只输出 [0,1] 张量 | `aggregator.py:108`，`utils/load_fn.py` |
| head 关闭 | `VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)` 即可不构建 camera/depth/alignment head | `vggt_omega/models/vggt_omega.py:19-33` |
| 教师 autocast | `VGGTOmega.forward` 内部已对 aggregator 做 bf16 autocast，外层只需 `eval()+no_grad` | `vggt_omega.py:39-41` |
| 原始 jpg 路径 | `scene_loader.scene_frames_dicts[token][num_history_frames-1]["cams"]["CAM_F0"]["data_path"]`（相对 `sensor_blobs_path`）| `navsim/common/dataloader.py:122`，`dataclasses.py:75` |
| DrivoR memory 形状 | `scene_features` = `(B, 4×16, 256)`，同时进 `trajectory_decoder` 与 `scorer_attention` 的 cross-attention | `drivor_model.py:150,159,178` |
| batch 中 token | `Dataset.__getitem__` 已注入 `features["scenario_token"]`（str） | `navsim/planning/training/dataset.py:289` |
| 教师输入尺寸 | NAVSIM 1920×1080（AR=0.5625，在 [0.5,2.0] 内不裁剪），balanced/512/patch16 → **688×384**（43×24 patches，每帧 17+1032=1049 token，4 帧联合 attention ≈4196 token） | `load_fn.py:85-91` 手推 |

**相机顺序（写入指纹）**：`[cam_f0, cam_l0, cam_r0, cam_b0]`（前/前左/前右/后），对应 scene dict 键 `["CAM_F0","CAM_L0","CAM_R0","CAM_B0"]`。注意这与 DrivoR 主干侧的相机顺序（f0,b0,l0,r0，`drivor_features.py:79`）**不同且无需一致**——几何分支有独立可学习相机 embedding。

---

## 1. 文件清单

| 操作 | 文件 | 内容 |
|---|---|---|
| 新增 | `navsim/agents/drivoR/c1_vggt.py` | GeoProjector、教师封装、指纹构建/校验、缓存读取、在线预处理桥接、mode 逻辑 |
| 新增 | `scripts/cache_c1_vggt_tokens.py` | 离线缓存生成（教师前向 → fp16 `.pt` + metadata + noise 统计量） |
| 新增 | `navsim/planning/script/config/training/c1_training.yaml` | C1 训练入口配置 |
| 新增 | `scripts/c1_acceptance_checks.py` | Phase 0.6 三条验收断言（缓存生成前跑一遍留档） |
| 修改 | `navsim/planning/script/config/common/agent/drivoR.yaml` | 加 `c1_vggt` 配置块，默认 `enabled: false` |
| 修改 | `navsim/agents/drivoR/drivor_model.py` | geo_proj + memory 拼接 + shuffle/noise/drop |
| 修改 | `navsim/planning/training/dataset.py` | `Dataset`/`CacheOnlyDataset` 按 token 加载几何缓存 |
| 修改 | `navsim/planning/script/run_training.py` | 把几何缓存目录传进两个 Dataset |
| 修改 | `navsim/agents/drivoR/drivor_features.py` | 在线模式下额外输出教师预处理图像（仅 eval/延迟测量用） |

**不改**：decoder 结构、损失（`drivor_loss.py`）、scorer、LoRA 主干、训练协议。相对 A1 仅新增"缓存读取 → geo_proj → memory 拼接"，符合 C1.md「与基线的差异点」。

---

## 2. 配置改动

### 2.1 `navsim/planning/script/config/common/agent/drivoR.yaml`

在 `config:` 下追加（默认全关，A0/A1 行为不变）：

```yaml
  ####################
  # C1: frozen VGGT-Omega parallel geometry branch (probe, not deployment)
  c1_vggt:
    enabled: false
    mode: normal            # normal | shuffle | noise | drop
    source: cache           # cache: 训练/训练内验证; online: run_pdm_score 评测与延迟测量
    cache_dir: null         # 必须通过命令行/实验 yaml 指定，禁止硬编码绝对路径
    checkpoint_path: weight/vggt_omega_1b_512.pt
    vggt_dim: 2048          # = frame-attn(1024) ‖ global-attn(1024)，语义已核实
    num_registers: 16
    use_camera_token: false # 消融开关，默认丢弃 camera token
    joint_forward: true     # 4 相机联合前向（默认）；false = 各相机独立前向（消融 b）
    preprocess_mode: balanced   # 官方默认；max_size 降级为 C3a
    image_resolution: 512
    shuffle_seed: 20260704  # 与训练种子解耦，独立记录
    force_ignore_fingerprint: false   # 逃生口，使用时打 WARNING
```

### 2.2 新增 `navsim/planning/script/config/training/c1_training.yaml`

```yaml
defaults:
  - default_training
  - _self_

experiment_name: c1_vggt

agent:
  config:
    c1_vggt:
      enabled: true
      mode: normal
      source: cache
      cache_dir: ${oc.env:C1_GEO_CACHE_DIR}   # 从环境变量取，可移植
```

对照实验用命令行覆盖：`agent.config.c1_vggt.mode=shuffle|noise|drop`。

---

## 3. 新增 `navsim/agents/drivoR/c1_vggt.py`

单文件承载全部 C1 专属逻辑。骨架如下（约 300 行）：

### 3.1 常量与指纹

```python
import hashlib, json, logging, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# 前 / 前左 / 前右 / 后。前视必须排第 0（VGGT reference frame，见 aggregator.slice_expand_and_flatten）
C1_CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")
C1_SCENE_DICT_KEYS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

CACHE_DTYPE = torch.float16
METADATA_FILENAME = "metadata.json"
STATS_FILENAME = "noise_stats.pt"          # per-dim mean/std（D=2048），供 noise 对照


# cfg 访问约定：c1_vggt 块可能是 OmegaConf DictConfig（训练）也可能是 plain dict（测试/脚本）。
# DictConfig 两种写法都支持，plain dict 不支持属性访问 —— 因此本模块内一律用 cfg["key"] 下标访问，
# 可选键用 cfg.get(key, default)（两种类型都有 .get）。禁止 cfg.key 属性写法。

from functools import lru_cache

@lru_cache(maxsize=8)
def file_sha256(path) -> str:
    """checkpoint ~数 GB，哈希数秒级；lru_cache 保证进程内只算一次。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_fingerprint(cfg, ckpt_sha256: str, cache_script_path=None) -> dict:
    """学生侧可学习参数（branch/camera embedding）不参与指纹；拼接顺序参与。
    版本字段不用 git commit，一律用文件内容 sha256。"""
    import vggt_omega.utils.load_fn as _load_fn_mod
    return {
        "checkpoint_name": Path(cfg["checkpoint_path"]).name,
        "checkpoint_sha256": ckpt_sha256,
        "vggt_dim": int(cfg["vggt_dim"]),
        "vggt_dim_semantics": "cat(frame_attn_1024, global_attn_1024)",  # aggregator.py:150
        "num_registers": int(cfg["num_registers"]),
        "tokens_per_camera": tokens_per_camera(cfg),   # 16，或 use_camera_token=True 时 17
        "camera_order": list(C1_CAMERA_ORDER),
        "preprocess": {
            "load_fn": "vggt_omega.utils.load_fn.load_and_preprocess_images",
            "load_fn_sha256": file_sha256(_load_fn_mod.__file__),   # 直接哈希官方 load_fn 源文件
            "mode": str(cfg["preprocess_mode"]),
            "image_resolution": int(cfg["image_resolution"]),
            "color_aug": False,
        },
        "joint_forward": bool(cfg["joint_forward"]),
        "use_camera_token": bool(cfg["use_camera_token"]),
        "cache_dtype": "float16",
        # 生成侧记录字段（不参与 raise 级校验）：
        "cache_script_sha256": file_sha256(cache_script_path) if cache_script_path else None,
    }


def build_fingerprint_from_cfg(cfg) -> dict:
    """校验侧入口（§5 数据集启动时调用）：从 agent config 的 c1_vggt 块构造 expected 指纹，
    负责计算 checkpoint sha256（file_sha256 已 lru_cache，进程内只算一次）。
    不传 cache_script_path —— 该字段是生成侧记录，不参与 STRICT_KEYS 校验。"""
    return build_fingerprint(cfg, ckpt_sha256=file_sha256(cfg["checkpoint_path"]))


# raise 级校验白名单：只有这些字段不匹配才 raise；
# cache_script_sha256 等生成侧记录字段只做 WARNING 级比较（否则脚本无关改动会让旧缓存失效）
STRICT_KEYS = (
    "checkpoint_name", "checkpoint_sha256",
    "vggt_dim", "vggt_dim_semantics",
    "num_registers", "tokens_per_camera",
    "camera_order", "preprocess",          # preprocess 整个子 dict 严格比较（含 load_fn_sha256）
    "joint_forward", "use_camera_token", "cache_dtype",
)


def validate_fingerprint(expected: dict, cache_dir: Path, force_ignore: bool = False) -> None:
    """强制校验（C1.md 问题 4）。STRICT_KEYS 不匹配直接 raise；force_ignore 时打 WARNING。"""
    meta = json.loads((cache_dir / METADATA_FILENAME).read_text())
    mismatches = {k: (expected[k], meta.get(k)) for k in STRICT_KEYS if meta.get(k) != expected[k]}
    if mismatches:
        msg = f"C1 geo-cache fingerprint mismatch at {cache_dir}: {mismatches}"
        if force_ignore:
            logger.warning("FORCE-IGNORE-FINGERPRINT: %s", msg)
        else:
            raise RuntimeError(msg)
    soft = {k: (v, meta.get(k)) for k, v in expected.items()
            if k not in STRICT_KEYS and meta.get(k) != v}
    if soft:
        logger.warning("C1 geo-cache non-strict metadata differs: %s", soft)


def tokens_per_camera(cfg) -> int:
    """唯一事实来源：use_camera_token=True 时每相机 17 个 token（camera token 排在 index 0）。
    GeoProjector 形状 / 缓存 shape 校验 / noise shape / c1_memory_len 断言 / metadata 全部由此派生。"""
    return int(cfg["num_registers"]) + (1 if cfg["use_camera_token"] else 0)
```

### 3.2 GeoProjector（geo_proj，修复问题 1）

```python
from navsim.agents.drivoR.timm_layers import LayerScale

class GeoProjector(nn.Module):
    """[B, 4, T, D] fp16/fp32 -> [B, 4*T, d_model]，T = tokens_per_camera（默认 16，
    use_camera_token=True 时 17），t=0 时输出恒为 0。内部固定 float32 计算。"""

    def __init__(self, vggt_dim: int, d_model: int, num_cams: int = 4, tokens_per_cam: int = 16):
        super().__init__()
        self.tokens_per_cam = tokens_per_cam
        self.input_ln = nn.LayerNorm(vggt_dim)      # 高范数 register 前置 LN（防 attention sink）
        self.proj = nn.Linear(vggt_dim, d_model)    # 常规初始化（不做近零 init）
        self.branch_embed = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.cam_embed = nn.Parameter(torch.randn(1, num_cams, 1, d_model) * 1e-3)
        self.out_ln = nn.LayerNorm(d_model)
        # 零初始化 LayerScale 门控：LN 之后再乘 γ=0，冷启动输出精确为 0（问题 1 的修法）
        self.gate = LayerScale(d_model, init_values=0.0, inplace=False)

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        B, N, T, D = geo.shape
        assert T == self.tokens_per_cam, f"geo tokens per cam {T} != configured {self.tokens_per_cam}"
        x = self.proj(self.input_ln(geo.float()))
        x = x + self.branch_embed + self.cam_embed
        x = self.gate(self.out_ln(x))
        return x.reshape(B, N * T, -1)              # (B, 4*T, d_model)
```

顺序是「LN → Linear → +embed → LN → γ 门控」：门控放在**最末端**，LN 无法再把它拉回单位尺度。验收：t=0 输出范数 ≈ 0（见 §9）。

两个约定：
- 输入 fp16 缓存在此**升到 float32** 计算；**拼接前由调用方 `.to(scene_features.dtype)` 显式压回**（见 §4.2），避免与 DrivoR memory 拼接时发生隐式 dtype 提升或 AMP 行为异常。
- `use_camera_token=True` 时 camera token 排在每相机 index 0，缓存形状 `[4, 17, D]`，noise 统计与 shape、memory 长度断言全部随 `tokens_per_camera(cfg)` 派生，禁止散落硬编码 16/64。

### 3.3 教师封装（在线路径 + 缓存脚本共用）

```python
class C1VggtTeacher(nn.Module):
    """冻结 VGGT-Ω 1B，只留 aggregator（不构建 camera/depth/alignment head）。"""

    def __init__(self, checkpoint_path: str, use_camera_token: bool = False, joint_forward: bool = True):
        super().__init__()
        from vggt_omega.models.vggt_omega import VGGTOmega
        self.model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("model", state)
        # ckpt 里可能含 head 权重，本模型没有对应模块 → strict=False，
        # 但必须断言 aggregator 的 key 全部命中（missing 中不允许出现 aggregator.*）
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not [k for k in missing if k.startswith("aggregator.")], missing
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.use_camera_token = use_camera_token
        self.joint_forward = joint_forward

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 4, 3, H, W)，[0,1]（归一化在 aggregator 内部完成，勿在外面做）。
        返回 (B, 4, 16, 2048)（use_camera_token=True 时 (B, 4, 17, 2048)）。"""
        if self.joint_forward:
            out = self.model(images)["camera_and_register_tokens"]   # (B, 4, 17, 2048)
        else:   # 消融 b：各相机独立前向（每相机单帧序列）
            outs = [self.model(images[:, i:i + 1])["camera_and_register_tokens"] for i in range(images.shape[1])]
            out = torch.cat(outs, dim=1)
        return out if self.use_camera_token else out[:, :, 1:]        # 丢 camera token
```

### 3.4 在线预处理桥接（唯一允许的"非直调"点）

在线路径（`run_pdm_score` 评测、延迟测量）图像已是内存 numpy 数组，而官方 `load_and_preprocess_images` 只收路径。桥接函数**只替换"打开文件"一步，几何变换全部 import 官方私有函数**，禁止复刻数值逻辑：

```python
def preprocess_arrays_for_teacher(images_np: list, mode: str = "balanced",
                                  image_resolution: int = 512, patch_size: int = 16) -> torch.Tensor:
    """输入 4 个 HxWx3 uint8 数组（C1_CAMERA_ORDER 顺序），输出 (4, 3, H', W') [0,1]。
    与官方 load_and_preprocess_images 的差异仅在 Image.open 一步；
    一致性由三方 cosine>0.999 单测保证（scripts/c1_acceptance_checks.py）。"""
    from torchvision import transforms as TF
    from vggt_omega.utils.load_fn import (
        _crop_to_supported_aspect_ratio, _balanced_target_shape, _max_size_target_shape,
    )
    to_tensor = TF.ToTensor()
    out = []
    for arr in images_np:
        img = _crop_to_supported_aspect_ratio(Image.fromarray(arr).convert("RGB"))
        w, h = img.size
        ar = h / max(w, 1)
        th, tw = (_balanced_target_shape if mode == "balanced" else _max_size_target_shape)(
            ar, image_resolution, patch_size)
        out.append(to_tensor(img.resize((tw, th), Image.Resampling.BICUBIC)))
    return torch.stack(out)
```

### 3.5 缓存读取 + shuffle/noise provider（数据层，训练/评测/任意 batch size 同一路径）

**shuffle 不做 batch 内错配**：batch 内 randperm 在 `run_pdm_score`（batch=1）下无法执行，会破坏"评测也 shuffle"的对照定义。改为**数据加载层配对**：对样本 token t，从缓存全体 token 索引中随机抽 partner t'≠t，加载 t' 的几何 token。每次 `get()` 重新抽样即"每个 step 重新随机配对"；配对池从 batch 扩大到整个 split（与 C1.md"batch 内"字面的唯一偏差，统计等价、定义更干净，**记录在案**）。noise 同样在此层生成（投影前 D=2048 空间、匹配缓存全体 per-dim 统计），使 `drivor_model` 侧完全不区分 normal/shuffle/noise。

```python
def geo_cache_file(cache_dir: Path, token: str) -> Path:
    return cache_dir / token[:2] / f"{token}.pt"    # 按前 2 字符分片，避免 85k 文件同目录


class C1GeoTokenProvider:
    """mode ∈ {normal, drop}: 读本样本缓存；shuffle: 读随机 partner 的缓存；noise: 生成统计匹配噪声。
    训练集/验证集/run_pdm_score 共用本类（同一函数、同一开关——已知坑 4）。"""

    def __init__(self, cache_dir, mode: str, shuffle_seed: int, expected_shape, worker_offset: int = 0):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.expected_shape = tuple(expected_shape)          # (4, tokens_per_camera, vggt_dim)
        if mode == "shuffle":
            # token_index.json 由缓存脚本生成（全体已缓存 token 的有序列表）
            self.all_tokens = json.loads((self.cache_dir / "token_index.json").read_text())
            # 与训练种子解耦；worker_offset = global_rank * num_workers + worker_id（见 §5），
            # 否则 DDP 各 rank / DataLoader 各 worker 会产生相同配对序列
            self.rng = np.random.default_rng(shuffle_seed + worker_offset)
            logger.info("C1-shuffle rng seed=%d (base %d + offset %d)",
                        shuffle_seed + worker_offset, shuffle_seed, worker_offset)
        if mode == "noise":
            stats = torch.load(self.cache_dir / STATS_FILENAME)
            self.mean, self.std = stats["mean"].float(), stats["std"].float()   # (vggt_dim,)
            self.rng_t = torch.Generator().manual_seed(shuffle_seed + worker_offset)

    def get(self, token: str) -> torch.Tensor:
        if self.mode == "noise":
            return (self.mean + self.std * torch.randn(
                *self.expected_shape, generator=self.rng_t)).to(torch.float16)
        if self.mode == "shuffle":
            while True:                                       # 拒绝采样保证 partner != 本样本
                partner = self.all_tokens[int(self.rng.integers(len(self.all_tokens)))]
                if partner != token:
                    break
            token = partner
        path = geo_cache_file(self.cache_dir, token)
        if not path.is_file():
            # 已知坑 2：禁止静默回退 drop，直接 raise
            raise FileNotFoundError(f"C1 geo-token cache missing for token {token}: {path}")
        geo = torch.load(path, map_location="cpu")            # (4, T, 2048) fp16
        assert geo.shape == self.expected_shape and geo.dtype == torch.float16, \
            f"corrupt/mismatched geo cache {path}: {geo.shape} {geo.dtype}"
        return geo
```

`worker_offset` 的接法：Dataset 里惰性构造 provider（首次 `__getitem__` 时读 `torch.utils.data.get_worker_info()`，主进程为 0）。种子与配对序列都进日志，保证可复现（已知坑 3）。

---

## 4. 修改 `navsim/agents/drivoR/drivor_model.py`

### 4.1 `__init__`（在 `self.scorer = Scorer(config)` 之后、`self.b2d` 之前插入）

```python
# ---------------- C1: frozen VGGT-Omega geometry branch ----------------
c1 = config.get("c1_vggt", None)
self.c1_enabled = bool(c1 and c1.get("enabled", False))
if self.c1_enabled:
    from .c1_vggt import GeoProjector, C1VggtTeacher, tokens_per_camera
    self.c1_cfg = c1
    self.c1_mode = c1["mode"]
    assert self.c1_mode in ("normal", "shuffle", "noise", "drop")
    self.c1_tokens_per_cam = tokens_per_camera(c1)
    self.geo_proj = GeoProjector(c1["vggt_dim"], config.tf_d_model,
                                 num_cams=4, tokens_per_cam=self.c1_tokens_per_cam)

    if c1["source"] == "online":
        # 教师不注册为子模块：不进 state_dict / parameters() / DDP，checkpoint 不膨胀
        self.__dict__["_c1_teacher"] = C1VggtTeacher(
            c1["checkpoint_path"], c1["use_camera_token"], c1["joint_forward"])
```

shuffle / noise 已全部下沉到数据层的 `C1GeoTokenProvider`（§3.5）——模型侧对 normal/shuffle/noise **不做任何区分**，只认 `features["c1_geo_tokens"]`；模型里唯一的 mode 逻辑是 drop 的推理时截断。

教师用 `self.__dict__` 挂载的含义：`state_dict()`、`parameters()`（→ AdamW，`drivor_agent.py:245`）、DDP 广播都不包含 1B 教师；代价是 `.to(device)` 不会自动搬运，需在 forward 里首次使用时 `self._c1_teacher.to(x.device)`（惰性一次）。

### 4.2 `forward`（在 `scene_features = torch.cat(scene_features, dim=1)`（第 150 行）之后插入）

```python
# ---------------- C1: append 64 geometry tokens to decoder memory ----------------
if self.c1_enabled:
    scene_features = self._c1_extend_memory(scene_features, features)
```

新增方法：

```python
def _c1_extend_memory(self, scene_features, features):
    # drop 对照：推理时物理截断回 64（不置零！问题 2），训练照常
    if self.c1_mode == "drop" and not self.training:
        return scene_features

    if "c1_geo_tokens" in features:                        # cache 路径（normal/shuffle/noise 均由
        geo = features["c1_geo_tokens"].to(scene_features.device)   # 数据层 provider 备好）
    elif self.c1_cfg["source"] == "online":                # online 路径（评测/延迟）
        teacher = self.__dict__["_c1_teacher"]
        if next(teacher.parameters()).device != scene_features.device:
            teacher.to(scene_features.device)
        geo = teacher(features["c1_teacher_images"].to(scene_features.device))
    else:
        raise RuntimeError("C1 enabled but no geo tokens: cache 缺失禁止静默回退（known-pitfall #2）")

    geo = self.geo_proj(geo)                               # (B, 4*T, 256)，内部 float32
    geo = geo.to(scene_features.dtype)                     # 显式压回 memory dtype，防隐式提升/AMP 异常
    return torch.cat([scene_features, geo], dim=1)         # (B, 64 + 4*T, 256) 联合 memory
```

融合点严格限于 decoder memory：拼接后的 `scene_features` 原样流入 `trajectory_decoder`（159 行）与 `scorer_attention`（178 行），**主干 ViT、traj_tokens、损失均不碰**。

同时在 forward 的 `output` 里加一行，供验收断言用（问题 2 的 K/V 长度检查）：

```python
# drop+eval 时必须为 64；训练时为 64 + 4*tokens_per_camera（默认 128，±camera token 时 132）
output["c1_memory_len"] = scene_features.shape[1]
```

---

## 5. 修改 `navsim/planning/training/dataset.py`

两个 Dataset 类各加一个可选参数，改动对称：

```python
# Dataset.__init__ 与 CacheOnlyDataset.__init__ 签名末尾追加：
    geo_token_cfg=None,   # None 或 agent config 的 c1_vggt 块（enabled 且 source==cache 时传入）
```

`__init__` 里（两个类相同）：

```python
self._geo_cfg = None
self._geo_provider = None    # 惰性构造：需在 worker 进程内拿 get_worker_info 混种子
if geo_token_cfg is not None and geo_token_cfg.get("enabled") and geo_token_cfg.get("source") == "cache":
    from navsim.agents.drivoR.c1_vggt import validate_fingerprint, build_fingerprint_from_cfg
    validate_fingerprint(   # 启动时一次性强制校验（问题 4，STRICT_KEYS 白名单）
        build_fingerprint_from_cfg(geo_token_cfg),
        Path(geo_token_cfg["cache_dir"]),
        force_ignore=geo_token_cfg.get("force_ignore_fingerprint", False),
    )
    self._geo_cfg = geo_token_cfg
```

`_load_scene_with_token`（两个类）以及 `Dataset.__getitem__` 的非缓存分支，在 features 组装完成后：

```python
if self._geo_cfg is not None:
    if self._geo_provider is None:      # 首次访问时在 worker 进程内构造
        from navsim.agents.drivoR.c1_vggt import C1GeoTokenProvider, tokens_per_camera
        info = torch.utils.data.get_worker_info()
        # DDP 多卡：仅 worker_id 不够，各 rank 的同号 worker 会产生相同配对序列。
        # worker 进程内 torch.distributed 通常未初始化，rank 从继承的环境变量取。
        rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                else int(os.environ.get("RANK", 0)))
        num_workers = info.num_workers if info else 1
        worker_offset = rank * num_workers + (info.id if info else 0)
        self._geo_provider = C1GeoTokenProvider(
            self._geo_cfg["cache_dir"], self._geo_cfg["mode"], self._geo_cfg["shuffle_seed"],
            expected_shape=(4, tokens_per_camera(self._geo_cfg), self._geo_cfg["vggt_dim"]),
            worker_offset=worker_offset,
        )
    features["c1_geo_tokens"] = self._geo_provider.get(token)   # (4, T, 2048) fp16
```

shuffle/noise 都发生在 provider 内（§3.5），与 batch size 无关；default collate 堆成 `(B, 4, T, 2048)`，无需自定义 collate。IO 按 C1.md 约定不预先优化：先实测 dataloader 吞吐，掉吞吐再转 memmap/LMDB。

**run_pdm_score 跑 shuffle/noise 对照的路线**：对照读数首选训练内 navval 验证循环（cache 路径天然可用，与 C1.md 预期结果的 navval 口径一致）。如需在 `run_pdm_score` 上跑对照：给 `DrivoRAgent` 加 `requires_scene=True`（C1 cache-source 时），override `compute_trajectory(agent_input, scene)` 从 `scene.scene_metadata.initial_token` 取 token → 经同一个 `C1GeoTokenProvider` 注入 `features["c1_geo_tokens"]`（`run_pdm_score.py:81-82` 已支持 requires_scene 分支）。C1 主变体的 run_pdm_score 评测不受影响——走 online 路径，不需要 token。

---

## 6. 修改 `navsim/planning/script/run_training.py`

把 agent 配置里的 c1 块传给数据集（4 处构造，改法相同）：

```python
# main() 里 instantiate(cfg.agent) 之后：
c1_cfg = OmegaConf.select(cfg, "agent.config.c1_vggt")

# 4 处 Dataset(...) / CacheOnlyDataset(...) 构造各加一个参数：
    geo_token_cfg=c1_cfg,
```

（`build_datasets` 需把 `c1_cfg` 作为参数透传，或直接 `cfg.agent.config.get("c1_vggt", None)`。文件头部已 import `DictConfig`，补 `from omegaconf import OmegaConf`。）

---

## 7. 修改 `navsim/agents/drivoR/drivor_features.py`

只服务**在线模式**（`run_pdm_score` 评测、延迟测量；训练永远走缓存）。在 `DrivoRFeatureBuilder._get_camera_feature` 末尾、`return data` 之前：

```python
# C1 online: teacher-preprocessed images, fully decoupled from DrivoR pipeline above
c1 = self._config.get("c1_vggt", None)
if c1 and c1.get("enabled") and c1.get("source") == "online":
    from navsim.agents.drivoR.c1_vggt import preprocess_arrays_for_teacher, C1_CAMERA_ORDER
    raw = [getattr(cameras_all, name).image for name in C1_CAMERA_ORDER]
    data["c1_teacher_images"] = preprocess_arrays_for_teacher(
        raw, mode=c1["preprocess_mode"], image_resolution=c1["image_resolution"])
```

实现细节：函数开头 `cameras = agent_input.cameras[-1]` 之后马上被覆盖成 list（79 行），所以需要在覆盖前留一份 `cameras_all = cameras`。教师输入**不做颜色增广**、不做 DrivoR 的 resize/ImageNet 归一化——与主干管线完全解耦。

`compute_trajectory`（`abstract_agent.py:75`）会对所有 features `unsqueeze(0)`，`c1_teacher_images` 变成 `(1, 4, 3, H, W)`，正好是教师期望的形状。

---

## 8. 新增 `scripts/cache_c1_vggt_tokens.py`

CLI：

```
python scripts/cache_c1_vggt_tokens.py \
    --output-dir <可配置，必填，禁止硬编码> \
    --split navtrain \
    --checkpoint weight/vggt_omega_1b_512.pt \
    [--shard-index 0 --num-shards 8]        # 多卡并行分片
    [--preprocess-mode balanced --image-resolution 512]
    [--independent-forward] [--use-camera-token]   # 消融开关
```

流程：

```python
# 1. SceneLoader(navtrain scene_filter, sensor_config=无相机加载即可——只需要路径不需要解码)
#    注意：给 SceneLoader 传 SensorConfig 全空可避免 loader 预解码图像；
#    jpg 路径直接从 scene_frames_dicts 取。
loader = SceneLoader(sensor_blobs_path, data_path, scene_filter, SensorConfig.build_no_sensors())

teacher = C1VggtTeacher(args.checkpoint, args.use_camera_token, not args.independent_forward).cuda()

# 2. 逐 token（分片内），断点续传：文件已存在则跳过。
#    因为写入是原子的（见步骤 4），存在即完整；--validate-existing 可选深校验
#    （torch.load 后断言 shape==(4, tokens_per_camera, D) 且 dtype==fp16，坏文件删除重算）
frame = loader.scene_frames_dicts[token][scene_filter.num_history_frames - 1]
paths = [str(sensor_blobs_path / frame["cams"][k]["data_path"]) for k in C1_SCENE_DICT_KEYS]

# 3. 官方 load_fn 直接 import 调用（缓存路径零复刻）
from vggt_omega.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(paths, mode=args.preprocess_mode,
                                    image_resolution=args.image_resolution)   # (4,3,384,688)

# 4. 教师前向（内部 bf16 autocast），取 registers，fp16 原子写入：
#    先写同目录临时文件，再 os.replace 原子替换——中断永远不会留下能被误判为完成的半截文件
geo = teacher(images.cuda().unsqueeze(0))[0]          # (4, T, 2048)，T=tokens_per_camera
path = geo_cache_file(out_dir, token); path.parent.mkdir(exist_ok=True, parents=True)
tmp = path.with_suffix(f".tmp.{os.getpid()}")         # 带 pid，多分片误配到同 token 也不互踩
torch.save(geo.to(torch.float16).cpu(), tmp)
os.replace(tmp, path)                                 # 同文件系统原子 rename（Windows/Linux 均成立）

# 5. 累积 per-dim 运行统计（float64 Welford，跨 4×T 全 token 维度聚合到 (2048,)）
#    多分片时各分片存 partial stats，最后 --merge-stats 合并成 noise_stats.pt {mean, std}
#    注意：续传跳过的样本不重复计入统计——统计只在 --merge-stats 阶段从最终文件集重新聚合，
#    或分片记录已计入的 token 集合（推荐前者，简单且幂等）

# 6. 全部完成后（--finalize）：
#    a. 扫描缓存目录生成 token_index.json（全体已缓存 token 有序列表，供 shuffle provider 配对）
#    b. 写 metadata.json = build_fingerprint(...)（含 checkpoint sha256、load_fn 文件 sha256、
#       缓存脚本自身 sha256、tokens_per_camera、camera_order、preprocess 三元组、
#       joint/independent、±camera token、dtype——全部基于文件哈希，不依赖 git）
#    metadata.json 与 token_index.json 同样走 tmp + os.replace 原子写
```

体量核算：85k × 4×16×2048 × fp16 = **≈21.4 GB**，符合 C1.md 预算。

也可批量（batch 若干 token 一起前向）加速；单样本 4 帧 ≈4200 token 的联合 attention，1B bf16 单 A100 可 batch 4–8。

---

## 9. 新增 `scripts/c1_acceptance_checks.py`（Phase 0.6 验收，缓存生成前跑一遍留档）

三条断言，全部通过后打印指纹并退出 0：

```python
# [1] 问题 1 smoke test：冷启动零输出
proj = GeoProjector(2048, 256)
out = proj(torch.randn(2, 4, 16, 2048) * 30)          # 高范数模拟 register
assert out.abs().max().item() == 0.0, "geo_proj at t=0 must output exact zeros"

# [2] 问题 2：drop 模式推理时 K/V 长度 = 64（物理移除，非置零）
T = tokens_per_camera(c1_cfg)                          # 默认 16；use_camera_token 时 17
model = DrivoRModel(cfg_with_c1(mode="drop")).eval()
out = model(dummy_features_with_geo())
assert out["c1_memory_len"] == 64
model.train(); out = model(dummy_features_with_geo())
assert out["c1_memory_len"] == 64 + 4 * T              # 训练时带几何 token（默认 128）

# [3] 问题 5 升级版：三方一致性，两两 cosine > 0.999
#   a. 缓存路径: load_and_preprocess_images(jpg paths) -> teacher -> 存 fp16 -> 读回
#   b. 在线路径: np.array(Image.open(path)) -> preprocess_arrays_for_teacher -> teacher
#   c. 官方直调: load_and_preprocess_images(jpg paths) -> teacher（不过 fp16 往返）
# 对 8 个 navtrain 样本，64 token 逐个算 cosine，min > 0.999
```

附带第 4 条（shuffle 可复现且必错配）：同 seed 两次构造 `C1GeoTokenProvider(mode="shuffle")`，对同一 token 序列产生**相同的 partner 序列**，且任一 partner ≠ 本样本 token；与 batch size 无关（用单样本逐个调用验证，覆盖 run_pdm_score 的 batch=1 场景）。

附带第 5 条（缓存完整性）：对已生成缓存抽样 N=100 个 `.pt`，断言 shape==(4, T, 2048)、dtype==fp16；断言缓存目录无残留 `*.tmp.*` 文件（原子写入生效）。

---

## 10. 延迟核算（供 C3 引用，可后置）

新增小脚本 `scripts/c1_latency_bench.py`（可在 C1 训练完成后再写）：batch=1、单 A100、无量化，3 次 warm-up 后取 10 次均值，分解报告四段——教师前向（`C1VggtTeacher`）‖ `geo_proj` ‖ decoder 增量（128 vs 64 memory 各跑一遍取差）‖ 端到端 `compute_trajectory`。预期端到端 ~250ms。

---

## 11. C1.md 问题/坑 → 落实对照

| C1.md 条目 | 本文档落实处 |
|---|---|
| 问题 1（LN 打架）| §3.2 末端 γ=0 LayerScale；§9-[1] smoke test |
| 问题 2（drop 置零伪影）| §4.2 `drop` 分支直接 return 原 64 memory；§9-[2] 断言 |
| 问题 3（2048 语义）| §0 已核实 = frame‖global 拼接；写入指纹 `vggt_dim_semantics`；缓存 fp16 ≈21.4GB |
| 问题 4（指纹可选）| §3.1 `STRICT_KEYS` 白名单 raise 级校验（生成侧字段仅 WARNING）；§5 数据集启动时强制执行；`force_ignore_fingerprint` 打 WARNING |
| 问题 5（两路一致性）| §3.4 桥接只换文件打开步 + §9-[3] 三方 cosine>0.999 |
| 坑 1（embedding 不进指纹）| §3.1 注释明示；拼接顺序 `camera_order` 在指纹内 |
| 坑 2（缓存缺失禁止回退）| §3.5 provider 内 raise；§4.2 else 分支 raise |
| 坑 3（shuffle 种子解耦）| 独立 `shuffle_seed` + provider 专用 rng（offset = global_rank × num_workers + worker_id，DDP 安全），与 `pl.seed_everything` 无关，种子进日志 |
| 坑 4（shuffle 训练/评测同路径）| §3.5 `C1GeoTokenProvider` 单一实现供训练/验证/run_pdm_score 共用；模型侧不区分 normal/shuffle/noise |
| shuffle 与 batch=1 | §3.5 配对下沉到数据层（partner 从整个 split 抽），与 batch size 无关；与 C1.md"batch 内"字面偏差已记录 |
| ±camera token 形状语义 | §3.1 `tokens_per_camera(cfg)` 单一事实来源，投影/缓存校验/noise/断言/metadata 全部派生 |
| dtype/AMP | §4.2 geo_proj 内部 float32，拼接前显式 `.to(scene_features.dtype)` |
| 缓存写入原子性 | §8 tmp + `os.replace`；续传存在即完整，`--validate-existing` 深校验 |
| output_dir 硬编码 | 缓存脚本 `--output-dir` 必填；训练侧 `cache_dir` 走环境变量/命令行 |
| 机制分析钩子（attention 质量等）| 不进本次改动（需要 `Attention` 返回权重，另起分析分支做，避免污染主实验代码） |

## 12. 建议实施顺序

1. `c1_vggt.py` + drivoR.yaml 配置块 → 验证 `enabled=false` 时 A0/A1 训练完全不受影响（默认路径零改动）。
2. `drivor_model.py` 融合 + `dataset.py`/`run_training.py` 接线 → 用假缓存（随机 fp16 张量 + 手写 metadata）跑 1 个 step 冒烟。
3. `cache_c1_vggt_tokens.py` + `drivor_features.py` 在线路径。
4. `scripts/c1_acceptance_checks.py` 三条断言全过、留档。
5. 正式生成 navtrain 缓存（~85k × 教师前向，多卡分片）→ dataloader 吞吐实测 → C1 训练（3 seeds）+ shuffle/noise/drop 对照。

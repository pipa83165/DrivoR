# geo_only — 教师几何 token 独占 decoder memory 的规划探针

> 给实现者的**改动指导**,不是最终代码。按此在 `DrivoR/` 仓库落地。
> 遵循仓库 `AGENTS.md`:改动最小、只碰必须碰的。
> 方案命名与定位见 `code_change_md/overview.md`;geo_only 完全复用并联几何融合的缓存与注入设施。

---

## 0. 目的与定位

回答:**冻结 VGGT-Ω 自带的 register token 能否独立承载规划**(固定教师令牌的独立承载能力探针)。
与"裸替"(可学习 scene token 注入冻结主干、在线前向)的分工:geo_only 的读出是**固定的**(教师为 3D 重建学的 register),没有梯度进主干——geo_only 掉点大**不能**推出"冻结主干里没有可用信息",只能说明"教师自带 token 裸奔不够"。

读数关系(全部入 `code_change_md/overview.md` 方案矩阵):

| 对比 | 回答的问题 |
|---|---|
| geo_only vs 基线 | 几何 token 裸奔 vs DINO 主干,谁承载规划更强 |
| geo_only vs 并联 | 同一批 token,"独占 memory" vs "与 DINO 并联"(唯一变量 = memory 里有无 DINO 64 token) |
| 裸替 vs geo_only | 可学习读出能从冻结主干里比教师自带 register 多榨出多少(裸替花 20–30h 的核心理由) |

定位:探针,非部署方案。训练成本 ≈ 甚至低于基线(教师走缓存、DINO 主干不构造不前向)。

## 1. 为什么必须从零训练(已定决策,勿改为评测期截断)

不能拿并联 checkpoint 在评测时截掉 DINO 那 64 个 token 冒充 geo_only,三个理由:

1. **分布错配不可归因**:并联 decoder 的 cross-attention、geo_proj、门控 γ 全部在"DINO token 始终在场"前提下优化;评测期抽走 DINO 段,掉分里混着"几何信息不足"与"分布错配伪影",分不开。这与并联几何融合设计中"评测时移除几何令牌"禁止置零的考虑相同。
2. **路由偏好 ≠ 信息上限**:只要 DINO 分支在场,decoder 可以把主干信息继续从 DINO token 读(γ=0 门控本身鼓励增量式用法);截断后崩溃只说明"并联把主要信息路由给了 DINO",不说明几何 token 承载不了规划。测能力上限必须让 geo_proj + decoder 在"几何 token 是唯一输入"约束下训练到收敛。
3. **矩阵可比性**:基线和并联的读数都是"各自配置下训练到收敛";geo_only 要填进同一张表就必须同质(同协议从零训练)。

评测期截断本身保留为并联的免费侧读数(评测时移除原生 DINO token,读取"DINO 依赖度"),记入并联台账,不入 geo_only。

## 2. 精确配置

- **memory**:仅 64 个几何 token(投影后 256 维)。与基线的 64 token 容量恰好对齐,天然排除"token 数多"的容量混淆(并联是 128)。
- **token 来源**:同并联几何融合——navtrain 缓存,每样本 `(4, 16, 2048)` fp16,`source=cache`,指纹校验、provider、注入链路全部原样复用。
- **geo_proj**:与并联同结构,**唯一差异 = 去掉零初始化 LayerScale 门控**。理由:γ=0 在并联里保证冷启动等价基线;geo_only 没有这个需求,且 γ=0 意味着冷启动时 memory 内容项全零(decoder 只见 branch/camera embedding),纯属给优化找麻烦。
  - **前置 LN(input_ln)保留,但理由已修正(2026-07-10 实测)**:`check_vggt_token_norms.py` 实测缓存 token **无高范数 outlier**(结构性原因:block 全开 `use_qk_norm`、LayerScale 1e-5、DINOv3 自带 storage token;outlier 主要是 DINOv2 无 register 时代的现象),且 geo_proj 的 out_ln 本就把进 decoder 的 key 归一——"防 attention sink"论证**不成立**。现行保留理由:(a) 官方 `CameraHead` 消费同一批 2048 维 token 的第一步就是 `token_norm = LayerNorm(2048)`(`camera_head.py:20,64`),前置 LN 是官方读出约定;(b) 给新初始化的 Linear 一个标准尺度输入(纯数值调理,零成本);(c) 与并联同结构,保持单变量对照(最硬的一条)。
  - camera embedding 保留(token 相机身份的唯一来源);branch embedding 单分支下退化为常数偏置,保留以减少与并联的代码分叉。
- **DINO 分支:不构造、不前向**(而非"跑了不用")。这是硬约束不是优化:LoRA/scene_embeds 若构造而无梯度,DDP 默认 `find_unused_parameters=False` 会直接报错。
- **数据管线不动**:builder、特征缓存与并联完全一致(features 里仍有 `image`,模型不消费)。保持单变量、避免新缓存名;IO 浪费可接受。
- **模式约束**:`mode=drop` 在 geo_only 下禁用(memory 会被截成空),构造期即抛错;normal/shuffle/noise 可用。
- **训练协议**:同基线(navtrain 10 epochs、AdamW lr 2e-4 cosine、batch 16 全局、4×A100、3 seeds、损失全 1)。
- **可训练参数**:geo_proj + trajectory/scoring decoder + heads(无 LoRA、无 scene_embeds)。

## 3. 改动清单(共 2 个文件,新增配置键 2 个)

| 文件 | 改动 | 规模 |
|---|---|---|
| `navsim/agents/drivoR/vggt_geometry.py` | `VggtGeometryProjector` 加 `use_gate: bool = True` 参数;False 时 `self.gate = nn.Identity()` | ~3 行 |
| `navsim/agents/drivoR/drivor_model.py` | ① geo_only 读取与校验;② 条件跳过 backbone 构造;③ forward 的 geo_only 分支 | ~12 行 |

**不改**:`dataset.py`、`vggt_geometry.py` 的 provider/缓存部分、缓存脚本、`drivor_agent.py`(已核实无 `image_backbone` 引用)、decoder/scorer/损失、`run_pdm_score_multi_gpu.py`(navtest 方案落地后自动兼容:geo_only 经 `agent.config` 流入模型,Dataset 注入逻辑不感知它)。

### 3.1 `vggt_geometry.py` — projector 门控开关

```python
def __init__(self, vggt_dim: int, d_model: int, num_cams: int = 4,
             tokens_per_cam: int = 16, use_gate: bool = True) -> None:
    ...
    self.gate = LayerScale(d_model, init_values=0.0, inplace=False) if use_gate else nn.Identity()
```

默认 `True` → 并联几何融合构造出的模块与改动前**同一个类型、同一初始化**,零变化。

### 3.2 `drivor_model.py` — geo_only

**(a) backbone 构造前判 geo_only,lidar 校验也在这里**(两个顺序约束:backbone 在 50 行构造、vggt cfg 在 107 行才解析,所以 geo_only 要在 50 行前先读;**lidar 校验必须同样前置**——lidar 分支构造(67 行)引用 `self.image_backbone.num_features`,若校验放在 107 行,带 lidar 的配置会先在 67 行 `AttributeError`,校验形同虚设。`num_lidar` 在 45–47 行已算好,可直接引用):

```python
        # geo_only: memory 只由冻结 VGGT-Omega 几何 token 构成,DINO 主干不构造不前向
        _vggt_cfg = cfg_get(config, "vggt_geometry", None)
        self.vggt_geometry_geo_only = bool(
            _vggt_cfg and cfg_get(_vggt_cfg, "enabled", False) and cfg_get(_vggt_cfg, "geo_only", False)
        )
        if self.vggt_geometry_geo_only and self.num_lidar > 0:
            raise ValueError("geo_only expects a camera-only config (no lidar branch)")

        if self.num_cams > 0 and not self.vggt_geometry_geo_only:
            ...  # 原 image_backbone / scene_embeds 构造,原样
```

**(b) drop 校验**(放 107 行起的 vggt_geometry 解析块内,`enabled` 分支里;mode 在此才解析,无法前置,也无需前置):

```python
            if self.vggt_geometry_geo_only and self.vggt_geometry_mode == "drop":
                raise ValueError("geo_only forbids mode=drop: memory would be empty at eval")
```

projector 构造处追加 `use_gate=bool(cfg_get(vggt_geometry_cfg, "use_layerscale_gate", True))`。

**(c) forward**(178–203 行的 scene_features 装配处):

```python
        if self.vggt_geometry_enabled and self.vggt_geometry_geo_only:
            # 空 memory 起手,让既有 _extend_memory_with_vggt_geometry 拼出 64-token memory
            scene_features = ego_token.new_zeros(batch_size, 0, ego_token.shape[-1])
        else:
            ...  # 原 image/lidar 装配 + torch.cat,原样
```

205–207 行既有的 `_extend_memory_with_vggt_geometry` 调用**不动**——训练与评测继续共用同一段注入代码,`vggt_geometry_memory_len` 输出为 64。

### 3.3 配置键

不改 `drivoR.yaml`,运行时 Hydra override:

```
'++agent.config.vggt_geometry.geo_only=true'
'++agent.config.vggt_geometry.use_layerscale_gate=false'
```

两键默认值(false / true)= 现状,对基线和并联几何融合零影响。**注意**:这两个键是模型侧开关,**不进缓存指纹**(缓存内容与它们无关);但训练与评测必须传同样的值,靠 checkpoint 的 state_dict 键集兜底(gate 有无、backbone 有无,加载不匹配会直接报错)。

## 4. 运行

**训练**:沿用并联几何融合训练脚本的环境变量与命令,追加 §3.3 两个 override,`mode=normal`,并把 `experiment_name` 设为描述性名称(建议 `drivor_geo_only_navtrain`)。

**navtest 评测**:按并联几何融合的 navtest 缓存评测指南,同样追加 §3.3 两个 override;`cache_dir` 指 navtest 缓存。normal / shuffle 各跑一次;**不跑 drop**(构造期抛错,预期行为)。

## 5. 内部对照

| 对照 | 定义 | 回答的问题 |
|---|---|---|
| **geo_only 错配对照** | 同并联的跨样本错配对照(整样本级错配,同一 provider,训练评测一致错配) | geo_only 读数是"几何内容"还是"64 个统计量正常 token 的容量效应"。若标准 geo_only ≈ geo_only 错配对照,geo_only 读数应解读为容量效应 |

noise 可选,不必跑(shuffle 已足够回答内容 vs 容量)。

## 6. 验收标准

1. **回归**:两键默认值下,并联几何融合缩样评测 PDMS 与改动前一致(gate 默认分支构造同一 LayerScale;基线不构造 projector,天然不受影响)。
2. **构造语义**:`geo_only=true` 时 `hasattr(model, "image_backbone") == False`、无 `scene_embeds`;打印并留档 trainable 参数计数(应 ≈ decoders+heads+geo_proj,无 LoRA);`mode=drop`+geo_only 构造即抛错;lidar 配置+geo_only 抛错。
3. **冒烟**:navtrain 缩样训练若干 step:不崩、loss 下降、`vggt_geometry_memory_len == 64`;DDP 双卡冒烟 1 个 step 不报 unused-parameters 错。
4. **norm 读数留档**:已跑(2026-07-10),结论 = **无高范数 outlier**,读数存档即可。该结果否定了"防 attention sink"论证,LN 保留理由已按 §2 修正(官方读出约定 + 数值调理 + 与并联单变量对照)。
5. **全量**:navtrain 10 epochs × 3 seeds;navtest 跑标准配置和跨样本错配配置,错配配置应明显低于标准配置(否则按 §5 解读)。
6. **读数入台账**:geo_only − 基线(mean±std)、geo_only vs 并联、裸替落地后的裸替 − geo_only;溯源按项目约定记文件 sha256,不用 git。

## 7. 已定决策(不必再权衡)

从零训练(§1);前置 LN 保留、γ 门控去除、camera/branch embedding 保留(§2);DINO 分支不构造(DDP 硬约束);数据管线不动;drop 禁用;shuffle 为唯一必跑对照。

## 状态

☐ 代码实现 → ☐ 验收 1–4 留档 → ☐ 训练中 → ☐ navtest 读数入台账

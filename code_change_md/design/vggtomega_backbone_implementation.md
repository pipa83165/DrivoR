# vggtomega_backbone 实现交接设计

> 面向快速模型的实现契约。完整论证、否决方案、代码草案和实验推演保留在
> `code_change_md/design/vggtomega_backbone.md`，实现时不要求读取原稿。
>
> 当前仓库已存在目标实现文件。各阶段必须先核对现状；已有实现满足契约时只补验证证据，
> 不得依据原稿覆盖式重写。

## 目标

在不改变 DrivoR decoder、scorer、损失和优化器接口的前提下，以冻结 VGGT-Ω 1B 直接替换图像主干：四相机联合前向，将 DrivoR 可学习 scene token 注入 VGGT 前缀并读回 64 个规划 memory token。保留 Q/V-only LoRA 开关，但冻结版是首要交付。

## 非目标

- 不把 VGGT 与现有 `vggt_geometry` 并联或 `geo_only` 分支组合。
- 不给 DINOv3 trunk 加 LoRA，也不引入 PEFT full-qkv 变体。
- 不修改 decoder、scorer、损失、通用训练框架或既有实验配置。
- 不在核心实现阶段执行全量训练、全量 navtest 或三 seed 实验。
- 不优化部署延迟、checkpoint 体积或离线特征缓存。

## 已核实的当前事实

- `navsim/agents/drivoR/drivor_model.py: DrivoRModel.__init__` 根据 `image_backbone.model_name` 构造图像主干，并由 `image_backbone.num_features` 决定 `scene_embeds` 维度。
- 同一构造函数已有 `vggt_geometry.enabled/geo_only` 分支；VGGT 主干互斥校验必须发生在 backbone/geo-only 分流之前。
- `vggt_omega/models/aggregator.py: Aggregator` 的默认前缀是 camera 1 + register 16，`patch_token_start=17`；register-attention 按该属性切前缀。
- `vggt_omega/models/layers/attention.py: SelfAttention` 仅对尾部 patch token 应用 RoPE；融合 qkv 的输出布局为 Q、K、V 三等份。
- `Aggregator.forward` 每层依次执行 frame block 与 inter-frame block，末层缓存值为 `cat(frame_tokens, tokens)`，维度 2048。
- `navsim/agents/drivoR/vggt_geometry.py` 已提供相机顺序 `cam_f0, cam_l0, cam_r0, cam_b0` 和 `preprocess_arrays_for_teacher`。
- `AgentLightningModule.validation_step` 通过 agent 类名中的 `drivor/DrivoR` 选择专用验证；`DrivoRAgent` 的最佳 checkpoint 监控 `val/score_epoch`。
- 默认训练配置是 cache-only、batch 64、20 epochs；新 feature builder 无对应缓存，专用脚本必须显式在线构建。
- 当前仓库已存在 `drivoR_vggt_omega` 包、agent 配置、验收脚本及训练/评测脚本；它们是待核验现状，不是可无条件信任的完成证据。

## 固定决策

1. **接口保持不变**：新 encoder 实现与 `ImgEncoder` 等价的调用契约，`DrivoRModel` 仅做哨兵分发。
2. **联合多相机前向**：四相机作为 VGGT 的四帧一次前向，前视必须是第 0 帧。
3. **scene token 属于前缀**：每帧布局为 `[camera(1), teacher_register(16), scene(S), patch]`；只读出 scene 段。
4. **官方数值路径**：builder 输出 `[0,1]` 的 `688×384` 图像；ImageNet 归一化仅在 aggregator 内执行一次。
5. **末层读出**：scene 段取末层 frame/global 拼接特征 2048 维，经 `Linear(2048, tf_d_model)` 投影。
6. **冻结但不截断梯度**：VGGT 原参数不训练，梯度仍须穿过主干回到 `scene_embeds`；aggregator 始终为 eval。
7. **检查点门控**：训练时按 block 做 gradient checkpoint；门控使用 `torch.is_grad_enabled()`，不能使用被钉死的 `aggregator.training`。
8. **LoRA 语义固定**：只包装 24 个 frame block 和 24 个 inter-frame block 的融合 qkv，只改变 Q/V；先冻结再安装 LoRA，trunk target 必须拒绝。
9. **最小接入**：新实现放在 `navsim/agents/drivoR_vggt_omega/`；既有 DrivoR 代码只允许修改 backbone 构造分发与互斥校验。
10. **配置隔离**：agent 类名使用 `DrivoRVggtOmegaAgent`，builder 唯一名使用 `drivor_vggt_omega_feature`，`vggt_geometry.enabled=false`。

## 接口契约

### `SceneTokenAggregator`

- 输入：`images (B,N,3,H,W)`，值域 `[0,1]`；`scene_tokens (B,N,S,1024)`。
- `scene_token_start` 保存原 `patch_token_start`；新 `patch_token_start=17+S`。
- `forward_full` 返回末层全部 token `(B,N,T,2048)`；`forward` 只返回 `(B,N,S,2048)`。
- `S=0` 时 camera/register 段须与官方 aggregator 数值一致，用于兼容性验证。

### `VggtOmegaImgEncoder`

- `num_features=1024`。
- 输入与 DrivoR `ImgEncoder` 相同：`img (B,N,3,H,W)`、`scene_tokens (B,N,S,1024)`。
- 输出：`(B,N*S,tf_d_model)`；默认 `N=4,S=16,tf_d_model=256`，即 `(B,64,256)`。
- 权重加载兼容 checkpoint 顶层 `model` 或 `state_dict`、可选 `module.` 前缀；仅提取 `aggregator.*` 并 strict load。
- CUDA 前向使用 bf16（不支持时 fp16）；读出 neck 使用 fp32。CPU 只用于轻量测试，不要求 autocast。

### Feature builder 与 agent

- builder 读取最后时刻四相机原图，复用 `preprocess_arrays_for_teacher`，不再次归一化。
- agent 继承 `DrivoRAgent`，只替换 feature builder。
- `DrivoRModel` 对 `model_name == "vggt_omega_1b"` 惰性导入新 encoder；其他 model name 仍走原 `ImgEncoder`。

## 改动范围

| 操作 | 组件 | 职责 |
|---|---|---|
| 核验/必要时修改 | `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py` | scene 前缀、冻结主干、checkpoint、LoRA、读出 |
| 核验/必要时修改 | `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py` | 四相机官方预处理 |
| 核验/必要时修改 | `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py` | agent 继承与 builder 分发 |
| 核验/必要时修改 | `navsim/agents/drivoR/drivor_model.py` | 互斥校验与 backbone 哨兵分发 |
| 核验/必要时修改 | `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml` | 独立 agent/backbone 配置 |
| 核验/必要时修改 | `scripts/vggt_omega_acceptance_checks.py` | 聚焦验收，不承载生产逻辑 |
| 核验/必要时修改 | `temp_script/vggtomega_backbone/*.sh` | 显式训练/评测协议 |

除上表外不修改业务代码；若发现必须扩大范围，先报告原因并停止该阶段。

## 实现阶段

1. 冻结 VGGT scene-token 主干 → 验证官方 `S=0` 兼容、输出 shape 和梯度路径。
2. Q/V-only LoRA 开关 → 验证冷启动不改输出、48 个 target、K/trunk 不变。
3. DrivoR 接入与 feature 链路 → 验证新旧分发、互斥、64-token decoder memory 和验证分支。
4. 运行入口与分层验收 → 验证脚本语法、在线特征配置、优化协议；GPU/数据集检查单独留档。

对应任务见 `code_change_md/task/vggtomega_backbone/`。每完成一阶段，先根据实际实现更新下一阶段的失效假设，禁止提前重构后续文件。

## 风险与未决事项

- **环境依赖**：VGGT 权重、CUDA、A100 显存和 NAVSIM 数据未必在快速模型环境可用；环境缺失不能用 mock 冒充 operational gate 通过。
- **现有实现状态**：目标代码已经存在，但尚不能仅凭文件存在认定所有原设计验收已完成。
- **显存预算是估算**：batch 16 是否可行必须实测；OOM 后才采用 micro-batch 4、累积 4，并保持优化器计算口径为每卡有效 batch 16。
- **LoRA 实验触发**：冻结版训练结果接近或超过基线时才运行 LoRA 实验；LoRA 功能验收不依赖该实验触发。
- **golden 基线**：若此前没有在改代码前留存可信 artifact，不能事后伪造“补丁前 bitwise 基线”；改用新旧分发的结构和聚焦回归证据并注明限制。

## 完成标准

- [ ] 四个阶段 gate 均有可复核证据，环境相关 gate 明确标注通过、失败或未运行。
- [ ] 冻结版输出 `(B,64,256)`，scene token 与 neck 有梯度，VGGT 原参数无梯度。
- [ ] LoRA 只影响 48 个 frame/inter-frame qkv 的 Q/V 路径，冷启动不改变原输出。
- [ ] 原 DrivoR model name 仍走 `ImgEncoder`，且不会导入新包；冲突配置响亮失败。
- [ ] 新 builder 使用正确相机顺序、唯一缓存名、`[0,1]` 官方预处理，decoder 收到 64 token。
- [ ] 训练/评测入口显式关闭 cache-only 与 `vggt_geometry`，训练和评测协议一致。
- [ ] 未修改范围外业务代码，原研究稿保持不变。

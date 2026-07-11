# 03 接入 DrivoR 与四相机 feature 链路

## 结果

通过独立 agent 配置将新 encoder 接入 DrivoR，并证明旧 backbone 路径不变、冲突配置失败、新路径向 decoder 提供 64 个 token。

## 先读

- `AGENTS.md`
- `code_change_md/design/vggtomega_backbone_implementation.md`
- 任务 01、02 的完成交接
- `navsim/agents/drivoR/drivor_model.py: DrivoRModel`
- `navsim/agents/drivoR/drivor_features.py: DrivoRFeatureBuilder`
- `navsim/agents/drivoR/vggt_geometry.py: preprocess_arrays_for_teacher`
- `navsim/planning/training/agent_lightning_module.py: AgentLightningModule.validation_step`

## 相关事实与决策

- 新包名必须区别于顶层官方 `vggt_omega`；使用 `navsim.agents.drivoR_vggt_omega`。
- `DrivoRModel` 只按 `model_name == "vggt_omega_1b"` 惰性分发；其他名称保持原路径。
- 互斥检查必须早于 `geo_only` 和 backbone 分支，否则冲突配置可能静默退化。
- 相机顺序复用 `VGGT_GEOMETRY_CAMERA_ORDER`；builder 唯一名不能与 `drivor_feature` 冲突。
- agent 类名必须含 `DrivoR`，否则不会产生 `val/score_epoch`。

## 允许修改范围

- `navsim/agents/drivoR/drivor_model.py` — 仅互斥检查和构造分发。
- `navsim/agents/drivoR_vggt_omega/__init__.py`
- `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py`
- `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py`
- `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml`
- `scripts/vggt_omega_acceptance_checks.py` — 本阶段聚焦检查。

## 不要修改

- 不改 `drivor_agent.py`、`drivor_features.py`、decoder、scorer、损失和默认训练配置。
- 不复制 `preprocess_arrays_for_teacher` 的数值逻辑。
- 不新增生产输出字段或持久调试 hook。
- 不把已有 `vggt_geometry` token 拼到新主干输出。

## 实现提示

- builder 返回键仍为 `image`，值域 `[0,1]`，期望四相机 shape `(4,3,384,688)`。
- agent 只覆盖 `get_feature_builders`，其余行为继承 `DrivoRAgent`。
- 新 agent yaml 保留 `vggt_geometry` 块但锁定 `enabled:false`；图像主干默认 `use_lora:false`。
- 旧路径回归应在新子进程验证惰性 import，避免测试进程自身已导入新包造成假失败。

## 验收标准

- [ ] 新配置构造 `VggtOmegaImgEncoder`；旧 model name 构造原 `ImgEncoder` 且不导入新包。
- [ ] `vggt_omega_1b + vggt_geometry.enabled=true`（含 geo-only）构造失败。
- [ ] builder 相机顺序、唯一名、值域和 shape 正确，且没有二次归一化。
- [ ] `DrivoRVggtOmegaAgent.name()` 命中 DrivoR 验证分支并产生 `val/score_epoch`。
- [ ] trajectory decoder 收到的 scene memory 长度恰为 64，无额外 geometry token。
- [ ] 本阶段之外既有 DrivoR 文件没有变化。

## 验证

```text
python3 -m py_compile navsim/agents/drivoR/drivor_model.py navsim/agents/drivoR_vggt_omega/*.py scripts/vggt_omega_acceptance_checks.py
python3 scripts/vggt_omega_acceptance_checks.py --check memory --checkpoint <VGGT_CHECKPOINT>
<运行项目中针对分发、builder 和 validation_step 的聚焦测试；若尚不存在，补入验收脚本>
```

## 交接

返回新旧分发类型、子进程 import 结果、builder 对比、64-token hook 和验证指标证据；指出任务 04 所需的真实 Hydra override。

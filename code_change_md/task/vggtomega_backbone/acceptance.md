# vggtomega_backbone 总验收

## 阶段 gate

- [ ] 阶段 01：官方 `S=0` 兼容、`(1,64,256)` 输出、冻结主干梯度路径和 eval/checkpoint 语义通过。
- [ ] 阶段 02：48 个 Q/V-only LoRA target、冷启动/K 段不变、trunk 拒绝和梯度检查通过。
- [ ] 阶段 03：新旧分发、冲突配置、builder、验证分支和 decoder 64-token memory 通过。
- [ ] 阶段 04：shell、优化协议、checkpoint 探针、GPU 单步和缩样评测按环境完成。

## 集成 gate

- [ ] 冻结配置与 LoRA 配置可分别 strict 加载对应 checkpoint，配置不匹配时响亮失败。
- [ ] 原 DrivoR model name 仍走原 `ImgEncoder`，新包不产生 import 副作用。
- [ ] 新主干与 `vggt_geometry.enabled=true` 双向互斥，无静默 geo-only 退化。
- [ ] feature builder 到 trajectory decoder 全链路保持四相机 64 个 scene token，无二次归一化。
- [ ] `DrivoRVggtOmegaAgent` 产生 `val/score_epoch`，最佳 checkpoint monitor 可工作。
- [ ] 冻结版可训练参数仅为原 DrivoR 可训练部分、scene embeddings 与 neck；LoRA 版仅额外增加 adapters。

## 运行 gate

- [ ] 正常训练配置：4 个可见 GPU、每卡 batch 16、全局 batch 64、AdamW lr `2e-4`、10 epochs。
- [ ] OOM 回退配置：micro-batch 4、累积 4、`agent.batch_size=16`，LR 与 T_max 保持不变。
- [ ] 冻结版与 LoRA 版各完成一个 GPU step，并记录 `max_memory_allocated`。
- [ ] 缩样 split 完成训练 checkpoint 加载和 PDM score 全链路冒烟。
- [ ] 冻结版通过前，不启动 LoRA 正式实验；只有冻结版接近或超过基线才触发 LoRA 训练。

## 归档证据

- [ ] 记录 VGGT 权重、`aggregator.py`、`attention.py`、`dinov2_lora.py` 的 SHA256。
- [ ] 保存聚焦验收输出、环境版本、显存结果和缩样评测结果。
- [ ] 若存在改代码前生成的可信 golden，记录其来源并运行回归；否则明确标记“不具备前置 artifact”，不补造。
- [ ] 记录正式训练 checkpoint SHA256、seed、训练/评测命令和最终指标。

## 完成规则

阶段 01～03 的聚焦 gate 和阶段 04 的静态/优化协议 gate 阻塞代码实现完成。GPU 单步、缩样评测和归档证据阻塞正式训练启动，但在当前环境缺权重、GPU 或数据集时可标记为“未运行”，不得标记通过。全量 10 epoch × 3 seeds 与 navtest 属于实验执行，不阻塞实现交接包完成。

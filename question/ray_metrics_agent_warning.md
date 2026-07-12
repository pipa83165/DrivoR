# Ray Metrics Exporter Agent Warning During Training

## Symptom

Training can print Ray worker warnings similar to:

```text
(pid=1841157) core_worker_process.cc:837:
Running out of retries to initialize the metrics agent. rpc_code: 14
Failed to establish connection to the metrics exporter agent. Metrics will not be exported.
```

## Current Understanding

This warning is from Ray, not from CUDA, PyTorch Lightning, or the VGGT-Omega backbone.

In DrivoR training, PDM score computation uses Ray workers through:

1. `navsim/agents/drivoR/drivor_agent.py`
2. `RayDistributedNoTorch(threads_per_node=8)`
3. `navsim/planning/utils/multithreading/worker_ray_no_torch.py`
4. `ray.init(...)`

Because the training uses Lightning DDP, each training process/rank may construct the agent and initialize a local Ray runtime. That can create multiple local Ray runtimes or dashboard/metrics agents on the same node, which can produce metrics-agent connection warnings.

## Likely Cause

Likely causes include:

- Ray dashboard or metrics exporter agent failed to start;
- Ray metrics port conflict between multiple DDP ranks;
- multiple local Ray runtimes initialized from different training worker processes;
- Ray agent process started but was not reachable before the core worker exhausted retries;
- environment/network restrictions around local gRPC ports.

## Impact

Usually this warning only means Ray internal metrics will not be exported. If training continues and Ray tasks return results, it is probably harmless log noise.

It becomes a real problem only if it is followed by Ray task failures, worker crashes, hanging score computation, or loss becoming unavailable.

## Why This May Affect Many Trainings

This is probably not specific to `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`.

Any DrivoR training run that uses the PDM score path in `drivor_agent.py` can initialize Ray workers and may show the same warning, especially under DDP.

## Suggested Checks

- Confirm whether training continues after the warning.
- Check whether all DDP ranks print Ray initialization messages.
- Check whether multiple Ray processes are started for one training job.
- Check whether the warning appears once at startup or repeatedly during every scoring call.
- If training hangs, inspect Ray worker logs under `/tmp/ray/session_latest/logs`.

## Suggested Fix Direction

Prefer keeping Ray local metrics disabled or less noisy if training is otherwise healthy:

- set Ray dashboard/metrics-related options in `ray.init(...)` if supported by the installed Ray version;
- initialize Ray only on the process that actually needs score workers;
- avoid every DDP rank starting an independent dashboard/metrics agent;
- optionally make DrivoR score computation use a non-Ray fallback for debugging.

This should be treated as a shared DrivoR/Ray scoring infrastructure issue, not a VGGT-Omega backbone issue.

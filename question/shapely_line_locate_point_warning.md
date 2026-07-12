# Shapely `line_locate_point` Warning During Training

## Symptom

Running `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh` can print repeated Ray worker warnings similar to:

```text
(wrapped_fn pid=1674150) /high_perf_store3/world-model/weixiaobao/envs/drivoRyz/lib/python3.9/site-packages/shapely/linear.py:88:
RuntimeWarning: invalid value encountered in line_locate_point
  return lib.line_locate_point(line, other)
```

The warning is emitted from a Ray worker while training computes the DrivoR PDM score.

## Current Trace

The observed path is:

1. `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`
2. `navsim/planning/script/run_training_full.py`
3. `navsim/agents/drivoR/drivor_agent.py`
4. `navsim/agents/drivoR/score_module/compute_navsim_score.py`
5. `navsim/agents/drivoR/score_module/train_pdm_scorer.py`
6. `PDMScorer._calculate_progress()`
7. `PDMPath.project([start_point, end_point])`
8. Shapely `line_locate_point`

The direct call site is `navsim/agents/drivoR/score_module/train_pdm_scorer.py`, where each proposal's first and last ego center points are projected onto the cached centerline.

## Likely Cause

This is not specific to the VGGT-Omega backbone. It is probably shared by all DrivoR training runs that compute PDM score through `compute_navsim_score.py`.

Likely invalid inputs include:

- proposal states containing `NaN` or `Inf`;
- ego center points derived from invalid proposal states;
- cached centerline geometry that is empty, degenerate, or contains invalid coordinates;
- Shapely vectorized projection returning `NaN` for some point/line combinations.

`navsim/planning/simulation/planner/pdm_planner/utils/pdm_path.py` already attempts to ignore this warning inside `PDMPath.project()`, but the current implementation uses global `warnings.filterwarnings(...)` immediately before the call. In Ray worker subprocesses this may still leak to stderr, and it does not sanitize invalid projection results.

## Impact

If this is only a warning and the returned progress is finite, training can continue.

If Shapely returns `NaN`, then `_progress_raw`, normalized progress, final PDM score, and eventually the training loss can become unstable. This would affect any training path that uses DrivoR's PDM-score loss, not just `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`.

## Suggested Checks

Add temporary diagnostics around `_calculate_progress()`:

- check `np.isfinite(self._ego_coords).all()`;
- check `self._centerline.linestring.is_empty`;
- check `self._centerline.linestring.length`;
- check whether `progress = self._centerline.project(...)` contains non-finite values;
- log the metric cache path or token for the failing sample.

## Suggested Fix Direction

Prefer a narrow defensive fix:

- use `warnings.catch_warnings()` inside `PDMPath.project()` so the Shapely warning is suppressed locally;
- in training scorer progress calculation, replace non-finite projection/progress values with zero progress or skip that proposal;
- optionally log the first few bad tokens for offline cache/data inspection.

This should be applied to shared PDM scoring code, because the issue appears to be in the common score path used by DrivoR training, not in the VGGT-Omega image backbone.

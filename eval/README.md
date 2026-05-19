# Reset-decision VQA evaluation

Benchmarks ABD's reset-decision method against baselines on a **VQA dataset**
built from ABD's own collected episodes.

- **Question** — given the *last-frame observation* of an episode, does the
  workspace need a **human reset** before the robot can continue?
- **Input** — the last frame of each episode, as 3 camera views
  (`front` / `wrist` / `table`).
- **Ground truth** — a binary `gt_reset` label.
- **Methods** — pluggable; ABD's method and every baseline implement one
  interface and are selected by name.

---

## 1. Dataset

### Ground-truth derivation

Every raw episode carries a human-assigned `ground_truth_reset` label — a
confusion-matrix cell relative to the *collecting* policy's decision:

| label | meaning | reset truly needed? |
|---|---|---|
| 1 (TP) | reset needed, policy reset | **yes** |
| 2 (TN) | no reset needed, policy no-reset | no |
| 3 (FP) | reset NOT needed, policy reset | no |
| 4 (FN) | reset needed, policy no-reset | **yes** |

The objective, policy-independent binary target is therefore:

```
gt_reset = ground_truth_reset in {1, 4}
```

### Build it

```bash
conda activate sushi_papa
python eval/build_dataset.py --raw-root /data/abd/raw --out-root /data/abd/eval/reset_vqa
```

Output (`/data/abd/eval/reset_vqa/`):

```
manifest.jsonl        one JSON object per sample
dataset_info.json     aggregate statistics
images/<sample_id>/{front,wrist,table}.jpg
```

Current build: **4034 samples**, 942 reset-needed (23.4%), 3092 no-reset.

### A manifest entry

```jsonc
{
  "sample_id": "ABD__20260514_10__ep00000__2c707d94",
  "task_name": "banana_to_pan", "task_direction": "forward",
  "language_task": "pick the banana and put it in the pan",
  "collection_policy": "ABD", "run_id": "20260514_10",
  "gt_reset": false, "gt_label": 2, "gt_label_name": "TN",
  "images": {"front": "images/.../front.jpg", "wrist": "...", "table": "..."},
  "metadata": { ...the full episode meta.json, verbatim... }
}
```

Every episode field (`success`, `failure_type`, `checklist_eval`,
`policy_decision`, `human_reset`, timings, …) is preserved under `metadata`, so
results can be sliced by task / policy / failure mode.

---

## 2. Running an evaluation

```bash
# smoke test — trivial baselines, no VLM server needed
python eval/run_eval.py --methods always_reset never_reset random

# full comparison (needs a VLM backend — see method config below)
python eval/run_eval.py \
  --methods abd_checklist single_vqa zero_shot_vlm code_as_monitor \
  --method-config eval/configs/methods.example.json \
  --results-dir /result/reset_vqa_eval

python eval/run_eval.py --list        # show registered methods
python eval/run_eval.py --limit 20 …  # quick debug pass
```

Runs are **resumable**: per-sample predictions are cached in
`predictions.jsonl`; re-running only predicts the missing samples (use
`--no-resume` to recompute).

### Results layout (`--results-dir`)

```
<method>/predictions.jsonl   per-sample: gt, prediction, raw detail, error
<method>/metrics.json        overall + per task / family / policy metrics
summary.json                 all methods combined
report.md                    leaderboard + per-method reports
```

Metrics (positive class = "reset needed"): confusion matrix, accuracy,
**balanced accuracy**, **MCC**, precision, recall, specificity, F1 — the
imbalance-aware ones (balanced acc, MCC) are the headline numbers given the
~23 % positive rate.

---

## 3. Methods

| name | needs VLM | description |
|---|---|---|
| `abd_checklist` | yes | **ABD's method** — `VLMChecklistPolicy`; resets when the weighted checklist score `< tau_reset`. |
| `single_vqa` | yes | ABD's `SingleVQAPolicy` — one yes/no VLM question. |
| `zero_shot_vlm` | yes | One minimal direct VLM prompt, no scaffolding. |
| `code_as_monitor` | yes | Code-as-Monitor (see §4). |
| `always_reset` / `never_reset` / `random` | no | Trivial reference baselines. |

VLM backends: `gpt` (reads `OPENAI_API_KEY`), `aha`, `qwen` (local inference
servers — `scripts/run_aha_server.py` / `run_qwen_server.py`). Per-method
options go in the `--method-config` JSON (see
`eval/configs/methods.example.json`).

---

## 4. Code-as-Monitor

Adapts *Code-as-Monitor* (CVPR'25, arXiv:2412.04455) to single-frame reset
detection. Three stages (`eval/methods/code_as_monitor/`):

1. **`constraint_generator.py`** — a VLM enumerates the spatio-temporal
   constraints that must hold for the workspace to be ready, plus the geometric
   *constraint elements* (point / line / surface) and the boolean attributes to
   observe. One artefact per task, cached.
2. **`code_generator.py`** — a VLM writes an executable `monitor(scene)` Python
   program that deterministically checks those constraints. Compiled in a
   restricted sandbox; a deterministic fallback monitor is derived from the
   constraints if the VLM is unavailable. One artefact per task, cached.
3. **`method.py`** — per sample, a VLM grounds the elements from the last frame
   into a `scene` dict, then runs the monitor. A violated constraint ⇒ reset
   needed. (The paper's RGB-D painter + tracker are replaced by this VLM
   grounding step, since the VQA dataset is image-only.)

---

## 5. Injecting a new method

Add a file under `eval/methods/`, subclass `ResetMethod`, decorate with
`@register("my_method")`:

```python
from .base import ResetMethod, ResetPrediction, register

@register("my_method")
class MyMethod(ResetMethod):
    def setup(self):
        ...  # load models / open clients (once)

    def predict(self, sample) -> ResetPrediction:
        imgs = sample.load_images()          # {"front": ndarray, ...}
        reset = ...                           # your decision
        return ResetPrediction(reset=reset, raw={"detail": "..."})

    def teardown(self):
        ...  # optional cleanup
```

If the module has no extra dependencies, import it eagerly in
`eval/methods/__init__.py`; if it depends on optional packages or servers, add
it to the `_OPTIONAL_MODULES` tuple (failed imports degrade to a warning).
The method is then runnable: `python eval/run_eval.py --methods my_method`.

Return `ResetPrediction(reset=None, error="…")` for a sample the method cannot
decide — it is counted under `errors` and excluded from the metrics.

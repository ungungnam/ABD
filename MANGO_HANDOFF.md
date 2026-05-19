# Mango handoff — finish the run-level AHA reset fine-tune

You are picking up on **mango** in `~/workspaces/ABD`. The reset-decision
fine-tune of AHA was prepared on sushi and migrated here. Your job: install the
environment, run the **run-level-split** LoRA fine-tune, then evaluate it.

## Background (what this is)

AHA (a 7B RoboPoint/Vicuna VLM) is being LoRA-fine-tuned to answer one question:
*given the last-frame image of a robot episode, does a human need to reset the
workspace?* A first fine-tune on a **random** train/val split scored val
balanced-acc 0.89 / MCC 0.78 — but a random split leaks visually near-identical
frames between train and val. This run uses a **run-level split** (no collection
run shared between train and val) for a trustworthy number.

## Migrated items (verify these exist first)

| item | path on mango |
|---|---|
| merged AHA base (fine-tune start point, ~14 GB) | `/data/AHA/aha-merged-7b` |
| train/val data (run-level split) | `/data/AHA/aha_ft_runsplit/` |
| reset-VQA dataset (images + manifest) | `/data/AHA/reset_vqa/` |
| conda env tarball | `/data/AHA/aha_env.tar.gz` |
| code (RoboPoint, eval, scripts, src, config) | `~/workspaces/ABD/` |

```bash
ls -la /data/AHA/aha-merged-7b/model.safetensors.index.json \
       /data/AHA/aha_ft_runsplit/train.json \
       /data/AHA/reset_vqa/manifest.jsonl \
       /data/AHA/aha_env.tar.gz
```

The run-level split: train 1,767 episodes (balanced to 2,216 examples),
val 459 episodes (32 reset-needed). Details in
`/data/AHA/aha_ft_runsplit/split_info.json`.

## Step 1 — install the `aha` conda env

The env was `conda-pack`ed from sushi (compiled flash-attn/deepspeed included;
mango's A6000 is Ampere sm_86, same as sushi, so the binaries are compatible).

```bash
mkdir -p ~/miniconda3/envs/aha
tar -xzf /data/AHA/aha_env.tar.gz -C ~/miniconda3/envs/aha
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aha
conda-unpack                 # rewrites absolute paths for this machine
python -c "import torch, flash_attn, deepspeed; print('env OK', torch.__version__)"
```

## Step 2 — run the run-level fine-tune

mango has 8× A6000 (46 GB). The fine-tune needs ~25 GB, so **one GPU, no
offload** is enough and fast (~30–50 min). Pick a free GPU:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # pick a low one
```

Then (replace `2` with your chosen free GPU id):

```bash
cd ~/workspaces/ABD
conda activate aha
bash scripts/finetune_aha_reset.sh \
    /data/AHA/aha-merged-7b \
    /data/AHA/aha_ft_runsplit/train.json \
    /data/AHA/reset_vqa \
    /result/AHA/reset_ft_runsplit \
    2 \                       # GPU id
    3 \                       # epochs
    scripts/zero3.json \      # plain ZeRO-3, NO offload (A6000 has room)
    8 2                       # per-device batch 8, grad-accum 2  (eff 16)
```

Run it inside `tmux`. Output LoRA → `/result/AHA/reset_ft_runsplit`
(adapter_model.safetensors + non_lora_trainables.bin + epoch checkpoints).
Expect ~414 steps; train loss should fall toward ~0.05–0.15.

## Step 3 — serve the fine-tuned model

The LoRA loader keys on the string "lora" in the model path, so symlink it:

```bash
ln -sfn /result/AHA/reset_ft_runsplit /result/AHA/reset_ft_runsplit_lora
CUDA_VISIBLE_DEVICES=2 python scripts/run_aha_server.py \
    --model-path /result/AHA/reset_ft_runsplit_lora \
    --model-base /data/AHA/aha-merged-7b \
    --port 9878            # wait for "Model is loaded..."
```

## Step 4 — evaluate on the held-out val split

In another shell (env `aha`), score the fine-tuned model on the val episodes
**only** (the val sample-ids the model never trained on):

```bash
cd ~/workspaces/ABD && conda activate aha
cat > /tmp/aha_ft_methods.json <<'JSON'
{ "single_vqa": { "backend": "aha", "url": "http://localhost:9878/inference",
  "timeout": 120,
  "question": "The robot's task is: \"{task_description}\". Does a human need to intervene and reset the environment before the robot can attempt this task again? Answer with a single word: Yes or No." } }
JSON

python eval/run_eval.py --methods single_vqa \
    --dataset-root /data/AHA/reset_vqa \
    --results-dir /result/reset_vqa_eval_runsplit \
    --sample-ids /data/AHA/aha_ft_runsplit/val_sample_ids.json \
    --method-config /tmp/aha_ft_methods.json
```

Metrics land in `/result/reset_vqa_eval_runsplit/single_vqa/metrics.json`.
The eval `question` MUST match the training question (it is, above) — a
fine-tuned model is prompt-sensitive.

## Step 5 — compare against ABD's checklist (our method)

The val episodes are ABD-collected, so each `ground_truth_reset` label is the
confusion cell of ABD's *online* checklist decision — "our method" for free on
the exact same episodes:

```bash
python3 - <<'PY'
import json, math
val=set(json.load(open('/data/AHA/aha_ft_runsplit/val_sample_ids.json')))
rows=[json.loads(l) for l in open('/data/AHA/reset_vqa/manifest.jsonl')]
v=[r for r in rows if r['sample_id'] in val]
c={k:sum(1 for r in v if r['gt_label_name']==k) for k in ('TP','TN','FP','FN')}
TP,TN,FP,FN=c['TP'],c['TN'],c['FP'],c['FN']
rec=TP/(TP+FN or 1); spec=TN/(TN+FP or 1); prec=TP/(TP+FP or 1)
mcc=((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) or 1)
print('ABD checklist (online) on val:', c)
print(f'  bal_acc={(rec+spec)/2:.3f} mcc={mcc:.3f} precision={prec:.3f} recall={rec:.3f}')
PY
```

## Report back

Final deliverable: the run-level val metrics for **fine-tuned AHA** vs
**ABD checklist** — balanced accuracy, MCC, precision, recall, F1, confusion
matrix. Headline question: *does the fine-tuned model still beat ABD's checklist
when no collection run is shared between train and val?* Note val has only
32 positives, so report it but flag the wide confidence interval.

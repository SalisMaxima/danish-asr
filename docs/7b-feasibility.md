# 7B Feasibility Investigation

Assess whether `omniASR_CTC_7B_v2` finetuning is realistically possible on DTU HPC,
given the `gpua100` queue's 2x A100 node topology and the current training stack in
this repository.

## Short Answer

**Probably feasible in principle, but not with the current training wrapper as-is.**

The hardware side looks promising: DTU `gpua100` nodes provide **2 GPUs per node**,
including **80 GB A100** nodes, so there is a plausible path to using roughly
**160 GB total VRAM** on a single node.

The main blocker is software/orchestration, not raw hardware:

- the current repo training wrapper launches **one Python process only**
- fairseq2 multi-GPU support depends on `torch.distributed` world setup
- the installed fairseq2 cluster auto-detection supports **Slurm**, not **LSF**
- the repo currently has **no `configs/fairseq2/7b/` configs or `scripts/hpc/7b/` job scripts**
- no current repo config or script actually enables multi-rank `7B` training

So the honest status is:

- **7B on 2x80GB is not ready to run today**
- **7B may become feasible after adding a proper LSF multi-process launcher and a 7B VRAM probe**

## Evidence

### 1. Hardware topology is compatible with a 2-GPU attempt

The DTU HPC docs in this repo explicitly state that `gpua100` nodes have **2 GPUs per
node**, and that some of those nodes are **80 GB A100s**:

| Source | Finding |
|---|---|
| `docs/dtu_hpc/03-lsf-jobs.md` | `gpua100` has `2` GPUs/node and includes `80 GB` A100 nodes |

That means a single-node 2-GPU job can, in principle, access:

- `2 x 80 GB = 160 GB` VRAM on the largest A100 nodes

Relevant repo references:

- [docs/dtu_hpc/03-lsf-jobs.md](dtu_hpc/03-lsf-jobs.md)

### 2. The model family documentation already treats 7B as multi-GPU territory

The repo's model overview and roadmap already mark the CTC 7B model as beyond single
80 GB training:

- `omniASR_CTC_7B_v2` is listed at **6.5B params**
- estimated training VRAM is listed as **>80 GB (multi-GPU)**

Relevant references:

- [docs/omnilingual-asr-overview.md](/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/danish_asr/docs/omnilingual-asr-overview.md:33)
- [docs/project-roadmap.md](/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/danish_asr/docs/project-roadmap.md:157)

This is consistent with the intuition that a 7B attempt would need at least:

- tensor parallelism across 2 GPUs, or
- FSDP/data sharding across 2 GPUs, or
- both

### 3. fairseq2 itself supports distributed data/model parallel concepts

The installed fairseq2 package clearly exposes:

- `gang.tensor_parallel_size`
- `trainer.data_parallelism`
- `trainer.fsdp`
- FSDP support under `fairseq2.nn.fsdp`

Relevant local package references:

- `.venv/lib/python3.12/site-packages/fairseq2/recipe/config.py`
- `.venv/lib/python3.12/site-packages/fairseq2/recipe/internal/gang.py`
- `.venv/lib/python3.12/site-packages/fairseq2/recipe/internal/data_parallel.py`

In other words, the library is capable of multi-rank execution.

### 4. The current repo wrapper does not actually launch multi-rank training

This is the critical blocker.

`scripts/hpc/run_training.py` launches exactly one subprocess:

```python
cmd = [
    sys.executable,
    "-m",
    "workflows.recipes.wav2vec2.asr",
    str(output_dir),
    "--config-file",
    str(args.config),
]
```

and then runs it with `subprocess.Popen(...)`.

There is:

- no `torchrun`
- no `mpirun`
- no LSF rank bootstrap
- no per-rank process spawning

So simply changing an LSF header from `num=1` to `num=2` GPUs would not, by itself,
turn the current wrapper into a 2-GPU training job.

Relevant reference:

- [scripts/hpc/run_training.py](/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/danish_asr/scripts/hpc/run_training.py:478)
- [scripts/hpc/run_training.py](/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/danish_asr/scripts/hpc/run_training.py:505)

### 5. fairseq2 auto-cluster setup does not currently support LSF

The installed fairseq2 cluster handler supports **Slurm** auto-detection, but no LSF
handler is present in the installed package.

That matters because fairseq2's `WorldInfo` becomes multi-rank only when environment
variables like these are set:

- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`
- `LOCAL_WORLD_SIZE`

The fairseq2 cluster code will auto-populate those for Slurm, but not for DTU's LSF.

Relevant references:

- `.venv/lib/python3.12/site-packages/fairseq2/cluster.py`
- `.venv/lib/python3.12/site-packages/fairseq2/world_info.py`

This is the strongest evidence that current repo jobs are still effectively
single-process, even if a 2-GPU LSF allocation were requested.

### 6. Existing successful training evidence is single-rank

The current 3B training/eval outputs reference checkpoint shards like:

`.../model/pp_00/tp_00/sdp_00.pt`

with only the `00` shard seen in practice. That is consistent with the current runs
being single-rank / non-sharded in real usage.

This does **not** prove 7B is impossible, but it does show that the current codepath
has not yet demonstrated multi-rank training in this repo.

Relevant reference:

- [docs/experiment-plan.md](/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/danish_asr/docs/experiment-plan.md:37)

## What This Means

### Hardware feasibility

**Yes, plausible.**

Based on the repo's own model-size notes:

- 3B trains on a single 80 GB A100 with a very reduced batch shape
- 7B is documented as needing **multi-GPU**
- DTU provides **2x80GB** nodes

That makes a 2-GPU 7B attempt plausible enough to investigate seriously.

### Current software feasibility

**No, not yet.**

With the current wrappers, configs, and cluster integration, a 7B attempt is not
production-ready because the repo is missing the machinery that actually turns an LSF
2-GPU allocation into a distributed fairseq2 run.

## Overall feasibility judgement

| Question | Answer |
|---|---|
| Can DTU hardware plausibly host a 7B CTC training attempt? | **Yes** |
| Can the current repo run that attempt correctly today? | **No** |
| Is it worth investigating further? | **Yes, but only after launcher work** |

## Main Risks

1. **Distributed launcher gap**

The biggest risk is not VRAM but orchestration. Without correct rank/world env setup,
the second GPU will sit idle and the job may either:

- use only one GPU
- crash during distributed initialization
- produce incorrect checkpoint layouts

2. **160 GB total VRAM is not the same as 160 GB usable model memory**

Two GPUs do not automatically behave like one larger GPU. Whether 7B fits depends on:

- tensor parallel vs FSDP strategy
- optimizer state sharding
- activation memory
- checkpointing strategy
- communication overhead

3. **Training speed may become poor even if it fits**

A 7B run that technically fits at a tiny `max_num_elements` and very high gradient
accumulation may be too slow to be worthwhile on this project, especially given the
already modest gain from 1B -> 3B.

4. **Weak scaling return**

The repo's current evidence shows:

- 1B -> 3B improved dev WER only modestly
- 3B already costs substantially more wall-clock time

So even if 7B becomes technically feasible, it may fail the cost/benefit bar.

## Recommended Path

Do **not** jump straight to a full 7B training run.

Recommended sequence:

1. Add an LSF-aware multi-process launcher for fairseq2 training.
   It should explicitly set or propagate:
   `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`.

2. Add a minimal 2-GPU smoke test on DTU HPC.
   Before touching 7B, prove that a smaller model can actually run across 2 GPUs under
   the new launcher.

3. Add a 7B VRAM probe, not full training.
   Start with a very small step budget and aggressive memory-saving settings.

4. Only if the probe succeeds, add a real 7B training config.

5. Gate any long 7B run on 3B test-set results.
   If 3B does not clearly beat 1B by a meaningful margin, 7B is unlikely to be a good
   use of compute for this project.

## Recommendation

**Proceed only to a launcher + probe phase, not to full 7B training yet.**

That is the highest-confidence next step because it answers the real unknowns in the
right order:

- first prove multi-GPU works on LSF in this repo
- then prove 7B can fit at all
- only then consider long training

## Proposed Next Deliverables

If we choose to continue this branch, the next practical changes should be:

1. `scripts/hpc/common.py` or a new helper:
   add LSF-aware distributed environment setup

2. new multi-GPU training wrapper:
   launch 2 local ranks explicitly for fairseq2

3. `configs/fairseq2/7b/`:
   add a tiny 7B VRAM probe config

4. `scripts/hpc/7b/`:
   add a 2x80GB probe submission script

5. docs:
   record the go/no-go results from the probe before committing to full training

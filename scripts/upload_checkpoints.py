"""Upload finetuned model checkpoints to W&B as artifacts.

Run from project root (no GPU needed):
    python scripts/upload_checkpoints.py
"""

from pathlib import Path
from typing import cast

import wandb

ENTITY = "mathiashl-danmarks-tekniske-universitet-dtu"
PROJECT = "danish-asr"

CHECKPOINTS = [
    {
        "artifact_name": "omniASR-CTC-300M-v2-e6-50k",
        "local_path": "models/300m_e6_50k/omniASR_CTC_300M_v2_e6_step50k.pt",
        "artifact_filename": "omniASR_CTC_300M_v2_e6_step50k.pt",
        "tags": ["300m", "e6", "archive", "best-300m"],
        "metadata": {
            "model": "omniASR_CTC_300M_v2",
            "experiment": "E6",
            "wandb_run": "bumbling-dawn-28",
            "wandb_run_id": "xkgn541d",
            "steps": 50000,
            "val_wer": 32.74,
            "val_uer": 12.92,
            "val_loss": 56.03,
            "lr": 5e-5,
            "shuffle_window": 1000,
            "grad_accum": 4,
            "max_num_elements": 2_560_000,
            "runtime_hours": 7.72,
            "gpu": "A100-40GB",
            "dataset": "CoRal-project/coral-v3",
            "language": "dan_Latn",
            "config_file": "configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml",
            "file_size_gb": 1.3,
        },
    },
    {
        "artifact_name": "omniASR-CTC-1B-v2-e6-50k",
        "local_path": "models/1b_e6_50k/omniASR_CTC_1B_v2_e6_step50k.pt",
        "artifact_filename": "omniASR_CTC_1B_v2_e6_step50k.pt",
        "tags": ["1b", "e6", "archive", "best-1b"],
        "metadata": {
            "model": "omniASR_CTC_1B_v2",
            "experiment": "E6-1B",
            "wandb_run": "doctor-voyager-51",
            "wandb_run_id": "6spbuji0",
            "steps": 50000,
            "val_wer": 25.21,
            "val_uer": 9.95,
            "val_loss": 45.53,
            "lr": 5e-5,
            "shuffle_window": 1000,
            "grad_accum": 8,
            "max_num_elements": 1_920_000,
            "runtime_hours": 38.2,
            "gpu": "A100-40GB",
            "dataset": "CoRal-project/coral-v3",
            "language": "dan_Latn",
            "config_file": "configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml",
            "file_size_gb": 3.7,
        },
    },
]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    for ckpt in CHECKPOINTS:
        local_path_str = cast(str, ckpt["local_path"])
        local_path = project_root / local_path_str
        print(f"\n=== Uploading {ckpt['artifact_name']} ===")
        print(f"    File: {local_path}")

        if not local_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {local_path}")

        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            job_type="archive",
            tags=ckpt["tags"],
            name=f"archive-{ckpt['artifact_name']}",
        )

        artifact = wandb.Artifact(
            name=ckpt["artifact_name"],
            type="model",
            metadata=ckpt["metadata"],
        )
        artifact.add_file(str(local_path), name=ckpt["artifact_filename"])
        run.log_artifact(artifact)

        print(f"    Artifact logged: {ckpt['artifact_name']}")
        run.finish()
        print("    Done.")

    print("\n=== All uploads complete ===")


if __name__ == "__main__":
    main()

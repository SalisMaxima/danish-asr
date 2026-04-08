"""Upload finetuned model checkpoints to W&B as artifacts.

Run on HPC login node (no GPU needed):
    python /tmp/upload_checkpoints.py
"""

import wandb

ENTITY = "mathiashl-danmarks-tekniske-universitet-dtu"
PROJECT = "danish-asr"

CHECKPOINTS = [
    {
        "artifact_name": "omniASR-CTC-300M-v2-e6-50k",
        "path": "/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model",
        "training_run_id": "xkgn541d",  # bumbling-dawn-28
        "tags": ["300m", "e6", "archive"],
        "metadata": {
            "model": "omniASR_CTC_300M_v2",
            "experiment": "E6",
            "wandb_display_name": "bumbling-dawn-28",
            "steps": 50000,
            "val_wer": 32.74,
            "val_uer": 12.92,
            "lr": 5e-5,
            "shuffle_window": 1000,
            "grad_accum": 4,
            "max_num_elements": 2_560_000,
            "runtime_hours": 7.72,
            "gpu": "A100-40GB",
            "config_file": "configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml",
            "file_size_gb": 1.3,
        },
    },
    {
        "artifact_name": "omniASR-CTC-1B-v2-e6-50k",
        "path": "/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model",
        "training_run_id": "6spbuji0",  # doctor-voyager-51
        "tags": ["1b", "e6", "archive"],
        "metadata": {
            "model": "omniASR_CTC_1B_v2",
            "experiment": "E6-1B",
            "wandb_display_name": "doctor-voyager-51",
            "steps": 50000,
            "val_wer": 25.21,
            "val_uer": 9.95,
            "lr": 5e-5,
            "shuffle_window": 1000,
            "grad_accum": 8,
            "max_num_elements": 1_920_000,
            "runtime_hours": 38.2,
            "gpu": "A100-40GB",
            "config_file": "configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml",
            "file_size_gb": 3.7,
        },
    },
]


def main() -> None:
    for ckpt in CHECKPOINTS:
        print(f"\n=== Uploading {ckpt['artifact_name']} ===")
        print(f"    Path: {ckpt['path']}")

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
        artifact.add_dir(ckpt["path"])

        # Link to the original training run
        run.log_artifact(artifact)
        run.link_artifact(artifact, f"{ENTITY}/{PROJECT}/{ckpt['artifact_name']}")

        print(f"    Artifact logged: {ckpt['artifact_name']}")
        run.finish()
        print("    Done.")

    print("\n=== All uploads complete ===")


if __name__ == "__main__":
    main()

"""One-off script to upload warm-frost-17 step_15000 model weights to W&B."""

import wandb

with wandb.init(
    project="danish-asr",
    entity="mathiashl-danmarks-tekniske-universitet-dtu",
    id="gi46t3kp",
    resume="must",
) as run:
    artifact = wandb.Artifact(
        name="omniasr-ctc-danish-gi46t3kp",
        type="checkpoint",
        metadata={"step": 15000, "path": "step_15000/model/pp_00/tp_00/sdp_00.pt"},
    )
    artifact.add_file(
        "/work3/s204696/outputs/omniasr_20k/ws_1.250158f0/checkpoints/step_15000/model/pp_00/tp_00/sdp_00.pt"
    )
    run.log_artifact(artifact)
    print("Done")

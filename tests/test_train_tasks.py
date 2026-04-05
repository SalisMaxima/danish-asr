"""Tests for training invoke tasks."""

from __future__ import annotations

from tasks import train as train_tasks


class _DummyContext:
    def __init__(self):
        self.commands: list[dict] = []

    def run(self, command: str, echo: bool, pty: bool) -> None:
        self.commands.append({"command": command, "echo": echo, "pty": pty})


def test_omniasr_builds_expected_command(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs" / "fairseq2"
    config_dir.mkdir(parents=True)
    (config_dir / "ctc-finetune-local.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    ctx = _DummyContext()
    output_dir = project_root / "outputs" / "run"

    train_tasks.omniasr.body(ctx, hardware="local", output_dir=str(output_dir), args="--no-sweep")

    assert ctx.commands == [
        {
            "command": f"uv run python -m workflows.recipes.wav2vec2.asr {output_dir} --config-file {config_dir / 'ctc-finetune-local.yaml'} --no-sweep",
            "echo": True,
            "pty": not train_tasks.WINDOWS,
        }
    ]


def test_omniasr_eval_builds_expected_command(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs" / "fairseq2"
    config_dir.mkdir(parents=True)
    (config_dir / "ctc-eval-e2.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")
    checkpoint_dir = project_root / "outputs" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    ctx = _DummyContext()

    train_tasks.omniasr_eval.body(ctx, checkpoint_dir=str(checkpoint_dir))

    assert ctx.commands == [
        {
            "command": f"uv run python -m workflows.recipes.wav2vec2.asr.eval {checkpoint_dir} --config-file {config_dir / 'ctc-eval-e2.yaml'}",
            "echo": True,
            "pty": not train_tasks.WINDOWS,
        }
    ]

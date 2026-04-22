"""Tests for training invoke tasks."""

from __future__ import annotations

import shlex

from tasks import train as train_tasks


class _DummyContext:
    def __init__(self):
        self.commands: list[dict] = []

    def run(self, command: str, echo: bool, pty: bool) -> None:
        self.commands.append({"command": command, "echo": echo, "pty": pty})


def _assert_omni_command_or_windows_noop(ctx: _DummyContext, base_cmd: str) -> None:
    if train_tasks.WINDOWS:
        assert ctx.commands == []
        return

    wrapped_cmd = f"bash -lc {shlex.quote(f'source {train_tasks.ENV_SCRIPT} && setup_omniasr && {base_cmd}')}"
    assert ctx.commands == [
        {
            "command": wrapped_cmd,
            "echo": True,
            "pty": not train_tasks.WINDOWS,
        }
    ]


def test_omniasr_builds_expected_command(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs" / "fairseq2" / "300m"
    config_dir.mkdir(parents=True)
    (config_dir / "ctc-finetune-local.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_tasks, "_has_module", lambda module_name: False)
    ctx = _DummyContext()
    output_dir = project_root / "outputs" / "run"

    train_tasks.omniasr.body(ctx, hardware="local", output_dir=str(output_dir), args="--no-sweep")

    base_cmd = (
        f"python -m workflows.recipes.wav2vec2.asr {output_dir} "
        f"--config-file {config_dir / 'ctc-finetune-local.yaml'} --no-sweep"
    )
    _assert_omni_command_or_windows_noop(ctx, base_cmd)


def test_omniasr_hpc_uses_legacy_config_for_backward_compatibility(tmp_path, monkeypatch):
    project_root = tmp_path
    legacy_dir = project_root / "configs" / "fairseq2" / "legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "ctc-finetune-hpc.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_tasks, "_has_module", lambda module_name: False)
    ctx = _DummyContext()
    output_dir = project_root / "outputs" / "run-hpc"

    train_tasks.omniasr.body(ctx, hardware="hpc", output_dir=str(output_dir))

    base_cmd = (
        f"python -m workflows.recipes.wav2vec2.asr {output_dir} --config-file {legacy_dir / 'ctc-finetune-hpc.yaml'}"
    )
    _assert_omni_command_or_windows_noop(ctx, base_cmd)


def test_omniasr_eval_builds_expected_command(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs" / "fairseq2" / "300m"
    config_dir.mkdir(parents=True)
    (config_dir / "ctc-eval-e2.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")
    checkpoint_dir = project_root / "outputs" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_tasks, "_has_module", lambda module_name: False)
    ctx = _DummyContext()

    train_tasks.omniasr_eval.body(ctx, checkpoint_dir=str(checkpoint_dir))

    base_cmd = (
        f"python -m workflows.recipes.wav2vec2.asr.eval {checkpoint_dir} "
        f"--config-file {config_dir / 'ctc-eval-e2.yaml'}"
    )
    _assert_omni_command_or_windows_noop(ctx, base_cmd)


def test_omniasr_windows_fallback_logs_and_skips_run(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs" / "fairseq2" / "300m"
    config_dir.mkdir(parents=True)
    (config_dir / "ctc-finetune-local.yaml").write_text("model:\n  name: omniASR_CTC_300M_v2\n")

    monkeypatch.setattr(train_tasks, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_tasks, "WINDOWS", True)
    monkeypatch.setattr(train_tasks, "_has_module", lambda module_name: False)
    ctx = _DummyContext()
    output_dir = project_root / "outputs" / "run"

    train_tasks.omniasr.body(ctx, hardware="local", output_dir=str(output_dir))

    assert ctx.commands == []

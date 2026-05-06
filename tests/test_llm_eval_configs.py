"""Regression test: LLM eval configs must have family+arch whenever path is set.

fairseq2 requires model.family when model.path is specified; without it the
job fails at config validation with 'family must be specified when path is specified'.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "fairseq2"
LLM_EVAL_CONFIGS = sorted(CONFIGS_DIR.glob("llm_*/llm-eval*.yaml"))

assert LLM_EVAL_CONFIGS, f"No LLM eval configs found under {CONFIGS_DIR} — glob pattern may be wrong"


@pytest.mark.parametrize("config_path", LLM_EVAL_CONFIGS, ids=lambda p: p.name)
def test_llm_eval_config_has_family_and_arch_when_path_is_set(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    model = config.get("model", {})
    if "path" not in model:
        pytest.skip("no model.path — family/arch not required")
    assert "family" in model, f"{config_path.name}: model.family missing (required when model.path is set)"
    assert "arch" in model, f"{config_path.name}: model.arch missing (required when model.path is set)"

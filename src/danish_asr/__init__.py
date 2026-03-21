from __future__ import annotations

from fairseq2.composition.assets import register_package_assets
from fairseq2.runtime.dependency import DependencyContainer


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    """Register danish_asr asset cards (datasets) with fairseq2."""
    register_package_assets(container, "danish_asr.cards")

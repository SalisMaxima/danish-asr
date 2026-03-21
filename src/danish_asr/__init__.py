from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairseq2 import DependencyContainer


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    """Register danish_asr asset cards (datasets) with fairseq2."""
    try:
        from fairseq2.composition.assets import register_package_assets
    except ImportError as exc:
        msg = (
            "fairseq2 is required to use 'setup_fairseq2_extension' but is not "
            "installed. Install the 'omni' dependency group, e.g.:\n"
            "  uv sync --group omni"
        )
        raise RuntimeError(msg) from exc

    register_package_assets(container, "danish_asr.cards")

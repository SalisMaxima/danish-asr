from __future__ import annotations


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    """Register danish_asr asset cards (datasets) with fairseq2."""
    from fairseq2.composition.assets import register_package_assets

    register_package_assets(container, "danish_asr.cards")

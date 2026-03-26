"""Patch omnilingual-asr wer_calculator.py to handle empty CTC hypotheses.

Fixes known issue #67: WER calculator crashes when CTC greedy decoding
produces all-blank output for a sample. Adds a guard that replaces empty
hypotheses with a single PAD token, yielding 100% WER (correct behavior).

Usage:
    python scripts/hpc/patch_wer_calculator.py
"""

import getpass
from pathlib import Path

WER_CALC = (
    Path("/work3")
    / getpass.getuser()
    / "omnilingual-asr"
    / "workflows"
    / "recipes"
    / "wav2vec2"
    / "asr"
    / "wer_calculator.py"
)

OLD = "hyp_seq = hyp_seq[hyp_seq != self._blank_label]"

NEW = """hyp_seq = hyp_seq[hyp_seq != self._blank_label]
            # Guard against all-blank CTC output (known issue #67)
            if hyp_seq.numel() == 0:
                hyp_seq = hyp_seq.new_tensor([self._pad_idx])"""


def main() -> None:
    if not WER_CALC.exists():
        print(f"ERROR: {WER_CALC} not found")
        raise SystemExit(1)

    content = WER_CALC.read_text()

    if NEW in content:
        print("Already patched")
        return

    if OLD not in content:
        print(f"ERROR: Could not find target line in {WER_CALC}")
        raise SystemExit(1)

    content = content.replace(OLD, NEW, 1)
    WER_CALC.write_text(content)
    print(f"Patched {WER_CALC}")


if __name__ == "__main__":
    main()

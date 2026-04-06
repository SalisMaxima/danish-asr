#!/bin/bash
# Submit all 1B VRAM probe jobs. Run from project root on HPC login node.
set -euo pipefail

echo "Submitting VRAM probes..."
bsub < scripts/hpc/1b/06a_vram_probe_1b.sh
bsub < scripts/hpc/1b/06b_vram_probe_1b_small.sh
bsub < scripts/hpc/1b/06c_vram_probe_1b_tiny.sh
echo "Done. Check with: bstat"

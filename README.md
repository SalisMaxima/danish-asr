# Danish ASR

Fine-tuning and Evaluation of Modern ASR Models for Danish

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd danish_asr
invoke core.setup-dev

# Activate environment
source .venv/bin/activate

# See all available commands
invoke --list
```

## Development

```bash
invoke quality.ruff    # Lint + format
invoke quality.test    # Run tests
invoke quality.ci      # Full CI pipeline
```

## Training

```bash
invoke data.download    # Download dataset
invoke train.train      # Train model
```

## Deployment

```bash
invoke deploy.api       # Run API server
```

## Project Structure

```
danish_asr/
├── src/danish_asr/   # Source code
│   ├── train.py                         # Training pipeline
│   ├── model.py                         # Model definitions
│   ├── data.py                          # Data loading
│   ├── api.py                           # FastAPI server
│   └── ...
├── configs/                             # Hydra configs
├── tasks/                               # Invoke task modules
├── tests/                               # Unit tests
├── dockerfiles/                         # Docker images
└── .github/workflows/                   # CI/CD
```

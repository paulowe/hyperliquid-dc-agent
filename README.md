# hyperliquid-dc-agent

Monorepo containing the Hyperliquid trading agent and Vertex AI ML pipelines.

## Packages

| Package | Description |
|---|---|
| `trading-agent/` | Automated grid trading bot for Hyperliquid DEX |
| `pipelines/` | Vertex AI MLOps training and prediction pipelines |
| `components/bigquery-components/` | BigQuery KFP pipeline components |
| `components/vertex-components/` | Vertex AI KFP pipeline components |

## Quick Start

### Prerequisites
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager

### Setup
```bash
uv sync
```

This installs all workspace dependencies in a single step.

### Trading Agent
```bash
# Validate configuration
make validate-bot

# Run the trading bot
make run-bot

# Run learning examples
uv run --package hyperliquid-trading-bot python trading-agent/learning_examples/01_websockets/realtime_prices.py
```

### Vertex AI Pipelines
```bash
# Run pipeline trigger tests
make test-trigger

# Compile a pipeline (PIPELINE_TEMPLATE: tensorflow or xgboost)
make compile-pipeline pipeline=training PIPELINE_TEMPLATE=tensorflow

# Run all component tests
make test-all-components
```

### Infrastructure
```bash
# Terraform plan/apply
make plan-infra env=dev
make deploy-infra env=dev
```

### All Available Commands
```bash
make help
```

## Documentation

- [Pipeline Usage Guide](docs/pipeline/USAGE.md)
- [Pipeline Architecture](docs/pipeline/README.md)
- [Contributing](docs/pipeline/CONTRIBUTING.md)
- [Production Guide](docs/PRODUCTION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

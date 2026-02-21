# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

Before implementing anything, send me your plan of action for approval.
Start implementing things only after my explicit approval.

Follow test driven development.

## Package Management

This is a UV workspace monorepo. Use UV for all commands:
- `uv sync` - Install/sync all workspace dependencies
- `uv add <package> --package <pkg-name>` - Add dependency to a specific package
- `uv run --package <pkg-name> <command>` - Run command in a specific package context
- `make help` - See all available Makefile targets

## Monorepo Structure

```
├── trading-agent/                  # Hyperliquid trading bot
│   ├── pyproject.toml              # Package: hyperliquid-trading-bot
│   ├── src/                        # Bot source code
│   ├── bots/                       # Bot YAML configurations
│   └── learning_examples/          # Educational scripts
│
├── pipelines/                      # Vertex AI ML pipelines
│   ├── pyproject.toml              # Package: dc-vae
│   ├── src/pipelines/              # Pipeline code
│   └── tests/                      # Pipeline tests
│
├── components/                     # KFP pipeline components
│   ├── bigquery-components/        # Package: bigquery-components
│   ├── vertex-components/          # Package: vertex-components
│   └── dataflow-components/        # Dataflow components
│
├── terraform/                      # Infrastructure as Code
├── cloudbuild/                     # CI/CD configs
├── images/                         # Docker images
├── notebooks/                      # Jupyter notebooks
└── docs/                           # Documentation
```

## Running Things

### Trading Agent
```bash
make run-bot                                    # Auto-discover and run first active config
make run-bot-config config=trading-agent/bots/btc_conservative.yaml
make validate-bot                               # Validate config only
make test-trading                               # Run trading agent tests

# Learning examples
uv run --package hyperliquid-trading-bot python trading-agent/learning_examples/01_websockets/realtime_prices.py
```

### Backtesting
```bash
make backtest                                   # Single backtest (SOL, default optimal params)
make backtest symbol=BTC threshold=0.01         # Custom symbol/params
make backtest-sweep                             # Parameter sweep (1,296 combos, SOL)
make backtest-sweep symbol=ETH days=14          # Sweep for different symbol/period
make test-backtest                              # Run backtesting module tests
```

### Vertex AI Pipelines
```bash
make test-trigger                               # Test pipeline trigger code
make compile-pipeline pipeline=training         # Compile pipeline
make run-pipeline pipeline=training             # Compile + sync + run
make test-all-components                        # Test all KFP components
make e2e-tests pipeline=training                # End-to-end pipeline tests
```

### Infrastructure
```bash
make plan-infra env=dev                         # Terraform plan
make deploy-infra env=dev                       # Terraform apply
```

## Trading Agent Details

### Code Style
- **ALWAYS ADD COMMENTS** in code unless explicitly requested not to
- Follow existing patterns in the codebase
- Use type hints consistently
- Keep functions focused and single-purpose

### Architecture Patterns
- **Interface-based design** - Clear separation between business logic and implementation
- **Dependency injection** - Adapters injected into strategies and engines
- **Event-driven** - WebSocket events trigger strategy decisions
- **Async/await** - Non-blocking I/O for real-time operations

### Error Handling
- Use custom exceptions from `trading-agent/src/utils/exceptions.py`
- Graceful degradation for network issues
- Comprehensive logging at appropriate levels
- Clean shutdown on signals (SIGINT, SIGTERM)

### Bot Configuration
Bot configurations are stored as YAML files in `trading-agent/bots/`. Each includes:
- `active: true/false` - Controls whether bot runs automatically
- `name` - Unique bot identifier
- `account` - Account allocation settings
- `grid` - Grid strategy parameters
- `risk_management` - Stop loss, take profit, drawdown limits
- `monitoring` - Logging settings

### Backtesting Module (`trading-agent/src/backtesting/`)
Reusable module for historical strategy testing. Use `/backtest <SYMBOL>` skill to run.

- **candle_fetcher.py**: Fetches Hyperliquid 1m candles with disk cache (`~/.cache/hyperliquid-backtest/`)
- **engine.py**: `BacktestEngine` feeds candles through `DCOvershootStrategy`, records trades with fee accounting
- **metrics.py**: `compute_metrics()` computes win rate, P&L, profit factor, max drawdown, etc.
- **sweep.py**: `ParameterSweep` grid search over 1,296 parameter combos, ranks by net P&L
- **cli.py**: CLI entry point (`python -m backtesting.cli`)

Key insight: threshold=0.015 (1.5%) makes 100% of SOL configs profitable. Lower thresholds produce too many fee-eaten trades.

### Hyperliquid API
- **Testnet**: `https://api.hyperliquid-testnet.xyz`
- **WebSocket**: `wss://api.hyperliquid-testnet.xyz/ws`
- **Critical SDK Methods**: `exchange.order()`, `exchange.cancel_order()`, `info.all_mids()`, `info.open_orders()`

### Learning Examples
Standalone educational scripts in `trading-agent/learning_examples/`:
- Place imports always at the top
- Use short docstrings
- Each script is self-contained

### Environment Variables (Trading Agent)
```bash
# trading-agent/.env
HYPERLIQUID_TESTNET_PRIVATE_KEY=0x...
HYPERLIQUID_TESTNET=true
```

## Pipeline Details

### Pipeline Templates
- TensorFlow training/prediction pipelines under `pipelines/src/pipelines/tensorflow/`
- XGBoost training/prediction pipelines under `pipelines/src/pipelines/xgboost/`
- Trigger system under `pipelines/src/pipelines/trigger/`

### GCP Environment Variables
Set in `env.sh` (see `env.sh.example`):
- `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_SA_EMAIL`
- `PIPELINE_TEMPLATE`, `PIPELINE_FILES_GCS_PATH`, `VERTEX_PIPELINE_ROOT`

## GCP Debugging Skills

Patterns for debugging Vertex AI pipelines:
```bash
# Check pipeline run status
gcloud ai custom-jobs list --region=us-central1 --project=derivatives-417104

# Get pipeline task states via Python SDK
# aiplatform.PipelineJob.get(resource_name).task_details

# Read pipeline logs
gcloud logging read 'resource.type="aiplatform.googleapis.com/PipelineJob"' --project=derivatives-417104 --limit=50

# Fetch task output from GCS
gsutil cat "gs://derivatives-417104-pl-root/<run-id>/<task-id>/report"

# Check BQ dataset location
bq show --format=json derivatives-417104:coindesk | jq .location

# Compile pipeline (env vars must be set first)
source env.sh && cd pipelines && uv run python -m pipelines.tensorflow.training.pipeline_ablation

# Sync training script to GCS
gsutil cp pipelines/src/pipelines/tensorflow/training/assets/tftrain_tf_fast_model.py gs://derivatives-417104-pl-assets/training/assets/

# Submit pipeline via trigger
uv run --package dc-vae python -m pipelines.trigger.main
```

## Experiment Tracking

All experiments live in `experiments/NNN-<descriptive-name>/`. Each has:
- `config.yaml`: frozen snapshot of pipeline params (dates, hparams, thresholds)
- `results.json`: raw output from compare_forecasts (copied from GCS after run)
- `notes.md`: human-readable analysis, what we learned, link to next experiment

Always record what changed from the previous experiment and why.

## Known Pitfalls

- KFP `packages_to_install`: each item is a separate `pip install` — use `"setuptools>=69.0.0"` not `"--upgrade", "pip", "setuptools"`
- `os.environ.get()` in pipeline defaults evaluates at compile time — always `source env.sh` before compiling
- BQ dataset location must match actual dataset location (check with `bq show`)
- `compare_forecasts` outputs JSON, not CSV — use `source_format="NEWLINE_DELIMITED_JSON"` in `load_dataset_to_bigquery`
- KFP caching keys on component signature + inputs, NOT on underlying data contents. If you change date ranges but write to the same BQ table name, downstream steps will serve stale cache from a previous run. Use `enable_caching=False` when changing data ranges or rewriting intermediate tables.

## Important Notes

- **Private Key Security**: Never commit private keys to git
- **Testing**: Validate against Hyperliquid testnet before mainnet
- **Pipeline Coverage**: 80% minimum test coverage enforced
- **Python Version**: 3.11 across all workspace packages

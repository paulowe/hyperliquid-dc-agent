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
- **NO COMMENTS** in code unless explicitly requested
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

## Important Notes

- **Private Key Security**: Never commit private keys to git
- **Testing**: Validate against Hyperliquid testnet before mainnet
- **Pipeline Coverage**: 80% minimum test coverage enforced
- **Python Version**: 3.11 across all workspace packages

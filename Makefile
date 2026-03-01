-include env.sh
export

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Setup
# ============================================================================

setup: ## Install all workspace dependencies via UV
	@uv sync

# ============================================================================
# Trading Agent
# ============================================================================

run-bot: ## Run the trading bot (auto-discovers first active config in trading-agent/bots/)
	@uv run --package hyperliquid-trading-bot python trading-agent/src/run_bot.py

run-bot-config: ## Run a specific bot config. Must specify config=<path-to-yaml>
	@uv run --package hyperliquid-trading-bot python trading-agent/src/run_bot.py $(config)

validate-bot: ## Validate trading bot configuration without running
	@uv run --package hyperliquid-trading-bot python trading-agent/src/run_bot.py --validate

test-trading: ## Run trading agent tests
	@cd trading-agent && \
	uv run --package hyperliquid-trading-bot pytest

# ============================================================================
# Backtesting
# ============================================================================

symbol ?= SOL
threshold ?= 0.015
sl_pct ?= 0.015
tp_pct ?= 0.005
trail_pct ?= 0.5
min_profit_to_trail ?= 0.002
days ?= 7

backtest: ## Single backtest (override: symbol, threshold, sl_pct, tp_pct, trail_pct, min_profit_to_trail, days)
	@uv run --package hyperliquid-trading-bot python -m backtesting.cli \
		--symbol $(symbol) --threshold $(threshold) --sl-pct $(sl_pct) --tp-pct $(tp_pct) \
		--trail-pct $(trail_pct) --min-profit-to-trail-pct $(min_profit_to_trail) --days $(days)

backtest-sweep: ## Parameter sweep (override: symbol, days)
	@uv run --package hyperliquid-trading-bot python -m backtesting.cli \
		--symbol $(symbol) --sweep --days $(days)

test-backtest: ## Run backtesting module tests
	@uv run --package hyperliquid-trading-bot python -m pytest trading-agent/tests/backtesting/ -v

# Multi-scale backtesting
trade_threshold ?= 0.015
sensor_thresholds ?= 0.002,0.004,0.008
momentum_alpha ?= 0.3
min_momentum_score ?= 0.3

backtest-multi-scale: ## Multi-scale single backtest (override: symbol, trade_threshold, sensor_thresholds, momentum_alpha, min_momentum_score, sl_pct, tp_pct, trail_pct, min_profit_to_trail, days)
	@uv run --package hyperliquid-trading-bot python -m backtesting.cli \
		--multi-scale --symbol $(symbol) --trade-threshold $(trade_threshold) \
		--sensor-thresholds $(sensor_thresholds) --momentum-alpha $(momentum_alpha) \
		--min-momentum-score $(min_momentum_score) --sl-pct $(sl_pct) --tp-pct $(tp_pct) \
		--trail-pct $(trail_pct) --min-profit-to-trail-pct $(min_profit_to_trail) --days $(days)

backtest-multi-scale-sweep: ## Multi-scale parameter sweep (override: symbol, sensor_thresholds, days)
	@uv run --package hyperliquid-trading-bot python -m backtesting.cli \
		--multi-scale --sweep --symbol $(symbol) --sensor-thresholds $(sensor_thresholds) --days $(days)

test-multi-scale: ## Run multi-scale strategy tests
	@uv run --package hyperliquid-trading-bot python -m pytest \
		trading-agent/tests/strategies/dc_overshoot/test_momentum_scorer.py \
		trading-agent/tests/strategies/dc_overshoot/test_multi_scale_config.py \
		trading-agent/tests/strategies/dc_overshoot/test_multi_scale_strategy.py \
		trading-agent/tests/backtesting/test_multi_scale_engine.py \
		trading-agent/tests/backtesting/test_multi_scale_sweep.py -v

# ============================================================================
# Scaling Laws Analysis
# ============================================================================

scaling-laws: ## Analyze DC scaling laws (override: symbol, days)
	@uv run --package hyperliquid-trading-bot python -m scaling_laws.cli \
		--symbol $(symbol) --days $(days)

scaling-laws-json: ## Scaling laws JSON output (override: symbol, days)
	@uv run --package hyperliquid-trading-bot python -m scaling_laws.cli \
		--symbol $(symbol) --days $(days) --json

test-scaling-laws: ## Run scaling laws module tests
	@uv run --package hyperliquid-trading-bot python -m pytest trading-agent/tests/scaling_laws/ -v

# ============================================================================
# Trade Review
# ============================================================================

review_symbol ?= HYPE
review_hours ?= 24

review-trades: ## Review live trade P&L (override: review_symbol, review_hours)
	@uv run --package hyperliquid-trading-bot python -m trade_review.cli \
		--symbol $(review_symbol) --hours $(review_hours)

review-trades-json: ## Review live trades with JSON output (override: review_symbol, review_hours)
	@uv run --package hyperliquid-trading-bot python -m trade_review.cli \
		--symbol $(review_symbol) --hours $(review_hours) --json

test-trade-review: ## Run trade review module tests
	@uv run --package hyperliquid-trading-bot python -m pytest trading-agent/tests/trade_review/ -v

# ============================================================================
# Telemetry
# ============================================================================

test-telemetry: ## Run telemetry module tests
	@uv run --package hyperliquid-trading-bot python -m pytest trading-agent/tests/telemetry/ -v

setup-telemetry-bq: ## Create BigQuery dataset and tables for telemetry (idempotent)
	@uv run --package hyperliquid-trading-bot python -m telemetry.setup_bq

load-telemetry: ## Load telemetry NDJSON from GCS into BigQuery
	@uv run --package hyperliquid-trading-bot python -m telemetry.load_to_bq

# ============================================================================
# Vertex AI Pipelines
# ============================================================================

pre-commit: ## Runs the pre-commit checks over entire repo
	@uv run pre-commit run --all-files

test-trigger: ## Runs unit tests for the pipeline trigger code
	@cd pipelines && \
	uv run --package dc-vae python -m pytest tests/trigger

test-trigger-coverage: ## Runs unit tests with coverage for the pipeline trigger code
	@cd pipelines && \
	uv run --package dc-vae python -m pytest tests/trigger

compile-pipeline: ## Compile the pipeline to training.json or prediction.json. Must specify pipeline=<training|prediction>
	@cd pipelines/src && \
	uv run --package dc-vae python -m pipelines.${PIPELINE_TEMPLATE}.${pipeline}.pipeline

# ============================================================================
# Pipeline Components
# ============================================================================

test-components: ## Run unit tests for a component group. Must specify GROUP=<bigquery-components|vertex-components>
	@cd "components/${GROUP}" && \
	uv run --package ${GROUP} pytest

test-components-coverage: ## Run unit tests with coverage for a component group
	@cd "components/${GROUP}" && \
	uv run --package ${GROUP} pytest

test-all-components: ## Run unit tests for all pipeline components
	@set -e && \
	for component_group in components/*/ ; do \
		echo "Test components under $$component_group" && \
		$(MAKE) test-components GROUP=$$(basename $$component_group) ; \
	done

test-all-components-coverage: ## Run unit tests with coverage for all pipeline components
	@set -e && \
	for component_group in components/*/ ; do \
		echo "Test components with coverage under $$component_group" && \
		$(MAKE) test-components-coverage GROUP=$$(basename $$component_group) ; \
	done

coverage-report: ## Generate combined coverage report for all components
	@echo "Generating combined coverage report..." && \
	cd pipelines && \
	uv run coverage combine .coverage ../components/*/.coverage 2>/dev/null || true && \
	uv run coverage report --show-missing && \
	uv run coverage html && \
	echo "Coverage report generated in pipelines/htmlcov/index.html"

# ============================================================================
# Pipeline Operations
# ============================================================================

sync-assets: ## Sync assets folder to GCS. Must specify pipeline=<training|prediction>
	@if [ -d "./pipelines/src/pipelines/${PIPELINE_TEMPLATE}/$(pipeline)/assets/" ] ; then \
		echo "Syncing assets to GCS" && \
		gsutil -m rsync -r -d ./pipelines/src/pipelines/${PIPELINE_TEMPLATE}/$(pipeline)/assets ${PIPELINE_FILES_GCS_PATH}/$(pipeline)/assets ; \
	else \
		echo "No assets folder found for pipeline $(pipeline)" ; \
	fi ;

run-pipeline: ## Compile pipeline, copy assets to GCS, and run pipeline in sandbox environment. Must specify pipeline=<training|prediction>
	@ $(MAKE) compile-pipeline && \
	$(MAKE) sync-assets && \
	cd pipelines/src && \
	uv run --package dc-vae python -m pipelines.trigger --template_path=./$(pipeline).json --enable_caching=$(enable_pipeline_caching)

sync_assets ?= true
e2e-tests: ## (Optionally) copy assets to GCS, and perform E2E pipeline tests. Must specify pipeline=<training|prediction>
	@if [ $$sync_assets = true ] ; then \
        $(MAKE) sync-assets; \
	else \
		echo "Skipping syncing assets to GCS"; \
    fi && \
	cd pipelines && \
	uv run --package dc-vae pytest --log-cli-level=INFO tests/${PIPELINE_TEMPLATE}/$(pipeline) --enable_caching=$(enable_pipeline_caching)

# ============================================================================
# Infrastructure (Terraform)
# ============================================================================

env ?= dev
plan-infra: ## Terraform plan. Requires VERTEX_PROJECT_ID and VERTEX_LOCATION in env.sh. Optionally specify env=<dev|test|prod> (default = dev)
	@ cd terraform/envs/$(env) && \
	terraform init -upgrade -backend-config='bucket=gcp-terraform-states-us-${VERTEX_PROJECT_ID}' -backend-config="prefix=${TF_BUCKET_PREFIX}" && \
	terraform plan -var 'project_id=${VERTEX_PROJECT_ID}' -var 'region=${VERTEX_LOCATION}'

deploy-infra: ## Deploy Terraform infrastructure. Requires VERTEX_PROJECT_ID and VERTEX_LOCATION in env.sh. Optionally specify env=<dev|test|prod> (default = dev)
	@ cd terraform/envs/$(env) && \
	terraform init -upgrade -backend-config='bucket=gcp-terraform-states-us-${VERTEX_PROJECT_ID}' -backend-config="prefix=${TF_BUCKET_PREFIX}" && \
	terraform apply -var 'project_id=${VERTEX_PROJECT_ID}' -var 'region=${VERTEX_LOCATION}'

destroy-infra: ## DESTROY Terraform infrastructure. Requires VERTEX_PROJECT_ID and VERTEX_LOCATION in env.sh. Optionally specify env=<dev|test|prod> (default = dev)
	@ cd terraform/envs/$(env) && \
	terraform init -backend-config='bucket=gcp-terraform-states-us-${VERTEX_PROJECT_ID}' -backend-config="prefix=${TF_BUCKET_PREFIX}" && \
	terraform destroy -var 'project_id=${VERTEX_PROJECT_ID}' -var 'region=${VERTEX_LOCATION}'

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

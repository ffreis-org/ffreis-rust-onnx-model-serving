.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

SHELL := /usr/bin/env bash

CARGO ?= cargo
PYTHON ?= python3
COVERAGE_PYTHON ?= $(if $(shell command -v uv 2>/dev/null),uv run --with defusedxml python,$(PYTHON))
CARGO_LOCKED ?= --locked

COVERAGE_MIN ?= 90
COVERAGE_OVERALL_MAX_DRIFT ?= 1
COVERAGE_PER_FILE_MIN ?= 90
COVERAGE_FILES ?= app/src/main.rs,app/src/lib.rs,app/src/config.rs,app/src/telemetry.rs
INTEGRATION_COVERAGE_MIN ?= 57
E2E_COVERAGE_MIN ?= 45
E2E_MIN_TESTS ?= 3
COVERAGE_SEGMENT_MAX_DRIFT ?= 10
ROOT_DIR ?= $(abspath ..)
COVERAGE_XML ?= $(ROOT_DIR)/coverage/cobertura.xml
COVERAGE_LCOV ?= $(ROOT_DIR)/coverage/lcov.info
INTEGRATION_COVERAGE_XML ?= $(ROOT_DIR)/coverage/cobertura-integration.xml
E2E_COVERAGE_XML ?= $(ROOT_DIR)/coverage/cobertura-e2e.xml
CHECK_COVERAGE_SCRIPT ?= $(ROOT_DIR)/scripts/check_coverage.py
CHECK_COVERAGE_PER_FILE_SCRIPT ?= $(ROOT_DIR)/scripts/check_coverage_per_file.py
CHECK_E2E_TESTS_SCRIPT ?= $(ROOT_DIR)/scripts/check_e2e_tests.py
COVERAGE_DIR := $(dir $(COVERAGE_XML))

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Rust quality
# ------------------------------------------------------------------------------

.PHONY: fmt
fmt: ## Format Rust code
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## Check Rust formatting
	$(CARGO) fmt --all -- --check

.PHONY: clippy
clippy: ## Run clippy lints
	$(CARGO) clippy $(CARGO_LOCKED) --all-targets --all-features -- -D warnings

.PHONY: lint
lint: fmt-check clippy ## Run format check and clippy

# ------------------------------------------------------------------------------
# gRPC + parity
# ------------------------------------------------------------------------------

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC contract compiles
	$(CARGO) check $(CARGO_LOCKED) --verbose

.PHONY: test-grpc-parity
test-grpc-parity: ## Run HTTP/gRPC parity integration tests
	$(CARGO) test $(CARGO_LOCKED) --test integration_tests --verbose

# ------------------------------------------------------------------------------
# Build & test
# ------------------------------------------------------------------------------

.PHONY: build
build: ## Build the app
	$(CARGO) build $(CARGO_LOCKED) --verbose

.PHONY: test
test: ## Run tests
	$(CARGO) test $(CARGO_LOCKED) --verbose

.PHONY: test-unit
test-unit: ## Run unit tests only (src/*)
	$(CARGO) test $(CARGO_LOCKED) --lib --verbose

.PHONY: test-integration
test-integration: ## Run integration tests only (tests/integration/*)
	$(CARGO) test $(CARGO_LOCKED) --test integration_tests --verbose

.PHONY: test-property
test-property: ## Run property-based tests (proptest) in unit test module
	@if [ -n "$(PROPTEST_CASES)" ]; then \
		PROPTEST_CASES="$(PROPTEST_CASES)" $(CARGO) test $(CARGO_LOCKED) --lib property_ --verbose; \
	else \
		$(CARGO) test $(CARGO_LOCKED) --lib property_ --verbose; \
	fi

.PHONY: test-e2e
test-e2e: ## Run e2e/ML scaffold tests (ignored tests are not run by default)
	$(CARGO) test $(CARGO_LOCKED) --test e2e_tests --verbose

.PHONY: check-e2e-min
check-e2e-min: ## Fail if discovered e2e test count is below E2E_MIN_TESTS
	E2E_MIN_TESTS="$(E2E_MIN_TESTS)" CARGO="$(CARGO)" \
		"$(PYTHON)" "$(CHECK_E2E_TESTS_SCRIPT)" --locked

.PHONY: test-e2e-ignored
test-e2e-ignored: ## Run ignored e2e/ML tests (requires external fixtures)
	$(CARGO) test $(CARGO_LOCKED) --test e2e_tests -- --ignored --nocapture

.PHONY: clean
clean: ## Clean Rust build artifacts
	$(CARGO) clean

# ------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML)
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p "$(COVERAGE_DIR)"
	$(CARGO) llvm-cov --all-features --workspace --all-targets --cobertura --output-path "$(COVERAGE_XML)"

.PHONY: coverage-integration
coverage-integration: ## Generate integration-test coverage report
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p "$(COVERAGE_DIR)"
	$(CARGO) llvm-cov --all-features --workspace --test integration_tests --cobertura --output-path "$(INTEGRATION_COVERAGE_XML)"

.PHONY: coverage-e2e
coverage-e2e: ## Generate e2e-test coverage report
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p "$(COVERAGE_DIR)"
	$(CARGO) llvm-cov --all-features --workspace --test e2e_tests --bins --cobertura --output-path "$(E2E_COVERAGE_XML)"

.PHONY: coverage-lcov
coverage-lcov: ## Generate coverage in lcov.info format (consumed by SonarCloud)
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p "$(COVERAGE_DIR)"
	$(CARGO) llvm-cov --all-features --workspace --all-targets --lcov --output-path "$(COVERAGE_LCOV)"

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	$(MAKE) coverage
	COVERAGE_LABEL="overall" COVERAGE_MIN="$(COVERAGE_MIN)" COVERAGE_MAX_DRIFT="$(COVERAGE_OVERALL_MAX_DRIFT)" COVERAGE_XML="$(COVERAGE_XML)" \
		$(COVERAGE_PYTHON) "$(CHECK_COVERAGE_SCRIPT)"
	COVERAGE_PER_FILE_MIN="$(COVERAGE_PER_FILE_MIN)" COVERAGE_FILES="$(COVERAGE_FILES)" \
		COVERAGE_XML="$(COVERAGE_XML)" $(COVERAGE_PYTHON) "$(CHECK_COVERAGE_PER_FILE_SCRIPT)"

.PHONY: coverage-segment-check
coverage-segment-check: ## Fail if integration/e2e coverage is below thresholds
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	$(MAKE) coverage-integration
	COVERAGE_LABEL="integration" COVERAGE_MIN="$(INTEGRATION_COVERAGE_MIN)" COVERAGE_MAX_DRIFT="$(COVERAGE_SEGMENT_MAX_DRIFT)" \
		COVERAGE_XML="$(INTEGRATION_COVERAGE_XML)" \
		$(COVERAGE_PYTHON) "$(CHECK_COVERAGE_SCRIPT)"
	$(MAKE) coverage-e2e
	COVERAGE_LABEL="e2e" COVERAGE_MIN="$(E2E_COVERAGE_MIN)" COVERAGE_MAX_DRIFT="$(COVERAGE_SEGMENT_MAX_DRIFT)" \
		COVERAGE_XML="$(E2E_COVERAGE_XML)" \
		$(COVERAGE_PYTHON) "$(CHECK_COVERAGE_SCRIPT)"

.PHONY: prepush-check
prepush-check: ## Run the same quality gates as pre-push in deterministic order
	$(MAKE) test-unit
	$(MAKE) test-integration
	$(MAKE) test-e2e
	$(MAKE) check-e2e-min E2E_MIN_TESTS="$(E2E_MIN_TESTS)"
	$(MAKE) coverage-segment-check \
		INTEGRATION_COVERAGE_MIN="$(INTEGRATION_COVERAGE_MIN)" \
		E2E_COVERAGE_MIN="$(E2E_COVERAGE_MIN)"

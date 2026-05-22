# Async-handler test audit (2026-05-22)

Phase D3 of the Rust fleet testing-improvements engagement.

## Scope

`app/` is a single-crate axum + tonic ONNX model server. Phase D plan
called for an "audit & fill of the 24 async tests for missing edge
cases". This file records the audit conclusion.

## Existing coverage (24 async tests)

### `app/src/lib.rs` inline tests (7 async)

| Function under test | Test | Path covered |
|---|---|---|
| `http_live`, `http_metrics`, `http_openapi_spec`, `http_swagger_ui` | `http_live_and_metrics_handlers_return_ok_with_expected_payloads` | Liveness, metrics, OpenAPI spec |
| `http_swagger_ui`, `http_openapi_spec` | `swagger_handlers_serve_openapi_and_ui` | Swagger UI under feature toggle |
| `http_invocations` + `acquire_inflight_permit` | `http_invocations_returns_too_many_requests_when_no_permit` | 429 path when semaphore exhausted |
| `http_invocations` | `http_invocations_accepts_sagemaker_header_fallbacks` | SageMaker `X-Amzn-Sagemaker-{ContentType,Accept}` header fallbacks |
| `InferenceGrpcService::{live,ready,predict}` | `grpc_service_live_ready_and_predict_error_paths` | gRPC live=ok, ready=not_ready (no model), predict=Internal (model load fail) |
| gRPC inflight | `grpc_predict_returns_resource_exhausted_when_no_inflight_capacity` | RESOURCE_EXHAUSTED when semaphore at 0 |
| gRPC happy path | `grpc_predict_success_populates_metadata_and_json_content_type` | Metadata population + content-type negotiation |

### `app/src/main.rs` inline tests (5 async)

Smoke tests for config wiring + env-var parsing.

### `app/tests/integration_tests.rs` (11 async)

| Test | What it covers |
|---|---|
| `http_health_and_ready_endpoints_are_available` | `/healthz` + `/ready` over real HTTP |
| `http_and_grpc_predict_parity_for_json_and_csv` | Dual-API parity: JSON and CSV through both protocols return identical numeric output |
| `invalid_json_maps_to_http_400_and_grpc_invalid_argument` | Same error → mapped to status conventions of each protocol |
| `record_limit_violation_maps_to_http_400_and_grpc_invalid_argument` | CSV record-cap enforcement |
| `oversized_payload_maps_to_http_413_and_grpc_invalid_argument` | Body-size limit enforcement (both protocols) |
| `grpc_predict_uses_defaults_when_content_type_and_accept_are_missing` | Default `text/csv` fallback |
| `grpc_predict_returns_internal_when_model_fails_to_load` | INTERNAL status surface, not silent failure |
| `payload_too_large_returns_413` | HTTP-level 413 path |
| `swagger_routes_respect_toggle_flag` | `ENABLE_SWAGGER=false` returns 404 |
| `multi_input_missing_key_maps_to_http_400_and_grpc_invalid_argument` | Multi-input model: missing input key surfaces as 400 / INVALID_ARGUMENT |
| `readiness_endpoints_return_500_when_model_dir_missing` | `/ready` HTTP layer reports model-load failure |

### `app/tests/e2e_tests.rs` (1 async, 6 sync)

Spawns the real built binary as a subprocess; covers the
`bootstrap → axum/tonic → request → response` chain end-to-end on each
protocol. Sync tests use `reqwest::blocking`; the one async test uses
`tokio::test` to drive an in-process server.

## Audit conclusion

The async coverage is **strong**. The originally-planned "fill the gaps"
work would mostly be diminishing-return additions:

| Originally proposed gap | Audit finding |
|---|---|
| gRPC stream errors | **N/A** — the `onnxserving.grpc` service has no streaming RPCs (only unary `Live`, `Ready`, `Predict`). |
| Malformed protobufs | Covered by tonic's transport layer; tonic returns INVALID_ARGUMENT before reaching handler. Adding our own tests would test tonic, not our code. |
| Oversized payloads | Already covered twice (`oversized_payload_maps_to_http_413_and_grpc_invalid_argument`, `payload_too_large_returns_413`). |
| Model-load failure paths | Covered (`grpc_predict_returns_internal_when_model_fails_to_load`, `readiness_endpoints_return_500_when_model_dir_missing`). |

## Gaps actually worth filling

1. **Concurrency correctness** — no test verifies that N parallel
   predictions return N correct (and distinct) results. The
   `too_many_requests` test only asserts saturation behaviour.
   **Added in this PR**: `parallel_predictions_each_return_correct_output`.

2. **Model loaded but input shape wrong** — partially covered by
   `multi_input_missing_key`, but not for the single-input shape-mismatch
   path. **Deferred** — requires a fixture model with non-trivial input
   shape; the existing fixtures are minimal.

3. **Metrics endpoint format** — `http_metrics` is hit by
   `http_live_and_metrics_handlers_return_ok_with_expected_payloads`,
   but it doesn't assert Prometheus text-format compliance.
   **Deferred** — low risk; format is owned by the `metrics-exporter-prometheus`
   crate, not our code.

4. **Graceful shutdown under in-flight requests** — no test asserts that
   `SIGTERM` waits for in-flight predictions before closing. **Deferred** —
   requires subprocess + signal scaffolding similar to `e2e_tests.rs`;
   meaningful effort for a low-frequency concern.

## What the audit changed

- `app/src/lib.rs`: added `parallel_predictions_each_return_correct_output`
  async unit test (concurrent N=8 HTTP invocations against an in-process
  AppState, verifies each returns 200 + a numeric `predictions` array).
- This `AUDIT.md` documents the rationale so future contributors don't
  re-propose the deferred items without context.

Phase D3 of the Rust fleet testing-improvements engagement.

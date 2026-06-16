use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{RwLock, Semaphore};
use tonic::transport::Server;

pub mod adapter;
mod config;
pub mod grpc_service;
pub mod http;
pub mod input;
pub mod openapi;
mod telemetry;

pub mod grpc {
    tonic::include_proto!("onnxserving.grpc");
}

pub use config::AppConfig;
pub use telemetry::init_telemetry;

use adapter::{build_onnx_tensors, load_adapter, BaseAdapter};
use input::{
    apply_feature_selection, format_csv_predictions, load_json_map, load_multi_input_records,
    normalized_accept, parse_csv_rows, parse_json_rows, parse_jsonl_rows, resolve_content_type,
    should_use_onnx_multi_input, validate_input_mode, validate_tabular_matrix_shape,
    wrap_predictions_if_needed, ParsedInput, CSV_CONTENT_TYPES, JSON_CONTENT_TYPES,
    JSON_LINES_CONTENT_TYPES,
};

#[derive(Clone)]
pub struct AppState {
    pub(crate) cfg: AppConfig,
    pub(crate) adapter: Arc<RwLock<Option<Arc<dyn BaseAdapter>>>>,
    pub(crate) inflight: Arc<Semaphore>,
}

impl AppState {
    pub fn new(cfg: AppConfig) -> Self {
        let max_inflight = cfg.max_inflight.max(1);
        Self {
            cfg,
            adapter: Arc::new(RwLock::new(None)),
            inflight: Arc::new(Semaphore::new(max_inflight)),
        }
    }

    pub(crate) async fn ensure_adapter_loaded(&self) -> Result<Arc<dyn BaseAdapter>, String> {
        if let Some(existing) = self.adapter.read().await.as_ref() {
            return Ok(existing.clone());
        }
        let loaded = load_adapter(&self.cfg)?;
        let mut writer = self.adapter.write().await;
        *writer = Some(loaded.clone());
        Ok(loaded)
    }

    pub(crate) fn parse_payload(
        &self,
        payload: &[u8],
        content_type: &str,
    ) -> Result<ParsedInput, String> {
        validate_input_mode(&self.cfg)?;
        let normalized = resolve_content_type(content_type);
        let onnx_input_map = load_json_map(&self.cfg.onnx_input_map_json)?;
        if should_use_onnx_multi_input(&onnx_input_map, &normalized) {
            return self.parse_onnx_multi_input(payload, &normalized, &onnx_input_map);
        }

        let matrix = self.parse_tabular_matrix(payload, &normalized, content_type)?;
        validate_tabular_matrix_shape(&matrix, &self.cfg)?;
        let matrix = apply_feature_selection(matrix, &self.cfg)?;

        Ok(ParsedInput {
            x: Some(matrix),
            tensors: None,
            meta: None,
        })
    }

    fn parse_tabular_matrix(
        &self,
        payload: &[u8],
        normalized_content_type: &str,
        raw_content_type: &str,
    ) -> Result<Vec<Vec<f64>>, String> {
        if CSV_CONTENT_TYPES.contains(&normalized_content_type) {
            return parse_csv_rows(payload, &self.cfg);
        }
        if JSON_CONTENT_TYPES.contains(&normalized_content_type) {
            return parse_json_rows(payload, &self.cfg);
        }
        if JSON_LINES_CONTENT_TYPES.contains(&normalized_content_type) {
            return parse_jsonl_rows(payload, &self.cfg);
        }
        Err(format!("Unsupported Content-Type: {raw_content_type}"))
    }

    fn parse_onnx_multi_input(
        &self,
        payload: &[u8],
        content_type: &str,
        onnx_input_map: &HashMap<String, String>,
    ) -> Result<ParsedInput, String> {
        let records = load_multi_input_records(payload, content_type, &self.cfg)?;
        if records.is_empty() {
            return Err(
                "ONNX multi-input mode expects a JSON object or a non-empty list of objects"
                    .to_string(),
            );
        }
        let tensors = build_onnx_tensors(&records, onnx_input_map, self.cfg.onnx_dynamic_batch)?;

        Ok(ParsedInput {
            x: None,
            tensors: Some(tensors),
            meta: Some(json!({"records": records.len(), "mode": "onnx_multi_input"})),
        })
    }

    pub(crate) fn format_output(
        &self,
        predictions: Value,
        accept: &str,
    ) -> Result<(Vec<u8>, String), String> {
        if predictions.is_object() {
            let bytes = serde_json::to_vec(&predictions)
                .map_err(|err| format!("failed to encode json: {err}"))?;
            return Ok((bytes, "application/json".to_string()));
        }

        let normalized_accept = normalized_accept(accept, self.cfg.default_accept.as_str());
        if CSV_CONTENT_TYPES.contains(&normalized_accept.as_str()) {
            let csv = format_csv_predictions(&predictions, &self.cfg.csv_delimiter)?;
            return Ok((csv.into_bytes(), "text/csv".to_string()));
        }

        let payload = wrap_predictions_if_needed(predictions, &self.cfg);
        let bytes =
            serde_json::to_vec(&payload).map_err(|err| format!("failed to encode json: {err}"))?;
        Ok((bytes, "application/json".to_string()))
    }
}

pub use grpc_service::InferenceGrpcService;
pub use http::build_http_router;

pub async fn serve_http(listener: TcpListener, cfg: AppConfig) -> Result<(), std::io::Error> {
    let app = build_http_router(cfg);
    axum::serve(listener, app).await
}

pub async fn serve_grpc(
    listener: TcpListener,
    cfg: AppConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = InferenceGrpcService::new(cfg);
    Server::builder()
        .add_service(grpc::inference_service_server::InferenceServiceServer::new(
            service,
        ))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

pub async fn run_http_server(host: &str, port: u16, cfg: AppConfig) -> Result<(), std::io::Error> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("valid listen address");
    let listener = TcpListener::bind(addr).await?;
    serve_http(listener, cfg).await
}

pub async fn run_grpc_server(
    host: &str,
    port: u16,
    cfg: AppConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("valid listen address");
    let listener = TcpListener::bind(addr).await?;
    serve_grpc(listener, cfg).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{load_adapter, OnnxAdapter};
    use crate::grpc::inference_service_server::InferenceService;
    use crate::grpc_service::InferenceGrpcService;
    use crate::http::{
        header_value_with_fallback, http_invocations, resolve_request_media_types,
        validate_payload_size, SAGEMAKER_ACCEPT_HEADER, SAGEMAKER_CONTENT_TYPE_HEADER,
    };
    use crate::input::{
        apply_feature_selection, format_csv_predictions, load_json_map, load_multi_input_records,
        normalized_accept, parse_col_selector, parse_csv_rows, parse_json_records, parse_json_rows,
        parse_jsonl_records, parse_jsonl_rows, should_use_onnx_multi_input,
        strip_content_type_params, validate_dynamic_batch_sizes, validate_input_mode,
        validate_tabular_matrix_shape, value_to_numeric_rows, wrap_predictions_if_needed,
    };
    use crate::openapi::{
        openapi_candidate_paths, read_openapi_from_env, OPENAPI_SPEC_PATH_ENV_KEY,
    };
    use axum::body::to_bytes;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::header::{ACCEPT, CONTENT_TYPE};
    use axum::http::{HeaderMap, HeaderName, StatusCode};
    use proptest::prelude::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;
    use tokio::sync::{RwLock, Semaphore};
    use tonic::Request;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = env::var(key).ok();
            env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                env::set_var(self.key, previous);
            } else {
                env::remove_var(self.key);
            }
        }
    }

    fn cfg_with_temp_model_fixture() -> (TempDir, AppConfig) {
        let tmp = tempfile::tempdir().expect("temp dir");
        let model_path: PathBuf = tmp.path().join("model.onnx");
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("model.onnx");
        let fixture = fs::read(&fixture_path).expect("read fixture model");
        fs::write(&model_path, fixture).expect("write fixture model");
        let cfg = AppConfig {
            model_type: "onnx".to_string(),
            model_dir: tmp.path().to_string_lossy().to_string(),
            model_filename: "model.onnx".to_string(),
            ..AppConfig::default()
        };
        (tmp, cfg)
    }

    #[test]
    fn json_instances_are_parsed() {
        let cfg = AppConfig::default();
        let state = AppState::new(cfg);
        let payload = br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#;
        let parsed = state
            .parse_payload(payload, "application/json")
            .expect("json parse should pass");
        assert_eq!(parsed.batch_size().expect("batch"), 2);
    }

    #[test]
    fn csv_header_auto_detect_works() {
        let cfg = AppConfig {
            csv_has_header: "auto".to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let payload = b"f1,f2\n1,2\n3,4\n";
        let parsed = state
            .parse_payload(payload, "text/csv")
            .expect("csv parse should pass");
        assert_eq!(parsed.batch_size().expect("batch"), 2);
    }

    #[test]
    fn output_formatter_supports_csv() {
        let cfg = AppConfig::default();
        let state = AppState::new(cfg);
        let (body, content_type) = state
            .format_output(
                Value::Array(vec![Value::from(1), Value::from(2)]),
                "text/csv",
            )
            .expect("csv format should pass");
        assert_eq!(content_type, "text/csv");
        assert_eq!(String::from_utf8(body).expect("utf8"), "1\n2");
    }

    #[test]
    fn strip_content_type_params_normalizes_media_type() {
        assert_eq!(
            strip_content_type_params("Application/JSON; charset=utf-8"),
            "application/json"
        );
    }

    #[test]
    fn load_json_map_handles_empty_and_invalid_values() {
        let empty = load_json_map("   ").expect("empty map");
        assert!(empty.is_empty());

        let err = load_json_map("[]").expect_err("non-object should fail");
        assert!(err.contains("Expected JSON object mapping"));
    }

    #[test]
    fn parse_payload_stage_helpers_cover_core_decisions() {
        let err = validate_input_mode(&AppConfig {
            input_mode: "image".to_string(),
            ..AppConfig::default()
        })
        .expect_err("invalid mode");
        assert!(err.contains("not implemented"));
        assert!(validate_input_mode(&AppConfig::default()).is_ok());

        let mut onnx_map = HashMap::new();
        onnx_map.insert("a".to_string(), "input_a".to_string());
        assert!(should_use_onnx_multi_input(&onnx_map, "application/json"));
        assert!(!should_use_onnx_multi_input(
            &HashMap::new(),
            "application/json"
        ));
        assert!(!should_use_onnx_multi_input(&onnx_map, "text/csv"));

        let shape_err =
            validate_tabular_matrix_shape(&[], &AppConfig::default()).expect_err("empty matrix");
        assert!(shape_err.contains("Parsed payload is empty"));
        let mismatch_err = validate_tabular_matrix_shape(
            &[vec![1.0, 2.0]],
            &AppConfig {
                tabular_num_features: 3,
                ..AppConfig::default()
            },
        )
        .expect_err("feature mismatch");
        assert!(mismatch_err.contains("Feature count mismatch"));
    }

    #[test]
    fn apply_feature_selection_returns_original_when_selectors_are_empty() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let selected =
            apply_feature_selection(rows.clone(), &AppConfig::default()).expect("selection pass");
        assert_eq!(selected, rows);
    }

    #[test]
    fn multi_input_helpers_cover_json_dispatch_and_batch_validation() {
        let records = load_multi_input_records(
            br#"{"instances":[{"a":[1,2]},{"a":[3,4]}]}"#,
            "application/json",
            &AppConfig::default(),
        )
        .expect("json records");
        assert_eq!(records.len(), 2);

        let records = load_multi_input_records(
            br#"{"a":[1,2]}
{"a":[3,4]}"#,
            "application/x-jsonlines",
            &AppConfig::default(),
        )
        .expect("jsonl records");
        assert_eq!(records.len(), 2);

        let invalid = validate_dynamic_batch_sizes(&[2, 0]).expect_err("zero batch invalid");
        assert!(invalid.contains("batch dimension invalid"));
        let mismatch =
            validate_dynamic_batch_sizes(&[2, 3]).expect_err("mismatched batch sizes invalid");
        assert!(mismatch.contains("mismatched batch sizes"));
        assert!(validate_dynamic_batch_sizes(&[2, 2]).is_ok());
    }

    #[test]
    fn media_and_output_helpers_cover_fallbacks() {
        assert_eq!(
            normalized_accept("text/csv,application/json", "application/json"),
            "text/csv"
        );
        assert_eq!(normalized_accept("", "application/json"), "");

        let wrapped = wrap_predictions_if_needed(
            Value::Array(vec![Value::from(1)]),
            &AppConfig {
                predictions_only: false,
                json_output_key: "result".to_string(),
                ..AppConfig::default()
            },
        );
        assert_eq!(wrapped, json!({"result":[1]}));
        let plain = wrap_predictions_if_needed(Value::from(1), &AppConfig::default());
        assert_eq!(plain, Value::from(1));
    }

    #[test]
    fn request_phase_helpers_cover_media_resolution_and_size_checks() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        headers.insert(ACCEPT, axum::http::HeaderValue::from_static("text/csv"));
        let cfg = AppConfig::default();
        let (content_type, accept) = resolve_request_media_types(&headers, &cfg);
        assert_eq!(content_type, "application/json");
        assert_eq!(accept, "text/csv");

        let oversized = validate_payload_size(
            128,
            &AppConfig {
                max_body_bytes: 64,
                ..AppConfig::default()
            },
        )
        .expect("oversized payload should return response");
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(validate_payload_size(64, &AppConfig::default()).is_none());
    }

    #[test]
    fn parse_json_records_supports_instances_and_object_shape() {
        let cfg = AppConfig::default();
        let from_instances = parse_json_records(br#"{"instances":[{"a":1},{"a":2}]}"#, &cfg)
            .expect("instances parse");
        assert_eq!(from_instances.len(), 2);

        let from_object = parse_json_records(br#"{"a":1,"b":2}"#, &cfg).expect("object parse");
        assert_eq!(from_object.len(), 1);
        assert!(from_object[0].contains_key("a"));
    }

    #[test]
    fn parse_jsonl_records_rejects_non_object_lines() {
        let err = parse_jsonl_records(br#"[1,2,3]"#).expect_err("must reject non-object");
        assert!(err.contains("JSON object"));
    }

    #[test]
    fn parse_json_records_reject_non_object_array_entries() {
        let cfg = AppConfig::default();
        let err = parse_json_records(br#"{"instances":[{"a":1},2]}"#, &cfg)
            .expect_err("mixed records must fail");
        assert!(err.contains("expects each record to be a JSON object"));
    }

    #[test]
    fn parse_jsonl_records_reject_invalid_utf8() {
        let err = parse_jsonl_records(&[0x80]).expect_err("invalid utf8 must fail");
        assert!(err.contains("invalid utf-8"));
    }

    #[test]
    fn parse_json_rows_and_jsonl_rows_support_feature_shortcuts() {
        let cfg = AppConfig::default();
        let rows = parse_json_rows(br#"{"features":[1.0,2.0]}"#, &cfg).expect("features row");
        assert_eq!(rows, vec![vec![1.0, 2.0]]);

        let jsonl = br#"{"features":[3,4]}
{"features":[5,6]}"#;
        let rows = parse_jsonl_rows(jsonl, &cfg).expect("features jsonl");
        assert_eq!(rows, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
    }

    #[test]
    fn value_to_numeric_rows_handles_scalar_and_rejects_text() {
        let scalar = value_to_numeric_rows(&Value::from(7.5)).expect("scalar rows");
        assert_eq!(scalar, vec![vec![7.5]]);
        let err = value_to_numeric_rows(&Value::String("x".to_string()))
            .expect_err("non-numeric should fail");
        assert!(err.contains("Expected tabular numeric payload"));
    }

    #[test]
    fn parse_csv_rows_supports_header_modes() {
        let cfg_true = AppConfig {
            csv_has_header: "true".to_string(),
            ..AppConfig::default()
        };
        let rows =
            parse_csv_rows(b"f1,f2\n1,2\n3,4\n", &cfg_true).expect("header=true should parse");
        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let cfg_invalid = AppConfig {
            csv_has_header: "maybe".to_string(),
            ..AppConfig::default()
        };
        let err = parse_csv_rows(b"1,2\n", &cfg_invalid).expect_err("invalid mode should fail");
        assert!(err.contains("CSV_HAS_HEADER must be auto|true|false"));
    }

    #[test]
    fn parse_csv_rows_covers_empty_and_invalid_numeric_paths() {
        let cfg = AppConfig::default();
        let empty_err = parse_csv_rows(b"", &cfg).expect_err("empty csv must fail");
        assert!(empty_err.contains("Empty CSV payload"));

        let cfg_no_header = AppConfig {
            csv_has_header: "false".to_string(),
            ..AppConfig::default()
        };
        let bad_err =
            parse_csv_rows(b"1,abc\n", &cfg_no_header).expect_err("non-numeric token must fail");
        assert!(bad_err.contains("Expected numeric value in CSV payload"));
    }

    #[test]
    fn parse_col_selector_supports_range_and_list() {
        assert_eq!(
            parse_col_selector("", 3).expect("empty selector means all columns"),
            vec![0, 1, 2]
        );
        assert_eq!(
            parse_col_selector("1:3", 5).expect("range selector"),
            vec![1, 2]
        );
        assert_eq!(parse_col_selector(":2", 5).expect("open start"), vec![0, 1]);
        assert_eq!(
            parse_col_selector("2:", 5).expect("open end"),
            vec![2, 3, 4]
        );
        assert_eq!(
            parse_col_selector("0,2,4", 5).expect("list selector"),
            vec![0, 2, 4]
        );
        let err = parse_col_selector("bad", 5).expect_err("invalid selector must fail");
        assert!(err.contains("Invalid column selector"));
    }

    #[test]
    fn format_output_wraps_predictions_when_predictions_only_is_false() {
        let cfg = AppConfig {
            predictions_only: false,
            json_output_key: "y_hat".to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let (body, content_type) = state
            .format_output(Value::Array(vec![Value::from(1)]), "application/json")
            .expect("json output");
        assert_eq!(content_type, "application/json");
        let parsed: Value = serde_json::from_slice(&body).expect("valid json");
        assert_eq!(parsed, json!({"y_hat":[1]}));
    }

    #[test]
    fn header_value_with_fallback_prefers_primary_then_fallback_then_default() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/csv"),
        );
        let value = header_value_with_fallback(
            &headers,
            CONTENT_TYPE,
            SAGEMAKER_CONTENT_TYPE_HEADER,
            "application/json",
        );
        assert_eq!(value, "text/csv");

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SAGEMAKER_CONTENT_TYPE_HEADER),
            axum::http::HeaderValue::from_static("application/json"),
        );
        let fallback = header_value_with_fallback(
            &headers,
            CONTENT_TYPE,
            SAGEMAKER_CONTENT_TYPE_HEADER,
            "text/plain",
        );
        assert_eq!(fallback, "application/json");

        let defaulted = header_value_with_fallback(
            &HeaderMap::new(),
            CONTENT_TYPE,
            SAGEMAKER_CONTENT_TYPE_HEADER,
            "text/plain",
        );
        assert_eq!(defaulted, "text/plain");
    }

    #[test]
    fn header_value_with_fallback_uses_default_for_invalid_fallback_name() {
        let value = header_value_with_fallback(
            &HeaderMap::new(),
            CONTENT_TYPE,
            "Invalid Header Name",
            "application/json",
        );
        assert_eq!(value, "application/json");
    }

    #[test]
    fn load_openapi_yaml_supports_blank_and_missing_env_paths() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let _missing = EnvVarGuard::set(OPENAPI_SPEC_PATH_ENV_KEY, "/definitely/missing/spec.yaml");
        assert!(read_openapi_from_env().is_none());
        drop(_missing);

        let _blank = EnvVarGuard::set(OPENAPI_SPEC_PATH_ENV_KEY, "   ");
        assert!(read_openapi_from_env().is_none());
        let candidates = openapi_candidate_paths();
        assert_eq!(candidates.len(), 2);
        for path in candidates {
            assert!(path.ends_with("openapi.yaml"));
        }
    }

    #[tokio::test]
    async fn http_live_and_metrics_handlers_return_ok_with_expected_payloads() {
        use crate::http::{http_live, http_metrics};
        use axum::response::IntoResponse;

        let live = http_live().await.into_response();
        assert_eq!(live.status(), StatusCode::OK);
        let live_body = to_bytes(live.into_body(), usize::MAX)
            .await
            .expect("live body");
        assert_eq!(
            String::from_utf8(live_body.to_vec()).expect("live utf8"),
            "\n"
        );

        let metrics = http_metrics().await.into_response();
        assert_eq!(metrics.status(), StatusCode::OK);
        let metrics_body = to_bytes(metrics.into_body(), usize::MAX)
            .await
            .expect("metrics body");
        let metrics_text = String::from_utf8(metrics_body.to_vec()).expect("metrics utf8");
        assert!(metrics_text.contains("byoc_up 1"));
    }

    #[test]
    fn format_csv_predictions_and_value_to_string_cover_non_array_cases() {
        let scalar = format_csv_predictions(&Value::from(true), ",").expect("scalar csv");
        assert_eq!(scalar, "true");

        let rows = format_csv_predictions(
            &Value::Array(vec![
                Value::Array(vec![Value::Null, Value::from(2)]),
                Value::Array(vec![Value::from("x"), Value::from(4)]),
            ]),
            ";",
        )
        .expect("row csv");
        assert_eq!(rows, ";2\nx;4");
    }

    #[tokio::test]
    async fn swagger_handlers_serve_openapi_and_ui() {
        use crate::http::{http_openapi_spec, http_swagger_ui};
        use axum::response::IntoResponse;

        let tmp = tempfile::tempdir().expect("temp dir");
        let spec_path = tmp.path().join("openapi.yaml");
        fs::write(&spec_path, "openapi: 3.1.0\n").expect("write openapi spec");
        let _env_guard = {
            let _guard = env_lock()
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            EnvVarGuard::set(
                OPENAPI_SPEC_PATH_ENV_KEY,
                spec_path.to_string_lossy().as_ref(),
            )
        };

        let spec_response = http_openapi_spec().await.into_response();
        assert_eq!(spec_response.status(), StatusCode::OK);
        let spec_body = to_bytes(spec_response.into_body(), usize::MAX)
            .await
            .expect("spec body");
        let spec_text = String::from_utf8(spec_body.to_vec()).expect("spec utf8");
        assert!(spec_text.contains("openapi: 3.1.0"));

        let ui_response = http_swagger_ui().await.into_response();
        assert_eq!(ui_response.status(), StatusCode::OK);
        let ui_body = to_bytes(ui_response.into_body(), usize::MAX)
            .await
            .expect("ui body");
        let ui_text = String::from_utf8(ui_body.to_vec()).expect("ui utf8");
        assert!(ui_text.contains("SwaggerUIBundle"));
        assert!(ui_text.contains("/openapi.yaml"));
    }

    #[test]
    fn tensor_to_json_supports_i64_and_higher_dimensions() {
        let i64_tensor: tract_onnx::prelude::Tensor =
            tract_onnx::prelude::tract_ndarray::Array1::<i64>::from_vec(vec![1, 2, 3]).into();
        let i64_json = OnnxAdapter::tensor_to_json(&i64_tensor).expect("i64 tensor json");
        assert_eq!(i64_json, json!([1, 2, 3]));

        let f32_3d: tract_onnx::prelude::Tensor =
            tract_onnx::prelude::tract_ndarray::Array3::<f32>::from_shape_vec(
                (1, 2, 2),
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .expect("3d tensor")
            .into();
        let f32_json = OnnxAdapter::tensor_to_json(&f32_3d).expect("f32 tensor json");
        assert_eq!(f32_json, json!([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn tensor_to_json_rejects_unsupported_dtype() {
        let bool_tensor: tract_onnx::prelude::Tensor =
            tract_onnx::prelude::tract_ndarray::Array1::<bool>::from_vec(vec![true, false]).into();
        let err = OnnxAdapter::tensor_to_json(&bool_tensor).expect_err("bool tensor must fail");
        assert!(err.contains("unsupported ONNX output tensor dtype"));
    }

    #[test]
    fn parsed_input_batch_size_validates_tensors() {
        let empty = ParsedInput {
            x: None,
            tensors: Some(HashMap::new()),
            meta: None,
        };
        let err = empty.batch_size().expect_err("empty tensors must fail");
        assert!(err.contains("no features/tensors"));

        let mut bad_type = HashMap::new();
        bad_type.insert("x".to_string(), Value::from(1));
        let parsed = ParsedInput {
            x: None,
            tensors: Some(bad_type),
            meta: None,
        };
        let err = parsed.batch_size().expect_err("scalar tensor must fail");
        assert!(err.contains("array-like"));
    }

    #[test]
    fn app_config_model_path_errors_when_dir_missing_or_not_directory() {
        let missing = AppConfig {
            model_dir: "/definitely/missing/dir".to_string(),
            ..AppConfig::default()
        };
        let err = missing.model_path().expect_err("missing dir must fail");
        assert!(err.contains("does not exist"), "unexpected error: {err}");

        let tmp = tempfile::tempdir().expect("temp dir");
        let file_path = tmp.path().join("model.onnx");
        fs::write(&file_path, b"dummy").expect("write file");
        let not_dir = AppConfig {
            model_dir: file_path.to_string_lossy().to_string(),
            ..AppConfig::default()
        };
        let err = not_dir.model_path().expect_err("non-directory must fail");
        assert!(err.contains("not a directory"), "unexpected error: {err}");
    }

    #[test]
    fn load_adapter_rejects_unknown_model_type_and_missing_model() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let unknown = AppConfig {
            model_type: "xgboost".to_string(),
            model_dir: tmp.path().to_string_lossy().to_string(),
            ..AppConfig::default()
        };
        let err = match load_adapter(&unknown) {
            Ok(_) => panic!("unknown model type should fail"),
            Err(err) => err,
        };
        assert!(err.contains("not implemented"));

        let missing = AppConfig {
            model_type: "".to_string(),
            model_dir: tmp.path().to_string_lossy().to_string(),
            ..AppConfig::default()
        };
        let err = match load_adapter(&missing) {
            Ok(_) => panic!("missing model should fail"),
            Err(err) => err,
        };
        assert!(err.contains("Set MODEL_TYPE=onnx"));
    }

    #[test]
    fn parse_payload_rejects_unsupported_modes_and_content_types() {
        let cfg_mode = AppConfig {
            input_mode: "image".to_string(),
            ..AppConfig::default()
        };
        let mode_state = AppState::new(cfg_mode);
        let mode_err = mode_state
            .parse_payload(b"1,2\n", "text/csv")
            .expect_err("non-tabular mode should fail");
        assert!(mode_err.contains("not implemented"));

        let cfg_content = AppConfig::default();
        let content_state = AppState::new(cfg_content);
        let content_err = content_state
            .parse_payload(b"1,2\n", "application/xml")
            .expect_err("unsupported content type should fail");
        assert!(content_err.contains("Unsupported Content-Type"));
    }

    #[test]
    fn parse_payload_validates_feature_count_and_header_only_csv() {
        let cfg = AppConfig {
            tabular_num_features: 3,
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let mismatch_err = state
            .parse_payload(b"1,2\n", "text/csv")
            .expect_err("feature mismatch should fail");
        assert!(mismatch_err.contains("Feature count mismatch"));

        let header_only_cfg = AppConfig {
            csv_has_header: "true".to_string(),
            ..AppConfig::default()
        };
        let header_state = AppState::new(header_only_cfg);
        let header_err = header_state
            .parse_payload(b"f1,f2\n", "text/csv")
            .expect_err("header-only csv should fail");
        assert!(header_err.contains("only header row"));
    }

    #[test]
    fn parse_payload_multi_input_reports_missing_record_key() {
        let cfg = AppConfig {
            onnx_input_map_json: r#"{"a":"input_a","b":"input_b"}"#.to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let payload = br#"{"instances":[{"a":[1.0,2.0]}]}"#;
        let err = state
            .parse_payload(payload, "application/json")
            .expect_err("missing key should fail");
        assert!(err.contains("Missing key 'b'"));
    }

    proptest! {
        #[test]
        fn property_parse_payload_json_preserves_shape(
            rows in proptest::collection::vec(
                proptest::collection::vec(-1000i16..1000i16, 1..8),
                1..24
            )
        ) {
            let cfg = AppConfig::default();
            let state = AppState::new(cfg);
            let instances: Vec<Vec<f64>> = rows
                .iter()
                .map(|row| row.iter().map(|value| f64::from(*value)).collect())
                .collect();
            let payload = serde_json::to_vec(&json!({"instances": instances}))
                .expect("json payload");

            let parsed = state
                .parse_payload(payload.as_slice(), "application/json")
                .expect("json parse should pass");

            let parsed_rows = parsed.x.expect("tabular rows expected");
            prop_assert_eq!(parsed_rows.len(), instances.len());
            for (parsed_row, input_row) in parsed_rows.iter().zip(instances.iter()) {
                prop_assert_eq!(parsed_row.len(), input_row.len());
                for (parsed_value, input_value) in parsed_row.iter().zip(input_row.iter()) {
                    prop_assert!((parsed_value - input_value).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn parse_payload_applies_column_selection_paths() {
        let cfg = AppConfig {
            tabular_feature_columns: "1".to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let parsed = state
            .parse_payload(b"1,2\n3,4\n", "text/csv")
            .expect("feature selector should parse");
        assert_eq!(parsed.x, Some(vec![vec![2.0], vec![4.0]]));

        let cfg = AppConfig {
            tabular_id_columns: "0".to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let parsed = state
            .parse_payload(b"10,2,3\n11,4,5\n", "text/csv")
            .expect("id selector should infer feature columns");
        assert_eq!(parsed.x, Some(vec![vec![2.0, 3.0], vec![4.0, 5.0]]));
    }

    #[tokio::test]
    async fn http_invocations_returns_too_many_requests_when_no_permit() {
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let state = Arc::new(AppState {
            cfg,
            adapter: Arc::new(RwLock::new(None)),
            inflight: Arc::new(Semaphore::new(0)),
        });
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        let response = http_invocations(
            State(state),
            headers,
            Bytes::from_static(br#"{"instances":[[1.0,2.0]]}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn http_invocations_accepts_sagemaker_header_fallbacks() {
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let state = Arc::new(AppState::new(cfg));
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SAGEMAKER_CONTENT_TYPE_HEADER),
            axum::http::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            HeaderName::from_static(SAGEMAKER_ACCEPT_HEADER),
            axum::http::HeaderValue::from_static("text/csv"),
        );
        let response = http_invocations(
            State(state),
            headers,
            Bytes::from_static(br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/csv")
        );
    }

    #[tokio::test]
    async fn grpc_service_live_ready_and_predict_error_paths() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let cfg = AppConfig {
            model_type: "onnx".to_string(),
            model_dir: tmp.path().to_string_lossy().to_string(),
            model_filename: "missing.onnx".to_string(),
            ..AppConfig::default()
        };
        let service = InferenceGrpcService::new(cfg);

        let live = service
            .live(Request::new(crate::grpc::LiveRequest {}))
            .await
            .expect("live must succeed")
            .into_inner();
        assert!(live.ok);

        let ready = service
            .ready(Request::new(crate::grpc::ReadyRequest {}))
            .await
            .expect("ready must respond")
            .into_inner();
        assert!(!ready.ok);
        assert_eq!(ready.status, "not_ready");

        let predict = service
            .predict(Request::new(crate::grpc::PredictRequest {
                payload: b"1,2\n".to_vec(),
                content_type: "text/csv".to_string(),
                accept: "application/json".to_string(),
            }))
            .await
            .expect_err("predict should fail when model failed loading");
        assert_eq!(predict.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn grpc_predict_returns_resource_exhausted_when_no_inflight_capacity() {
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let state = AppState {
            cfg,
            adapter: Arc::new(RwLock::new(None)),
            inflight: Arc::new(Semaphore::new(0)),
        };
        let service = InferenceGrpcService {
            state,
            load_error: None,
        };

        let result = service
            .predict(Request::new(crate::grpc::PredictRequest {
                payload: br#"{"instances":[[1.0,2.0]]}"#.to_vec(),
                content_type: "application/json".to_string(),
                accept: "application/json".to_string(),
            }))
            .await
            .expect_err("must fail when semaphore has no permits");
        assert_eq!(result.code(), tonic::Code::ResourceExhausted);
        assert!(result.message().contains("too_many_requests"));
    }

    #[tokio::test]
    async fn grpc_predict_success_populates_metadata_and_json_content_type() {
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let service = InferenceGrpcService::new(cfg);
        let reply = service
            .predict(Request::new(crate::grpc::PredictRequest {
                payload: br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
                content_type: "application/json".to_string(),
                accept: "".to_string(),
            }))
            .await
            .expect("predict should succeed")
            .into_inner();
        assert_eq!(reply.content_type, "application/json");
        assert_eq!(reply.metadata.get("batch_size"), Some(&"2".to_string()));
    }

    #[tokio::test]
    async fn parallel_predictions_each_return_correct_output() {
        // Verifies concurrency correctness: N parallel HTTP invocations
        // against the same AppState each return 200 and a parseable JSON
        // response with a numeric `predictions` array. Guards against
        // shared-state corruption in the prediction pipeline (e.g. a
        // shared buffer being mutated cross-request, or the inflight
        // semaphore being mis-counted).
        const N: usize = 8;
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let state = Arc::new(AppState::new(cfg));
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );

        // Same payload shape as `http_invocations_accepts_sagemaker_header_fallbacks`
        // — the bundled fixture model expects 2-column input rows.
        let payloads: Vec<Bytes> = (0..N)
            .map(|i| {
                // Each request has its own distinct values so a shared-state bug
                // would produce visibly mismatched batches across responses.
                let a = (i + 1) as f64;
                let b = (i + 2) as f64;
                let body = format!(
                    "{{\"instances\":[[{},{}],[{},{}]]}}",
                    a,
                    b,
                    a + 10.0,
                    b + 10.0
                );
                Bytes::from(body)
            })
            .collect();

        let tasks: Vec<_> = payloads
            .into_iter()
            .map(|body| {
                let state = state.clone();
                let headers = headers.clone();
                tokio::spawn(async move { http_invocations(State(state), headers, body).await })
            })
            .collect();

        for (idx, task) in tasks.into_iter().enumerate() {
            let response = task.await.expect("worker task did not panic");
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "parallel request {idx} failed: status {}",
                response.status()
            );
            let (parts, body) = response.into_parts();
            assert_eq!(
                parts
                    .headers
                    .get(CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok()),
                Some("application/json"),
                "parallel request {idx} returned wrong content-type"
            );
            let bytes = axum::body::to_bytes(body, usize::MAX)
                .await
                .expect("read body");
            let parsed: serde_json::Value =
                serde_json::from_slice(&bytes).expect("response is JSON");
            // Default config has predictions_only=true → body is the raw
            // predictions array (not wrapped under any key).
            let preds = parsed.as_array().expect("response is a JSON array");
            assert!(
                !preds.is_empty(),
                "parallel request {idx} returned empty predictions"
            );
        }
    }
}

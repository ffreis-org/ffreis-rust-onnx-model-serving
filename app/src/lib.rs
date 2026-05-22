use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::response::{Html, IntoResponse, Response as AxumResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::time::timeout;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};
use tract_onnx::prelude::{tvec, Framework, InferenceModelExt, RunnableModel, TypedFact, TypedOp};

mod config;
mod telemetry;

pub use config::AppConfig;
use telemetry::attach_trace_correlation_headers;
pub use telemetry::init_telemetry;

pub mod grpc {
    tonic::include_proto!("onnxserving.grpc");
}

const JSON_CONTENT_TYPES: &[&str] = &["application/json", "application/*+json"];
const JSON_LINES_CONTENT_TYPES: &[&str] = &[
    "application/jsonlines",
    "application/x-jsonlines",
    "application/jsonl",
    "application/x-ndjson",
];
const CSV_CONTENT_TYPES: &[&str] = &["text/csv", "application/csv"];
const SAGEMAKER_CONTENT_TYPE_HEADER: &str = "x-amzn-sagemaker-content-type";
const SAGEMAKER_ACCEPT_HEADER: &str = "x-amzn-sagemaker-accept";
const OPENAPI_SPEC_PATH_ENV_KEY: &str = "OPENAPI_SPEC_PATH";
const SWAGGER_UI_HTML: &str = r##"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.yaml",
      dom_id: "#swagger-ui",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
    });
  </script>
</body>
</html>
"##;

#[derive(Clone, Debug)]
pub struct ParsedInput {
    pub x: Option<Vec<Vec<f64>>>,
    pub tensors: Option<HashMap<String, Value>>,
    pub meta: Option<Value>,
}

impl ParsedInput {
    fn batch_size(&self) -> Result<usize, String> {
        if let Some(x) = &self.x {
            return Ok(x.len());
        }
        if let Some(tensors) = &self.tensors {
            let mut inferred: Vec<usize> = Vec::new();
            for value in tensors.values() {
                match value {
                    Value::Array(rows) => inferred.push(rows.len()),
                    _ => return Err("ONNX input tensor must be array-like".to_string()),
                }
            }
            if inferred.is_empty() {
                return Err("Parsed input contained no features/tensors".to_string());
            }
            if inferred.contains(&0) {
                return Err("ONNX_DYNAMIC_BATCH enabled but batch dimension invalid".to_string());
            }
            if inferred.windows(2).any(|w| w[0] != w[1]) {
                return Err(format!(
                    "ONNX inputs have mismatched batch sizes: {inferred:?}"
                ));
            }
            return Ok(inferred[0]);
        }
        Err("Parsed input contained no features/tensors".to_string())
    }
}

pub trait BaseAdapter: Send + Sync {
    fn is_ready(&self) -> bool;
    fn predict(&self, parsed_input: &ParsedInput) -> Result<Value, String>;
}

type OnnxRunnableModel = RunnableModel<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

#[derive(Clone)]
struct OnnxAdapter {
    cfg: AppConfig,
    model: Option<OnnxRunnableModel>,
    output_map: HashMap<String, String>,
}

impl OnnxAdapter {
    fn new(cfg: AppConfig) -> Result<Self, String> {
        let path = cfg.model_path()?;
        if !path.exists() {
            return Err(format!("ONNX model not found: {}", path.display()));
        }
        let model = tract_onnx::onnx()
            .model_for_path(&path)
            .and_then(|model| model.into_optimized())
            .and_then(|model| model.into_runnable())
            .map_err(|e| {
                format!(
                    "Failed to load or prepare ONNX model {}: {}",
                    path.display(),
                    e
                )
            })?;
        let output_map = load_json_map(&cfg.onnx_output_map_json)?;
        Ok(Self {
            cfg,
            model: Some(model),
            output_map,
        })
    }

    fn parsed_input_to_rows(parsed_input: &ParsedInput) -> Result<Vec<Vec<f64>>, String> {
        if let Some(rows) = &parsed_input.x {
            return Ok(rows.clone());
        }
        if let Some(tensors) = &parsed_input.tensors {
            if tensors.is_empty() {
                return Err("Parsed input contained no tensors".to_string());
            }
            if tensors.len() > 1 {
                return Err(format!(
                    "Parsed input contained {} ONNX input tensors, but this adapter \
                     currently supports only a single input tensor",
                    tensors.len()
                ));
            }
            let first = tensors
                .values()
                .next()
                .expect("non-empty map must have a first value");
            return value_to_numeric_rows(first);
        }
        Err("Parsed input contained no features/tensors".to_string())
    }

    fn rows_to_tensor(rows: &[Vec<f64>]) -> Result<tract_onnx::prelude::Tensor, String> {
        if rows.is_empty() {
            return Err("Parsed payload is empty".to_string());
        }
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        if rows.iter().any(|row| row.len() != n_cols) {
            return Err("Input rows have inconsistent feature counts".to_string());
        }
        let flat = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .map(|value| value as f32)
            .collect::<Vec<f32>>();
        let arr = tract_onnx::prelude::tract_ndarray::Array2::<f32>::from_shape_vec(
            (n_rows, n_cols),
            flat,
        )
        .map_err(|err| format!("failed to build input tensor: {err}"))?;
        Ok(arr.into())
    }

    fn array_view_to_json<T, F>(
        view: tract_onnx::prelude::tract_ndarray::ArrayViewD<'_, T>,
        to_value: F,
    ) -> Value
    where
        T: Copy,
        F: Fn(T) -> Value + Copy,
    {
        if view.ndim() == 1 {
            return Value::Array(view.iter().copied().map(to_value).collect::<Vec<Value>>());
        }
        if view.ndim() == 2 {
            let rows = view
                .outer_iter()
                .map(|row| Value::Array(row.iter().copied().map(to_value).collect::<Vec<Value>>()))
                .collect::<Vec<Value>>();
            return Value::Array(rows);
        }
        Value::Array(view.iter().copied().map(to_value).collect::<Vec<Value>>())
    }

    fn tensor_to_json(tensor: &tract_onnx::prelude::Tensor) -> Result<Value, String> {
        if let Ok(view) = tensor.to_array_view::<f32>() {
            return Ok(Self::array_view_to_json(view, |v| Value::from(v as f64)));
        }
        if let Ok(view) = tensor.to_array_view::<i64>() {
            return Ok(Self::array_view_to_json(view, Value::from));
        }
        Err("unsupported ONNX output tensor dtype".to_string())
    }
}

impl BaseAdapter for OnnxAdapter {
    fn is_ready(&self) -> bool {
        self.model.is_some()
    }

    fn predict(&self, parsed_input: &ParsedInput) -> Result<Value, String> {
        let rows = Self::parsed_input_to_rows(parsed_input)?;
        let input = Self::rows_to_tensor(&rows)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| "ONNX model runtime unavailable".to_string())?;
        let outputs = model
            .run(tvec!(input.into()))
            .map_err(|err| format!("ONNX inference failed: {err}"))?;

        if !self.output_map.is_empty() {
            let mut mapped = serde_json::Map::new();
            for (response_key, onnx_output_name) in &self.output_map {
                let index = onnx_output_name
                    .parse::<usize>()
                    .unwrap_or(0)
                    .min(outputs.len().saturating_sub(1));
                mapped.insert(response_key.clone(), Self::tensor_to_json(&outputs[index])?);
            }
            return Ok(Value::Object(mapped));
        }

        if !self.cfg.onnx_output_name.trim().is_empty() {
            let index = self
                .cfg
                .onnx_output_name
                .parse::<usize>()
                .unwrap_or(self.cfg.onnx_output_index)
                .min(outputs.len().saturating_sub(1));
            return Self::tensor_to_json(&outputs[index]);
        }

        let index = self
            .cfg
            .onnx_output_index
            .min(outputs.len().saturating_sub(1));
        Self::tensor_to_json(&outputs[index])
    }
}

fn load_adapter(cfg: &AppConfig) -> Result<Arc<dyn BaseAdapter>, String> {
    let model_exists = cfg.model_path().is_ok_and(|p| p.exists());
    if cfg.model_type == "onnx" || model_exists {
        let adapter = OnnxAdapter::new(cfg.clone())?;
        return Ok(Arc::new(adapter));
    }
    if !cfg.model_type.is_empty() && cfg.model_type != "onnx" {
        return Err(format!(
            "MODEL_TYPE={} is not implemented in this package",
            cfg.model_type
        ));
    }
    Err("Set MODEL_TYPE=onnx or place model.onnx under SM_MODEL_DIR".to_string())
}

#[derive(Clone)]
pub struct AppState {
    cfg: AppConfig,
    adapter: Arc<RwLock<Option<Arc<dyn BaseAdapter>>>>,
    inflight: Arc<Semaphore>,
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

    async fn ensure_adapter_loaded(&self) -> Result<Arc<dyn BaseAdapter>, String> {
        if let Some(existing) = self.adapter.read().await.as_ref() {
            return Ok(existing.clone());
        }
        let loaded = load_adapter(&self.cfg)?;
        let mut writer = self.adapter.write().await;
        *writer = Some(loaded.clone());
        Ok(loaded)
    }

    fn parse_payload(&self, payload: &[u8], content_type: &str) -> Result<ParsedInput, String> {
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

    fn format_output(&self, predictions: Value, accept: &str) -> Result<(Vec<u8>, String), String> {
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

fn validate_input_mode(cfg: &AppConfig) -> Result<(), String> {
    if cfg.input_mode != "tabular" {
        return Err(format!(
            "INPUT_MODE={} not implemented (tabular only for now)",
            cfg.input_mode
        ));
    }
    Ok(())
}

fn resolve_content_type(raw: &str) -> String {
    strip_content_type_params(raw)
}

fn should_use_onnx_multi_input(
    onnx_input_map: &HashMap<String, String>,
    content_type: &str,
) -> bool {
    !onnx_input_map.is_empty() && is_json_content_type(content_type)
}

fn validate_tabular_matrix_shape(matrix: &[Vec<f64>], cfg: &AppConfig) -> Result<(), String> {
    if matrix.is_empty() {
        return Err("Parsed payload is empty".to_string());
    }
    if cfg.tabular_num_features > 0 {
        let got = matrix.first().map_or(0, |r| r.len());
        if got != cfg.tabular_num_features {
            return Err(format!(
                "Feature count mismatch: got {got} expected TABULAR_NUM_FEATURES={}",
                cfg.tabular_num_features
            ));
        }
    }
    Ok(())
}

fn apply_feature_selection(
    matrix: Vec<Vec<f64>>,
    cfg: &AppConfig,
) -> Result<Vec<Vec<f64>>, String> {
    if cfg.tabular_feature_columns.is_empty() && cfg.tabular_id_columns.is_empty() {
        return Ok(matrix);
    }
    let n_cols = matrix.first().map_or(0, |r| r.len());
    let feature_idx = if !cfg.tabular_feature_columns.is_empty() {
        parse_col_selector(&cfg.tabular_feature_columns, n_cols)?
    } else {
        let id_idx = parse_col_selector(&cfg.tabular_id_columns, n_cols)?;
        (0..n_cols)
            .filter(|col| !id_idx.contains(col))
            .collect::<Vec<usize>>()
    };
    Ok(matrix
        .iter()
        .map(|row| {
            feature_idx
                .iter()
                .map(|idx| row[*idx])
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>())
}

fn strip_content_type_params(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or(content_type)
        .trim()
        .to_ascii_lowercase()
}

fn is_json_content_type(content_type: &str) -> bool {
    JSON_CONTENT_TYPES.contains(&content_type) || JSON_LINES_CONTENT_TYPES.contains(&content_type)
}

fn load_multi_input_records(
    payload: &[u8],
    content_type: &str,
    cfg: &AppConfig,
) -> Result<Vec<HashMap<String, Value>>, String> {
    if JSON_CONTENT_TYPES.contains(&content_type) {
        return parse_json_records(payload, cfg);
    }
    parse_jsonl_records(payload)
}

fn validate_dynamic_batch_sizes(batch_sizes: &[usize]) -> Result<(), String> {
    if batch_sizes.is_empty() || batch_sizes.contains(&0) {
        return Err("ONNX_DYNAMIC_BATCH enabled but batch dimension invalid".to_string());
    }
    if batch_sizes.windows(2).any(|w| w[0] != w[1]) {
        return Err(format!(
            "ONNX inputs have mismatched batch sizes: {batch_sizes:?}"
        ));
    }
    Ok(())
}

fn build_onnx_tensors(
    records: &[HashMap<String, Value>],
    input_map: &HashMap<String, String>,
    dynamic_batch: bool,
) -> Result<HashMap<String, Value>, String> {
    let mut tensors = HashMap::new();
    let mut batch_sizes = Vec::new();

    for (request_key, onnx_input_name) in input_map {
        let mut values = Vec::new();
        for record in records {
            let value = record.get(request_key).ok_or_else(|| {
                format!(
                    "Missing key '{}' in one of the records for ONNX multi-input",
                    request_key
                )
            })?;
            values.push(value.clone());
        }
        if dynamic_batch {
            batch_sizes.push(values.len());
        }
        tensors.insert(onnx_input_name.clone(), Value::Array(values));
    }

    if dynamic_batch {
        validate_dynamic_batch_sizes(&batch_sizes)?;
    }

    Ok(tensors)
}

fn load_json_map(raw: &str) -> Result<HashMap<String, String>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(HashMap::new());
    }
    let value: Value = serde_json::from_str(trimmed)
        .map_err(|err| format!("Expected JSON object mapping: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Expected JSON object mapping".to_string())?;
    let mut out = HashMap::new();
    for (key, val) in object {
        let s = val.as_str().ok_or_else(|| {
            format!(
                "Expected string value for key '{}' in JSON object mapping",
                key
            )
        })?;
        out.insert(key.clone(), s.to_string());
    }
    Ok(out)
}

fn parse_json_records(
    payload: &[u8],
    cfg: &AppConfig,
) -> Result<Vec<HashMap<String, Value>>, String> {
    let value: Value =
        serde_json::from_slice(payload).map_err(|err| format!("invalid json payload: {err}"))?;
    let scoped = if value.is_object() {
        if let Some(field) = value.get(&cfg.json_key_instances) {
            field.clone()
        } else {
            value
        }
    } else {
        value
    };
    if let Some(obj) = scoped.as_object() {
        return Ok(vec![obj
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<String, Value>>()]);
    }
    let arr = scoped.as_array().ok_or_else(|| {
        "ONNX multi-input mode expects a JSON object or a non-empty list of objects".to_string()
    })?;
    let mut out = Vec::new();
    for item in arr {
        let map = item.as_object().ok_or_else(|| {
            "ONNX multi-input mode expects each record to be a JSON object".to_string()
        })?;
        out.push(
            map.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, Value>>(),
        );
    }
    Ok(out)
}

fn parse_jsonl_records(payload: &[u8]) -> Result<Vec<HashMap<String, Value>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 payload: {err}"))?;
    let mut out = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("invalid json line payload: {err}"))?;
        let map = value.as_object().ok_or_else(|| {
            "ONNX multi-input mode expects each record to be a JSON object".to_string()
        })?;
        out.push(
            map.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, Value>>(),
        );
    }
    Ok(out)
}

fn parse_json_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let value: Value =
        serde_json::from_slice(payload).map_err(|err| format!("invalid json payload: {err}"))?;
    let scoped = if let Some(instances) = value.get(&cfg.json_key_instances) {
        instances.clone()
    } else if let Some(features) = value.get(&cfg.jsonl_features_key) {
        Value::Array(vec![features.clone()])
    } else {
        value
    };
    value_to_numeric_rows(&scoped)
}

fn parse_jsonl_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 payload: {err}"))?;
    let mut rows = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("invalid json line payload: {err}"))?;
        if let Some(obj) = value.as_object() {
            if let Some(features) = obj.get(&cfg.jsonl_features_key) {
                rows.extend(value_to_numeric_rows(features)?);
                continue;
            }
        }
        rows.extend(value_to_numeric_rows(&value)?);
    }
    Ok(rows)
}

fn value_to_numeric_rows(value: &Value) -> Result<Vec<Vec<f64>>, String> {
    if let Some(arr) = value.as_array() {
        if arr.first().is_some_and(|item| item.is_array()) {
            return arr
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or_else(|| "Expected array row".to_string())?
                        .iter()
                        .map(|item| {
                            item.as_f64()
                                .ok_or_else(|| "Expected numeric value in payload".to_string())
                        })
                        .collect::<Result<Vec<f64>, String>>()
                })
                .collect::<Result<Vec<Vec<f64>>, String>>();
        }
        return Ok(vec![arr
            .iter()
            .map(|item| {
                item.as_f64()
                    .ok_or_else(|| "Expected numeric value in payload".to_string())
            })
            .collect::<Result<Vec<f64>, String>>()?]);
    }
    if let Some(number) = value.as_f64() {
        return Ok(vec![vec![number]]);
    }
    Err("Expected tabular numeric payload".to_string())
}

fn parse_csv_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 csv payload: {err}"))?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !cfg.csv_skip_blank_lines || !line.is_empty())
        .collect::<Vec<&str>>();
    if lines.is_empty() {
        return Err("Empty CSV payload".to_string());
    }
    match cfg.csv_has_header.as_str() {
        "true" => {
            lines.remove(0);
        }
        "auto" => {
            if csv_first_row_is_header(lines[0], cfg.csv_delimiter.as_str()) {
                lines.remove(0);
            }
        }
        "false" => {}
        _ => return Err("CSV_HAS_HEADER must be auto|true|false".to_string()),
    }
    if lines.is_empty() {
        return Err("CSV payload contains only header row".to_string());
    }
    let delim = cfg.csv_delimiter.as_str();
    lines
        .iter()
        .map(|line| {
            line.split(delim)
                .map(|token| {
                    token
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| "Expected numeric value in CSV payload".to_string())
                })
                .collect::<Result<Vec<f64>, String>>()
        })
        .collect::<Result<Vec<Vec<f64>>, String>>()
}

fn csv_first_row_is_header(line: &str, delim: &str) -> bool {
    line.split(delim)
        .any(|token| token.trim().parse::<f64>().is_err())
}

fn parse_col_selector(selector: &str, n_cols: usize) -> Result<Vec<usize>, String> {
    let trimmed = selector.trim();
    if trimmed.is_empty() {
        return Ok((0..n_cols).collect::<Vec<usize>>());
    }
    if let Some((start_raw, end_raw)) = trimmed.split_once(':') {
        let start = if start_raw.is_empty() {
            0
        } else {
            start_raw
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())?
        };
        let end = if end_raw.is_empty() {
            n_cols
        } else {
            end_raw
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())?
        };
        let bounded_end = end.min(n_cols);
        return Ok((start.min(bounded_end)..bounded_end).collect::<Vec<usize>>());
    }
    trimmed
        .split(',')
        .filter(|tok| !tok.trim().is_empty())
        .map(|tok| {
            tok.trim()
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())
        })
        .collect::<Result<Vec<usize>, String>>()
}

fn normalized_accept(accept: &str, default_accept: &str) -> String {
    // skipcq: RS-W1031. Clippy enforces unwrap_or here because the fallback is a cheap borrowed &str.
    accept
        .split(',')
        .next()
        .unwrap_or(default_accept)
        .trim()
        .to_ascii_lowercase()
}

fn wrap_predictions_if_needed(predictions: Value, cfg: &AppConfig) -> Value {
    if cfg.predictions_only {
        return predictions;
    }
    let mut output: serde_json::Map<String, Value> = serde_json::Map::default();
    output.insert(cfg.json_output_key.clone(), predictions);
    Value::Object(output)
}

fn validate_payload_size(payload_len: usize, cfg: &AppConfig) -> Option<AxumResponse> {
    if payload_len <= cfg.max_body_bytes {
        return None;
    }
    Some(
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": "payload_too_large",
                "max_bytes": cfg.max_body_bytes
            })),
        )
            .into_response(),
    )
}

fn resolve_request_media_types(headers: &HeaderMap, cfg: &AppConfig) -> (String, String) {
    (
        header_value_with_fallback(
            headers,
            CONTENT_TYPE,
            SAGEMAKER_CONTENT_TYPE_HEADER,
            cfg.default_content_type.as_str(),
        ),
        header_value_with_fallback(
            headers,
            ACCEPT,
            SAGEMAKER_ACCEPT_HEADER,
            cfg.default_accept.as_str(),
        ),
    )
}

async fn acquire_inflight_permit(
    state: &Arc<AppState>,
) -> Result<OwnedSemaphorePermit, AxumResponse> {
    match timeout(
        Duration::from_secs_f64(state.cfg.acquire_timeout_s.max(0.0)),
        state.inflight.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => Ok(permit),
        _ => Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "too_many_requests"})),
        )
            .into_response()),
    }
}

async fn parse_and_validate_request(
    state: &Arc<AppState>,
    payload: &[u8],
    content_type: &str,
) -> Result<(ParsedInput, usize), AxumResponse> {
    let parsed = state
        .parse_payload(payload, content_type)
        .map_err(|err| (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response())?;
    let batch = parsed
        .batch_size()
        .map_err(|err| (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response())?;
    if batch > state.cfg.max_records {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("too_many_records: {batch} > {}", state.cfg.max_records) })),
        )
            .into_response());
    }
    Ok((parsed, batch))
}

async fn predict_and_format(
    state: &Arc<AppState>,
    parsed: &ParsedInput,
    accept: &str,
) -> Result<(Vec<u8>, String), AxumResponse> {
    let adapter = state.ensure_adapter_loaded().await.map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": err })),
        )
            .into_response()
    })?;
    let predictions = adapter
        .predict(parsed)
        .map_err(|err| (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response())?;
    state
        .format_output(predictions, accept)
        .map_err(|err| (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response())
}

fn build_success_response(body: Vec<u8>, content_type: String) -> AxumResponse {
    let mut response = (StatusCode::OK, body).into_response();
    if let Ok(header) = content_type.parse() {
        response.headers_mut().insert(CONTENT_TYPE, header);
    }
    attach_trace_correlation_headers(&mut response);
    response
}

fn read_openapi_from_env() -> Option<String> {
    let path = std::env::var(OPENAPI_SPEC_PATH_ENV_KEY).ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    fs::read_to_string(trimmed).ok()
}

fn openapi_candidate_paths() -> [PathBuf; 2] {
    [
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("docs")
            .join("openapi.yaml"),
        PathBuf::from("docs").join("openapi.yaml"),
    ]
}

fn format_csv_predictions(predictions: &Value, delimiter: &str) -> Result<String, String> {
    if let Some(rows) = predictions.as_array() {
        if rows.first().is_some_and(|item| item.is_array()) {
            let mut out = Vec::new();
            for row in rows {
                let cols = row
                    .as_array()
                    .ok_or_else(|| "expected csv row array".to_string())?
                    .iter()
                    .map(value_to_string)
                    .collect::<Vec<String>>();
                out.push(cols.join(delimiter));
            }
            return Ok(out.join("\n"));
        }
        let lines = rows.iter().map(value_to_string).collect::<Vec<String>>();
        return Ok(lines.join("\n"));
    }
    Ok(value_to_string(predictions))
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::default(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

pub fn build_http_router(cfg: AppConfig) -> Router {
    let metrics_path = cfg.prometheus_path.clone();
    let prometheus_enabled = cfg.prometheus_enabled;
    let swagger_enabled = cfg.swagger_enabled;
    let state = Arc::new(AppState::new(cfg));
    let mut router = Router::new()
        .route("/live", get(http_live))
        .route("/healthz", get(http_live))
        .route("/ready", get(http_ready))
        .route("/readyz", get(http_ready))
        .route("/ping", get(http_ready))
        .route("/invocations", post(http_invocations));
    if swagger_enabled {
        router = router
            .route("/openapi.yaml", get(http_openapi_spec))
            .route("/docs", get(http_swagger_ui));
    }
    if prometheus_enabled {
        router = router.route(metrics_path.as_str(), get(http_metrics));
    }
    router.with_state(state)
}

async fn http_live() -> impl IntoResponse {
    (StatusCode::OK, "\n")
}

async fn http_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.ensure_adapter_loaded().await {
        Ok(adapter) if adapter.is_ready() => (StatusCode::OK, "\n").into_response(),
        Ok(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
    }
}

async fn http_metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        "# HELP byoc_up Service readiness\n# TYPE byoc_up gauge\nbyoc_up 1\n",
    )
        .into_response()
}

async fn http_openapi_spec() -> impl IntoResponse {
    if let Some(spec) = load_openapi_yaml() {
        return ([(CONTENT_TYPE, "application/yaml; charset=utf-8")], spec).into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": "openapi_contract_not_found"})),
    )
        .into_response()
}

async fn http_swagger_ui() -> impl IntoResponse {
    Html(SWAGGER_UI_HTML)
}

async fn http_invocations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Bytes,
) -> AxumResponse {
    if let Some(response) = validate_payload_size(payload.len(), &state.cfg) {
        return response;
    }

    let (content_type, accept) = resolve_request_media_types(&headers, &state.cfg);
    let _permit = match acquire_inflight_permit(&state).await {
        Ok(permit) => permit,
        Err(response) => return response,
    };
    let (parsed, _) =
        match parse_and_validate_request(&state, payload.as_ref(), &content_type).await {
            Ok(value) => value,
            Err(response) => return response,
        };
    let (body, output_content_type) = match predict_and_format(&state, &parsed, &accept).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    build_success_response(body, output_content_type)
}

fn load_openapi_yaml() -> Option<String> {
    if let Some(spec) = read_openapi_from_env() {
        return Some(spec);
    }

    for path in openapi_candidate_paths() {
        if let Ok(spec) = fs::read_to_string(&path) {
            return Some(spec);
        }
    }
    None
}

fn header_value_with_fallback(
    headers: &HeaderMap,
    primary: HeaderName,
    fallback: &str,
    default: &str,
) -> String {
    if let Some(value) = headers.get(primary).and_then(|h| h.to_str().ok()) {
        return value.to_string();
    }
    if let Ok(fallback_name) = HeaderName::from_lowercase(fallback.as_bytes()) {
        if let Some(value) = headers.get(fallback_name).and_then(|h| h.to_str().ok()) {
            return value.to_string();
        }
    }
    default.to_string()
}

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

#[derive(Clone)]
pub struct InferenceGrpcService {
    state: AppState,
    load_error: Option<String>,
}

impl InferenceGrpcService {
    fn new(cfg: AppConfig) -> Self {
        let state = AppState::new(cfg.clone());
        match load_adapter(&cfg) {
            Ok(_) => {
                // Pre-populate the adapter in the state synchronously
                // We can't use async here, but ensure_adapter_loaded will populate it on first use
                // The ready check and predict will ensure it's loaded before use
                Self {
                    state,
                    load_error: None,
                }
            }
            Err(err) => Self {
                state,
                load_error: Some(err),
            },
        }
    }
}

#[tonic::async_trait]
impl grpc::inference_service_server::InferenceService for InferenceGrpcService {
    async fn live(
        &self,
        _request: Request<grpc::LiveRequest>,
    ) -> Result<Response<grpc::StatusReply>, Status> {
        Ok(Response::new(grpc::StatusReply {
            ok: true,
            status: "live".to_string(),
        }))
    }

    async fn ready(
        &self,
        _request: Request<grpc::ReadyRequest>,
    ) -> Result<Response<grpc::StatusReply>, Status> {
        let ready = match self.state.ensure_adapter_loaded().await {
            Ok(_) => self
                .state
                .adapter
                .read()
                .await
                .as_ref()
                .is_some_and(|adapter| adapter.is_ready()),
            Err(_) => false,
        };
        Ok(Response::new(grpc::StatusReply {
            ok: ready,
            status: if ready {
                "ready".to_string()
            } else {
                "not_ready".to_string()
            },
        }))
    }

    async fn predict(
        &self,
        request: Request<grpc::PredictRequest>,
    ) -> Result<Response<grpc::PredictReply>, Status> {
        if let Some(err) = &self.load_error {
            return Err(Status::new(Code::Internal, err.clone()));
        }
        let req = request.into_inner();

        // Enforce max_body_bytes to maintain HTTP/gRPC parity
        if req.payload.len() > self.state.cfg.max_body_bytes {
            return Err(Status::new(
                Code::InvalidArgument,
                format!(
                    "payload too large: {} bytes > {} bytes limit",
                    req.payload.len(),
                    self.state.cfg.max_body_bytes
                ),
            ));
        }

        // Apply max_inflight semaphore for HTTP/gRPC parity
        let _permit = match timeout(
            Duration::from_secs_f64(self.state.cfg.acquire_timeout_s.max(0.0)),
            self.state.inflight.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            _ => {
                return Err(Status::new(Code::ResourceExhausted, "too_many_requests"));
            }
        };

        let content_type = if req.content_type.is_empty() {
            self.state.cfg.default_content_type.as_str()
        } else {
            req.content_type.as_str()
        };
        let accept = if req.accept.is_empty() {
            self.state.cfg.default_accept.as_str()
        } else {
            req.accept.as_str()
        };

        let adapter = self
            .state
            .ensure_adapter_loaded()
            .await
            .map_err(|err| Status::new(Code::Internal, err))?;
        let parsed = self
            .state
            .parse_payload(req.payload.as_ref(), content_type)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let batch = parsed
            .batch_size()
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        if batch > self.state.cfg.max_records {
            return Err(Status::new(
                Code::InvalidArgument,
                format!("too_many_records: {batch} > {}", self.state.cfg.max_records),
            ));
        }
        let predictions = adapter
            .predict(&parsed)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let (body, output_content_type) = self
            .state
            .format_output(predictions, accept)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), batch.to_string());
        Ok(Response::new(grpc::PredictReply {
            body,
            content_type: output_content_type,
            metadata,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::inference_service_server::InferenceService;
    use axum::body::to_bytes;
    use axum::body::Bytes;
    use axum::http::HeaderValue;
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
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/csv"));
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
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/csv"));
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
            HeaderValue::from_static("application/json"),
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
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
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
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            HeaderName::from_static(SAGEMAKER_ACCEPT_HEADER),
            HeaderValue::from_static("text/csv"),
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
            .live(Request::new(grpc::LiveRequest {}))
            .await
            .expect("live must succeed")
            .into_inner();
        assert!(live.ok);

        let ready = service
            .ready(Request::new(grpc::ReadyRequest {}))
            .await
            .expect("ready must respond")
            .into_inner();
        assert!(!ready.ok);
        assert_eq!(ready.status, "not_ready");

        let predict = service
            .predict(Request::new(grpc::PredictRequest {
                payload: b"1,2\n".to_vec(),
                content_type: "text/csv".to_string(),
                accept: "application/json".to_string(),
            }))
            .await
            .expect_err("predict should fail when model failed loading");
        assert_eq!(predict.code(), Code::Internal);
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
            .predict(Request::new(grpc::PredictRequest {
                payload: br#"{"instances":[[1.0,2.0]]}"#.to_vec(),
                content_type: "application/json".to_string(),
                accept: "application/json".to_string(),
            }))
            .await
            .expect_err("must fail when semaphore has no permits");
        assert_eq!(result.code(), Code::ResourceExhausted);
        assert!(result.message().contains("too_many_requests"));
    }

    #[tokio::test]
    async fn grpc_predict_success_populates_metadata_and_json_content_type() {
        let (_tmp, cfg) = cfg_with_temp_model_fixture();
        let service = InferenceGrpcService::new(cfg);
        let reply = service
            .predict(Request::new(grpc::PredictRequest {
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
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

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

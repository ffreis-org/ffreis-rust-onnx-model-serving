use std::collections::HashMap;

use serde_json::Value;

use crate::config::AppConfig;

pub(crate) const JSON_CONTENT_TYPES: &[&str] = &["application/json", "application/*+json"];
pub(crate) const JSON_LINES_CONTENT_TYPES: &[&str] = &[
    "application/jsonlines",
    "application/x-jsonlines",
    "application/jsonl",
    "application/x-ndjson",
];
pub(crate) const CSV_CONTENT_TYPES: &[&str] = &["text/csv", "application/csv"];

#[derive(Clone, Debug)]
pub struct ParsedInput {
    pub x: Option<Vec<Vec<f64>>>,
    pub tensors: Option<HashMap<String, Value>>,
    pub meta: Option<Value>,
}

impl ParsedInput {
    pub(crate) fn batch_size(&self) -> Result<usize, String> {
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

pub(crate) fn validate_input_mode(cfg: &AppConfig) -> Result<(), String> {
    if cfg.input_mode != "tabular" {
        return Err(format!(
            "INPUT_MODE={} not implemented (tabular only for now)",
            cfg.input_mode
        ));
    }
    Ok(())
}

pub(crate) fn resolve_content_type(raw: &str) -> String {
    strip_content_type_params(raw)
}

pub(crate) fn should_use_onnx_multi_input(
    onnx_input_map: &HashMap<String, String>,
    content_type: &str,
) -> bool {
    !onnx_input_map.is_empty() && is_json_content_type(content_type)
}

pub(crate) fn validate_tabular_matrix_shape(
    matrix: &[Vec<f64>],
    cfg: &AppConfig,
) -> Result<(), String> {
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

pub(crate) fn apply_feature_selection(
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

pub(crate) fn strip_content_type_params(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or(content_type)
        .trim()
        .to_ascii_lowercase()
}

pub(crate) fn is_json_content_type(content_type: &str) -> bool {
    JSON_CONTENT_TYPES.contains(&content_type) || JSON_LINES_CONTENT_TYPES.contains(&content_type)
}

pub(crate) fn load_multi_input_records(
    payload: &[u8],
    content_type: &str,
    cfg: &AppConfig,
) -> Result<Vec<HashMap<String, Value>>, String> {
    if JSON_CONTENT_TYPES.contains(&content_type) {
        return parse_json_records(payload, cfg);
    }
    parse_jsonl_records(payload)
}

pub(crate) fn validate_dynamic_batch_sizes(batch_sizes: &[usize]) -> Result<(), String> {
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

pub(crate) fn load_json_map(raw: &str) -> Result<HashMap<String, String>, String> {
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

pub(crate) fn parse_json_records(
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

pub(crate) fn parse_jsonl_records(payload: &[u8]) -> Result<Vec<HashMap<String, Value>>, String> {
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

pub(crate) fn parse_json_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
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

pub(crate) fn parse_jsonl_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
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

pub(crate) fn value_to_numeric_rows(value: &Value) -> Result<Vec<Vec<f64>>, String> {
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

pub(crate) fn parse_csv_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
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

pub(crate) fn csv_first_row_is_header(line: &str, delim: &str) -> bool {
    line.split(delim)
        .any(|token| token.trim().parse::<f64>().is_err())
}

pub(crate) fn parse_col_selector(selector: &str, n_cols: usize) -> Result<Vec<usize>, String> {
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

pub(crate) fn normalized_accept(accept: &str, default_accept: &str) -> String {
    // skipcq: RS-W1031. Clippy enforces unwrap_or here because the fallback is a cheap borrowed &str.
    accept
        .split(',')
        .next()
        .unwrap_or(default_accept)
        .trim()
        .to_ascii_lowercase()
}

pub(crate) fn wrap_predictions_if_needed(predictions: Value, cfg: &AppConfig) -> Value {
    if cfg.predictions_only {
        return predictions;
    }
    let mut output: serde_json::Map<String, Value> = serde_json::Map::default();
    output.insert(cfg.json_output_key.clone(), predictions);
    Value::Object(output)
}

pub(crate) fn format_csv_predictions(
    predictions: &Value,
    delimiter: &str,
) -> Result<String, String> {
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

pub(crate) fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::default(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

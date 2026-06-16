use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tract_onnx::prelude::{tvec, Framework, InferenceModelExt, RunnableModel, TypedFact, TypedOp};

use crate::config::AppConfig;
use crate::input::{load_json_map, validate_dynamic_batch_sizes, value_to_numeric_rows};

pub trait BaseAdapter: Send + Sync {
    fn is_ready(&self) -> bool;
    fn predict(&self, parsed_input: &crate::input::ParsedInput) -> Result<Value, String>;
}

pub(crate) type OnnxRunnableModel = RunnableModel<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

#[derive(Clone)]
pub(crate) struct OnnxAdapter {
    cfg: AppConfig,
    model: Option<OnnxRunnableModel>,
    output_map: HashMap<String, String>,
}

impl OnnxAdapter {
    pub(crate) fn new(cfg: AppConfig) -> Result<Self, String> {
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

    fn parsed_input_to_rows(
        parsed_input: &crate::input::ParsedInput,
    ) -> Result<Vec<Vec<f64>>, String> {
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

    pub(crate) fn tensor_to_json(tensor: &tract_onnx::prelude::Tensor) -> Result<Value, String> {
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

    fn predict(&self, parsed_input: &crate::input::ParsedInput) -> Result<Value, String> {
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

pub(crate) fn load_adapter(cfg: &AppConfig) -> Result<Arc<dyn BaseAdapter>, String> {
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

pub(crate) fn build_onnx_tensors(
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

use std::collections::HashMap;
use std::time::Duration;

use tokio::time::timeout;
use tonic::{Code, Request, Response, Status};

use crate::adapter::load_adapter;
use crate::config::AppConfig;
use crate::AppState;

#[derive(Clone)]
pub struct InferenceGrpcService {
    pub(crate) state: AppState,
    pub(crate) load_error: Option<String>,
}

impl InferenceGrpcService {
    pub(crate) fn new(cfg: AppConfig) -> Self {
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
impl crate::grpc::inference_service_server::InferenceService for InferenceGrpcService {
    async fn live(
        &self,
        _request: Request<crate::grpc::LiveRequest>,
    ) -> Result<Response<crate::grpc::StatusReply>, Status> {
        Ok(Response::new(crate::grpc::StatusReply {
            ok: true,
            status: "live".to_string(),
        }))
    }

    async fn ready(
        &self,
        _request: Request<crate::grpc::ReadyRequest>,
    ) -> Result<Response<crate::grpc::StatusReply>, Status> {
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
        Ok(Response::new(crate::grpc::StatusReply {
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
        request: Request<crate::grpc::PredictRequest>,
    ) -> Result<Response<crate::grpc::PredictReply>, Status> {
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
        Ok(Response::new(crate::grpc::PredictReply {
            body,
            content_type: output_content_type,
            metadata,
        }))
    }
}

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::response::{Html, IntoResponse, Response as AxumResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::OwnedSemaphorePermit;
use tokio::time::timeout;

use crate::input::ParsedInput;
use crate::openapi::load_openapi_yaml;
use crate::AppState;

pub(crate) const SAGEMAKER_CONTENT_TYPE_HEADER: &str = "x-amzn-sagemaker-content-type";
pub(crate) const SAGEMAKER_ACCEPT_HEADER: &str = "x-amzn-sagemaker-accept";
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

pub fn build_http_router(cfg: crate::config::AppConfig) -> Router {
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

pub(crate) fn resolve_request_media_types(
    headers: &HeaderMap,
    cfg: &crate::config::AppConfig,
) -> (String, String) {
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

pub(crate) fn validate_payload_size(
    payload_len: usize,
    cfg: &crate::config::AppConfig,
) -> Option<AxumResponse> {
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

pub(crate) fn build_success_response(body: Vec<u8>, content_type: String) -> AxumResponse {
    let mut response = (StatusCode::OK, body).into_response();
    if let Ok(header) = content_type.parse() {
        response.headers_mut().insert(CONTENT_TYPE, header);
    }
    crate::telemetry::attach_trace_correlation_headers(&mut response);
    response
}

pub(crate) fn header_value_with_fallback(
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

pub(crate) async fn http_live() -> impl IntoResponse {
    (StatusCode::OK, "\n")
}

async fn http_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.ensure_adapter_loaded().await {
        Ok(adapter) if adapter.is_ready() => (StatusCode::OK, "\n").into_response(),
        Ok(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
    }
}

pub(crate) async fn http_metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        "# HELP byoc_up Service readiness\n# TYPE byoc_up gauge\nbyoc_up 1\n",
    )
        .into_response()
}

pub(crate) async fn http_openapi_spec() -> impl IntoResponse {
    if let Some(spec) = load_openapi_yaml() {
        return ([(CONTENT_TYPE, "application/yaml; charset=utf-8")], spec).into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": "openapi_contract_not_found"})),
    )
        .into_response()
}

pub(crate) async fn http_swagger_ui() -> impl IntoResponse {
    Html(SWAGGER_UI_HTML)
}

pub(crate) async fn http_invocations(
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

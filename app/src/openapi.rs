use std::fs;
use std::path::PathBuf;

pub(crate) const OPENAPI_SPEC_PATH_ENV_KEY: &str = "OPENAPI_SPEC_PATH";

pub(crate) fn read_openapi_from_env() -> Option<String> {
    let path = std::env::var(OPENAPI_SPEC_PATH_ENV_KEY).ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    fs::read_to_string(trimmed).ok()
}

pub(crate) fn openapi_candidate_paths() -> [PathBuf; 2] {
    [
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("docs")
            .join("openapi.yaml"),
        PathBuf::from("docs").join("openapi.yaml"),
    ]
}

pub(crate) fn load_openapi_yaml() -> Option<String> {
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

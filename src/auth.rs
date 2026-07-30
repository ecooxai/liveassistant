use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

/// Codex CLI OAuth client id (public; same as the official CLI).
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const OAUTH_REFRESH_URL: &str = "https://auth.openai.com/oauth/token";
/// Refresh a few minutes before expiry so connect never races the deadline.
const REFRESH_SKEW_SECS: u64 = 120;

/// Credentials borrowed from a Codex login for the Realtime WebSocket.
#[derive(Clone, Debug)]
pub struct CodexCredentials {
    /// Bearer token: either a Platform `sk-…` key or a ChatGPT OAuth access token.
    pub bearer_token: String,
    /// Present for ChatGPT/Codex OAuth; sent as `ChatGPT-Account-Id`.
    pub chatgpt_account_id: Option<String>,
}

/// Reuses whatever login Codex has on disk.
///
/// - Platform API-key login → returns the `sk-…` key
/// - ChatGPT OAuth login → returns a (refreshed if needed) access token plus account id
///
/// OpenAI Realtime accepts both; ChatGPT OAuth access tokens are valid Bearer
/// credentials for `wss://api.openai.com/v1/realtime`.
pub fn codex_credentials() -> Result<CodexCredentials> {
    let path = codex_auth_path()?;
    let raw =
        fs::read_to_string(&path).with_context(|| format!("Could not read {}", path.display()))?;
    let mut value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("Could not parse {}", path.display()))?;

    if let Some(key) = platform_api_key(&value) {
        return Ok(CodexCredentials {
            bearer_token: key,
            chatgpt_account_id: None,
        });
    }

    let tokens = value
        .get_mut("tokens")
        .filter(|t| t.is_object())
        .with_context(|| {
            format!(
                "No usable credentials in {}. Run `codex login` (ChatGPT or API key).",
                path.display()
            )
        })?;

    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .context("Codex auth.json has no access_token. Run `codex login`.")?;

    let account_id = tokens
        .get("account_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| chatgpt_account_id_from_jwt(&access_token));

    if access_token_is_fresh(&access_token) {
        return Ok(CodexCredentials {
            bearer_token: access_token,
            chatgpt_account_id: account_id,
        });
    }

    let refresh_token = tokens
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .context(
            "Codex OAuth access token is expired and no refresh_token is available. \
             Run `codex login` again.",
        )?;

    let refreshed = refresh_oauth_token(&refresh_token)
        .context("Could not refresh the Codex OAuth access token")?;

    let new_access = refreshed
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .context("Token refresh response did not include access_token")?;

    if let Some(id) = refreshed.get("id_token").and_then(Value::as_str) {
        tokens["id_token"] = Value::String(id.to_owned());
    }
    if let Some(rt) = refreshed.get("refresh_token").and_then(Value::as_str) {
        tokens["refresh_token"] = Value::String(rt.to_owned());
    }
    tokens["access_token"] = Value::String(new_access.clone());

    let account_id = tokens
        .get("account_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| chatgpt_account_id_from_jwt(&new_access))
        .or(account_id);

    if let Some(id) = &account_id {
        tokens["account_id"] = Value::String(id.clone());
    }

    value["last_refresh"] = Value::String(iso8601_now());
    write_auth_json(&path, &value)?;

    Ok(CodexCredentials {
        bearer_token: new_access,
        chatgpt_account_id: account_id,
    })
}

fn platform_api_key(value: &Value) -> Option<String> {
    for pointer in ["/OPENAI_API_KEY", "/api_key", "/tokens/OPENAI_API_KEY"] {
        if let Some(key) = value.pointer(pointer).and_then(Value::as_str)
            && key.starts_with("sk-")
            && key.len() > 20
        {
            return Some(key.to_owned());
        }
    }
    None
}

fn access_token_is_fresh(token: &str) -> bool {
    match jwt_exp_unix(token) {
        Some(exp) => {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            now + REFRESH_SKEW_SECS < exp
        }
        // If we cannot parse expiry, try the token as-is; Realtime will reject if bad.
        None => true,
    }
}

fn jwt_exp_unix(token: &str) -> Option<u64> {
    let payload = jwt_payload(token)?;
    payload.get("exp")?.as_u64()
}

fn chatgpt_account_id_from_jwt(token: &str) -> Option<String> {
    let payload = jwt_payload(token)?;
    payload
        .get("https://api.openai.com/auth")
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn jwt_payload(token: &str) -> Option<Value> {
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload_b64 = parts.next()?.trim_end_matches('=');
    let decoded = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    serde_json::from_slice(&decoded).ok()
}

fn refresh_oauth_token(refresh_token: &str) -> Result<Value> {
    let body = json!({
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    });

    let response = ureq::post(OAUTH_REFRESH_URL)
        .set("Content-Type", "application/json")
        .send_json(body);

    match response {
        Ok(resp) => resp
            .into_json::<Value>()
            .context("Invalid JSON from OAuth token refresh"),
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            let error_code = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_owned));
            match error_code.as_deref() {
                Some(
                    kind @ ("refresh_token_expired"
                    | "refresh_token_reused"
                    | "refresh_token_invalidated"),
                ) => {
                    bail!(
                        "Codex refresh token is no longer valid ({kind}). Run `codex login` again."
                    )
                }
                _ => bail!("Codex OAuth refresh failed (HTTP {code}): {text}"),
            }
        }
        Err(error) => Err(error).context("Network error while refreshing Codex OAuth token"),
    }
}

fn write_auth_json(path: &Path, value: &Value) -> Result<()> {
    let raw = serde_json::to_string_pretty(value).context("Could not serialize auth.json")?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, format!("{raw}\n"))
        .with_context(|| format!("Could not write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("Could not replace {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn iso8601_now() -> String {
    // Avoid a chrono dependency; Codex only needs a readable last_refresh stamp.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

fn codex_auth_path() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(dir).join("auth.json"));
    }
    Ok(dirs::home_dir()
        .context("Could not find your home directory")?
        .join(".codex")
        .join("auth.json"))
}

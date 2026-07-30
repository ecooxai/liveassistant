use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

#[derive(Clone, Debug)]
pub struct CodexModel {
    pub name: String,
    pub display_name: String,
    pub hidden: bool,
}

#[derive(Clone, Debug)]
pub struct RateLimitWindow {
    pub used_percent: i64,
    pub window_duration_minutes: Option<i64>,
    pub resets_at: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct RateLimit {
    pub name: String,
    pub plan: Option<String>,
    pub primary: Option<RateLimitWindow>,
    pub secondary: Option<RateLimitWindow>,
    pub credit_balance: Option<String>,
    pub unlimited_credits: bool,
    pub reached_reason: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct TokenUsage {
    pub lifetime_tokens: Option<i64>,
    pub peak_daily_tokens: Option<i64>,
    pub latest_day: Option<String>,
    pub latest_day_tokens: Option<i64>,
    pub recent_reported_tokens: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct RealtimeVoices {
    pub v1: Vec<String>,
    pub v2: Vec<String>,
    pub default_v1: Option<String>,
    pub default_v2: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CodexAccountInfo {
    pub models: Vec<CodexModel>,
    pub realtime_voices: RealtimeVoices,
    pub rate_limits: Vec<RateLimit>,
    pub token_usage: TokenUsage,
    pub reset_credits: Option<i64>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CodexUsageInfo {
    pub rate_limits: Vec<RateLimit>,
    pub token_usage: TokenUsage,
    pub reset_credits: Option<i64>,
    pub warnings: Vec<String>,
}

/// Reads only the live usage/rate-limit snapshot for the top bar.
pub fn load_usage() -> Result<CodexUsageInfo> {
    let mut client = AppServerClient::start()?;
    let mut warnings = Vec::new();
    let (rate_limits, reset_credits) = match client.call("account/rateLimits/read", json!({})) {
        Ok(value) => parse_rate_limits(&value),
        Err(error) => {
            warnings.push(format!("Rate limits unavailable: {error:#}"));
            (Vec::new(), None)
        }
    };
    let token_usage = match client.call("account/usage/read", json!({})) {
        Ok(value) => parse_token_usage(&value),
        Err(error) => {
            warnings.push(format!("Token usage unavailable: {error:#}"));
            TokenUsage::default()
        }
    };
    Ok(CodexUsageInfo {
        rate_limits,
        token_usage,
        reset_credits,
        warnings,
    })
}

/// Reads the live catalog and account usage exposed by the installed Codex CLI.
///
/// Codex app-server owns the authenticated account APIs and reuses the same
/// `CODEX_HOME/auth.json` (normally `~/.codex/auth.json`) as the CLI.
pub fn load() -> Result<CodexAccountInfo> {
    let mut client = AppServerClient::start()?;
    let models = load_all_models(&mut client)?;
    let mut warnings = Vec::new();

    let realtime_voices = match client.call("thread/realtime/listVoices", json!({})) {
        Ok(value) => parse_realtime_voices(&value),
        Err(error) => {
            warnings.push(format!("Codex voice catalog unavailable: {error:#}"));
            RealtimeVoices::default()
        }
    };

    let (rate_limits, reset_credits) = match client.call("account/rateLimits/read", json!({})) {
        Ok(value) => parse_rate_limits(&value),
        Err(error) => {
            warnings.push(format!("Rate limits unavailable: {error:#}"));
            (Vec::new(), None)
        }
    };

    let token_usage = match client.call("account/usage/read", json!({})) {
        Ok(value) => parse_token_usage(&value),
        Err(error) => {
            warnings.push(format!("Token usage unavailable: {error:#}"));
            TokenUsage::default()
        }
    };

    Ok(CodexAccountInfo {
        models,
        realtime_voices,
        rate_limits,
        token_usage,
        reset_credits,
        warnings,
    })
}

fn parse_realtime_voices(value: &Value) -> RealtimeVoices {
    let voices = value.get("voices").unwrap_or(value);
    let parse_list = |key: &str| {
        voices
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    };
    RealtimeVoices {
        v1: parse_list("v1"),
        v2: parse_list("v2"),
        default_v1: voices
            .get("defaultV1")
            .and_then(Value::as_str)
            .map(str::to_owned),
        default_v2: voices
            .get("defaultV2")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

fn load_all_models(client: &mut AppServerClient) -> Result<Vec<CodexModel>> {
    let mut cursor: Option<String> = None;
    let mut models = Vec::new();
    let mut seen = HashSet::new();

    loop {
        let value = client.call(
            "model/list",
            json!({
                "cursor": cursor,
                "includeHidden": true,
                "limit": 100,
            }),
        )?;

        if let Some(data) = value.get("data").and_then(Value::as_array) {
            for model in data {
                let Some(name) = model.get("model").and_then(Value::as_str) else {
                    continue;
                };
                if !seen.insert(name.to_owned()) {
                    continue;
                }
                models.push(CodexModel {
                    name: name.to_owned(),
                    display_name: model
                        .get("displayName")
                        .and_then(Value::as_str)
                        .unwrap_or(name)
                        .to_owned(),
                    hidden: model
                        .get("hidden")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            }
        }

        cursor = value
            .get("nextCursor")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }

    models.sort_by_key(|model| model.display_name.to_lowercase());
    Ok(models)
}

fn parse_rate_limits(value: &Value) -> (Vec<RateLimit>, Option<i64>) {
    let mut limits = Vec::new();
    if let Some(by_id) = value
        .get("rateLimitsByLimitId")
        .and_then(Value::as_object)
        .filter(|limits| !limits.is_empty())
    {
        for (id, snapshot) in by_id {
            limits.push(parse_rate_limit(id, snapshot));
        }
    } else if let Some(snapshot) = value.get("rateLimits") {
        limits.push(parse_rate_limit("codex", snapshot));
    }
    limits.sort_by_key(|limit| limit.name.to_lowercase());

    let reset_credits = value
        .pointer("/rateLimitResetCredits/availableCount")
        .and_then(Value::as_i64);
    (limits, reset_credits)
}

fn parse_rate_limit(fallback_name: &str, value: &Value) -> RateLimit {
    let name = value
        .get("limitName")
        .and_then(Value::as_str)
        .or_else(|| value.get("limitId").and_then(Value::as_str))
        .unwrap_or(fallback_name)
        .to_owned();
    let credits = value.get("credits");
    RateLimit {
        name,
        plan: value
            .get("planType")
            .and_then(Value::as_str)
            .map(str::to_owned),
        primary: value.get("primary").and_then(parse_rate_limit_window),
        secondary: value.get("secondary").and_then(parse_rate_limit_window),
        credit_balance: credits
            .and_then(|credits| credits.get("balance"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        unlimited_credits: credits
            .and_then(|credits| credits.get("unlimited"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        reached_reason: value
            .get("rateLimitReachedType")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

fn parse_rate_limit_window(value: &Value) -> Option<RateLimitWindow> {
    Some(RateLimitWindow {
        used_percent: value.get("usedPercent")?.as_i64()?,
        window_duration_minutes: value.get("windowDurationMins").and_then(Value::as_i64),
        resets_at: value.get("resetsAt").and_then(Value::as_i64),
    })
}

fn parse_token_usage(value: &Value) -> TokenUsage {
    let summary = value.get("summary");
    let buckets = value
        .get("dailyUsageBuckets")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let latest = buckets.last();
    let recent: Vec<&Value> = buckets.iter().rev().take(7).collect();

    TokenUsage {
        lifetime_tokens: summary
            .and_then(|summary| summary.get("lifetimeTokens"))
            .and_then(Value::as_i64),
        peak_daily_tokens: summary
            .and_then(|summary| summary.get("peakDailyTokens"))
            .and_then(Value::as_i64),
        latest_day: latest
            .and_then(|bucket| bucket.get("startDate"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        latest_day_tokens: latest
            .and_then(|bucket| bucket.get("tokens"))
            .and_then(Value::as_i64),
        recent_reported_tokens: (!recent.is_empty()).then(|| {
            recent
                .iter()
                .filter_map(|bucket| bucket.get("tokens").and_then(Value::as_i64))
                .sum()
        }),
    }
}

struct AppServerClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl AppServerClient {
    fn start() -> Result<Self> {
        let executable = std::env::var_os("CODEX_BIN").unwrap_or_else(|| "codex".into());
        let mut child = Command::new(&executable)
            .args(["app-server", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "Could not start Codex CLI at {:?}. Install Codex or set CODEX_BIN.",
                    executable
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .context("Codex app-server has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("Codex app-server has no stdout")?;
        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };
        client.call(
            "initialize",
            json!({
                "clientInfo": {
                    "name": "live-assistant",
                    "title": "Live Assistant",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "capabilities": {"experimentalApi": true},
            }),
        )?;
        Ok(client)
    }

    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        serde_json::to_writer(&mut self.stdin, &request)
            .context("Could not write to Codex app-server")?;
        self.stdin
            .write_all(b"\n")
            .context("Could not finish Codex app-server request")?;
        self.stdin
            .flush()
            .context("Could not flush Codex app-server request")?;

        loop {
            let mut line = String::new();
            let bytes = self
                .stdout
                .read_line(&mut line)
                .context("Could not read Codex app-server response")?;
            if bytes == 0 {
                bail!("Codex app-server closed before replying to {method}");
            }
            let message: Value = match serde_json::from_str(&line) {
                Ok(message) => message,
                Err(_) => continue,
            };
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                let detail = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                bail!("Codex {method} failed: {detail}");
            }
            return message
                .get("result")
                .cloned()
                .with_context(|| format!("Codex {method} returned no result"));
        }
    }
}

impl Drop for AppServerClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rate_limit_and_usage_snapshots() {
        let limits = json!({
            "rateLimits": {
                "limitId": "codex",
                "planType": "plus",
                "primary": {
                    "usedPercent": 35,
                    "windowDurationMins": 10080,
                    "resetsAt": 1_800_000_000
                },
                "credits": {"unlimited": false, "balance": "12.5"}
            },
            "rateLimitResetCredits": {"availableCount": 2}
        });
        let (parsed_limits, reset_credits) = parse_rate_limits(&limits);
        assert_eq!(parsed_limits.len(), 1);
        assert_eq!(parsed_limits[0].name, "codex");
        assert_eq!(parsed_limits[0].primary.as_ref().unwrap().used_percent, 35);
        assert_eq!(reset_credits, Some(2));

        let usage = parse_token_usage(&json!({
            "summary": {"lifetimeTokens": 5000, "peakDailyTokens": 1200},
            "dailyUsageBuckets": [
                {"startDate": "2026-07-25", "tokens": 100},
                {"startDate": "2026-07-26", "tokens": 250}
            ]
        }));
        assert_eq!(usage.lifetime_tokens, Some(5000));
        assert_eq!(usage.latest_day.as_deref(), Some("2026-07-26"));
        assert_eq!(usage.latest_day_tokens, Some(250));
        assert_eq!(usage.recent_reported_tokens, Some(350));
    }

    #[test]
    fn parses_codex_realtime_voice_catalog() {
        let voices = parse_realtime_voices(&json!({
            "voices": {
                "v1": ["juniper", "cove"],
                "v2": ["alloy", "marin", "cedar"],
                "defaultV1": "cove",
                "defaultV2": "marin"
            }
        }));
        assert_eq!(voices.v1, ["juniper", "cove"]);
        assert_eq!(voices.v2, ["alloy", "marin", "cedar"]);
        assert_eq!(voices.default_v1.as_deref(), Some("cove"));
        assert_eq!(voices.default_v2.as_deref(), Some("marin"));
    }
}

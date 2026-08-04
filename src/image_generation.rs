use crate::auth::CodexCredentials;
use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::io::Read;

/// Image sizes exposed by the settings UI and accepted by the create_image
/// tool. The landscape sizes are useful for wide desktop artwork and slides.
pub const AVAILABLE_RESOLUTIONS: &[&str] = &[
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2560x1440",
    "3840x2160",
];

pub const FALLBACK_MODELS: &[(&str, &str)] = &[
    ("gpt-image-2", "GPT Image 2"),
    ("gpt-image-1.5", "GPT Image 1.5"),
    ("gpt-image-1", "GPT Image 1"),
    ("gpt-image-1-mini", "GPT Image 1 Mini"),
];

const OPENAI_IMAGE_GENERATIONS_URL: &str = "https://api.openai.com/v1/images/generations";
const CODEX_IMAGE_GENERATIONS_URL: &str =
    "https://chatgpt.com/backend-api/codex/images/generations";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub model: String,
    pub resolution: String,
    pub turn_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageGenerationResult {
    pub data_url: String,
    pub model: String,
    pub resolution: String,
}

pub fn request_from_tool_arguments(
    arguments: &str,
    default_model: &str,
    default_resolution: &str,
    turn_id: &str,
) -> Result<ImageGenerationRequest> {
    let value: Value =
        serde_json::from_str(arguments).context("create_image arguments are not valid JSON")?;
    let object = value
        .as_object()
        .context("create_image arguments must be a JSON object")?;
    let prompt = object
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    if prompt.is_empty() {
        bail!("create_image requires a non-empty prompt");
    }

    let model = object
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .unwrap_or(default_model)
        .trim()
        .to_owned();
    if model.is_empty() {
        bail!("create_image requires an image model");
    }

    let resolution = object
        .get("resolution")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|resolution| !resolution.is_empty())
        .unwrap_or(default_resolution)
        .to_owned();
    if !AVAILABLE_RESOLUTIONS.contains(&resolution.as_str()) {
        bail!(
            "Unsupported image resolution {resolution}. Available resolutions: {}",
            AVAILABLE_RESOLUTIONS.join(", ")
        );
    }

    Ok(ImageGenerationRequest {
        prompt,
        model,
        resolution,
        turn_id: turn_id.to_owned(),
    })
}

pub fn generate(
    request: ImageGenerationRequest,
    credentials: &CodexCredentials,
) -> Result<ImageGenerationResult> {
    let body = json!({
        "prompt": request.prompt,
        "background": "auto",
        "model": request.model,
        "n": 1,
        "quality": "auto",
        "size": request.resolution,
    });

    // Codex OAuth credentials are issued for the ChatGPT Codex provider, not
    // the Platform API provider. Keep API-key users on the public endpoint,
    // while routing an account-scoped Codex token through its native image
    // generation base URL.
    let endpoint = if credentials.chatgpt_account_id.is_some() {
        CODEX_IMAGE_GENERATIONS_URL
    } else {
        OPENAI_IMAGE_GENERATIONS_URL
    };
    let mut call = ureq::post(endpoint)
        .set(
            "Authorization",
            &format!("Bearer {}", credentials.bearer_token),
        )
        .set("Content-Type", "application/json")
        .set("originator", "live-assistant");
    if let Some(account_id) = credentials
        .chatgpt_account_id
        .as_deref()
        .filter(|account_id| !account_id.is_empty())
    {
        call = call.set("ChatGPT-Account-Id", account_id);
    }
    if !request.turn_id.trim().is_empty() {
        call = call.set("x-codex-image-turn-id", request.turn_id.trim());
    }

    let response = match call.send_json(body) {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            bail!("Image generation failed (HTTP {code}): {detail}");
        }
        Err(error) => return Err(error).context("Network error while generating the image"),
    };
    let value = response
        .into_json::<Value>()
        .context("Image generation returned invalid JSON")?;
    let data_url = response_data_url(&value, credentials)?;

    Ok(ImageGenerationResult {
        data_url,
        model: request.model,
        resolution: request.resolution,
    })
}

fn response_data_url(value: &Value, credentials: &CodexCredentials) -> Result<String> {
    let first = value
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first())
        .context("Image generation response did not include an image")?;
    if let Some(encoded) = first
        .get("b64_json")
        .and_then(Value::as_str)
        .filter(|encoded| !encoded.is_empty())
    {
        return Ok(format!("data:image/png;base64,{encoded}"));
    }
    let Some(url) = first.get("url").and_then(Value::as_str) else {
        bail!("Image generation response did not include b64_json or url");
    };
    if url.starts_with("data:") {
        return Ok(url.to_owned());
    }

    let mut request = ureq::get(url).set(
        "Authorization",
        &format!("Bearer {}", credentials.bearer_token),
    );
    if let Some(account_id) = credentials
        .chatgpt_account_id
        .as_deref()
        .filter(|account_id| !account_id.is_empty())
    {
        request = request.set("ChatGPT-Account-Id", account_id);
    }
    let response = request.call().map_err(|error| anyhow::anyhow!(error))?;
    let mime = response
        .header("Content-Type")
        .and_then(|value| value.split(';').next())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("image/png")
        .to_owned();
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .context("Could not download generated image")?;
    if bytes.is_empty() {
        bail!("Downloaded generated image was empty");
    }
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults_and_supported_resolutions() {
        let request = request_from_tool_arguments(
            r#"{"prompt":"a tiny moonlit cabin"}"#,
            "gpt-image-2",
            "1024x1024",
            "call-1",
        )
        .unwrap();
        assert_eq!(request.model, "gpt-image-2");
        assert_eq!(request.resolution, "1024x1024");
        assert_eq!(request.turn_id, "call-1");

        let landscape = request_from_tool_arguments(
            r#"{"prompt":"wide landscape","resolution":"3840x2160","model":"custom-image"}"#,
            "gpt-image-2",
            "1024x1024",
            "call-2",
        )
        .unwrap();
        assert_eq!(landscape.model, "custom-image");
        assert_eq!(landscape.resolution, "3840x2160");
    }

    #[test]
    fn rejects_empty_prompt_and_unknown_resolution() {
        assert!(request_from_tool_arguments(
            r#"{"prompt":"  "}"#,
            "gpt-image-2",
            "1024x1024",
            "call-1",
        )
        .is_err());
        assert!(
            request_from_tool_arguments(
                r#"{"prompt":"test","resolution":"512x512"}"#,
                "gpt-image-2",
                "1024x1024",
                "call-1",
            )
            .is_err()
        );
    }

    #[test]
    fn parses_base64_image_response() {
        let credentials = CodexCredentials {
            bearer_token: "test".to_owned(),
            chatgpt_account_id: None,
        };
        let value = json!({"data":[{"b64_json":"aGVsbG8="}]});
        assert_eq!(
            response_data_url(&value, &credentials).unwrap(),
            "data:image/png;base64,aGVsbG8="
        );
    }
}

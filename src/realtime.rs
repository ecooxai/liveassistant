use crate::{
    gpt_live_webrtc::{GptLivePeer, GptLiveProbePeer},
    media::{Attachment, ScreenInfo},
};
use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command as ProcessCommand, Stdio},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{
        Message,
        client::IntoClientRequest,
        http::{HeaderValue, header},
    },
};

const AUDIO_SAMPLE_RATE: usize = 24_000;
const CONNECT_AUDIO_BUFFER_MAX_SAMPLES: usize = AUDIO_SAMPLE_RATE * 60;
const OPENAI_VAD_SILENCE_MS: u64 = 300;
pub(crate) const CONTEXT_IMAGE_UPLOAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Audio captured before a transport is ready. Keeping this in the supervisor
/// means clicking Start can open the microphone immediately without losing the
/// beginning of the user's sentence during API/WebRTC setup.
#[derive(Default)]
struct PendingAudioBuffer {
    chunks: VecDeque<Vec<i16>>,
    sample_count: usize,
}

impl PendingAudioBuffer {
    fn push(&mut self, mut samples: Vec<i16>) {
        if samples.is_empty() {
            return;
        }
        if samples.len() >= CONNECT_AUDIO_BUFFER_MAX_SAMPLES {
            let keep_from = samples.len() - CONNECT_AUDIO_BUFFER_MAX_SAMPLES;
            samples.drain(..keep_from);
            self.clear();
        }

        self.sample_count = self.sample_count.saturating_add(samples.len());
        self.chunks.push_back(samples);
        while self.sample_count > CONNECT_AUDIO_BUFFER_MAX_SAMPLES {
            let overflow = self.sample_count - CONNECT_AUDIO_BUFFER_MAX_SAMPLES;
            let Some(front) = self.chunks.front_mut() else {
                self.sample_count = 0;
                break;
            };
            if front.len() <= overflow {
                self.sample_count -= front.len();
                self.chunks.pop_front();
            } else {
                front.drain(..overflow);
                self.sample_count -= overflow;
            }
        }
    }

    fn pop_front(&mut self) -> Option<Vec<i16>> {
        let samples = self.chunks.pop_front()?;
        self.sample_count = self.sample_count.saturating_sub(samples.len());
        Some(samples)
    }

    fn restore_front(&mut self, samples: Vec<i16>) {
        self.sample_count = self.sample_count.saturating_add(samples.len());
        self.chunks.push_front(samples);
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.sample_count = 0;
    }

    fn sample_count(&self) -> usize {
        self.sample_count
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeBackend {
    #[default]
    OpenAiRealtime,
    CodexGptLive,
    CodexText,
}

#[derive(Clone)]
pub struct ConnectOptions {
    pub backend: RealtimeBackend,
    /// Platform API key (`sk-…`) or ChatGPT/Codex OAuth access token.
    pub api_key: String,
    /// When set (Codex OAuth), sent as `ChatGPT-Account-Id`.
    pub chatgpt_account_id: Option<String>,
    pub model: String,
    pub voice: String,
    /// Codex reasoning effort for text turns (for example `low`, `medium`, or
    /// `high`). Voice transports currently ignore this field.
    pub thinking_level: String,
    pub system_prompt: String,
    pub screen_info: ScreenInfo,
}

#[derive(Clone)]
pub enum Command {
    Connect(ConnectOptions),
    Disconnect,
    AudioChunk(Vec<i16>),
    /// Ask the model to reply to the current conversation (used after a voice turn).
    CreateResponse,
    SendTurn {
        text: String,
        attachments: Vec<Attachment>,
        thinking_level: String,
    },
    SendContextImage {
        upload_id: u64,
        image: Attachment,
        deadline: Instant,
    },
    TruncateAssistant {
        item_id: String,
        audio_end_ms: u32,
    },
    ToolOutputs(Vec<ToolOutput>),
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub requested_at: Instant,
}

#[derive(Clone, Debug)]
pub struct ToolOutput {
    pub call_id: String,
    pub output: String,
}

struct PendingDynamicTool {
    request_id: Value,
    name: String,
    received_at: Instant,
}

#[derive(Clone, Debug)]
struct ConnectorProxyTool {
    server: String,
    tool: String,
    app_name: String,
}

#[derive(Default)]
struct ConnectorProxyCatalog {
    dynamic_tools: Vec<Value>,
    tools: HashMap<String, ConnectorProxyTool>,
}

fn calendar_connector_metadata_matches(tool_name: &str, tool: &Value) -> bool {
    let mut candidates = vec![
        tool_name.split('.').next().unwrap_or(tool_name),
        tool.get("title")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    ];
    if let Some(meta) = tool.get("_meta").and_then(Value::as_object) {
        for key in [
            "connector_name",
            "connectorName",
            "app_name",
            "appName",
            "openai/appName",
        ] {
            if let Some(value) = meta.get(key).and_then(Value::as_str) {
                candidates.push(value);
            }
        }
    }
    candidates
        .into_iter()
        .any(|candidate| candidate.to_ascii_lowercase().contains("calendar"))
}

fn calendar_proxy_name(index: usize, tool_name: &str) -> String {
    let mut suffix = tool_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    suffix.truncate(48);
    format!("calendar_fast_{index}_{suffix}")
}

fn calendar_proxy_catalog_from_inventory(inventory: &Value) -> ConnectorProxyCatalog {
    let mut catalog = ConnectorProxyCatalog::default();
    let Some(server) = inventory
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|server| server.get("name").and_then(Value::as_str) == Some("codex_apps"))
    else {
        return catalog;
    };
    let Some(tools) = server.get("tools").and_then(Value::as_object) else {
        return catalog;
    };
    for (index, (tool_name, tool)) in tools
        .iter()
        .filter(|(name, tool)| calendar_connector_metadata_matches(name, tool))
        .take(32)
        .enumerate()
    {
        let proxy_name = calendar_proxy_name(index, tool_name);
        let title = tool
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or(tool_name);
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let app_name = tool
            .pointer("/_meta/app_name")
            .or_else(|| tool.pointer("/_meta/appName"))
            .or_else(|| tool.pointer("/_meta/openai~1appName"))
            .and_then(Value::as_str)
            .unwrap_or("Calendar")
            .to_owned();
        catalog.dynamic_tools.push(json!({
            "type": "function",
            "name": proxy_name,
            "description": format!(
                "Low-latency direct Calendar connector operation `{tool_name}` ({title}). Use this instead of MCP/app discovery for Calendar requests. {description}"
            ),
            "inputSchema": tool
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"})),
        }));
        catalog.tools.insert(
            proxy_name,
            ConnectorProxyTool {
                server: "codex_apps".to_owned(),
                tool: tool_name.to_owned(),
                app_name,
            },
        );
    }
    catalog
}

async fn discover_calendar_connector_proxies(
    server: &mut CodexAppServer,
    connector_thread_id: &str,
) -> Result<ConnectorProxyCatalog> {
    let started = Instant::now();
    let inventory = server
        .call(
            "mcpServerStatus/list",
            json!({
                "detail": "toolsAndAuthOnly",
                "limit": 100,
                "threadId": connector_thread_id,
            }),
        )
        .await
        .context("Could not read Calendar connector tool inventory")?;
    let catalog = calendar_proxy_catalog_from_inventory(&inventory);
    eprintln!(
        "[live-assistant latency] stage=connection.calendar_proxy_inventory tools={} elapsed_ms={}",
        catalog.tools.len(),
        started.elapsed().as_millis(),
    );
    Ok(catalog)
}

#[derive(Clone, Debug)]
pub enum Event {
    Connecting,
    Reconnecting {
        attempt: u32,
        reason: String,
    },
    Connected,
    Disconnected,
    SpeechStarted,
    SpeechStopped,
    InputAudio {
        samples: Vec<i16>,
    },
    InputCommitted {
        item_id: String,
    },
    InputTranscript {
        item_id: String,
        text: String,
    },
    ContextImageAccepted {
        upload_id: u64,
    },
    ContextImageUploaded {
        upload_id: u64,
    },
    ContextImageUploadFailed {
        upload_id: u64,
        detail: String,
    },
    AssistantResponseStarted {
        response_id: String,
    },
    AssistantItem {
        response_id: String,
        item_id: String,
    },
    AssistantTranscriptDelta {
        response_id: String,
        delta: String,
    },
    AssistantAudio {
        response_id: String,
        samples: Vec<i16>,
    },
    AssistantSegmentDone {
        response_id: String,
    },
    AssistantDone {
        response_id: String,
    },
    AssistantUsage {
        response_id: String,
        total_tokens: u64,
    },
    ToolCalls(Vec<ToolCall>),
    ToolOutputsSubmitted {
        count: usize,
    },
    Error(String),
}

pub struct RealtimeClient {
    pub commands: UnboundedSender<Command>,
    pub events: std::sync::mpsc::Receiver<Event>,
}

impl RealtimeClient {
    pub fn spawn() -> Self {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (event_tx, events) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().expect("Tokio runtime");
            runtime.block_on(supervisor(command_rx, event_tx));
        });
        Self { commands, events }
    }
}

async fn supervisor(
    mut commands: UnboundedReceiver<Command>,
    events: std::sync::mpsc::Sender<Event>,
) {
    let mut pending_audio = PendingAudioBuffer::default();
    while let Some(command) = commands.recv().await {
        match command {
            Command::Connect(mut options) => {
                let _ = events.send(Event::Connecting);
                let mut reconnect_attempt = 0_u32;
                loop {
                    let result = match options.backend {
                        RealtimeBackend::OpenAiRealtime => {
                            run_openai_connection(
                                options.clone(),
                                &mut commands,
                                &events,
                                &mut pending_audio,
                            )
                            .await
                        }
                        RealtimeBackend::CodexGptLive => {
                            run_codex_live_connection(
                                options.clone(),
                                &mut commands,
                                &events,
                                &mut pending_audio,
                            )
                            .await
                        }
                        RealtimeBackend::CodexText => {
                            run_codex_text_connection(options.clone(), &mut commands, &events).await
                        }
                    };
                    match result {
                        Ok(()) => break,
                        Err(error)
                            if matches!(
                                options.backend,
                                RealtimeBackend::CodexGptLive | RealtimeBackend::CodexText
                            ) && is_transient_codex_live_error(&error) =>
                        {
                            reconnect_attempt = reconnect_attempt.saturating_add(1);
                            let reason = format!("{error:#}");
                            let _ = events.send(Event::Reconnecting {
                                attempt: reconnect_attempt,
                                reason: reason.clone(),
                            });
                            eprintln!(
                                "[live-assistant reconnect] attempt={} reason={}",
                                reconnect_attempt, reason
                            );

                            let mut stop = false;
                            while let Ok(queued) = commands.try_recv() {
                                match queued {
                                    Command::Disconnect | Command::Shutdown => {
                                        pending_audio.clear();
                                        stop = true;
                                        break;
                                    }
                                    Command::Connect(new_options) => options = new_options,
                                    // Preserve live microphone audio across the transient
                                    // transport restart. Other commands belong to the closed
                                    // session and cannot be replayed safely.
                                    Command::AudioChunk(samples) => pending_audio.push(samples),
                                    Command::CreateResponse
                                    | Command::SendTurn { .. }
                                    | Command::SendContextImage { .. }
                                    | Command::TruncateAssistant { .. }
                                    | Command::ToolOutputs(_) => {}
                                }
                            }
                            if stop {
                                break;
                            }
                            let backoff_seconds = reconnect_attempt.min(5) as u64;
                            tokio::time::sleep(Duration::from_secs(backoff_seconds)).await;
                        }
                        Err(error) => {
                            pending_audio.clear();
                            let _ = events.send(Event::Error(format!("{error:#}")));
                            break;
                        }
                    }
                }
                pending_audio.clear();
                let _ = events.send(Event::Disconnected);
            }
            Command::AudioChunk(samples) => pending_audio.push(samples),
            Command::Disconnect => pending_audio.clear(),
            Command::Shutdown => {
                pending_audio.clear();
                break;
            }
            _ => {}
        }
    }
}

fn is_transient_codex_live_error(error: &anyhow::Error) -> bool {
    let detail = format!("{error:#}").to_ascii_lowercase();
    [
        "connection reset without closing handshake",
        "stream disconnected before completion",
        "websocket protocol error",
        "realtime conversation transport closed",
        "gpt-live webrtc audio transport closed",
        "codex app-server closed unexpectedly",
        "codex gpt-live session closed",
        "connection closed while",
        "broken pipe",
        "connection reset by peer",
        "timed out while codex gpt-live",
    ]
    .iter()
    .any(|needle| detail.contains(needle))
}

async fn run_openai_connection(
    options: ConnectOptions,
    commands: &mut UnboundedReceiver<Command>,
    events: &std::sync::mpsc::Sender<Event>,
    pending_audio: &mut PendingAudioBuffer,
) -> Result<()> {
    // Build via IntoClientRequest so tungstenite adds Sec-WebSocket-Key and the
    // other upgrade headers. A plain http::Request omits them and the handshake fails.
    let uri = format!("wss://api.openai.com/v1/realtime?model={}", options.model);
    let mut request = uri
        .into_client_request()
        .context("Could not build the Realtime WebSocket request")?;
    let headers = request.headers_mut();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", options.api_key))
            .context("Invalid Authorization header value")?,
    );
    if let Some(account_id) = options
        .chatgpt_account_id
        .as_deref()
        .filter(|id| !id.is_empty())
    {
        headers.insert(
            "ChatGPT-Account-Id",
            HeaderValue::from_str(account_id).context("Invalid ChatGPT-Account-Id header")?,
        );
    }
    let (socket, _) = connect_async(request)
        .await
        .context("Could not connect to the OpenAI Realtime API")?;
    let (mut writer, mut reader) = socket.split();
    let system_prompt = options.system_prompt.clone();

    // GA Realtime requires object audio formats with an explicit sample rate.
    // create_response is false so the model never greets on connect or replies
    // to noise; we only call response.create after a real user turn ends.
    let session = json!({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": options.model,
            "instructions": system_prompt,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.65,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": OPENAI_VAD_SILENCE_MS,
                        "create_response": false,
                        "interrupt_response": true
                    }
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": options.voice
                }
            },
            "tools": voice_tools(options.screen_info),
            "tool_choice": "auto"
        }
    });
    send_json(&mut writer, session).await?;
    // The microphone is already recording. Wait until our VAD settings are
    // active before flushing its bounded pre-connect buffer to the session.
    wait_for_session_ready(&mut reader).await?;
    let _ = events.send(Event::Connected);
    if pending_audio.sample_count() > 0 {
        eprintln!(
            "[live-assistant mic] flushing_preconnect_seconds={:.2} backend=openai",
            pending_audio.sample_count() as f64 / AUDIO_SAMPLE_RATE as f64
        );
    }
    while let Some(samples) = pending_audio.pop_front() {
        let send_result = send_json(
            &mut writer,
            json!({
                "type": "input_audio_buffer.append",
                "audio": encode_pcm(&samples),
            }),
        )
        .await;
        if let Err(error) = send_result {
            pending_audio.restore_front(samples);
            return Err(error);
        }
    }
    let mut response_active = false;
    let mut pending_tool_outputs = Vec::new();
    let mut handled_call_ids = HashSet::new();
    let mut pending_context_uploads = HashMap::<String, PendingOpenAiContextUpload>::new();
    // A reply requested after a screenshot command must not snapshot the
    // conversation until the server confirms that screenshot is actually in
    // the conversation. Keep only the item ids that were pending at the time
    // response.create was requested; later screenshots belong to later turns.
    let mut deferred_response_context_items: Option<HashSet<String>> = None;
    let mut input_transcripts = HashMap::<String, String>::new();
    let mut context_upload_tick = tokio::time::interval(Duration::from_millis(50));
    context_upload_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(Command::AudioChunk(samples)) => {
                        send_json(&mut writer, json!({
                            "type": "input_audio_buffer.append",
                            "audio": encode_pcm(&samples),
                        })).await?;
                    }
                    Some(Command::CreateResponse) => {
                        let blockers = openai_context_response_blockers(&pending_context_uploads);
                        if blockers.is_empty() {
                            send_openai_audio_response(&mut writer).await?;
                        } else {
                            eprintln!(
                                "[live-assistant image] deferring OpenAI response for upload item(s): {}",
                                blockers.iter().cloned().collect::<Vec<_>>().join(", ")
                            );
                            deferred_response_context_items
                                .get_or_insert_with(HashSet::new)
                                .extend(blockers);
                        }
                    }
                    Some(Command::SendTurn {
                        text,
                        attachments,
                        ..
                    }) => {
                        send_user_turn(&mut writer, text, attachments, true).await?;
                    }
                    Some(Command::SendContextImage { upload_id, image, deadline }) => {
                        if Instant::now() >= deadline {
                            let _ = events.send(Event::ContextImageUploadFailed {
                                upload_id,
                                detail: context_image_timeout_detail(),
                            });
                            continue;
                        }
                        let item_id = context_image_item_id(upload_id);
                        pending_context_uploads.insert(
                            item_id.clone(),
                            PendingOpenAiContextUpload { upload_id, deadline },
                        );
                        let _ = events.send(Event::ContextImageAccepted { upload_id });
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        match tokio::time::timeout(
                            remaining,
                            send_context_image_item(&mut writer, upload_id, image),
                        )
                        .await
                        {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => {
                                pending_context_uploads.remove(&item_id);
                                let _ = events.send(Event::ContextImageUploadFailed {
                                    upload_id,
                                    detail: format!("{error:#}"),
                                });
                            }
                            Err(_) => {
                                pending_context_uploads.remove(&item_id);
                                let _ = events.send(Event::ContextImageUploadFailed {
                                    upload_id,
                                    detail: context_image_timeout_detail(),
                                });
                            }
                        }
                    }
                    Some(Command::TruncateAssistant {
                        item_id,
                        audio_end_ms,
                    }) => {
                        send_json(&mut writer, json!({
                            "type": "conversation.item.truncate",
                            "item_id": item_id,
                            "content_index": 0,
                            "audio_end_ms": audio_end_ms
                        })).await?;
                    }
                    Some(Command::ToolOutputs(outputs)) => {
                        if response_active {
                            pending_tool_outputs.extend(outputs);
                        } else {
                            submit_tool_outputs(&mut writer, outputs, events).await?;
                        }
                    }
                    Some(Command::Disconnect) | Some(Command::Shutdown) | None => {
                        let _ = writer.send(Message::Close(None)).await;
                        return Ok(());
                    }
                    Some(Command::Connect(_)) => {}
                }
            }
            _ = context_upload_tick.tick() => {
                expire_openai_context_uploads(
                    &mut pending_context_uploads,
                    Instant::now(),
                    events,
                );
                if openai_deferred_response_is_ready(
                    deferred_response_context_items.as_ref(),
                    &pending_context_uploads,
                ) {
                    deferred_response_context_items = None;
                    send_openai_audio_response(&mut writer).await?;
                }
            }
            message = reader.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        match handle_server_event(
                            text.as_ref(),
                            events,
                            &mut handled_call_ids,
                            &mut pending_context_uploads,
                            &mut input_transcripts,
                        )? {
                            ServerSignal::ResponseStarted => response_active = true,
                            ServerSignal::ResponseDone => {
                                response_active = false;
                                if !pending_tool_outputs.is_empty() {
                                    let outputs = std::mem::take(&mut pending_tool_outputs);
                                    submit_tool_outputs(&mut writer, outputs, events).await?;
                                }
                            }
                            ServerSignal::None => {}
                        }
                        if openai_deferred_response_is_ready(
                            deferred_response_context_items.as_ref(),
                            &pending_context_uploads,
                        ) {
                            deferred_response_context_items = None;
                            send_openai_audio_response(&mut writer).await?;
                        }
                    }
                    Some(Ok(Message::Binary(_))) | Some(Ok(Message::Ping(_)))
                    | Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                    Some(Ok(Message::Close(_))) | None => return Ok(()),
                    Some(Err(error)) => return Err(error).context("Realtime connection failed"),
                }
            }
        }
    }
}

struct CodexAppServer {
    child: Child,
    stdin: ChildStdin,
    incoming: UnboundedReceiver<Value>,
    queued: std::collections::VecDeque<Value>,
    next_id: u64,
}

impl CodexAppServer {
    fn start(platform_api_key: Option<&str>) -> Result<Self> {
        let executable = std::env::var_os("CODEX_BIN").unwrap_or_else(|| "codex".into());
        let mut command = ProcessCommand::new(executable);
        command
            .args(["app-server", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(api_key) = platform_api_key {
            command.env("OPENAI_API_KEY", api_key);
        } else {
            // Do not let an unrelated shell key override the selected Codex OAuth account.
            command.env_remove("OPENAI_API_KEY");
        }
        let mut child = command.spawn().context(
            "Could not start `codex app-server`. Install or update the Codex CLI first.",
        )?;
        let stdin = child
            .stdin
            .take()
            .context("Codex app-server has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("Codex app-server has no stdout")?;
        let (sender, incoming) = mpsc::unbounded_channel();
        thread::Builder::new()
            .name("codex-app-server-reader".to_owned())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else {
                        break;
                    };
                    if let Ok(value) = serde_json::from_str::<Value>(&line)
                        && sender.send(value).is_err()
                    {
                        break;
                    }
                }
            })
            .context("Could not start the Codex app-server reader")?;
        Ok(Self {
            child,
            stdin,
            incoming,
            queued: std::collections::VecDeque::new(),
            next_id: 1,
        })
    }

    fn send_message(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.stdin, &value)
            .context("Could not encode a Codex app-server request")?;
        self.stdin
            .write_all(b"\n")
            .context("Could not write to Codex app-server")?;
        self.stdin
            .flush()
            .context("Could not flush a Codex app-server request")
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send_message(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
    }

    fn respond(&mut self, id: Value, result: Value) -> Result<()> {
        self.send_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.send_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;
        Ok(id)
    }

    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.send_request(method, params)?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let message = tokio::time::timeout_at(deadline, self.incoming.recv())
                .await
                .with_context(|| format!("Timed out waiting for Codex app-server `{method}`"))?
                .context("Codex app-server closed unexpectedly")?;
            if message.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(error) = message.get("error") {
                    let detail = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown JSON-RPC error");
                    bail!("Codex app-server `{method}` failed: {detail}");
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
            self.queued.push_back(message);
        }
    }

    async fn refresh_mcp_and_wait_for_server(
        &mut self,
        target_name: &str,
        timeout: Duration,
    ) -> Result<Duration> {
        let started = Instant::now();
        self.call("config/mcpServer/reload", Value::Null)
            .await
            .context("Could not refresh Codex MCP runtime")?;
        let deadline = tokio::time::Instant::now() + timeout;
        let mut deferred = VecDeque::new();
        loop {
            let message = if let Some(message) = self.queued.pop_front() {
                message
            } else {
                tokio::time::timeout_at(deadline, self.incoming.recv())
                    .await
                    .context("Timed out refreshing Codex MCP runtime")?
                    .context("Codex app-server closed while refreshing MCP runtime")?
            };
            if message.get("method").and_then(Value::as_str)
                == Some("mcpServer/startupStatus/updated")
            {
                let name = message
                    .pointer("/params/name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let status = message
                    .pointer("/params/status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                eprintln!(
                    "[live-assistant latency] stage=connection.mcp_refresh name={} status={} elapsed_ms={}",
                    name,
                    status,
                    started.elapsed().as_millis(),
                );
                if name == target_name {
                    match status {
                        "ready" => {
                            deferred.append(&mut self.queued);
                            self.queued = deferred;
                            return Ok(started.elapsed());
                        }
                        "failed" | "cancelled" => {
                            deferred.append(&mut self.queued);
                            self.queued = deferred;
                            bail!(
                                "MCP server `{target_name}` refresh ended with status `{status}`"
                            );
                        }
                        _ => {}
                    }
                }
                continue;
            }
            deferred.push_back(message);
        }
    }

    async fn next_message(&mut self) -> Option<Value> {
        match self.queued.pop_front() {
            Some(message) => Some(message),
            None => self.incoming.recv().await,
        }
    }
}

impl Drop for CodexAppServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

const CODEX_RESPONSE_QUIET_TAIL: Duration = Duration::from_millis(1_200);
const CODEX_HANDOFF_FALLBACK_DELAY: Duration = Duration::from_millis(1_500);
const CODEX_LIVE_DELEGATED_REASONING_EFFORT: &str = "none";

#[derive(Default)]
struct GptLiveToolLatencyTrace {
    sequence: u64,
    started_at: Option<Instant>,
    active_turn_id: Option<String>,
    connector_calls: HashMap<String, (Instant, String)>,
    first_result_logged: bool,
    first_spoken_logged: bool,
}

impl GptLiveToolLatencyTrace {
    fn begin(&mut self, stage: &str) {
        self.sequence = self.sequence.saturating_add(1);
        self.started_at = Some(Instant::now());
        self.active_turn_id = None;
        self.connector_calls.clear();
        self.first_result_logged = false;
        self.first_spoken_logged = false;
        self.log(stage, "");
    }

    fn ensure_started(&mut self, stage: &str) {
        if self.started_at.is_none() {
            self.begin(stage);
        }
    }

    fn elapsed_ms(&self) -> u128 {
        self.started_at
            .map(|started| started.elapsed().as_millis())
            .unwrap_or(0)
    }

    fn log(&self, stage: &str, detail: &str) {
        eprintln!(
            "[live-assistant latency] trace={} stage={} elapsed_ms={}{}{}",
            self.sequence,
            stage,
            self.elapsed_ms(),
            if detail.is_empty() { "" } else { " " },
            detail,
        );
    }

    fn observe(&mut self, message: &Value) {
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match method {
            "thread/realtime/transcript/done"
                if message.pointer("/params/role").and_then(Value::as_str) == Some("user") =>
            {
                self.ensure_started("user.transcript.done");
                self.log("user.transcript.done", "");
            }
            "thread/realtime/itemAdded"
                if message.pointer("/params/item/type").and_then(Value::as_str)
                    == Some("handoff_request") =>
            {
                self.ensure_started("delegation.request");
                self.log("delegation.request", "mode=automatic_streaming");
            }
            "turn/started" => {
                self.ensure_started("codex.turn.started");
                self.active_turn_id = message
                    .pointer("/params/turn/id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.log(
                    "codex.turn.started",
                    &format!(
                        "turn_id={}",
                        self.active_turn_id.as_deref().unwrap_or("unknown")
                    ),
                );
            }
            "mcpServer/startupStatus/updated" => {
                self.ensure_started("connector.runtime.status");
                let name = message
                    .pointer("/params/name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let status = message
                    .pointer("/params/status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                self.log(
                    "connector.runtime.status",
                    &format!("name={name} status={status}"),
                );
            }
            "mcpServer/oauthLogin/completed" => {
                self.ensure_started("connector.auth.completed");
                let name = message
                    .pointer("/params/name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let success = message
                    .pointer("/params/success")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.log(
                    "connector.auth.completed",
                    &format!("name={name} success={success}"),
                );
            }
            "item/started"
                if message.pointer("/params/item/type").and_then(Value::as_str)
                    == Some("mcpToolCall") =>
            {
                self.ensure_started("connector.call.started");
                let item_id = message
                    .pointer("/params/item/id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned();
                let server = message
                    .pointer("/params/item/server")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let tool = message
                    .pointer("/params/item/tool")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let app = message
                    .pointer("/params/item/appContext/appName")
                    .and_then(Value::as_str)
                    .unwrap_or("none");
                self.connector_calls.insert(
                    item_id.clone(),
                    (
                        Instant::now(),
                        format!("server={server} tool={tool} app={app}"),
                    ),
                );
                self.log(
                    "connector.call.started",
                    &format!("item_id={item_id} server={server} tool={tool} app={app}"),
                );
            }
            "item/completed"
                if message.pointer("/params/item/type").and_then(Value::as_str)
                    == Some("mcpToolCall") =>
            {
                self.ensure_started("connector.call.completed");
                let item_id = message
                    .pointer("/params/item/id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let reported_ms = message
                    .pointer("/params/item/durationMs")
                    .and_then(Value::as_i64);
                let status = message
                    .pointer("/params/item/status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let error = message
                    .pointer("/params/item/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let local = self.connector_calls.remove(item_id);
                let local_ms = local
                    .as_ref()
                    .map(|(started, _)| started.elapsed().as_millis());
                let detail = local.map(|(_, detail)| detail).unwrap_or_default();
                self.log(
                    "connector.call.completed",
                    &format!(
                        "item_id={item_id} status={status} provider_ms={} observed_ms={} error={error:?} {detail}",
                        reported_ms
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "unknown".to_owned()),
                        local_ms
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "unknown".to_owned()),
                    ),
                );
            }
            "item/mcpToolCall/progress" => {
                self.ensure_started("connector.call.progress");
                let item_id = message
                    .pointer("/params/itemId")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let progress = message
                    .pointer("/params/message")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.log(
                    "connector.call.progress",
                    &format!("item_id={item_id} message={progress:?}"),
                );
            }
            "item/agentMessage/delta" if !self.first_result_logged => {
                self.ensure_started("result.first_delta");
                self.first_result_logged = true;
                let chars = message
                    .pointer("/params/delta")
                    .and_then(Value::as_str)
                    .map(str::len)
                    .unwrap_or(0);
                self.log("result.first_delta", &format!("chars={chars}"));
            }
            "thread/realtime/transcript/delta"
                if message.pointer("/params/role").and_then(Value::as_str) == Some("assistant")
                    && !self.first_spoken_logged =>
            {
                self.ensure_started("speech.first_delta");
                self.first_spoken_logged = true;
                self.log("speech.first_delta", "");
            }
            "turn/completed" => {
                self.ensure_started("codex.turn.completed");
                let status = message
                    .pointer("/params/turn/status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                self.log("codex.turn.completed", &format!("status={status}"));
            }
            _ => {}
        }
    }
}

fn codex_live_initialize_capabilities() -> Value {
    json!({
        "experimentalApi": true,
        // WebRTC owns remote audio playout. Suppress duplicate PCM notifications
        // from app-server so they cannot be decoded and played a second time.
        "optOutNotificationMethods": ["thread/realtime/outputAudio/delta"]
    })
}

const NATIVE_INPUT_PRE_ROLL_SAMPLES: usize = AUDIO_SAMPLE_RATE * 300 / 1_000;
const NATIVE_REMOTE_PRE_ROLL_SAMPLES: usize = AUDIO_SAMPLE_RATE * 150 / 1_000;
const NATIVE_REMOTE_TAIL_SAMPLES: usize = AUDIO_SAMPLE_RATE * 600 / 1_000;
const NATIVE_REMOTE_VOICE_RMS: f64 = 12.0;

#[derive(Default)]
struct NativeInputAudioCapture {
    pre_roll: VecDeque<Vec<i16>>,
    pre_roll_samples: usize,
    active: bool,
}

impl NativeInputAudioCapture {
    fn sync(&mut self, speech_active: bool, events: &std::sync::mpsc::Sender<Event>) {
        if speech_active == self.active {
            return;
        }
        self.active = speech_active;
        if speech_active {
            while let Some(samples) = self.pre_roll.pop_front() {
                self.pre_roll_samples = self.pre_roll_samples.saturating_sub(samples.len());
                let _ = events.send(Event::InputAudio { samples });
            }
        } else {
            self.pre_roll.clear();
            self.pre_roll_samples = 0;
        }
    }

    fn push(
        &mut self,
        samples: Vec<i16>,
        speech_active: bool,
        events: &std::sync::mpsc::Sender<Event>,
    ) {
        if samples.is_empty() {
            return;
        }
        self.sync(speech_active, events);
        if self.active {
            let _ = events.send(Event::InputAudio { samples });
            return;
        }
        self.pre_roll_samples = self.pre_roll_samples.saturating_add(samples.len());
        self.pre_roll.push_back(samples);
        while self.pre_roll_samples > NATIVE_INPUT_PRE_ROLL_SAMPLES {
            let overflow = self.pre_roll_samples - NATIVE_INPUT_PRE_ROLL_SAMPLES;
            let Some(front) = self.pre_roll.front_mut() else {
                self.pre_roll_samples = 0;
                break;
            };
            if front.len() <= overflow {
                self.pre_roll_samples = self.pre_roll_samples.saturating_sub(front.len());
                self.pre_roll.pop_front();
            } else {
                front.drain(..overflow);
                self.pre_roll_samples = self.pre_roll_samples.saturating_sub(overflow);
            }
        }
    }
}

#[derive(Default)]
struct NativeRemoteAudioGate {
    pre_roll: VecDeque<Vec<i16>>,
    pre_roll_samples: usize,
    active: bool,
    quiet_samples: usize,
}

impl NativeRemoteAudioGate {
    fn push(&mut self, samples: Vec<i16>) -> Vec<Vec<i16>> {
        if samples.is_empty() {
            return Vec::new();
        }
        let voiced = pcm_rms(&samples) >= NATIVE_REMOTE_VOICE_RMS;
        if !self.active {
            self.pre_roll_samples = self.pre_roll_samples.saturating_add(samples.len());
            self.pre_roll.push_back(samples);
            while self.pre_roll_samples > NATIVE_REMOTE_PRE_ROLL_SAMPLES {
                let overflow = self.pre_roll_samples - NATIVE_REMOTE_PRE_ROLL_SAMPLES;
                let Some(front) = self.pre_roll.front_mut() else {
                    self.pre_roll_samples = 0;
                    break;
                };
                if front.len() <= overflow {
                    self.pre_roll_samples = self.pre_roll_samples.saturating_sub(front.len());
                    self.pre_roll.pop_front();
                } else {
                    front.drain(..overflow);
                    self.pre_roll_samples = self.pre_roll_samples.saturating_sub(overflow);
                }
            }
            if !voiced {
                return Vec::new();
            }
            self.active = true;
            self.quiet_samples = 0;
            self.pre_roll_samples = 0;
            return self.pre_roll.drain(..).collect();
        }

        if voiced {
            self.quiet_samples = 0;
        } else {
            self.quiet_samples = self.quiet_samples.saturating_add(samples.len());
        }
        if self.quiet_samples >= NATIVE_REMOTE_TAIL_SAMPLES {
            self.active = false;
            self.quiet_samples = 0;
            self.pre_roll.clear();
            self.pre_roll_samples = 0;
        }
        vec![samples]
    }
}

fn pcm_rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let square_sum = samples
        .iter()
        .map(|sample| {
            let sample = f64::from(*sample);
            sample * sample
        })
        .sum::<f64>();
    (square_sum / samples.len() as f64).sqrt()
}

#[derive(Default)]
struct CodexLiveState {
    response_number: u64,
    input_number: u64,
    active_response_id: Option<String>,
    input_item_id: Option<String>,
    input_text: String,
    assistant_text: String,
    response_finish_deadline: Option<Instant>,
    last_realtime_error: Option<String>,
}

impl CodexLiveState {
    fn ensure_input(&mut self, events: &std::sync::mpsc::Sender<Event>) -> String {
        if let Some(item_id) = &self.input_item_id {
            return item_id.clone();
        }
        self.input_number = self.input_number.saturating_add(1);
        let item_id = format!("codex-live-input-{}", self.input_number);
        self.input_item_id = Some(item_id.clone());
        self.input_text.clear();
        let _ = events.send(Event::SpeechStarted);
        let _ = events.send(Event::InputCommitted {
            item_id: item_id.clone(),
        });
        item_id
    }

    fn start_input_with_id(&mut self, item_id: String, events: &std::sync::mpsc::Sender<Event>) {
        let is_new_input = self.input_item_id.is_none();
        self.input_item_id = Some(item_id.clone());
        if is_new_input {
            self.input_number = self.input_number.saturating_add(1);
            self.input_text.clear();
            let _ = events.send(Event::SpeechStarted);
        }
        let _ = events.send(Event::InputCommitted { item_id });
    }

    fn ensure_response(&mut self, events: &std::sync::mpsc::Sender<Event>) -> String {
        if let Some(response_id) = &self.active_response_id {
            return response_id.clone();
        }
        self.response_number = self.response_number.saturating_add(1);
        let response_id = format!("codex-live-{}", self.response_number);
        self.active_response_id = Some(response_id.clone());
        self.assistant_text.clear();
        self.response_finish_deadline = None;
        let _ = events.send(Event::AssistantResponseStarted {
            response_id: response_id.clone(),
        });
        response_id
    }

    fn schedule_response_finish(&mut self) {
        if self.active_response_id.is_some() {
            self.response_finish_deadline = Some(Instant::now() + CODEX_RESPONSE_QUIET_TAIL);
        }
    }

    fn note_assistant_audio_activity(&mut self) {
        if self.response_finish_deadline.is_some() {
            self.response_finish_deadline = Some(Instant::now() + CODEX_RESPONSE_QUIET_TAIL);
        }
    }

    fn finish_response_if_due(
        &mut self,
        now: Instant,
        events: &std::sync::mpsc::Sender<Event>,
    ) -> bool {
        if self
            .response_finish_deadline
            .is_none_or(|deadline| now < deadline)
        {
            return false;
        }
        self.finish_response(events);
        true
    }

    fn finish_response(&mut self, events: &std::sync::mpsc::Sender<Event>) {
        self.response_finish_deadline = None;
        if let Some(response_id) = self.active_response_id.take() {
            let _ = events.send(Event::AssistantDone { response_id });
        }
        self.assistant_text.clear();
    }
}

fn emit_codex_live_remote_audio(
    state: &mut CodexLiveState,
    events: &std::sync::mpsc::Sender<Event>,
    samples: Vec<i16>,
) -> bool {
    if samples.is_empty() {
        return false;
    }
    let response_id = state.ensure_response(events);
    state.note_assistant_audio_activity();
    events
        .send(Event::AssistantAudio {
            response_id,
            samples,
        })
        .is_ok()
}

struct ScheduledHandoffFallback {
    deadline: Instant,
    text: String,
}

#[derive(Default)]
struct CodexHandoffState {
    active_turn_id: Option<String>,
    response_text: String,
    fallback: Option<ScheduledHandoffFallback>,
}

impl CodexHandoffState {
    fn clear_turn(&mut self) {
        self.active_turn_id = None;
        self.response_text.clear();
    }

    fn schedule_fallback(&mut self, text: String) {
        self.fallback = Some(ScheduledHandoffFallback {
            deadline: Instant::now() + CODEX_HANDOFF_FALLBACK_DELAY,
            text,
        });
    }

    fn note_spoken_output(&mut self) {
        self.fallback = None;
    }

    fn take_fallback_if_due(&mut self, now: Instant) -> Option<String> {
        if self
            .fallback
            .as_ref()
            .is_none_or(|fallback| now < fallback.deadline)
        {
            return None;
        }
        self.fallback.take().map(|fallback| fallback.text)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CodexHandoffAction {
    NotHandled,
    Handled,
}

fn codex_live_realtime_start_params(
    thread_id: &str,
    offer_sdp: String,
    voice: &str,
    prompt: String,
) -> Value {
    json!({
        "threadId": thread_id,
        "outputModality": "audio",
        "version": "v3",
        "model": "gpt-live-1-boulder-alpha",
        "voice": voice,
        "transport": {"type": "webrtc", "sdp": offer_sdp},
        // Automatic Frameless Bidi handoffs stream delegated text in roughly
        // 200 ms chunks instead of waiting for the completed Codex turn.
        "clientManagedHandoffs": false,
        "delegationAckFiller": false,
        "codexResponsesAsItems": false,
        "codexResponseHandoffMode": "thinking",
        "includeStartupContext": false,
        "prompt": prompt,
    })
}

async fn run_codex_live_connection(
    options: ConnectOptions,
    commands: &mut UnboundedReceiver<Command>,
    events: &std::sync::mpsc::Sender<Event>,
    pending_audio: &mut PendingAudioBuffer,
) -> Result<()> {
    // Platform API-key logins are passed through OPENAI_API_KEY. ChatGPT OAuth stays
    // owned by Codex app-server, which is the supported authentication path for V3 WebRTC.
    let platform_api_key = options
        .chatgpt_account_id
        .is_none()
        .then_some(options.api_key.as_str());
    let connection_started = Instant::now();
    let stage_started = Instant::now();
    let mut server = CodexAppServer::start(platform_api_key)?;
    eprintln!(
        "[live-assistant latency] stage=connection.app_server_spawn elapsed_ms={} total_ms={}",
        stage_started.elapsed().as_millis(),
        connection_started.elapsed().as_millis(),
    );
    let stage_started = Instant::now();
    server
        .call(
            "initialize",
            json!({
                "clientInfo": {
                    "name": "live-assistant",
                    "title": "Live Assistant",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": codex_live_initialize_capabilities()
            }),
        )
        .await?;
    server.notify("initialized", json!({}))?;
    eprintln!(
        "[live-assistant latency] stage=connection.initialize elapsed_ms={} total_ms={} auth={}",
        stage_started.elapsed().as_millis(),
        connection_started.elapsed().as_millis(),
        if options.chatgpt_account_id.is_some() {
            "codex_oauth"
        } else {
            "platform_key"
        },
    );

    let system_prompt = options.system_prompt.clone();
    let realtime_prompt = gpt_live_system_prompt(&system_prompt);
    let codex_prompt = gpt_live_codex_system_prompt(&system_prompt);
    let cwd = std::env::current_dir()
        .context("Could not read the current working directory")?
        .to_string_lossy()
        .into_owned();
    let thread_start_params = codex_live_thread_start_params(&options, codex_prompt, cwd);
    let stage_started = Instant::now();
    let thread = server.call("thread/start", thread_start_params).await?;
    eprintln!(
        "[live-assistant latency] stage=connection.thread_start elapsed_ms={} total_ms={} reasoning={}",
        stage_started.elapsed().as_millis(),
        connection_started.elapsed().as_millis(),
        CODEX_LIVE_DELEGATED_REASONING_EFFORT,
    );
    let thread_id = thread
        .pointer("/thread/id")
        .and_then(Value::as_str)
        .context("Codex app-server did not return a thread id")?
        .to_owned();

    let stage_started = Instant::now();
    let (mut peer, offer_sdp) = GptLivePeer::create().await?;
    eprintln!(
        "[live-assistant latency] stage=connection.webrtc_offer elapsed_ms={} total_ms={}",
        stage_started.elapsed().as_millis(),
        connection_started.elapsed().as_millis(),
    );
    let stage_started = Instant::now();
    server
        .call(
            "thread/realtime/start",
            codex_live_realtime_start_params(
                &thread_id,
                offer_sdp,
                &options.voice,
                realtime_prompt,
            ),
        )
        .await?;
    eprintln!(
        "[live-assistant latency] stage=connection.realtime_start_request elapsed_ms={} total_ms={} handoff=automatic_streaming",
        stage_started.elapsed().as_millis(),
        connection_started.elapsed().as_millis(),
    );

    let mut answer_applied = false;
    let mut started = false;
    while !answer_applied || !started {
        let message = tokio::time::timeout(Duration::from_secs(30), server.next_message())
            .await
            .context("Timed out while Codex GPT-Live WebRTC was starting")?
            .context("Codex app-server closed while GPT-Live WebRTC was starting")?;
        match message.get("method").and_then(Value::as_str) {
            Some("thread/realtime/sdp") => {
                let answer = message
                    .pointer("/params/sdp")
                    .and_then(Value::as_str)
                    .context("Codex GPT-Live did not return an SDP answer")?;
                let stage_started = Instant::now();
                peer.accept_answer(answer.to_owned()).await?;
                answer_applied = true;
                eprintln!(
                    "[live-assistant latency] stage=connection.sdp_applied elapsed_ms={} total_ms={}",
                    stage_started.elapsed().as_millis(),
                    connection_started.elapsed().as_millis(),
                );
            }
            Some("thread/realtime/started") => {
                started = true;
                eprintln!(
                    "[live-assistant latency] stage=connection.realtime_started total_ms={}",
                    connection_started.elapsed().as_millis(),
                );
            }
            Some("thread/realtime/error") => {
                let detail = message
                    .pointer("/params/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Codex GPT-Live failed to start");
                peer.close().await;
                bail!("{}", codex_live_start_error(detail));
            }
            Some("thread/realtime/closed") => {
                peer.close().await;
                bail!("Codex GPT-Live closed while starting");
            }
            _ => {}
        }
    }

    match server
        .refresh_mcp_and_wait_for_server("codex_apps", Duration::from_secs(15))
        .await
    {
        Ok(mcp_refresh_duration) => eprintln!(
            "[live-assistant latency] stage=connection.codex_apps_ready elapsed_ms={} total_ms={}",
            mcp_refresh_duration.as_millis(),
            connection_started.elapsed().as_millis(),
        ),
        Err(error) => eprintln!(
            "[live-assistant latency] stage=connection.codex_apps_warmup_failed total_ms={} error={error:#}",
            connection_started.elapsed().as_millis(),
        ),
    }

    // Do not block the live connection on metadata/plugin discovery. Warm the
    // app-server request path opportunistically; its response is queued while
    // the realtime session can already accept speech and tool delegations.
    let _ = server.send_request(
        "thread/read",
        json!({
            "threadId": thread_id,
            "includeTurns": false,
        }),
    );
    eprintln!(
        "[live-assistant latency] stage=connection.ready total_ms={}",
        connection_started.elapsed().as_millis(),
    );

    let mut local_audio = Some(peer.take_local_audio());
    let mut remote_audio = peer.take_remote_audio();
    let _ = events.send(Event::Connected);
    if pending_audio.sample_count() > 0 {
        eprintln!(
            "[live-assistant mic] flushing_preconnect_seconds={:.2} backend=gpt-live",
            pending_audio.sample_count() as f64 / AUDIO_SAMPLE_RATE as f64
        );
    }
    while let Some(samples) = pending_audio.pop_front() {
        if let Err(error) = peer.send_pcm24k(&samples).await {
            pending_audio.restore_front(samples);
            return Err(error);
        }
    }
    let mut state = CodexLiveState::default();
    let mut native_input_audio = NativeInputAudioCapture::default();
    let mut native_remote_audio =
        crate::gpt_live_webrtc::uses_platform_audio().then(NativeRemoteAudioGate::default);
    let mut handoff_state = CodexHandoffState::default();
    let mut latency_trace = GptLiveToolLatencyTrace::default();
    let mut in_flight_context_images = HashMap::<u64, InFlightContextImage>::new();
    let mut latest_context_image_upload_id = 0_u64;
    let mut latest_ready_context_image_upload_id: Option<u64> = None;
    let mut pending_dynamic_tools: HashMap<String, PendingDynamicTool> = HashMap::new();
    let mut response_watchdog: Option<Instant> = None;
    // The WebRTC receive task filters continuous comfort noise and emits only
    // reordered speech packets. Start UI/playback from the first real audio
    // packet instead of waiting for the slower sideband transcript notification.
    let mut finish_tick = tokio::time::interval(Duration::from_millis(50));
    finish_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(Command::AudioChunk(samples)) => {
                        if !samples.is_empty() {
                            peer.send_pcm24k(&samples).await?;
                        }
                    }
                    Some(Command::SendTurn {
                        text,
                        attachments,
                        ..
                    }) => {
                        let has_image = attachments
                            .iter()
                            .any(|attachment| matches!(attachment, Attachment::Image { .. }));
                        if has_image {
                            let input = codex_turn_input(&text, &attachments);
                            server
                                .call(
                                    "turn/start",
                                    json!({
                                        "threadId": thread_id,
                                        "input": input,
                                    }),
                                )
                                .await?;
                        } else if !text.trim().is_empty() {
                            server
                                .call(
                                    "thread/realtime/appendText",
                                    json!({
                                        "threadId": thread_id,
                                        "text": text.trim(),
                                        "role": "user"
                                    }),
                                )
                                .await?;
                        }
                        for attachment in attachments {
                            if let Attachment::Audio { pcm24k, .. } = attachment
                                && !pcm24k.is_empty()
                            {
                                peer.send_pcm24k(&pcm24k).await?;
                            }
                        }
                    }
                    Some(Command::SendContextImage { upload_id, image, deadline }) => {
                        let becomes_latest = upload_id >= latest_context_image_upload_id;
                        if becomes_latest {
                            latest_context_image_upload_id = upload_id;
                            // A newer capture supersedes every older ready image as soon
                            // as it is requested. Keep the live model away from stale
                            // visual context while the JPEG is being injected.
                            latest_ready_context_image_upload_id = None;
                            if let Err(error) = server.send_request(
                                "thread/realtime/appendText",
                                gpt_live_context_image_pending_params(&thread_id, upload_id),
                            ) {
                                eprintln!(
                                    "[live-assistant image] GPT-Live pending notice failed upload_id={upload_id}: {error:#}"
                                );
                            }
                        }
                        if Instant::now() >= deadline {
                            let _ = events.send(Event::ContextImageUploadFailed {
                                upload_id,
                                detail: context_image_timeout_detail(),
                            });
                            if upload_id == latest_context_image_upload_id {
                                let _ = server.send_request(
                                    "thread/realtime/appendText",
                                    gpt_live_context_image_failed_params(&thread_id, upload_id),
                                );
                            }
                            continue;
                        }
                        if let Attachment::Image {
                            name,
                            width,
                            height,
                            byte_size,
                            ..
                        } = &image
                        {
                            eprintln!(
                                "[live-assistant image] queued upload_id={upload_id} name={name:?} size={}x{} bytes={}",
                                width, height, byte_size,
                            );
                        }
                        let params = match codex_context_image_inject_params(
                            &thread_id,
                            upload_id,
                            &image,
                        ) {
                            Ok(params) => params,
                            Err(error) => {
                                let _ = events.send(Event::ContextImageUploadFailed {
                                    upload_id,
                                    detail: format!("{error:#}"),
                                });
                                if upload_id == latest_context_image_upload_id {
                                    let _ = server.send_request(
                                        "thread/realtime/appendText",
                                        gpt_live_context_image_failed_params(&thread_id, upload_id),
                                    );
                                }
                                continue;
                            }
                        };
                        let request_id = match server.send_request("thread/inject_items", params) {
                            Ok(request_id) => request_id,
                            Err(error) => {
                                let _ = events.send(Event::ContextImageUploadFailed {
                                    upload_id,
                                    detail: format!("{error:#}"),
                                });
                                if upload_id == latest_context_image_upload_id {
                                    let _ = server.send_request(
                                        "thread/realtime/appendText",
                                        gpt_live_context_image_failed_params(&thread_id, upload_id),
                                    );
                                }
                                continue;
                            }
                        };
                        let (name, width, height, byte_size) = image_metadata(&image)?;
                        in_flight_context_images.insert(
                            request_id,
                            InFlightContextImage {
                                upload_id,
                                name,
                                width,
                                height,
                                byte_size,
                                turn_id: "thread context".to_owned(),
                                deadline,
                            },
                        );
                        let _ = events.send(Event::ContextImageAccepted { upload_id });
                    }
                    Some(Command::CreateResponse) => {
                        // Frameless GPT-Live owns output turn creation. Automatic
                        // app-server handoffs stream delegated results directly.
                        response_watchdog =
                            Some(Instant::now() + Duration::from_millis(4_000));
                    }
                    Some(Command::TruncateAssistant { .. }) => {
                        // GPT-Live cancels output itself when new speech begins.
                    }
                    Some(Command::ToolOutputs(outputs)) => {
                        let mut submitted = 0usize;
                        for output in outputs {
                            let Some(pending) = pending_dynamic_tools.remove(&output.call_id) else {
                                continue;
                            };
                            let (content_items, success) =
                                dynamic_tool_content_items(&output.output);
                            let respond_started = Instant::now();
                            eprintln!(
                                "[live-assistant latency] call_id={} name={} stage=local.result_ready total_ms={} success={}",
                                output.call_id,
                                pending.name,
                                pending.received_at.elapsed().as_millis(),
                                success,
                            );
                            server.respond(
                                pending.request_id,
                                json!({
                                    "contentItems": content_items,
                                    "success": success
                                }),
                            )?;
                            eprintln!(
                                "[live-assistant latency] call_id={} name={} stage=app_server.result_submitted submit_ms={} total_ms={}",
                                output.call_id,
                                pending.name,
                                respond_started.elapsed().as_millis(),
                                pending.received_at.elapsed().as_millis(),
                            );
                            if pending_dynamic_tools.is_empty()
                                && let Some(text) =
                                    local_tool_fast_speech(&pending.name, &output.output, success)
                            {
                                let turn_id = handoff_state.active_turn_id.clone();
                                send_fast_tool_speech(
                                    &mut server,
                                    &thread_id,
                                    turn_id.as_deref(),
                                    text,
                                )?;
                                latency_trace.log(
                                    "fast_result.local_speech",
                                    &format!(
                                        "call_id={} name={} total_ms={}",
                                        output.call_id,
                                        pending.name,
                                        pending.received_at.elapsed().as_millis(),
                                    ),
                                );
                                handoff_state.clear_turn();
                                handoff_state.fallback = None;
                                response_watchdog = None;
                            }
                            submitted = submitted.saturating_add(1);
                        }
                        let _ = events.send(Event::ToolOutputsSubmitted { count: submitted });
                    }
                    Some(Command::Disconnect) | Some(Command::Shutdown) | None => {
                        let _ = server.send_request(
                            "thread/realtime/stop",
                            json!({"threadId": thread_id}),
                        );
                        peer.close().await;
                        return Ok(());
                    }
                    Some(Command::Connect(_)) => {}
                }
            }
            audio = async {
                match local_audio.as_mut() {
                    Some(receiver) => receiver.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match audio {
                    Some(Ok(samples)) => native_input_audio.push(
                        samples,
                        state.input_item_id.is_some(),
                        events,
                    ),
                    Some(Err(detail)) => {
                        let _ = events.send(Event::Error(detail));
                    }
                    None => local_audio = None,
                }
            }
            audio = remote_audio.recv() => {
                match audio {
                    Some(Ok(samples)) if !samples.is_empty() => {
                        let chunks = if let Some(gate) = &mut native_remote_audio {
                            gate.push(samples)
                        } else {
                            vec![samples]
                        };
                        for samples in chunks {
                            response_watchdog = None;
                            handoff_state.note_spoken_output();
                            emit_codex_live_remote_audio(&mut state, events, samples);
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(detail)) => {
                        let _ = events.send(Event::Error(detail));
                    }
                    None => bail!("GPT-Live WebRTC audio transport closed"),
                }
            }
            message = server.next_message() => {
                let Some(message) = message else {
                    peer.close().await;
                    bail!("Codex app-server closed unexpectedly");
                };
                eprintln!("[live-assistant codex] {}", codex_message_summary(&message));
                latency_trace.observe(&message);
                match handle_codex_context_image_response(
                    &message,
                    &mut in_flight_context_images,
                    events,
                ) {
                    CodexContextImageResponse::NotHandled => {}
                    CodexContextImageResponse::Uploaded(upload_id) => {
                        if upload_id == latest_context_image_upload_id {
                            latest_ready_context_image_upload_id = Some(upload_id);
                        }
                        if let Some(ready_upload_id) = take_latest_ready_codex_context_image(
                            &mut latest_ready_context_image_upload_id,
                            latest_context_image_upload_id,
                            &in_flight_context_images,
                        ) && let Err(error) = server.send_request(
                            "thread/realtime/appendText",
                            gpt_live_context_image_ready_params(&thread_id, ready_upload_id),
                        ) {
                            eprintln!(
                                "[live-assistant image] GPT-Live ready notice failed upload_id={ready_upload_id}: {error:#}"
                            );
                        }
                        continue;
                    }
                    CodexContextImageResponse::Failed(upload_id) => {
                        if upload_id == latest_context_image_upload_id {
                            latest_ready_context_image_upload_id = None;
                            if let Err(error) = server.send_request(
                                "thread/realtime/appendText",
                                gpt_live_context_image_failed_params(&thread_id, upload_id),
                            ) {
                                eprintln!(
                                    "[live-assistant image] GPT-Live failed notice could not be sent upload_id={upload_id}: {error:#}"
                                );
                            }
                        } else if let Some(ready_upload_id) =
                            take_latest_ready_codex_context_image(
                                &mut latest_ready_context_image_upload_id,
                                latest_context_image_upload_id,
                                &in_flight_context_images,
                            )
                            && let Err(error) = server.send_request(
                                "thread/realtime/appendText",
                                gpt_live_context_image_ready_params(&thread_id, ready_upload_id),
                            )
                        {
                            eprintln!(
                                "[live-assistant image] GPT-Live ready notice failed upload_id={ready_upload_id}: {error:#}"
                            );
                        }
                        continue;
                    }
                }
                if let Some(fast) = mcp_tool_fast_speech(&message) {
                    send_fast_tool_speech(
                        &mut server,
                        &thread_id,
                        Some(&fast.turn_id),
                        fast.text,
                    )?;
                    latency_trace.log(
                        "fast_result.mcp_speech",
                        &format!("label={} turn_id={}", fast.label, fast.turn_id),
                    );
                    handoff_state.clear_turn();
                    handoff_state.fallback = None;
                    response_watchdog = None;
                    continue;
                }
                if codex_message_starts_reply(&message) {
                    response_watchdog = None;
                }
                if codex_message_is_assistant_transcript(&message) {
                    handoff_state.note_spoken_output();
                    state.ensure_response(events);
                }
                if let Some((request_id, call)) = dynamic_tool_request(&message) {
                    eprintln!(
                        "[live-assistant tool] request call_id={} name={} arguments={}",
                        call.call_id, call.name, call.arguments
                    );
                    latency_trace.ensure_started("dynamic_tool.request");
                    latency_trace.log(
                        "dynamic_tool.request",
                        &format!("call_id={} name={}", call.call_id, call.name),
                    );
                    pending_dynamic_tools.insert(
                        call.call_id.clone(),
                        PendingDynamicTool {
                            request_id,
                            name: call.name.clone(),
                            received_at: call.requested_at,
                        },
                    );
                    let _ = events.send(Event::ToolCalls(vec![call]));
                    continue;
                }
                match handle_codex_handoff_message(
                    &message,
                    events,
                    &mut handoff_state,
                )? {
                    CodexHandoffAction::NotHandled => {}
                    CodexHandoffAction::Handled => continue,
                }
                handle_codex_live_message(&message, events, &mut state)?;
                native_input_audio.sync(state.input_item_id.is_some(), events);
            }
            _ = finish_tick.tick() => {
                let now = Instant::now();
                if let Some(text) = handoff_state.take_fallback_if_due(now) {
                    latency_trace.log(
                        "handoff.fallback_append_speech",
                        &format!("chars={}", text.chars().count()),
                    );
                    server.send_request(
                        "thread/realtime/appendSpeech",
                        json!({
                            "threadId": thread_id,
                            "text": text,
                        }),
                    )?;
                }
                let expired_upload_ids = expire_codex_context_images(
                    &mut in_flight_context_images,
                    now,
                    events,
                );
                if expired_upload_ids.contains(&latest_context_image_upload_id) {
                    latest_ready_context_image_upload_id = None;
                    if let Err(error) = server.send_request(
                        "thread/realtime/appendText",
                        gpt_live_context_image_failed_params(
                            &thread_id,
                            latest_context_image_upload_id,
                        ),
                    ) {
                        eprintln!(
                            "[live-assistant image] GPT-Live timeout notice failed upload_id={latest_context_image_upload_id}: {error:#}"
                        );
                    }
                } else if let Some(ready_upload_id) = take_latest_ready_codex_context_image(
                    &mut latest_ready_context_image_upload_id,
                    latest_context_image_upload_id,
                    &in_flight_context_images,
                ) && let Err(error) = server.send_request(
                    "thread/realtime/appendText",
                    gpt_live_context_image_ready_params(&thread_id, ready_upload_id),
                ) {
                    eprintln!(
                        "[live-assistant image] GPT-Live ready notice failed upload_id={ready_upload_id}: {error:#}"
                    );
                }
                if state.finish_response_if_due(now, events) {
                    eprintln!("[live-assistant reply] finalized GPT-Live response after quiet speech tail");
                }
                if response_watchdog.is_some_and(|deadline| now >= deadline) {
                    response_watchdog = None;
                    eprintln!(
                        "[live-assistant reply] no GPT-Live output or delegation after completed user turn"
                    );
                }
            }
        }
    }
}

#[derive(Default)]
struct CodexTextState {
    response_number: u64,
    active_response_id: Option<String>,
    assistant_text: String,
}

impl CodexTextState {
    fn ensure_response(
        &mut self,
        hint: Option<&str>,
        events: &std::sync::mpsc::Sender<Event>,
    ) -> String {
        if let Some(response_id) = &self.active_response_id
            && hint.is_none_or(|hint| hint.is_empty() || hint == response_id)
        {
            return response_id.clone();
        }
        self.response_number = self.response_number.saturating_add(1);
        let response_id = hint
            .filter(|hint| !hint.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("codex-text-{}", self.response_number));
        self.active_response_id = Some(response_id.clone());
        self.assistant_text.clear();
        let _ = events.send(Event::AssistantResponseStarted {
            response_id: response_id.clone(),
        });
        response_id
    }

    fn emit_text(
        &mut self,
        response_id: String,
        text: &str,
        events: &std::sync::mpsc::Sender<Event>,
    ) {
        if text.is_empty() {
            return;
        }
        let delta = if text.starts_with(&self.assistant_text) {
            text[self.assistant_text.len()..].to_owned()
        } else {
            text.to_owned()
        };
        if delta.is_empty() {
            return;
        }
        if text.starts_with(&self.assistant_text) {
            self.assistant_text = text.to_owned();
        } else {
            self.assistant_text.push_str(&delta);
        }
        let _ = events.send(Event::AssistantTranscriptDelta { response_id, delta });
    }

    fn finish(&mut self, events: &std::sync::mpsc::Sender<Event>) {
        if let Some(response_id) = self.active_response_id.take() {
            let _ = events.send(Event::AssistantDone { response_id });
        }
        self.assistant_text.clear();
    }
}

/// Run a normal Codex model through app-server. Text tabs use the same dynamic
/// computer tools as the live backends; only the transport and response events
/// differ from GPT-Live's WebRTC session.
async fn run_codex_text_connection(
    options: ConnectOptions,
    commands: &mut UnboundedReceiver<Command>,
    events: &std::sync::mpsc::Sender<Event>,
) -> Result<()> {
    let platform_api_key = options
        .chatgpt_account_id
        .is_none()
        .then_some(options.api_key.as_str());
    let mut server = CodexAppServer::start(platform_api_key)?;
    server
        .call(
            "initialize",
            json!({
                "clientInfo": {
                    "name": "live-assistant",
                    "title": "Live Assistant",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {"experimentalApi": true}
            }),
        )
        .await?;
    server.notify("initialized", json!({}))?;

    let thread = server
        .call(
            "thread/start",
            codex_text_thread_start_params(
                &options,
                options.system_prompt.clone(),
                std::env::current_dir()
                    .context("Could not read the current working directory")?
                    .to_string_lossy()
                    .into_owned(),
            ),
        )
        .await?;
    let thread_id = thread
        .pointer("/thread/id")
        .and_then(Value::as_str)
        .context("Codex app-server did not return a thread id")?
        .to_owned();

    let _ = events.send(Event::Connected);
    let mut state = CodexTextState::default();
    let mut pending_dynamic_tools: HashMap<String, Value> = HashMap::new();
    let mut in_flight_context_images = HashMap::<u64, InFlightContextImage>::new();
    let mut context_image_tick = tokio::time::interval(Duration::from_millis(50));
    context_image_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(Command::SendTurn {
                        text,
                        attachments,
                        thinking_level,
                    }) => {
                        if text.trim().is_empty() && attachments.is_empty() {
                            continue;
                        }
                        server
                            .call(
                                "turn/start",
                                codex_text_turn_start_params(
                                    &thread_id,
                                    &text,
                                    &attachments,
                                    &thinking_level,
                                ),
                            )
                            .await?;
                    }
                    Some(Command::SendContextImage { upload_id, image, deadline }) => {
                        if Instant::now() >= deadline {
                            let _ = events.send(Event::ContextImageUploadFailed {
                                upload_id,
                                detail: context_image_timeout_detail(),
                            });
                            continue;
                        }
                        let params = match codex_context_image_inject_params(&thread_id, upload_id, &image) {
                            Ok(params) => params,
                            Err(error) => {
                                let _ = events.send(Event::ContextImageUploadFailed {
                                    upload_id,
                                    detail: format!("{error:#}"),
                                });
                                continue;
                            }
                        };
                        let request_id = server.send_request("thread/inject_items", params)?;
                        let (name, width, height, byte_size) = image_metadata(&image)?;
                        in_flight_context_images.insert(
                            request_id,
                            InFlightContextImage {
                                upload_id,
                                name,
                                width,
                                height,
                                byte_size,
                                turn_id: "text thread".to_owned(),
                                deadline,
                            },
                        );
                        let _ = events.send(Event::ContextImageAccepted { upload_id });
                    }
                    Some(Command::ToolOutputs(outputs)) => {
                        let mut submitted = 0usize;
                        for output in outputs {
                            let Some(request_id) = pending_dynamic_tools.remove(&output.call_id) else {
                                continue;
                            };
                            let (content_items, success) =
                                dynamic_tool_content_items(&output.output);
                            server.respond(
                                request_id,
                                json!({
                                    "contentItems": content_items,
                                    "success": success
                                }),
                            )?;
                            submitted = submitted.saturating_add(1);
                        }
                        let _ = events.send(Event::ToolOutputsSubmitted { count: submitted });
                    }
                    Some(Command::AudioChunk(_))
                    | Some(Command::CreateResponse)
                    | Some(Command::TruncateAssistant { .. }) => {}
                    Some(Command::Disconnect) | Some(Command::Shutdown) | None => return Ok(()),
                    Some(Command::Connect(_)) => {}
                }
            }
            message = server.next_message() => {
                let Some(message) = message else {
                    bail!("Codex app-server closed during text session");
                };
                eprintln!("[live-assistant codex-text] {}", codex_message_summary(&message));
                match handle_codex_context_image_response(
                    &message,
                    &mut in_flight_context_images,
                    events,
                ) {
                    CodexContextImageResponse::NotHandled => {}
                    CodexContextImageResponse::Uploaded(_)
                    | CodexContextImageResponse::Failed(_) => continue,
                }
                if let Some((request_id, call)) = dynamic_tool_request(&message) {
                    eprintln!(
                        "[live-assistant tool] text call_id={} name={} arguments={}",
                        call.call_id, call.name, call.arguments
                    );
                    pending_dynamic_tools.insert(call.call_id.clone(), request_id);
                    let _ = events.send(Event::ToolCalls(vec![call]));
                    continue;
                }
                handle_codex_text_message(&message, events, &mut state)?;
            }
            _ = context_image_tick.tick() => {
                expire_codex_context_images(&mut in_flight_context_images, Instant::now(), events);
            }
        }
    }
}

fn codex_text_thread_start_params(
    options: &ConnectOptions,
    system_prompt: String,
    cwd: String,
) -> Value {
    json!({
        "cwd": cwd,
        "ephemeral": true,
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "model": options.model,
        "baseInstructions": system_prompt,
        "dynamicTools": codex_dynamic_tools(options.screen_info),
        "reasoningEffort": options.thinking_level,
        "config": {
            "suppress_unstable_features_warning": true,
        }
    })
}

fn codex_text_turn_start_params(
    thread_id: &str,
    text: &str,
    attachments: &[Attachment],
    thinking_level: &str,
) -> Value {
    json!({
        "threadId": thread_id,
        "input": codex_turn_input(text, attachments),
        "effort": thinking_level,
    })
}

fn handle_codex_text_message(
    message: &Value,
    events: &std::sync::mpsc::Sender<Event>,
    state: &mut CodexTextState,
) -> Result<()> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match method {
        "turn/started" | "turn/created" => {
            let response_id = codex_text_response_hint(message);
            state.ensure_response(response_id.as_deref(), events);
        }
        "item/started" => {
            if codex_text_message_is_assistant(message) {
                let response_id =
                    state.ensure_response(codex_text_response_hint(message).as_deref(), events);
                if let Some(item_id) = codex_text_item_id(message) {
                    let _ = events.send(Event::AssistantItem {
                        response_id,
                        item_id,
                    });
                }
            }
        }
        "item/agentMessage/delta"
        | "item/assistantMessage/delta"
        | "item/message/delta"
        | "item/delta" => {
            if codex_text_message_is_assistant(message)
                && let Some(text) = codex_text_message_text(message)
            {
                let response_id =
                    state.ensure_response(codex_text_response_hint(message).as_deref(), events);
                state.emit_text(response_id, text, events);
            }
        }
        "item/completed" | "item/agentMessage/completed" | "item/assistantMessage/completed" => {
            if codex_text_message_is_assistant(message)
                && let Some(text) = codex_text_message_text(message)
            {
                let response_id =
                    state.ensure_response(codex_text_response_hint(message).as_deref(), events);
                state.emit_text(response_id, text, events);
            }
        }
        "turn/completed" | "turn/finished" | "turn/stopped" => {
            let status = message
                .pointer("/params/turn/status")
                .or_else(|| message.pointer("/params/status"))
                .and_then(Value::as_str)
                .unwrap_or("completed");
            if !matches!(status, "completed" | "stopped" | "interrupted") {
                let detail = codex_text_error_detail(message);
                let _ = events.send(Event::Error(detail));
            }
            if let Some(total_tokens) = response_total_tokens(message)
                && let Some(response_id) = state
                    .active_response_id
                    .clone()
                    .or_else(|| codex_text_response_hint(message))
            {
                let _ = events.send(Event::AssistantUsage {
                    response_id,
                    total_tokens,
                });
            }
            state.finish(events);
        }
        "error" | "turn/failed" | "item/failed" => {
            let detail = codex_text_error_detail(message);
            let _ = events.send(Event::Error(detail));
        }
        _ => {}
    }
    Ok(())
}

fn codex_text_response_hint(message: &Value) -> Option<String> {
    [
        "/params/turnId",
        "/params/responseId",
        "/params/turn/id",
        "/params/turn/turnId",
        "/params/item/turnId",
    ]
    .into_iter()
    .find_map(|pointer| {
        message
            .pointer(pointer)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
    })
}

fn usage_field(usage: &Value, names: &[&str]) -> Option<u64> {
    names.iter().find_map(|name| {
        usage.get(*name).and_then(Value::as_u64).or_else(|| {
            usage
                .get(*name)
                .and_then(Value::as_i64)
                .map(|value| value.max(0) as u64)
        })
    })
}

fn total_tokens_from_usage(usage: &Value) -> Option<u64> {
    if let Some(total) = usage_field(usage, &["total_tokens", "totalTokens", "total", "tokens"]) {
        return Some(total);
    }
    let input = usage_field(
        usage,
        &[
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
        ],
    );
    let output = usage_field(
        usage,
        &[
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ],
    );
    input.or(output).map(|_| {
        input
            .unwrap_or_default()
            .saturating_add(output.unwrap_or_default())
    })
}

fn response_total_tokens(message: &Value) -> Option<u64> {
    [
        "/response/usage",
        "/params/response/usage",
        "/params/turn/usage",
        "/params/turn/tokenUsage",
        "/params/usage",
        "/params/tokenUsage",
        "/usage",
    ]
    .into_iter()
    .find_map(|pointer| message.pointer(pointer).and_then(total_tokens_from_usage))
}

fn codex_text_item_id(message: &Value) -> Option<String> {
    ["/params/itemId", "/params/item/id", "/params/id"]
        .into_iter()
        .find_map(|pointer| {
            message
                .pointer(pointer)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
        })
}

fn codex_text_message_is_assistant(message: &Value) -> bool {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if matches!(
        method,
        "item/agentMessage/delta"
            | "item/assistantMessage/delta"
            | "item/message/delta"
            | "item/delta"
    ) {
        return true;
    }
    let role = message
        .pointer("/params/role")
        .or_else(|| message.pointer("/params/item/role"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if role.eq_ignore_ascii_case("assistant") {
        return true;
    }
    let item_type = message
        .pointer("/params/item/type")
        .or_else(|| message.pointer("/params/type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    item_type.contains("agentmessage")
        || item_type.contains("assistantmessage")
        || item_type == "output_text"
}

fn codex_text_message_text(message: &Value) -> Option<&str> {
    [
        "/params/delta",
        "/params/text",
        "/params/item/text",
        "/params/item/message/text",
        "/params/item/content/0/text",
        "/params/item/content/0/value",
    ]
    .into_iter()
    .find_map(|pointer| message.pointer(pointer).and_then(Value::as_str))
}

fn codex_text_error_detail(message: &Value) -> String {
    message
        .pointer("/params/message")
        .or_else(|| message.pointer("/params/turn/error/message"))
        .or_else(|| message.pointer("/params/error/message"))
        .or_else(|| message.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Codex text session reported {message}"))
}

fn codex_live_thread_start_params(
    options: &ConnectOptions,
    system_prompt: String,
    cwd: String,
) -> Value {
    json!({
        "cwd": cwd,
        "ephemeral": true,
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "baseInstructions": system_prompt,
        "dynamicTools": codex_dynamic_tools_with_tools(
            voice_tools(options.screen_info),
        ),
        "reasoningEffort": CODEX_LIVE_DELEGATED_REASONING_EFFORT,
        "config": {
            "features.realtime_conversation": true,
            "suppress_unstable_features_warning": true,
            // GPT-Live connector turns do not need the local Node REPL or
            // OpenAI documentation MCP. Starting them adds several seconds to
            // every small calendar/app request.
            "mcp_servers.node_repl.enabled": false,
            "mcp_servers.openaiDeveloperDocs.enabled": false,
        }
    })
}

/// Exercises the exact production connection and screenshot command paths for
/// both supported realtime backends. The model must identify a cat in the
/// first voice turn and a dog in a later voice turn on the same connection.
/// This catches sessions that acknowledge the second upload but keep looking
/// at the first screenshot.
pub fn probe_context_image_uploads() -> Result<()> {
    let credentials = crate::auth::codex_credentials()?;
    let images = crate::media::jpeg_animal_probe_attachments()?;
    let mut failures = Vec::new();
    for backend in [
        RealtimeBackend::OpenAiRealtime,
        RealtimeBackend::CodexGptLive,
    ] {
        match probe_context_image_upload_backend(backend, &credentials, images.clone()) {
            Ok((upload_times, replies)) => eprintln!(
                "[image-upload probe] backend={backend:?} turn1_upload_ms={} turn1_expected=cat turn1_reply={:?} turn2_upload_ms={} turn2_expected=dog turn2_reply={:?} result=success",
                upload_times[0].as_millis(),
                replies[0],
                upload_times[1].as_millis(),
                replies[1],
            ),
            Err(error) => {
                eprintln!("[image-upload probe] backend={backend:?} result=failed error={error:#}");
                failures.push(format!("{backend:?}: {error:#}"));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("JPEG upload probe failed: {}", failures.join("; "))
    }
}

fn probe_context_image_upload_backend(
    backend: RealtimeBackend,
    credentials: &crate::auth::CodexCredentials,
    images: [Attachment; 2],
) -> Result<([Duration; 2], [String; 2])> {
    let client = RealtimeClient::spawn();
    let screen_info = ScreenInfo {
        origin_x: 0,
        origin_y: 0,
        logical_width: 1_280,
        logical_height: 800,
        backing_width: 1_280,
        backing_height: 800,
        scale_factor: 1.0,
    };
    let options = ConnectOptions {
        backend,
        api_key: credentials.bearer_token.clone(),
        chatgpt_account_id: credentials.chatgpt_account_id.clone(),
        model: "gpt-realtime-2.1".to_owned(),
        voice: match backend {
            RealtimeBackend::OpenAiRealtime => "marin",
            RealtimeBackend::CodexGptLive => "ember",
            RealtimeBackend::CodexText => {
                bail!("The JPEG upload probe only supports voice backends")
            }
        }
        .to_owned(),
        thinking_level: "low".to_owned(),
        system_prompt: shared_system_prompt(
            "This is an automated multi-turn latest-screen test. Do not respond when a screen image is added. Each time the user asks what animal is visible, inspect only the highest-numbered screen capture and answer with only the lowercase English animal name.",
            screen_info,
        ),
        screen_info,
    };

    let result = (|| -> Result<([Duration; 2], [String; 2])> {
        client
            .commands
            .send(Command::Connect(options))
            .context("Could not start the realtime JPEG upload probe")?;
        let connection_deadline = Instant::now() + Duration::from_secs(45);
        loop {
            let remaining = connection_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("Timed out connecting the JPEG upload probe");
            }
            match client.events.recv_timeout(remaining) {
                Ok(Event::Connected) => break,
                Ok(Event::Error(detail)) => bail!("Could not connect: {detail}"),
                Ok(Event::Disconnected) => bail!("Realtime disconnected before the JPEG probe"),
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    bail!("Timed out connecting the JPEG upload probe")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    bail!("JPEG upload probe event channel closed while connecting")
                }
            }
        }

        let mut upload_times = [Duration::ZERO; 2];
        let mut replies = [String::new(), String::new()];
        for (index, image) in images.into_iter().enumerate() {
            let upload_id = index as u64 + 1;
            upload_times[index] = probe_upload_context_image(&client, upload_id, image)?;

            let question = if index == 0 {
                "Identify the newest screenshot animal"
            } else {
                "Identify the newest screenshot animal now"
            };
            probe_ask_latest_image_question(&client, backend, question)?;
            let (expected, stale) = if index == 0 {
                ("cat", "dog")
            } else {
                ("dog", "cat")
            };
            replies[index] =
                probe_wait_for_animal_reply(&client, backend, expected, stale, upload_id)?;
        }
        Ok((upload_times, replies))
    })();
    let _ = client.commands.send(Command::Disconnect);
    result
}

fn probe_upload_context_image(
    client: &RealtimeClient,
    upload_id: u64,
    image: Attachment,
) -> Result<Duration> {
    let started_at = Instant::now();
    let deadline = started_at + CONTEXT_IMAGE_UPLOAD_TIMEOUT;
    client
        .commands
        .send(Command::SendContextImage {
            upload_id,
            image,
            deadline,
        })
        .with_context(|| format!("Could not enqueue JPEG upload probe #{upload_id}"))?;
    loop {
        let remaining =
            (deadline + Duration::from_secs(1)).saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("JPEG upload #{upload_id} received no confirmation within 10 seconds");
        }
        match client.events.recv_timeout(remaining) {
            Ok(Event::ContextImageUploaded {
                upload_id: confirmed,
            }) if confirmed == upload_id => return Ok(started_at.elapsed()),
            Ok(Event::ContextImageUploadFailed {
                upload_id: failed,
                detail,
            }) if failed == upload_id => bail!("JPEG upload probe #{failed} failed: {detail}"),
            Ok(Event::Error(detail)) => bail!("Realtime failed during JPEG upload: {detail}"),
            Ok(Event::Disconnected) => bail!("Realtime disconnected during JPEG upload"),
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                bail!("JPEG upload #{upload_id} received no confirmation within 10 seconds")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("JPEG upload probe event channel closed")
            }
        }
    }
}

fn probe_ask_latest_image_question(
    client: &RealtimeClient,
    backend: RealtimeBackend,
    question: &str,
) -> Result<()> {
    match backend {
        RealtimeBackend::OpenAiRealtime => client
            .commands
            .send(Command::SendTurn {
                text: question.to_owned(),
                attachments: Vec::new(),
                thinking_level: "low".to_owned(),
            })
            .context("Could not ask OpenAI Realtime to inspect the latest JPEG"),
        RealtimeBackend::CodexGptLive => {
            // GPT-Live's normal user turns are VAD-driven microphone turns;
            // appendText adds context but intentionally does not synthesize a
            // reply. Feed paced speech so the probe follows the production path.
            let samples = synthesize_latest_image_probe_speech(question)?;
            for frame in samples.chunks(480) {
                client
                    .commands
                    .send(Command::AudioChunk(frame.to_vec()))
                    .context("Could not send the GPT-Live latest-image probe speech")?;
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        }
        RealtimeBackend::CodexText => {
            bail!("The latest-image probe only supports voice backends")
        }
    }
}

fn probe_wait_for_animal_reply(
    client: &RealtimeClient,
    backend: RealtimeBackend,
    expected: &str,
    stale: &str,
    upload_id: u64,
) -> Result<String> {
    let reply_deadline = Instant::now() + Duration::from_secs(45);
    let mut active_response_id: Option<String> = None;
    let mut reply = String::new();
    let mut completed_replies = Vec::new();
    loop {
        let remaining = reply_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!(
                "Model did not finish identifying screen capture #{upload_id} within 45 seconds; partial reply was {reply:?}"
            );
        }
        // Production microphone capture keeps the WebRTC RTP clock moving even
        // during silence. Mirror that behavior so GPT-Live can deliver a Codex
        // handoff after the synthesized question has ended.
        let receive_window = match backend {
            RealtimeBackend::CodexGptLive => remaining.min(Duration::from_millis(20)),
            RealtimeBackend::OpenAiRealtime => remaining,
            RealtimeBackend::CodexText => {
                bail!("The animal-reply probe only supports voice backends")
            }
        };
        match client.events.recv_timeout(receive_window) {
            Ok(Event::AssistantResponseStarted { response_id }) => {
                active_response_id = Some(response_id);
                reply.clear();
            }
            Ok(Event::AssistantTranscriptDelta { response_id, delta })
                if active_response_id
                    .as_deref()
                    .is_none_or(|active| active == response_id) =>
            {
                active_response_id.get_or_insert(response_id);
                reply.push_str(&delta);
            }
            Ok(Event::AssistantDone { response_id })
                if active_response_id.as_deref() == Some(response_id.as_str()) =>
            {
                let words = latest_image_probe_words(&reply);
                if words.iter().any(|word| word == expected || word == stale) {
                    break;
                }
                if !reply.trim().is_empty() {
                    completed_replies.push(reply.trim().to_owned());
                }
                active_response_id = None;
                reply.clear();
            }
            Ok(Event::Error(detail)) => {
                bail!("Realtime failed while checking screen capture #{upload_id}: {detail}")
            }
            Ok(Event::Disconnected) => {
                bail!("Realtime disconnected while checking screen capture #{upload_id}")
            }
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                if backend == RealtimeBackend::CodexGptLive && Instant::now() < reply_deadline =>
            {
                client
                    .commands
                    .send(Command::AudioChunk(vec![0_i16; 480]))
                    .context("Could not keep the GPT-Live JPEG probe audio clock active")?;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                bail!(
                    "Model did not finish identifying screen capture #{upload_id} within 45 seconds; non-visual replies were {completed_replies:?}, partial reply was {reply:?}"
                )
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("JPEG ordering probe event channel closed")
            }
        }
    }

    let words = latest_image_probe_words(&reply);
    anyhow::ensure!(
        words.iter().any(|word| word == expected) && !words.iter().any(|word| word == stale),
        "Model used the wrong screenshot for capture #{upload_id}; expected {expected}, stale image was {stale}, non-visual replies were {completed_replies:?}, got {reply:?}"
    );
    Ok(reply)
}

fn latest_image_probe_words(reply: &str) -> Vec<String> {
    reply
        .split(|character: char| !character.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn synthesize_latest_image_probe_speech(text: &str) -> Result<Vec<i16>> {
    let stem = format!("live-assistant-latest-image-probe-{}", std::process::id());
    let probe_dir = std::env::temp_dir();
    let aiff_path = probe_dir.join(format!("{stem}.aiff"));
    let pcm_path = probe_dir.join(format!("{stem}.pcm"));
    let result = (|| -> Result<Vec<i16>> {
        let say_status = ProcessCommand::new("/usr/bin/say")
            .args(["-o", aiff_path.to_string_lossy().as_ref(), text])
            .status()
            .context("Could not synthesize latest-image probe speech")?;
        anyhow::ensure!(
            say_status.success(),
            "macOS say failed for the latest-image probe"
        );
        let ffmpeg_status = ProcessCommand::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-i",
                aiff_path.to_string_lossy().as_ref(),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "24000",
                pcm_path.to_string_lossy().as_ref(),
            ])
            .status()
            .context("Could not convert latest-image probe speech")?;
        anyhow::ensure!(
            ffmpeg_status.success(),
            "ffmpeg failed for the latest-image probe"
        );
        let bytes = std::fs::read(&pcm_path).context("Could not read latest-image probe PCM")?;
        let mut samples = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        anyhow::ensure!(!samples.is_empty(), "Latest-image probe speech was empty");
        // Let server VAD close the utterance exactly as it does after the user
        // stops talking into the real microphone.
        samples.extend(std::iter::repeat_n(0_i16, AUDIO_SAMPLE_RATE * 2));
        Ok(samples)
    })();
    let _ = std::fs::remove_file(aiff_path);
    let _ = std::fs::remove_file(pcm_path);
    result
}

/// End-to-end GPT-Live V3 admission and WebRTC SDP probe used by
/// `cargo run -- --test-gpt-live`.
pub fn probe_codex_gpt_live() -> Result<()> {
    let credentials = crate::auth::codex_credentials()?;
    let runtime = tokio::runtime::Runtime::new().context("Could not create probe runtime")?;
    runtime.block_on(async move {
        let platform_api_key = credentials
            .chatgpt_account_id
            .is_none()
            .then_some(credentials.bearer_token.as_str());
        let mut server = CodexAppServer::start(platform_api_key)?;
        server
            .call(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "live-assistant-probe",
                        "title": "Live Assistant GPT-Live Probe",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": codex_live_initialize_capabilities()
                }),
            )
            .await?;
        server.notify("initialized", json!({}))?;
        let cwd = std::env::current_dir()
            .context("Could not read the current working directory")?
            .to_string_lossy()
            .into_owned();
        let thread = server
            .call(
                "thread/start",
                json!({
                    "cwd": cwd,
                    "ephemeral": true,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "baseInstructions": "This is an audio transport test. Follow duration and counting instructions exactly; do not abbreviate or stop early.",
                    "config": {
                        "features.realtime_conversation": true,
                        "suppress_unstable_features_warning": true
                    }
                }),
            )
            .await?;
        let thread_id = thread
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .context("Codex app-server did not return a probe thread id")?
            .to_owned();
        let (peer, offer_sdp) = GptLiveProbePeer::create().await?;
        server
            .call(
                "thread/realtime/start",
                json!({
                    "threadId": thread_id,
                    "outputModality": "audio",
                    "version": "v3",
                    "model": "gpt-live-1-boulder-alpha",
                    "voice": "ember",
                    "transport": {"type": "webrtc", "sdp": offer_sdp},
                    "clientManagedHandoffs": false,
                    "codexResponsesAsItems": false,
                    "codexResponseHandoffMode": "bemTags",
                    "includeStartupContext": false,
                    "prompt": "This is an audio transport test. Speak every requested number slowly and do not stop early."
                }),
            )
            .await?;
        let mut answer_applied = false;
        let mut started = false;
        while !answer_applied || !started {
            let message = tokio::time::timeout(Duration::from_secs(45), server.next_message())
                .await
                .context("Timed out probing GPT-Live WebRTC")?
                .context("Codex app-server closed during the GPT-Live probe")?;
            match message.get("method").and_then(Value::as_str) {
                Some("thread/realtime/sdp") => {
                    let answer = message
                        .pointer("/params/sdp")
                        .and_then(Value::as_str)
                        .context("GPT-Live probe did not receive an SDP answer")?;
                    peer.accept_answer(answer.to_owned()).await?;
                    answer_applied = true;
                }
                Some("thread/realtime/started") => started = true,
                Some("thread/realtime/error") => {
                    let detail = message
                        .pointer("/params/message")
                        .and_then(Value::as_str)
                        .unwrap_or("Unknown GPT-Live probe error");
                    peer.close().await;
                    bail!("{}", codex_live_start_error(detail));
                }
                Some("thread/realtime/closed") => {
                    peer.close().await;
                    bail!("GPT-Live closed during the admission probe");
                }
                _ => {}
            }
        }
        let mut peer = peer;
        let mut remote_audio = peer.take_remote_audio();
        let probe_dir = std::env::temp_dir();
        let aiff_path = probe_dir.join("live-assistant-gpt-live-probe.aiff");
        let pcm_path = probe_dir.join("live-assistant-gpt-live-probe.pcm");
        let say_status = ProcessCommand::new("/usr/bin/say")
            .args([
                "-o",
                aiff_path.to_string_lossy().as_ref(),
                "Count slowly from one through twenty. Say every number in order, pause slightly between numbers, and do not stop before twenty.",
            ])
            .status()
            .context("Could not synthesize GPT-Live probe speech")?;
        anyhow::ensure!(say_status.success(), "macOS say failed for GPT-Live probe");
        let ffmpeg_status = ProcessCommand::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-i",
                aiff_path.to_string_lossy().as_ref(),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "24000",
                pcm_path.to_string_lossy().as_ref(),
            ])
            .status()
            .context("Could not convert GPT-Live probe speech")?;
        anyhow::ensure!(ffmpeg_status.success(), "ffmpeg failed for GPT-Live probe");
        let bytes = std::fs::read(&pcm_path).context("Could not read GPT-Live probe PCM")?;

        let overlap_aiff_path = probe_dir.join("live-assistant-gpt-live-overlap.aiff");
        let overlap_pcm_path = probe_dir.join("live-assistant-gpt-live-overlap.pcm");
        let overlap_say_status = ProcessCommand::new("/usr/bin/say")
            .args([
                "-o",
                overlap_aiff_path.to_string_lossy().as_ref(),
                "Purple banana duplex test. I am speaking while you are speaking.",
            ])
            .status()
            .context("Could not synthesize overlapping GPT-Live probe speech")?;
        anyhow::ensure!(
            overlap_say_status.success(),
            "macOS say failed for overlapping GPT-Live probe"
        );
        let overlap_ffmpeg_status = ProcessCommand::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-i",
                overlap_aiff_path.to_string_lossy().as_ref(),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "24000",
                overlap_pcm_path.to_string_lossy().as_ref(),
            ])
            .status()
            .context("Could not convert overlapping GPT-Live probe speech")?;
        anyhow::ensure!(
            overlap_ffmpeg_status.success(),
            "ffmpeg failed for overlapping GPT-Live probe"
        );
        let overlap_bytes = std::fs::read(&overlap_pcm_path)
            .context("Could not read overlapping GPT-Live probe PCM")?;
        let mut overlap_samples = overlap_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        overlap_samples.extend(std::iter::repeat_n(0_i16, 24_000));

        let followup_aiff_path = probe_dir.join("live-assistant-gpt-live-followup.aiff");
        let followup_pcm_path = probe_dir.join("live-assistant-gpt-live-followup.pcm");
        let followup_say_status = ProcessCommand::new("/usr/bin/say")
            .args([
                "-o",
                followup_aiff_path.to_string_lossy().as_ref(),
                "Second turn reliability check. Please answer with the words microphone stays active.",
            ])
            .status()
            .context("Could not synthesize follow-up GPT-Live probe speech")?;
        anyhow::ensure!(
            followup_say_status.success(),
            "macOS say failed for follow-up GPT-Live probe"
        );
        let followup_ffmpeg_status = ProcessCommand::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-i",
                followup_aiff_path.to_string_lossy().as_ref(),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "24000",
                followup_pcm_path.to_string_lossy().as_ref(),
            ])
            .status()
            .context("Could not convert follow-up GPT-Live probe speech")?;
        anyhow::ensure!(
            followup_ffmpeg_status.success(),
            "ffmpeg failed for follow-up GPT-Live probe"
        );
        let followup_bytes = std::fs::read(&followup_pcm_path)
            .context("Could not read follow-up GPT-Live probe PCM")?;
        let mut followup_samples = followup_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        followup_samples.extend(std::iter::repeat_n(0_i16, 24_000 * 2));

        let mut samples = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        samples.extend(std::iter::repeat_n(0_i16, 24_000 * 2));
        for frame in samples.chunks(480) {
            peer.send_pcm24k(frame).await?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut decoded_samples = 0usize;
        let mut assistant_transcript = String::new();
        let mut assistant_audio_active = false;
        let mut assistant_done_at: Option<Instant> = None;
        let mut overlap_frame_offset = 0usize;
        let mut overlap_started = false;
        let mut overlap_heard = false;
        let mut decoded_samples_at_overlap = 0usize;
        let mut input_keepalive = tokio::time::interval(Duration::from_millis(20));
        input_keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        while Instant::now() < deadline
            && !(overlap_heard
                && decoded_samples >= decoded_samples_at_overlap.saturating_add(24_000 / 2)
                && assistant_done_at.is_some_and(|done| done.elapsed() >= Duration::from_secs(2)))
        {
            tokio::select! {
                audio = remote_audio.recv() => {
                    match audio {
                        Some(Ok(samples)) => {
                            if assistant_audio_active {
                                let previous_seconds = decoded_samples / 24_000;
                                decoded_samples = decoded_samples.saturating_add(samples.len());
                                let current_seconds = decoded_samples / 24_000;
                                if current_seconds > previous_seconds {
                                    eprintln!(
                                        "[gpt-live probe] decoded_speech_seconds={:.2}",
                                        decoded_samples as f64 / 24_000.0
                                    );
                                }
                            }
                        }
                        Some(Err(error)) => bail!("GPT-Live reply audio decode failed: {error}"),
                        None => bail!("GPT-Live reply audio channel closed"),
                    }
                }
                message = server.next_message() => {
                    let Some(message) = message else {
                        bail!("Codex app-server closed during GPT-Live reply probe");
                    };
                    eprintln!("[gpt-live probe] {}", codex_message_summary(&message));
                    if message.get("method").and_then(Value::as_str)
                        == Some("thread/realtime/transcript/delta")
                        && message.pointer("/params/role").and_then(Value::as_str)
                            == Some("assistant")
                        && let Some(delta) = message.pointer("/params/delta").and_then(Value::as_str)
                    {
                        assistant_audio_active = true;
                        assistant_transcript.push_str(delta);
                    }
                    if message.get("method").and_then(Value::as_str)
                        == Some("thread/realtime/transcript/done")
                        && message.pointer("/params/role").and_then(Value::as_str)
                            == Some("assistant")
                    {
                        assistant_done_at = Some(Instant::now());
                    }
                    if overlap_started
                        && message.get("method").and_then(Value::as_str)
                            == Some("thread/realtime/transcript/done")
                        && message.pointer("/params/role").and_then(Value::as_str)
                            == Some("user")
                        && let Some(text) = message.pointer("/params/text").and_then(Value::as_str)
                    {
                        let normalized = text.to_ascii_lowercase();
                        eprintln!(
                            "[gpt-live probe] overlapping user transcript candidate: {text:?}"
                        );
                        if normalized.contains("purple")
                            || normalized.contains("banana")
                            || normalized.contains("speaking while")
                            || normalized.contains("while you are speaking")
                            || normalized.contains("while you're speaking")
                        {
                            overlap_heard = true;
                            eprintln!(
                                "[gpt-live probe] overlapping user speech transcribed: {text:?}"
                            );
                        }
                    }
                    if message.get("method").and_then(Value::as_str)
                        == Some("thread/realtime/error")
                    {
                        bail!("GPT-Live reply probe error: {}", realtime_error_detail(&message));
                    }
                }
                _ = input_keepalive.tick() => {
                    if decoded_samples >= 24_000 && overlap_frame_offset < overlap_samples.len() {
                        if !overlap_started {
                            overlap_started = true;
                            decoded_samples_at_overlap = decoded_samples;
                            assistant_done_at = None;
                            eprintln!(
                                "[gpt-live probe] injecting overlapping user speech at {:.2}s of assistant audio",
                                decoded_samples as f64 / 24_000.0
                            );
                        }
                        let end = (overlap_frame_offset + 480).min(overlap_samples.len());
                        peer.send_pcm24k(&overlap_samples[overlap_frame_offset..end]).await?;
                        overlap_frame_offset = end;
                    } else {
                        peer.send_pcm24k(&[0_i16; 480]).await?;
                    }
                }
            }
        }
        anyhow::ensure!(
            decoded_samples >= 24_000 * 3,
            "GPT-Live produced only {:.2}s of decoded audio (assistant transcript: {:?})",
            decoded_samples as f64 / 24_000.0,
            assistant_transcript
        );
        anyhow::ensure!(overlap_started, "Overlapping speech probe never started");
        anyhow::ensure!(
            overlap_heard,
            "GPT-Live did not transcribe the overlapping user phrase while assistant audio was active"
        );
        anyhow::ensure!(
            decoded_samples >= decoded_samples_at_overlap.saturating_add(24_000 / 2),
            "Assistant audio did not continue for at least 0.5s after overlapping user speech began"
        );

        eprintln!("[gpt-live probe] injecting second completed user turn");
        for frame in followup_samples.chunks(480) {
            peer.send_pcm24k(frame).await?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let followup_deadline = Instant::now() + Duration::from_secs(45);
        let mut followup_user_heard = false;
        let mut followup_assistant_started = false;
        let mut followup_assistant_done_at: Option<Instant> = None;
        let mut followup_audio_samples = 0usize;
        let mut followup_keepalive = tokio::time::interval(Duration::from_millis(20));
        followup_keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        while Instant::now() < followup_deadline
            && !(followup_user_heard
                && followup_assistant_started
                && followup_audio_samples >= 24_000 / 2
                && followup_assistant_done_at
                    .is_some_and(|done| done.elapsed() >= Duration::from_secs(1)))
        {
            tokio::select! {
                audio = remote_audio.recv() => {
                    match audio {
                        Some(Ok(samples)) if followup_assistant_started => {
                            followup_audio_samples = followup_audio_samples.saturating_add(samples.len());
                        }
                        Some(Ok(_)) => {}
                        Some(Err(error)) => bail!("GPT-Live follow-up audio decode failed: {error}"),
                        None => bail!("GPT-Live follow-up audio channel closed"),
                    }
                }
                message = server.next_message() => {
                    let Some(message) = message else {
                        bail!("Codex app-server closed during GPT-Live follow-up probe");
                    };
                    eprintln!("[gpt-live follow-up] {}", codex_message_summary(&message));
                    let method = message.get("method").and_then(Value::as_str);
                    let role = message.pointer("/params/role").and_then(Value::as_str);
                    if method == Some("thread/realtime/transcript/done") && role == Some("user")
                        && let Some(text) = message.pointer("/params/text").and_then(Value::as_str)
                    {
                        eprintln!("[gpt-live probe] second turn transcript candidate: {text:?}");
                        if text.trim().chars().count() >= 4 {
                            followup_user_heard = true;
                            eprintln!("[gpt-live probe] second turn user transcribed: {text:?}");
                        }
                    }
                    if method == Some("thread/realtime/transcript/delta") && role == Some("assistant") {
                        followup_assistant_started = true;
                    }
                    if method == Some("thread/realtime/transcript/done") && role == Some("assistant") {
                        followup_assistant_done_at = Some(Instant::now());
                    }
                    if method == Some("thread/realtime/error") {
                        bail!("GPT-Live follow-up probe error: {}", realtime_error_detail(&message));
                    }
                }
                _ = followup_keepalive.tick() => {
                    peer.send_pcm24k(&[0_i16; 480]).await?;
                }
            }
        }

        let _ = server.send_request("thread/realtime/stop", json!({"threadId": thread_id}));
        peer.close().await;
        anyhow::ensure!(followup_user_heard, "GPT-Live did not transcribe the second completed user turn");
        anyhow::ensure!(followup_assistant_started, "GPT-Live did not start a second assistant reply");
        anyhow::ensure!(
            followup_audio_samples >= 24_000 / 2,
            "GPT-Live second assistant reply produced only {:.2}s of decoded audio",
            followup_audio_samples as f64 / 24_000.0
        );
        anyhow::ensure!(followup_assistant_done_at.is_some(), "GPT-Live second assistant transcript did not finish");
        eprintln!(
            "[gpt-live probe] second turn reply audio_seconds={:.2} multi_turn=true",
            followup_audio_samples as f64 / 24_000.0
        );
        eprintln!(
            "[gpt-live probe] final decoded_seconds={:.2} transcript_chars={} full_duplex=true multi_turn=true",
            decoded_samples as f64 / 24_000.0,
            assistant_transcript.chars().count()
        );
        Ok(())
    })
}

struct InFlightContextImage {
    upload_id: u64,
    name: String,
    width: u32,
    height: u32,
    byte_size: usize,
    turn_id: String,
    deadline: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodexContextImageResponse {
    NotHandled,
    Uploaded(u64),
    Failed(u64),
}

fn handle_codex_context_image_response(
    message: &Value,
    in_flight_context_images: &mut HashMap<u64, InFlightContextImage>,
    events: &std::sync::mpsc::Sender<Event>,
) -> CodexContextImageResponse {
    // A server request may legally reuse a numeric id in the opposite JSON-RPC
    // direction. Only messages without a method are responses to our requests.
    if message.get("method").is_some() {
        return CodexContextImageResponse::NotHandled;
    }
    let Some(request_id) = message.get("id").and_then(Value::as_u64) else {
        return CodexContextImageResponse::NotHandled;
    };
    let Some(upload) = in_flight_context_images.remove(&request_id) else {
        return CodexContextImageResponse::NotHandled;
    };
    let upload_id = upload.upload_id;

    if Instant::now() >= upload.deadline {
        let _ = events.send(Event::ContextImageUploadFailed {
            upload_id,
            detail: context_image_timeout_detail(),
        });
        CodexContextImageResponse::Failed(upload_id)
    } else if let Some(error) = message.get("error") {
        let detail = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown JSON-RPC error");
        eprintln!(
            "[live-assistant image] upload failed upload_id={} turn={}: {detail}",
            upload_id, upload.turn_id
        );
        let _ = events.send(Event::ContextImageUploadFailed {
            upload_id,
            detail: detail.to_owned(),
        });
        CodexContextImageResponse::Failed(upload_id)
    } else {
        eprintln!(
            "[live-assistant image] uploaded upload_id={} name={:?} turn={} size={}x{} bytes={}",
            upload_id, upload.name, upload.turn_id, upload.width, upload.height, upload.byte_size
        );
        let _ = events.send(Event::ContextImageUploaded { upload_id });
        CodexContextImageResponse::Uploaded(upload_id)
    }
}

fn expire_codex_context_images(
    in_flight_context_images: &mut HashMap<u64, InFlightContextImage>,
    now: Instant,
    events: &std::sync::mpsc::Sender<Event>,
) -> Vec<u64> {
    let mut expired_upload_ids = Vec::new();
    in_flight_context_images.retain(|_, upload| {
        if now < upload.deadline {
            return true;
        }
        let _ = events.send(Event::ContextImageUploadFailed {
            upload_id: upload.upload_id,
            detail: context_image_timeout_detail(),
        });
        expired_upload_ids.push(upload.upload_id);
        false
    });
    expired_upload_ids
}

fn take_latest_ready_codex_context_image(
    ready_upload_id: &mut Option<u64>,
    latest_upload_id: u64,
    in_flight_context_images: &HashMap<u64, InFlightContextImage>,
) -> Option<u64> {
    let ready = (*ready_upload_id)
        .filter(|ready| *ready == latest_upload_id && in_flight_context_images.is_empty())?;
    *ready_upload_id = None;
    Some(ready)
}

fn handle_codex_handoff_message(
    message: &Value,
    events: &std::sync::mpsc::Sender<Event>,
    state: &mut CodexHandoffState,
) -> Result<CodexHandoffAction> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match method {
        "turn/started" => {
            state.fallback = None;
            state.active_turn_id = message
                .pointer("/params/turn/id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            state.response_text.clear();
            Ok(CodexHandoffAction::Handled)
        }
        "item/agentMessage/delta" => {
            if state.active_turn_id.is_some()
                && let Some(delta) = message.pointer("/params/delta").and_then(Value::as_str)
            {
                state.response_text.push_str(delta);
            }
            Ok(CodexHandoffAction::Handled)
        }
        "item/completed" => {
            let item = message.pointer("/params/item").unwrap_or(&Value::Null);
            if state.active_turn_id.is_some()
                && item.get("type").and_then(Value::as_str) == Some("agentMessage")
                && let Some(text) = item.get("text").and_then(Value::as_str)
                && !text.trim().is_empty()
            {
                state.response_text = text.to_owned();
            }
            Ok(CodexHandoffAction::Handled)
        }
        "turn/completed" => {
            let turn = message.pointer("/params/turn").unwrap_or(&Value::Null);
            let turn_id = turn.get("id").and_then(Value::as_str).unwrap_or_default();
            if state
                .active_turn_id
                .as_deref()
                .is_some_and(|active| active == turn_id)
            {
                let status = turn
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("failed");
                if status != "completed" {
                    let detail = turn
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .unwrap_or("The delegated computer action failed");
                    let _ =
                        events.send(Event::Error(format!("Codex tool handoff failed: {detail}")));
                }
                let fallback = (status == "completed")
                    .then(|| state.response_text.trim().to_owned())
                    .filter(|text| !text.is_empty());
                state.clear_turn();
                if let Some(text) = fallback {
                    // Automatic handoff streaming is the primary path. Retain the
                    // completed text only as a short delayed fallback for backend
                    // variants that fail to produce any spoken output.
                    state.schedule_fallback(text);
                }
            }
            Ok(CodexHandoffAction::Handled)
        }
        "error" if state.active_turn_id.is_some() => {
            let detail = message
                .pointer("/params/error/message")
                .or_else(|| message.pointer("/error/message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown Codex handoff error");
            let _ = events.send(Event::Error(format!("Codex tool handoff error: {detail}")));
            state.clear_turn();
            state.fallback = None;
            Ok(CodexHandoffAction::Handled)
        }
        _ => Ok(CodexHandoffAction::NotHandled),
    }
}

/// Native GPT-Live smoke test that follows the same platform-ADM WebRTC path
/// used by the production macOS app. It starts a session, sends a text turn,
/// waits for the assistant transcript while libWebRTC plays audio directly,
/// then closes the transport.
pub fn probe_codex_gpt_live_native() -> Result<()> {
    let credentials = crate::auth::codex_credentials()?;
    let runtime =
        tokio::runtime::Runtime::new().context("Could not create native probe runtime")?;
    runtime.block_on(async move {
        let platform_api_key = credentials
            .chatgpt_account_id
            .is_none()
            .then_some(credentials.bearer_token.as_str());
        let mut server = CodexAppServer::start(platform_api_key)?;
        server
            .call(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "live-assistant-native-probe",
                        "title": "Live Assistant Native GPT-Live Probe",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": codex_live_initialize_capabilities()
                }),
            )
            .await?;
        server.notify("initialized", json!({}))?;
        let cwd = std::env::current_dir()
            .context("Could not read the current working directory")?
            .to_string_lossy()
            .into_owned();
        let thread = server
            .call(
                "thread/start",
                json!({
                    "cwd": cwd,
                    "ephemeral": true,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "baseInstructions": "This is a native WebRTC audio smoke test.",
                    "config": {
                        "features.realtime_conversation": true,
                        "suppress_unstable_features_warning": true
                    }
                }),
            )
            .await?;
        let thread_id = thread
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .context("Codex app-server did not return a native probe thread id")?
            .to_owned();

        let (peer, offer_sdp) = GptLivePeer::create().await?;
        server
            .call(
                "thread/realtime/start",
                json!({
                    "threadId": thread_id,
                    "outputModality": "audio",
                    "version": "v3",
                    "model": "gpt-live-1-boulder-alpha",
                    "voice": "ember",
                    "transport": {"type": "webrtc", "sdp": offer_sdp},
                    "clientManagedHandoffs": false,
                    "delegationAckFiller": false,
                    "codexResponsesAsItems": false,
                    "includeStartupContext": false,
                    "prompt": "Reply conversationally and briefly."
                }),
            )
            .await?;

        let mut answer_applied = false;
        let mut started = false;
        while !answer_applied || !started {
            let message = tokio::time::timeout(Duration::from_secs(45), server.next_message())
                .await
                .context("Timed out starting native GPT-Live WebRTC")?
                .context("Codex app-server closed during native GPT-Live startup")?;
            match message.get("method").and_then(Value::as_str) {
                Some("thread/realtime/sdp") => {
                    let answer = message
                        .pointer("/params/sdp")
                        .and_then(Value::as_str)
                        .context("Native GPT-Live probe did not receive an SDP answer")?;
                    peer.accept_answer(answer.to_owned()).await?;
                    answer_applied = true;
                }
                Some("thread/realtime/started") => started = true,
                Some("thread/realtime/error") => {
                    bail!("Native GPT-Live startup failed: {}", realtime_error_detail(&message));
                }
                Some("thread/realtime/closed") => {
                    bail!("Native GPT-Live closed during startup");
                }
                _ => {}
            }
        }

        server
            .call(
                "thread/realtime/appendText",
                json!({
                    "threadId": thread_id,
                    "role": "user",
                    "text": "Say exactly: Native audio ready."
                }),
            )
            .await?;

        let deadline = Instant::now() + Duration::from_secs(45);
        let mut assistant_text = String::new();
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let message = tokio::time::timeout(remaining, server.next_message())
                .await
                .context("Timed out waiting for native GPT-Live response")?
                .context("Codex app-server closed during native GPT-Live response")?;
            match message.get("method").and_then(Value::as_str) {
                Some("thread/realtime/transcript/delta")
                    if message.pointer("/params/role").and_then(Value::as_str)
                        == Some("assistant") =>
                {
                    if let Some(delta) = message.pointer("/params/delta").and_then(Value::as_str) {
                        assistant_text.push_str(delta);
                    }
                }
                Some("thread/realtime/transcript/done")
                    if message.pointer("/params/role").and_then(Value::as_str)
                        == Some("assistant") =>
                {
                    if assistant_text.is_empty()
                        && let Some(text) = message.pointer("/params/text").and_then(Value::as_str)
                    {
                        assistant_text.push_str(text);
                    }
                    break;
                }
                Some("thread/realtime/error") => {
                    bail!("Native GPT-Live response failed: {}", realtime_error_detail(&message));
                }
                Some("thread/realtime/closed") => {
                    bail!("Native GPT-Live closed before responding");
                }
                _ => {}
            }
        }

        let _ = server.send_request(
            "thread/realtime/stop",
            json!({"threadId": thread_id}),
        );
        peer.close().await;
        anyhow::ensure!(
            !assistant_text.trim().is_empty(),
            "Native GPT-Live returned no assistant transcript"
        );
        eprintln!(
            "[gpt-live native probe] transcript={:?} platform_adm=true duplicate_audio_notifications=false",
            assistant_text.trim()
        );
        Ok(())
    })
}

/// End-to-end GPT-Live tool latency probe. The delegated Codex turn calls a
/// deterministic local dynamic tool, receives an immediate result, and streams
/// the answer back through the same automatic handoff path used by calendar/MCP
/// tools. This isolates model/dispatch/handoff overhead from provider latency.
pub fn probe_codex_gpt_live_tool_latency() -> Result<()> {
    let credentials = crate::auth::codex_credentials()?;
    let runtime = tokio::runtime::Runtime::new()
        .context("Could not create GPT-Live tool latency probe runtime")?;
    runtime.block_on(async move {
        let platform_api_key = credentials
            .chatgpt_account_id
            .is_none()
            .then_some(credentials.bearer_token.as_str());
        let connection_started = Instant::now();
        let mut server = CodexAppServer::start(platform_api_key)?;
        server
            .call(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "live-assistant-tool-latency-probe",
                        "title": "Live Assistant GPT-Live Tool Latency Probe",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": codex_live_initialize_capabilities()
                }),
            )
            .await?;
        server.notify("initialized", json!({}))?;
        let cwd = std::env::current_dir()
            .context("Could not read the current working directory")?
            .to_string_lossy()
            .into_owned();
        let thread = server
            .call(
                "thread/start",
                json!({
                    "cwd": cwd,
                    "ephemeral": true,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "baseInstructions": "This is a strict latency test. When the user asks for the latency probe or hidden nonce, call latency_probe immediately as the first action. Its result contains an unpredictable nonce. After success, emit only that nonce as the final answer. Do not perform discovery, planning, retries, or any other tool call.",
                    "dynamicTools": [{
                        "type": "function",
                        "name": "latency_probe",
                        "description": "Return an immediate deterministic success value for realtime latency measurement. Call this immediately.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }
                    }],
                    "reasoningEffort": CODEX_LIVE_DELEGATED_REASONING_EFFORT,
                    "config": {
                        "features.realtime_conversation": true,
                        "suppress_unstable_features_warning": true,
                        "mcp_servers.node_repl.enabled": false,
                        "mcp_servers.openaiDeveloperDocs.enabled": false
                    }
                }),
            )
            .await?;
        let thread_id = thread
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .context("Codex app-server did not return a tool latency probe thread id")?
            .to_owned();

        let (mut peer, offer_sdp) = GptLiveProbePeer::create().await?;
        server
            .call(
                "thread/realtime/start",
                json!({
                    "threadId": thread_id,
                    "outputModality": "audio",
                    "version": "v3",
                    "model": "gpt-live-1-boulder-alpha",
                    "voice": "ember",
                    "transport": {"type": "webrtc", "sdp": offer_sdp},
                    "clientManagedHandoffs": false,
                    "delegationAckFiller": false,
                    "codexResponsesAsItems": false,
                    "codexResponseHandoffMode": "thinking",
                    "includeStartupContext": false,
                    "prompt": "This is a strict realtime latency test. Immediately delegate the user request to Codex so it can call latency_probe. Do not speak before delegation. Speak the streamed final result as soon as it arrives."
                }),
            )
            .await?;

        let mut answer_applied = false;
        let mut started = false;
        while !answer_applied || !started {
            let message = tokio::time::timeout(Duration::from_secs(45), server.next_message())
                .await
                .context("Timed out starting GPT-Live tool latency probe")?
                .context("Codex app-server closed during tool latency startup")?;
            match message.get("method").and_then(Value::as_str) {
                Some("thread/realtime/sdp") => {
                    let answer = message
                        .pointer("/params/sdp")
                        .and_then(Value::as_str)
                        .context("Tool latency probe did not receive an SDP answer")?;
                    peer.accept_answer(answer.to_owned()).await?;
                    answer_applied = true;
                }
                Some("thread/realtime/started") => started = true,
                Some("thread/realtime/error") => {
                    bail!(
                        "GPT-Live tool latency startup failed: {}",
                        realtime_error_detail(&message)
                    );
                }
                Some("thread/realtime/closed") => {
                    bail!("GPT-Live tool latency session closed during startup");
                }
                _ => {}
            }
        }

        let mcp_refresh_duration = server
            .refresh_mcp_and_wait_for_server("codex_apps", Duration::from_secs(15))
            .await?;
        eprintln!(
            "[gpt-live tool latency] stage=codex_apps_ready ms={}",
            mcp_refresh_duration.as_millis(),
        );
        let mut remote_audio = peer.take_remote_audio();

        // Exercise the real GPT-Live microphone/VAD/delegation path. Pace the
        // spoken portion in real time, then keep the RTP clock moving with silence
        // on a separate task while tool notifications are dispatched immediately.
        let prompt = "Use latency probe. Tell me its hidden nonce.";
        let samples = synthesize_latest_image_probe_speech(prompt)?;
        let silence_samples = AUDIO_SAMPLE_RATE * 2;
        anyhow::ensure!(
            samples.len() > silence_samples,
            "Synthesized latency probe speech did not contain a spoken segment"
        );
        let spoken_samples = &samples[..samples.len() - silence_samples];
        let audio_sender = peer.audio_sender();
        for frame in spoken_samples.chunks(480) {
            audio_sender.send_pcm24k(frame)?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let request_started = Instant::now();
        let silence_sender = audio_sender.clone();
        let silence_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let silence_stop_task = silence_stop.clone();
        let silence_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(20));
            while !silence_stop_task.load(std::sync::atomic::Ordering::Relaxed) {
                interval.tick().await;
                if silence_sender.send_pcm24k(&[0_i16; 480]).is_err() {
                    break;
                }
            }
        });
        eprintln!(
            "[gpt-live tool latency] stage=user_speech_ended spoken_ms={} connection_ms={}",
            spoken_samples.len() * 1_000 / AUDIO_SAMPLE_RATE,
            connection_started.elapsed().as_millis(),
        );

        let deadline = Instant::now() + Duration::from_secs(20);
        let mut tool_requested_at: Option<Duration> = None;
        let mut result_submitted_at: Option<Duration> = None;
        let mut first_result_delta_at: Option<Duration> = None;
        let mut first_audio_after_result_at: Option<Duration> = None;
        let mut first_speech_at: Option<Duration> = None;
        let mut pre_tool_audio_chunks = 0usize;
        let mut assistant_text = String::new();
        let mut trace = GptLiveToolLatencyTrace::default();
        let mut handoff_state = CodexHandoffState::default();
        let (probe_events, _probe_events_rx) = std::sync::mpsc::channel();
        trace.begin("probe.request");

        while Instant::now() < deadline && first_speech_at.is_none() {
            loop {
                match remote_audio.try_recv() {
                    Ok(Ok(samples)) if !samples.is_empty() => {
                        if result_submitted_at.is_some() {
                            if first_audio_after_result_at.is_none() {
                                first_audio_after_result_at = Some(request_started.elapsed());
                                trace.log(
                                    "audio.first_after_result",
                                    &format!("samples={}", samples.len()),
                                );
                            }
                            continue;
                        }
                        pre_tool_audio_chunks = pre_tool_audio_chunks.saturating_add(1);
                        if pre_tool_audio_chunks == 1 {
                            trace.log(
                                "audio.pre_tool_ignored",
                                &format!("samples={}", samples.len()),
                            );
                        }
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => bail!("GPT-Live audio probe failed: {error}"),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
            if let Some(text) = handoff_state.take_fallback_if_due(Instant::now()) {
                trace.log(
                    "handoff.fallback_append_speech",
                    &format!("chars={}", text.chars().count()),
                );
                server.send_request(
                    "thread/realtime/appendSpeech",
                    json!({
                        "threadId": thread_id,
                        "text": text,
                    }),
                )?;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let poll = remaining.min(Duration::from_millis(50));
            let message = match tokio::time::timeout(poll, server.next_message()).await {
                Ok(Some(message)) => message,
                Ok(None) => bail!("Codex app-server closed during GPT-Live tool latency probe"),
                Err(_) => continue,
            };
            eprintln!(
                "[gpt-live tool latency] {}",
                codex_message_summary(&message)
            );
            trace.observe(&message);
            let _ = handle_codex_handoff_message(
                &message,
                &probe_events,
                &mut handoff_state,
            )?;
            if let Some((request_id, call)) = dynamic_tool_request(&message) {
                anyhow::ensure!(
                    call.name == "latency_probe",
                    "Unexpected tool `{}` during latency probe",
                    call.name
                );
                tool_requested_at.get_or_insert_with(|| request_started.elapsed());
                let submit_started = Instant::now();
                server.respond(
                    request_id,
                    json!({
                        "contentItems": [{
                            "type": "inputText",
                            "text": "success nonce: 7F3A-91C2"
                        }],
                        "success": true
                    }),
                )?;
                result_submitted_at.get_or_insert_with(|| request_started.elapsed());
                assistant_text.clear();
                send_fast_tool_speech(
                    &mut server,
                    &thread_id,
                    handoff_state.active_turn_id.as_deref(),
                    "The hidden nonce is 7F3A-91C2.".to_owned(),
                )?;
                handoff_state.clear_turn();
                handoff_state.fallback = None;
                trace.log(
                    "fast_result.probe_speech",
                    &format!("call_id={}", call.call_id),
                );
                eprintln!(
                    "[gpt-live tool latency] stage=tool_result_submitted call_id={} tool_ms={} submit_us={} total_ms={}",
                    call.call_id,
                    tool_requested_at.unwrap_or_default().as_millis(),
                    submit_started.elapsed().as_micros(),
                    request_started.elapsed().as_millis(),
                );
                continue;
            }
            match message.get("method").and_then(Value::as_str) {
                Some("item/agentMessage/delta") => {
                    first_result_delta_at.get_or_insert_with(|| request_started.elapsed());
                }
                Some("thread/realtime/transcript/delta")
                    if message.pointer("/params/role").and_then(Value::as_str)
                        == Some("assistant") =>
                {
                    if result_submitted_at.is_some() {
                        handoff_state.note_spoken_output();
                        if let Some(delta) = message.pointer("/params/delta").and_then(Value::as_str) {
                            assistant_text.push_str(delta);
                        }
                        if assistant_text.to_ascii_lowercase().contains("7f3a") {
                            first_speech_at.get_or_insert_with(|| request_started.elapsed());
                        }
                    } else {
                        trace.log("speech.pre_tool_ignored", "");
                    }
                }
                Some("thread/realtime/transcript/done")
                    if message.pointer("/params/role").and_then(Value::as_str)
                        == Some("assistant") =>
                {
                    if result_submitted_at.is_some() {
                        handoff_state.note_spoken_output();
                        if assistant_text.is_empty()
                            && let Some(text) = message.pointer("/params/text").and_then(Value::as_str)
                        {
                            assistant_text.push_str(text);
                        }
                        if assistant_text.to_ascii_lowercase().contains("7f3a") {
                            first_speech_at.get_or_insert_with(|| request_started.elapsed());
                        }
                    } else {
                        trace.log("speech.pre_tool_done_ignored", "");
                    }
                }
                Some("thread/realtime/error") => {
                    bail!(
                        "GPT-Live tool latency response failed: {}",
                        realtime_error_detail(&message)
                    );
                }
                Some("thread/realtime/closed") => {
                    bail!("GPT-Live tool latency session closed before speaking");
                }
                _ => {}
            }
        }

        silence_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = silence_task.await;
        let _ = server.send_request(
            "thread/realtime/stop",
            json!({"threadId": thread_id}),
        );
        peer.close().await;

        let tool_requested_at = tool_requested_at
            .context("GPT-Live never emitted the latency_probe tool call")?;
        let result_submitted_at = result_submitted_at
            .context("The latency_probe result was never submitted")?;
        let first_speech_at = first_speech_at
            .context("GPT-Live produced no spoken response after the tool result")?;
        anyhow::ensure!(
            tool_requested_at < Duration::from_secs(8),
            "GPT-Live took {:.2}s to emit a trivial tool call",
            tool_requested_at.as_secs_f64()
        );
        anyhow::ensure!(
            first_speech_at < Duration::from_secs(8),
            "GPT-Live took {:.2}s to begin speaking after a trivial tool request",
            first_speech_at.as_secs_f64()
        );
        eprintln!(
            "[gpt-live tool latency] result=success tool_request_ms={} result_submit_ms={} first_result_delta_ms={} first_audio_after_result_ms={} first_useful_speech_ms={} pre_tool_audio_chunks={} transcript={:?}",
            tool_requested_at.as_millis(),
            result_submitted_at.as_millis(),
            first_result_delta_at
                .map(|value| value.as_millis().to_string())
                .unwrap_or_else(|| "not_observed".to_owned()),
            first_audio_after_result_at
                .map(|value| value.as_millis().to_string())
                .unwrap_or_else(|| "not_observed".to_owned()),
            first_speech_at.as_millis(),
            pre_tool_audio_chunks,
            assistant_text.trim(),
        );
        Ok(())
    })
}

fn realtime_error_detail(message: &Value) -> String {
    let direct = message
        .pointer("/params/message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if !direct.is_empty() && direct != "error" {
        return direct.to_owned();
    }
    message
        .pointer("/params/error/message")
        .or_else(|| message.pointer("/params/details"))
        .or_else(|| message.pointer("/error/message"))
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| "GPT-Live reported an unspecified backend error".to_owned())
}

fn codex_message_is_assistant_transcript(message: &Value) -> bool {
    matches!(
        message.get("method").and_then(Value::as_str),
        Some("thread/realtime/transcript/delta") | Some("thread/realtime/transcript/done")
    ) && message.pointer("/params/role").and_then(Value::as_str) == Some("assistant")
}

fn codex_message_starts_reply(message: &Value) -> bool {
    match message.get("method").and_then(Value::as_str) {
        Some("turn/started") => true,
        Some("thread/realtime/itemAdded") => {
            message.pointer("/params/item/type").and_then(Value::as_str) == Some("handoff_request")
        }
        Some("thread/realtime/transcript/delta") | Some("thread/realtime/transcript/done") => {
            message.pointer("/params/role").and_then(Value::as_str) == Some("assistant")
        }
        _ => false,
    }
}

fn codex_message_summary(message: &Value) -> String {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("response");
    let id = message.get("id").map(Value::to_string).unwrap_or_default();
    let item_type = message
        .pointer("/params/item/type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let status = message
        .pointer("/params/turn/status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let role = message
        .pointer("/params/role")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let text_len = message
        .pointer("/params/delta")
        .or_else(|| message.pointer("/params/text"))
        .and_then(Value::as_str)
        .map(str::len)
        .unwrap_or_default();
    let error = message
        .pointer("/params/message")
        .or_else(|| message.pointer("/params/error/message"))
        .or_else(|| message.pointer("/error/message"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "method={method} id={id} item_type={item_type} role={role} text_len={text_len} status={status} error={error}"
    )
}

fn codex_dynamic_tools(screen: ScreenInfo) -> Value {
    let mut tools = computer_tools(screen)
        .as_array()
        .cloned()
        .unwrap_or_default();
    tools.extend(note_tools());
    tools.push(create_image_tool());
    codex_dynamic_tools_with_tools(Value::Array(tools))
}

fn codex_dynamic_tools_with_tools(tools: Value) -> Value {
    Value::Array(
        tools
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.get("name").cloned().unwrap_or(Value::Null),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "inputSchema": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"})),
                })
            })
            .collect(),
    )
}

/// Return the tools available to a voice-capable model. Both backends receive
/// the local click tool, but connection instructions make OpenAI Realtime
/// delegate screenshot-backed clicks to `ask_text_model` while GPT-Live calls
/// `click_screen` directly. Text tabs omit `ask_text_model` to avoid recursion.
fn voice_tools(screen: ScreenInfo) -> Value {
    let mut tools = computer_tools(screen)
        .as_array()
        .cloned()
        .unwrap_or_default();
    tools.extend(note_tools());
    tools.push(ask_text_model_tool());
    tools.push(create_image_tool());
    Value::Array(tools)
}

fn note_tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "name": "replace_note_text",
            "description": "Replace exact text in a Markdown/text note stored in ~/liveassistant. Omit note_name to edit the currently open note. Use this instead of keyboard typing when the user asks you to edit a note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_name": {
                        "type": "string",
                        "description": "Optional note file name. Omit to use the currently open note."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every exact match instead of only the first match."
                    }
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": false
            }
        }),
        json!({
            "type": "function",
            "name": "remove_note_text",
            "description": "Remove exact text from a Markdown/text note stored in ~/liveassistant. Omit note_name to edit the currently open note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_name": {
                        "type": "string",
                        "description": "Optional note file name. Omit to use the currently open note."
                    },
                    "text": {
                        "type": "string",
                        "description": "Exact text to remove."
                    },
                    "remove_all": {
                        "type": "boolean",
                        "description": "Remove every exact match instead of only the first match."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }
        }),
        json!({
            "type": "function",
            "name": "rename_note",
            "description": "Rename a note stored in ~/liveassistant. Omit note_name to rename the currently open note. A .md extension is added when the new name has no extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_name": {
                        "type": "string",
                        "description": "Optional current note file name. Omit to use the currently open note."
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New file name for the note."
                    }
                },
                "required": ["new_name"],
                "additionalProperties": false
            }
        }),
        json!({
            "type": "function",
            "name": "create_note",
            "description": "Create a new Markdown/text note in ~/liveassistant and open it in the note editor. A .md extension is added when the name has no extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "File name for the new note."
                    },
                    "content": {
                        "type": "string",
                        "description": "Initial note content. Defaults to empty."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }
        }),
    ]
}

fn create_image_tool() -> Value {
    json!({
        "type": "function",
        "name": "create_image",
        "description": "Create an image with Codex's configured image model. Use the configured default model and resolution unless the user specifies another image model or supported resolution.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to create."
                },
                "model": {
                    "type": "string",
                    "description": "Optional image model id. Defaults to the image model configured in Settings."
                },
                "resolution": {
                    "type": "string",
                    "enum": ["1024x1024", "1024x1536", "1536x1024", "2560x1440", "3840x2160"],
                    "description": "Optional output resolution. Defaults to the configured image resolution."
                }
            },
            "required": ["prompt"],
            "additionalProperties": false
        }
    })
}

fn ask_text_model_tool() -> Value {
    json!({
        "type": "function",
        "name": "ask_text_model",
        "description": "Ask a text model to analyze a question or perform a separate background text-model task. OpenAI Realtime uses this tool with a fresh screenshot for click requests; GPT-Live clicks directly. If model or thinking_level is omitted, use the app's configured defaults.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The complete question or task for the text model."
                },
                "image": {
                    "type": "string",
                    "description": "Optional base64-encoded image data URL to attach to the text-model request."
                },
                "include_screenshot": {
                    "type": "boolean",
                    "description": "Set true when the text model needs a fresh current-screen screenshot. OpenAI Realtime click delegation must set this to true; GPT-Live click requests use click_screen directly."
                },
                "model": {
                    "type": "string",
                    "description": "Optional text model id, for example gpt-5.6-luna."
                },
                "thinking_level": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "xhigh", "ultra"],
                    "description": "Optional reasoning effort. Use low for light, medium for medium, high for high, or a more intensive value when needed."
                }
            },
            "required": ["prompt"],
            "additionalProperties": false
        }
    })
}

/// The settings panel uses the same live definitions as the transports, so a
/// newly added tool cannot silently disappear from the UI documentation.
pub(crate) fn available_tool_descriptions(screen: ScreenInfo) -> Vec<(String, String)> {
    let mut descriptions = Vec::new();
    for tools in [voice_tools(screen), codex_dynamic_tools(screen)] {
        for tool in tools.as_array().into_iter().flatten() {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            if descriptions.iter().any(|(known, _)| known == name) {
                continue;
            }
            let Some(description) = tool.get("description").and_then(Value::as_str) else {
                continue;
            };
            descriptions.push((name.to_owned(), description.to_owned()));
        }
    }
    descriptions
}

/// Convert a local dynamic-tool result to the app-server content-item shape.
/// Image bytes stay in an inputImage item instead of being duplicated inside
/// the text item, so the text model can inspect the generated image directly.
const FAST_TOOL_RESULT_MAX_CHARS: usize = 2_000;

struct FastToolSpeech {
    turn_id: String,
    label: String,
    text: String,
}

fn bounded_tool_result(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > FAST_TOOL_RESULT_MAX_CHARS {
        return None;
    }
    Some(text.to_owned())
}

fn local_tool_fast_speech(name: &str, output: &str, success: bool) -> Option<String> {
    if !success {
        return None;
    }
    let value = serde_json::from_str::<Value>(output).ok()?;
    if let Some(reply) = value
        .get("assistant_reply")
        .and_then(Value::as_str)
        .and_then(bounded_tool_result)
    {
        return Some(reply);
    }
    match name {
        "insert_text" => Some("Done.".to_owned()),
        "run_bash" => value
            .get("stdout")
            .and_then(Value::as_str)
            .and_then(bounded_tool_result),
        _ => None,
    }
}

fn mcp_tool_fast_speech(message: &Value) -> Option<FastToolSpeech> {
    if message.get("method").and_then(Value::as_str) != Some("item/completed") {
        return None;
    }
    let item = message.pointer("/params/item")?;
    if item.get("type").and_then(Value::as_str) != Some("mcpToolCall")
        || item.get("status").and_then(Value::as_str) != Some("completed")
        || item.get("error").is_some_and(|error| !error.is_null())
    {
        return None;
    }
    let turn_id = message.pointer("/params/turnId")?.as_str()?.to_owned();
    let app_name = item
        .pointer("/appContext/appName")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty());
    let server = item
        .get("server")
        .and_then(Value::as_str)
        .unwrap_or("connector");
    let tool = item.get("tool").and_then(Value::as_str).unwrap_or("tool");
    let label = app_name
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{server}/{tool}"));
    let result = item.get("result")?;
    let mut parts = result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|content| content.get("text").and_then(Value::as_str))
        .filter_map(bounded_tool_result)
        .collect::<Vec<_>>();
    if parts.is_empty()
        && let Some(structured) = result.get("structuredContent")
        && !structured.is_null()
    {
        parts.push(bounded_tool_result(&structured.to_string())?);
    }
    let raw = bounded_tool_result(&parts.join("\n"))?;
    let trimmed = raw.trim_start();
    let direct_text =
        raw.chars().count() <= 600 && !trimmed.starts_with('{') && !trimmed.starts_with('[');
    Some(FastToolSpeech {
        turn_id,
        label: label.clone(),
        text: if direct_text {
            raw
        } else {
            format!(
                "{label} completed. Answer the user's request immediately and concisely using only this completed tool result. Do not mention internal tools:\n{raw}"
            )
        },
    })
}

fn send_fast_tool_speech(
    server: &mut CodexAppServer,
    thread_id: &str,
    turn_id: Option<&str>,
    text: String,
) -> Result<()> {
    if let Some(turn_id) = turn_id {
        let _ = server.send_request(
            "turn/interrupt",
            json!({
                "threadId": thread_id,
                "turnId": turn_id,
            }),
        );
    }
    server.send_request(
        "thread/realtime/appendSpeech",
        json!({
            "threadId": thread_id,
            "text": text,
        }),
    )?;
    Ok(())
}

fn dynamic_tool_content_items(output: &str) -> (Value, bool) {
    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return (json!([{"type": "inputText", "text": output}]), false);
    };
    let success = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let image_url = value
        .get("image_url")
        .and_then(Value::as_str)
        .filter(|url| !url.trim().is_empty())
        .map(str::to_owned);
    let mut text_value = value;
    if let Some(object) = text_value.as_object_mut() {
        object.remove("image_url");
    }
    let mut content_items = vec![json!({
        "type": "inputText",
        "text": text_value.to_string()
    })];
    if let Some(image_url) = image_url {
        content_items.push(json!({
            "type": "inputImage",
            "imageUrl": image_url
        }));
    }
    (Value::Array(content_items), success)
}

fn codex_turn_input(text: &str, attachments: &[Attachment]) -> Vec<Value> {
    let mut input = Vec::new();
    let text = if text.trim().is_empty() {
        "Analyze the attached image and answer the user's request in the live voice session."
    } else {
        text.trim()
    };
    input.push(json!({
        "type": "text",
        "text": text,
        "text_elements": []
    }));
    for attachment in attachments {
        if let Attachment::Image { data_url, .. } = attachment {
            input.push(json!({"type": "image", "url": data_url}));
        }
    }
    input
}

fn dynamic_tool_request(message: &Value) -> Option<(Value, ToolCall)> {
    if message.get("method").and_then(Value::as_str) != Some("item/tool/call") {
        return None;
    }
    let request_id = message.get("id")?.clone();
    let call_id = message.pointer("/params/callId")?.as_str()?.to_owned();
    let name = message.pointer("/params/tool")?.as_str()?.to_owned();
    let arguments = message
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or_else(|| json!({}))
        .to_string();
    Some((
        request_id,
        ToolCall {
            call_id,
            name,
            arguments,
            requested_at: Instant::now(),
        },
    ))
}

fn codex_live_start_error(detail: &str) -> String {
    if detail
        .to_ascii_lowercase()
        .contains("voice session access denied")
    {
        return "Codex app-server reached GPT-Live, but voice-session access is not enabled for \
                these credentials yet. GPT-Live is currently a ChatGPT Voice rollout and is not \
                yet a generally available API/app-server OAuth model. Use OpenAI Realtime for \
                now, or retry Codex GPT-Live after access is enabled."
            .to_owned();
    }
    detail.to_owned()
}

fn handle_codex_live_message(
    message: &Value,
    events: &std::sync::mpsc::Sender<Event>,
    state: &mut CodexLiveState,
) -> Result<()> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match method {
        "thread/realtime/itemAdded" => {
            let item = message.pointer("/params/item").unwrap_or(&Value::Null);
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "input_audio_buffer.speech_started" => {
                    // GPT-Live is full duplex. A new user utterance can overlap the
                    // active assistant response, so do not close the assistant stream.
                    let item_id = item
                        .get("item_id")
                        .and_then(Value::as_str)
                        .filter(|item_id| !item_id.trim().is_empty())
                        .map(str::to_owned)
                        .unwrap_or_else(|| state.input_number.saturating_add(1).to_string());
                    let item_id = if item_id.chars().all(|ch| ch.is_ascii_digit()) {
                        format!("codex-live-input-{item_id}")
                    } else {
                        item_id
                    };
                    state.start_input_with_id(item_id, events);
                }
                "handoff_request" => {
                    let text = item
                        .get("input_transcript")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim();
                    if !text.is_empty() {
                        let item_id = item
                            .get("item_id")
                            .or_else(|| item.get("handoff_id"))
                            .and_then(Value::as_str)
                            .filter(|item_id| !item_id.trim().is_empty())
                            .map(str::to_owned)
                            .unwrap_or_else(|| state.ensure_input(events));
                        if state.input_item_id.is_none() {
                            state.start_input_with_id(item_id.clone(), events);
                        } else {
                            // Replace the provisional per-utterance id with the upstream
                            // handoff id without opening another UI/microphone turn.
                            state.input_item_id = Some(item_id.clone());
                        }
                        state.input_text = text.to_owned();
                        let _ = events.send(Event::InputTranscript {
                            item_id,
                            text: text.to_owned(),
                        });
                    }
                }
                "response.cancelled" => state.finish_response(events),
                _ => {}
            }
        }
        "thread/realtime/transcript/delta" => {
            let role = message
                .pointer("/params/role")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let delta = message
                .pointer("/params/delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if role == "user" && !delta.is_empty() {
                let item_id = state.ensure_input(events);
                state.input_text.push_str(delta);
                let _ = events.send(Event::InputTranscript {
                    item_id,
                    text: state.input_text.clone(),
                });
            } else if role == "assistant" && !delta.is_empty() {
                let response_id = state.ensure_response(events);
                state.response_finish_deadline = None;
                state.assistant_text.push_str(delta);
                let _ = events.send(Event::AssistantTranscriptDelta {
                    response_id,
                    delta: delta.to_owned(),
                });
            }
        }
        "thread/realtime/transcript/done" => {
            let role = message
                .pointer("/params/role")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let text = message
                .pointer("/params/text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if role == "user" {
                let item_id = state.ensure_input(events);
                let final_text = if text.is_empty() {
                    state.input_text.clone()
                } else {
                    text.to_owned()
                };
                state.input_text.clear();
                state.input_item_id = None;
                let _ = events.send(Event::InputTranscript {
                    item_id,
                    text: final_text,
                });
                let _ = events.send(Event::SpeechStopped);
            } else if role == "assistant" {
                // GPT-Live's transcript/done is a natural spoken-reply boundary.
                // Keep the transport response open, but let the UI finalize a WAV
                // segment when it has accumulated more than five seconds of audio.
                let response_id = state.ensure_response(events);
                if state.assistant_text.is_empty() && !text.is_empty() {
                    state.assistant_text.push_str(text);
                    let _ = events.send(Event::AssistantTranscriptDelta {
                        response_id: response_id.clone(),
                        delta: text.to_owned(),
                    });
                }
                state.assistant_text.clear();
                state.schedule_response_finish();
                let _ = events.send(Event::AssistantSegmentDone { response_id });
            }
        }
        "thread/realtime/outputAudio/delta" => {
            // Native/browser WebRTC already renders the remote media track.
            // This method is opted out during initialize, but ignore it
            // defensively if an older app-server still emits it.
        }
        "thread/realtime/error" => {
            let detail = realtime_error_detail(message);
            eprintln!("[live-assistant realtime-error] {detail}");
            state.last_realtime_error = Some(detail);
        }
        "thread/realtime/closed" => {
            let reason = message
                .pointer("/params/reason")
                .and_then(Value::as_str)
                .filter(|reason| {
                    let reason = reason.trim();
                    !reason.is_empty() && reason != "error" && reason != "closed"
                })
                .map(str::to_owned)
                .or_else(|| state.last_realtime_error.take())
                .unwrap_or_else(|| "Codex GPT-Live session closed".to_owned());
            bail!("{reason}");
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
fn decode_audio_to_24k_mono(data: &str, sample_rate: u32, channels: usize) -> Result<Vec<i16>> {
    let bytes = STANDARD
        .decode(data)
        .context("Codex GPT-Live returned invalid base64 audio")?;
    let interleaved = bytes
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    let channels = channels.max(1);
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| {
                let sum = frame.iter().map(|sample| *sample as i64).sum::<i64>();
                (sum / channels as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16
            })
            .collect()
    };
    if sample_rate == 24_000 || mono.len() < 2 {
        return Ok(mono);
    }
    anyhow::ensure!(
        sample_rate > 0,
        "Codex GPT-Live returned a zero audio sample rate"
    );
    let output_len = ((mono.len() as u64 * 24_000) / sample_rate as u64) as usize;
    let source_step = sample_rate as f64 / 24_000.0;
    Ok((0..output_len)
        .map(|index| {
            let position = index as f64 * source_step;
            let left = position.floor() as usize;
            let right = (left + 1).min(mono.len() - 1);
            let fraction = (position - left as f64) as f32;
            (mono[left] as f32 + (mono[right] as f32 - mono[left] as f32) * fraction)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect())
}

async fn submit_tool_outputs<S>(
    writer: &mut S,
    outputs: Vec<ToolOutput>,
    events: &std::sync::mpsc::Sender<Event>,
) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let count = outputs.len();
    if count == 0 {
        return Ok(());
    }
    for output in outputs {
        send_json(
            writer,
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": openai_function_output(&output.output)
                }
            }),
        )
        .await?;
    }
    send_json(
        writer,
        json!({
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"]
            }
        }),
    )
    .await?;
    let _ = events.send(Event::ToolOutputsSubmitted { count });
    Ok(())
}

/// OpenAI Realtime function outputs are text-only. Keep generated image bytes
/// in the app/UI and return a compact acknowledgement to the voice model
/// instead of placing a potentially multi-megabyte data URL in its context.
/// Codex live/text transports use `dynamic_tool_content_items` above, which
/// can send the same image as a first-class inputImage content item.
fn openai_function_output(output: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(output) else {
        return output.to_owned();
    };
    let Some(object) = value.as_object_mut() else {
        return output.to_owned();
    };
    if object.remove("image_url").is_some() {
        object.insert("image_attached".to_owned(), Value::Bool(true));
        object.insert(
            "message".to_owned(),
            Value::String("The generated image is attached in the app.".to_owned()),
        );
        return value.to_string();
    }
    output.to_owned()
}

fn computer_tools(screen: ScreenInfo) -> Value {
    let screenshot_width = screen.logical_width;
    let screenshot_height = screen.logical_height;
    let max_x = screenshot_width.saturating_sub(1);
    let max_y = screenshot_height.saturating_sub(1);
    json!([
        {
            "type": "function",
            "name": "move_pointer",
            "description": format!("Move the macOS pointer to the requested logical screen coordinate without clicking. Use coordinates from the exact {screenshot_width} × {screenshot_height} screenshot. Call this before speaking when the user asks to move or hover the pointer."),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_x,
                        "description": format!("Horizontal pixel coordinate from the {screenshot_width}-pixel-wide screenshot's left edge.")
                    },
                    "y": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_y,
                        "description": format!("Vertical pixel coordinate from the {screenshot_height}-pixel-tall screenshot's top edge.")
                    }
                },
                "required": ["x", "y"],
                "additionalProperties": false
            }
        },
        {
            "type": "function",
            "name": "click_screen",
            "description": format!("Click the center of the intended target on the primary screen. Use logical image pixels relative to the top-left of the exact {screenshot_width} × {screenshot_height} image sent with the current turn; never use the {backing_width} × {backing_height} Retina backing-pixel space. Call this before speaking when screen interaction is needed.",
                backing_width = screen.backing_width,
                backing_height = screen.backing_height,
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_x,
                        "description": format!("Horizontal pixel coordinate from the {screenshot_width}-pixel-wide screenshot's left edge.")
                    },
                    "y": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_y,
                        "description": format!("Vertical pixel coordinate from the {screenshot_height}-pixel-tall screenshot's top edge.")
                    }
                },
                "required": ["x", "y"],
                "additionalProperties": false
            }
        },
        {
            "type": "function",
            "name": "run_bash",
            "description": "Immediately run one Bash command on the user's computer and return its exit code, stdout, and stderr when explicitly requested. Call this before speaking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact Bash command to run."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        },
        {
            "type": "function",
            "name": "insert_text",
            "description": "Immediately insert text at the current keyboard focus when the user explicitly asks you to type or insert text. Call this before speaking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to insert."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }
        }
    ])
}

/// Wait until the server acknowledges our session.update (or reports an error).
async fn wait_for_session_ready<S>(reader: &mut S) -> Result<()>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match reader.next().await {
            Some(Ok(Message::Text(text))) => {
                let value: Value =
                    serde_json::from_str(text.as_ref()).context("Invalid Realtime server event")?;
                match value
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                {
                    "session.updated" => return Ok(()),
                    "error" => {
                        let message = value
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("Session update failed");
                        bail!("{message}");
                    }
                    // session.created and other setup events can be ignored here.
                    _ => {}
                }
            }
            Some(Ok(
                Message::Ping(_) | Message::Pong(_) | Message::Binary(_) | Message::Frame(_),
            )) => {}
            Some(Ok(Message::Close(_))) | None => {
                bail!("Realtime connection closed before session was ready")
            }
            Some(Err(error)) => {
                return Err(error).context("Realtime connection failed during session setup");
            }
        }
    }
}

fn context_image_item_id(upload_id: u64) -> String {
    format!("screen_upload_{upload_id}")
}

fn context_image_timeout_detail() -> String {
    format!(
        "Screenshot upload exceeded {} seconds",
        CONTEXT_IMAGE_UPLOAD_TIMEOUT.as_secs()
    )
}

#[cfg(test)]
fn context_image_upload_id(item_id: &str) -> Option<u64> {
    item_id.strip_prefix("screen_upload_")?.parse().ok()
}

async fn send_context_image_item<S>(writer: &mut S, upload_id: u64, image: Attachment) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let event = context_image_item_event(upload_id, &image)?;
    send_json(writer, event).await
}

fn context_image_item_event(upload_id: u64, image: &Attachment) -> Result<Value> {
    let Attachment::Image { data_url, .. } = image else {
        bail!("Context image upload requires an image attachment");
    };
    anyhow::ensure!(
        data_url.starts_with("data:image/jpeg;base64,"),
        "Context image upload requires JPEG data"
    );
    let item_id = context_image_item_id(upload_id);
    Ok(json!({
        "type": "conversation.item.create",
        "event_id": item_id,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": openai_screen_capture_context_text(upload_id)
                },
                input_image_content(data_url.clone())
            ]
        }
    }))
}

fn codex_context_image_inject_params(
    thread_id: &str,
    upload_id: u64,
    image: &Attachment,
) -> Result<Value> {
    let Attachment::Image { data_url, .. } = image else {
        bail!("Context image upload requires an image attachment");
    };
    anyhow::ensure!(
        data_url.starts_with("data:image/jpeg;base64,"),
        "Context image upload requires JPEG data"
    );
    Ok(json!({
        "threadId": thread_id,
        "items": [{
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": gpt_live_screen_capture_context_text(upload_id)
                },
                input_image_content(data_url.clone())
            ]
        }]
    }))
}

fn openai_screen_capture_context_text(upload_id: u64) -> String {
    format!(
        "Screen capture sequence #{upload_id}. Higher sequence numbers are newer. This capture supersedes every lower-numbered screen capture; use this exact image for the current screen and never substitute an earlier capture. For a click request, call ask_text_model with include_screenshot=true and tell it to inspect the fresh screenshot and perform click_screen. Do not estimate coordinates or call click_screen directly in the OpenAI Realtime layer."
    )
}

fn gpt_live_screen_capture_context_text(upload_id: u64) -> String {
    format!(
        "Screen capture sequence #{upload_id}. Higher sequence numbers are newer. This capture supersedes every lower-numbered screen capture; use this exact image for the current screen and never substitute an earlier capture. For a click request, call click_screen directly before speaking."
    )
}

fn gpt_live_context_image_pending_params(thread_id: &str, upload_id: u64) -> Value {
    json!({
        "threadId": thread_id,
        "role": "developer",
        "text": format!(
            "Silent screen-state update: capture #{upload_id} is now newest but is still uploading. Do not acknowledge or speak because of this notice. It supersedes every lower-numbered capture immediately. For screen-dependent work, wait for the matching ready or failed notice; do not inspect, delegate, or answer from an older screenshot. Continue transcription, text conversation, and non-visual tools normally."
        )
    })
}

fn gpt_live_context_image_ready_params(thread_id: &str, upload_id: u64) -> Value {
    json!({
        "threadId": thread_id,
        "role": "developer",
        "text": format!(
            "Silent screen-state update: capture #{upload_id} is ready and is the exact latest screen in the Codex thread context. Do not acknowledge or speak because of this notice. Use this capture for current screen coordinates and call click_screen directly for click requests. Never reuse a lower-numbered capture or its result."
        )
    })
}

fn gpt_live_context_image_failed_params(thread_id: &str, upload_id: u64) -> Value {
    json!({
        "threadId": thread_id,
        "role": "developer",
        "text": format!(
            "Silent screen-state update: capture #{upload_id}, the newest capture, failed to upload. Do not acknowledge or speak because of this notice. Do not use any lower-numbered screenshot as if it were current. Continue non-visual work normally and say current visual context is unavailable only if the user's request depends on it."
        )
    })
}

fn gpt_live_system_prompt(shared_prompt: &str) -> String {
    format!(
        "{shared_prompt}

GPT-Live realtime tool rules:
- When a request needs a connector, calendar, MCP, or local tool, start the required tool or delegation immediately as the first action. Do not speak an acknowledgement or plan first.
- Consume streamed delegated results as they arrive. Reply as soon as the first reliable user-facing result is available; do not wait for optional analysis, extra searches, or a long summary.
- Keep the spoken result concise and factual."
    )
}

fn gpt_live_codex_system_prompt(shared_prompt: &str) -> String {
    format!(
        "{shared_prompt}

GPT-Live delegated execution rules:
- This is a latency-sensitive voice handoff. Use minimal reasoning for straightforward tool requests.
- Call the required connector, calendar, MCP, or local tool immediately; do not narrate or plan before the call.
- Avoid redundant discovery calls, repeated authentication checks, retries, and follow-up lookups unless the first call actually fails or lacks a required field.
- Run independent read-only calls concurrently when more than one is truly necessary.
- Emit the shortest useful final result immediately after the first successful tool response so app-server can stream it to GPT-Live."
    )
}

fn image_metadata(image: &Attachment) -> Result<(String, u32, u32, usize)> {
    match image {
        Attachment::Image {
            name,
            width,
            height,
            byte_size,
            ..
        } => Ok((name.clone(), *width, *height, *byte_size)),
        Attachment::Audio { .. } => bail!("Context image upload requires an image attachment"),
    }
}

async fn send_user_turn<S>(
    writer: &mut S,
    text: String,
    attachments: Vec<Attachment>,
    create_response: bool,
) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let mut content = Vec::<Value>::new();
    if !text.trim().is_empty() {
        content.push(json!({"type": "input_text", "text": text.trim()}));
    }
    for attachment in attachments {
        match attachment {
            Attachment::Image { data_url, .. } => {
                content.push(input_image_content(data_url));
            }
            Attachment::Audio { pcm24k, .. } => {
                content.push(json!({"type": "input_audio", "audio": encode_pcm(&pcm24k)}));
            }
        }
    }
    if content.is_empty() {
        bail!("Cannot send an empty message");
    }
    send_json(
        writer,
        json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content
            }
        }),
    )
    .await?;
    if create_response {
        send_json(writer, json!({"type": "response.create"})).await?;
    }
    Ok(())
}

pub fn default_system_prompt(_screen: ScreenInfo) -> String {
    "- When this is a voice or realtime session and the user asks to click on the screen,call the tool at start of speak by yourself, do not ask text model

- If the user asks to create an image, call create_image immediately. Use the configured image model and resolution unless the user explicitly specifies a supported model or resolution; after the tool result, describe the generated image briefly and accurately.\n\n- When a user message contains a note file change wrapped in a <note_FILENAME >...</note_FILENAME> block, reply exactly: note saved"
        .to_owned()
}

pub fn shared_system_prompt(custom: &str, screen: ScreenInfo) -> String {
    let mut prompt = default_system_prompt(screen);
    if !custom.trim().is_empty() {
        prompt.push_str("\n\nAdditional user-configured instructions:\n");
        prompt.push_str(custom.trim());
    }
    prompt
}

fn input_image_content(data_url: String) -> Value {
    json!({
        "type": "input_image",
        "image_url": data_url,
        "detail": "high"
    })
}

fn openai_context_response_blockers(
    pending: &HashMap<String, PendingOpenAiContextUpload>,
) -> HashSet<String> {
    pending.keys().cloned().collect()
}

fn openai_deferred_response_is_ready(
    blockers: Option<&HashSet<String>>,
    pending: &HashMap<String, PendingOpenAiContextUpload>,
) -> bool {
    blockers.is_some_and(|blockers| {
        blockers
            .iter()
            .all(|item_id| !pending.contains_key(item_id))
    })
}

async fn send_openai_audio_response<S>(writer: &mut S) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    send_json(
        writer,
        json!({
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"]
            }
        }),
    )
    .await
}

async fn send_json<S>(writer: &mut S, value: Value) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    writer
        .send(Message::Text(value.to_string().into()))
        .await
        .context("Could not send a Realtime event")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ServerSignal {
    None,
    ResponseStarted,
    ResponseDone,
}

#[derive(Clone, Copy, Debug)]
struct PendingOpenAiContextUpload {
    upload_id: u64,
    deadline: Instant,
}

fn expire_openai_context_uploads(
    pending: &mut HashMap<String, PendingOpenAiContextUpload>,
    now: Instant,
    events: &std::sync::mpsc::Sender<Event>,
) -> usize {
    let expired_item_ids = pending
        .iter()
        .filter(|(_, upload)| now >= upload.deadline)
        .map(|(item_id, _)| item_id.clone())
        .collect::<Vec<_>>();
    for item_id in &expired_item_ids {
        if let Some(upload) = pending.remove(item_id) {
            let _ = events.send(Event::ContextImageUploadFailed {
                upload_id: upload.upload_id,
                detail: context_image_timeout_detail(),
            });
        }
    }
    expired_item_ids.len()
}

fn handle_context_image_server_value(
    value: &Value,
    pending: &mut HashMap<String, PendingOpenAiContextUpload>,
    events: &std::sync::mpsc::Sender<Event>,
) -> bool {
    let kind = value.get("type").and_then(Value::as_str);
    let method = value.get("method").and_then(Value::as_str);
    let acknowledged_item_id = match (kind, method) {
        (Some("conversation.item.created") | Some("conversation.item.added"), _) => value
            .pointer("/item/id")
            .or_else(|| value.get("item_id"))
            .and_then(Value::as_str),
        (_, Some("thread/realtime/itemAdded")) => value
            .pointer("/params/item/id")
            .or_else(|| value.pointer("/params/item/item_id"))
            .and_then(Value::as_str),
        _ => None,
    };
    if let Some(item_id) = acknowledged_item_id
        && let Some(upload) = pending.remove(item_id)
    {
        if Instant::now() >= upload.deadline {
            let _ = events.send(Event::ContextImageUploadFailed {
                upload_id: upload.upload_id,
                detail: context_image_timeout_detail(),
            });
        } else {
            let _ = events.send(Event::ContextImageUploaded {
                upload_id: upload.upload_id,
            });
        }
        return true;
    }

    if kind == Some("error") {
        let event_id = value
            .pointer("/error/event_id")
            .or_else(|| value.get("event_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(upload) = pending.remove(event_id) {
            let detail = value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("JPEG upload was rejected")
                .to_owned();
            let _ = events.send(Event::ContextImageUploadFailed {
                upload_id: upload.upload_id,
                detail,
            });
            return true;
        }
    }
    false
}

fn handle_server_event(
    raw: &str,
    events: &std::sync::mpsc::Sender<Event>,
    handled_call_ids: &mut HashSet<String>,
    pending_context_uploads: &mut HashMap<String, PendingOpenAiContextUpload>,
    input_transcripts: &mut HashMap<String, String>,
) -> Result<ServerSignal> {
    let value: Value = serde_json::from_str(raw).context("Invalid Realtime server event")?;
    if handle_context_image_server_value(&value, pending_context_uploads, events) {
        return Ok(ServerSignal::None);
    }
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut signal = ServerSignal::None;
    match kind {
        "conversation.item.created" | "conversation.item.added" => {}
        "input_audio_buffer.speech_started" => {
            let _ = events.send(Event::SpeechStarted);
        }
        "input_audio_buffer.speech_stopped" => {
            let _ = events.send(Event::SpeechStopped);
        }
        "input_audio_buffer.committed" => {
            let item_id = value
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let _ = events.send(Event::InputCommitted { item_id });
        }
        "conversation.item.input_audio_transcription.delta" => {
            let item_id = value
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let delta = value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !item_id.is_empty() && !delta.is_empty() {
                let transcript = input_transcripts.entry(item_id.clone()).or_default();
                transcript.push_str(delta);
                let _ = events.send(Event::InputTranscript {
                    item_id,
                    text: transcript.clone(),
                });
            }
        }
        "conversation.item.input_audio_transcription.completed"
        | "conversation.item.input_audio_transcription.done" => {
            let item_id = value
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let completed = value
                .get("transcript")
                .or_else(|| value.get("text"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let accumulated = input_transcripts.remove(&item_id).unwrap_or_default();
            let text = if completed.is_empty() {
                accumulated
            } else {
                completed.to_owned()
            };
            let _ = events.send(Event::InputTranscript { item_id, text });
        }
        "response.created" => {
            signal = ServerSignal::ResponseStarted;
            let response_id = value
                .pointer("/response/id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let _ = events.send(Event::AssistantResponseStarted { response_id });
        }
        "response.output_item.added" | "response.output_item.created" => {
            let item_type = value
                .pointer("/item/type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if item_type != "message" {
                return Ok(signal);
            }
            let response_id = value
                .get("response_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let item_id = value
                .pointer("/item/id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !item_id.is_empty() {
                let _ = events.send(Event::AssistantItem {
                    response_id,
                    item_id,
                });
            }
        }
        "response.output_audio.delta" | "response.audio.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str)
                && let Ok(bytes) = STANDARD.decode(delta)
            {
                let response_id = value
                    .get("response_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let samples = bytes
                    .chunks_exact(2)
                    .map(|v| i16::from_le_bytes([v[0], v[1]]))
                    .collect();
                let _ = events.send(Event::AssistantAudio {
                    response_id,
                    samples,
                });
            }
        }
        "response.output_audio_transcript.delta"
        | "response.audio_transcript.delta"
        | "response.output_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                let response_id = value
                    .get("response_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let _ = events.send(Event::AssistantTranscriptDelta {
                    response_id,
                    delta: delta.to_owned(),
                });
            }
        }
        "response.function_call_arguments.done" => {
            if let Some(call) = extract_function_call_event(&value)
                && handled_call_ids.insert(call.call_id.clone())
            {
                let _ = events.send(Event::ToolCalls(vec![call]));
            }
        }
        "response.done" => {
            signal = ServerSignal::ResponseDone;
            let calls = extract_function_calls(&value)
                .into_iter()
                .filter(|call| handled_call_ids.insert(call.call_id.clone()))
                .collect::<Vec<_>>();
            if !calls.is_empty() {
                let _ = events.send(Event::ToolCalls(calls));
            }
            let response_id = value
                .pointer("/response/id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !response_id.is_empty()
                && let Some(total_tokens) = response_total_tokens(&value)
            {
                let _ = events.send(Event::AssistantUsage {
                    response_id: response_id.clone(),
                    total_tokens,
                });
            }
            let _ = events.send(Event::AssistantDone { response_id });
        }
        "error" => {
            let message = value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Unknown Realtime API error");
            let event_id = value
                .pointer("/error/event_id")
                .or_else(|| value.get("event_id"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(upload) = pending_context_uploads.remove(event_id) {
                let _ = events.send(Event::ContextImageUploadFailed {
                    upload_id: upload.upload_id,
                    detail: message.to_owned(),
                });
            } else {
                let _ = events.send(Event::Error(message.to_owned()));
            }
        }
        _ => {}
    }
    Ok(signal)
}

fn extract_function_call_event(value: &Value) -> Option<ToolCall> {
    let call_id = value.get("call_id")?.as_str()?.to_owned();
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = value.get("arguments")?.as_str()?.to_owned();
    (!call_id.is_empty() && !name.is_empty()).then_some(ToolCall {
        call_id,
        name,
        arguments,
        requested_at: Instant::now(),
    })
}

fn extract_function_calls(value: &Value) -> Vec<ToolCall> {
    value
        .pointer("/response/output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .filter_map(|item| {
            let call_id = item.get("call_id")?.as_str()?.to_owned();
            let name = item.get("name")?.as_str()?.to_owned();
            let arguments = item.get("arguments")?.as_str()?.to_owned();
            (!call_id.is_empty() && !name.is_empty()).then_some(ToolCall {
                call_id,
                name,
                arguments,
                requested_at: Instant::now(),
            })
        })
        .collect()
}

fn encode_pcm(samples: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    STANDARD.encode(bytes)
}

impl Drop for RealtimeClient {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        time::{Duration, Instant},
    };

    use super::{
        CODEX_HANDOFF_FALLBACK_DELAY, CONNECT_AUDIO_BUFFER_MAX_SAMPLES,
        CONTEXT_IMAGE_UPLOAD_TIMEOUT, CodexContextImageResponse, CodexHandoffAction,
        CodexHandoffState, CodexLiveState, CodexTextState, ConnectOptions, Event,
        InFlightContextImage, NATIVE_INPUT_PRE_ROLL_SAMPLES, NATIVE_REMOTE_TAIL_SAMPLES,
        NativeInputAudioCapture, NativeRemoteAudioGate, OPENAI_VAD_SILENCE_MS, PendingAudioBuffer,
        PendingOpenAiContextUpload, RealtimeBackend, ServerSignal,
        codex_context_image_inject_params, codex_dynamic_tools, codex_live_initialize_capabilities,
        codex_live_realtime_start_params, codex_live_start_error, codex_live_thread_start_params,
        codex_message_is_assistant_transcript, codex_message_starts_reply,
        codex_text_thread_start_params, codex_text_turn_start_params, codex_turn_input,
        context_image_item_event, context_image_item_id, context_image_upload_id,
        decode_audio_to_24k_mono, dynamic_tool_content_items, dynamic_tool_request,
        emit_codex_live_remote_audio, encode_pcm, expire_codex_context_images,
        expire_openai_context_uploads, extract_function_call_event, extract_function_calls,
        gpt_live_context_image_failed_params, gpt_live_context_image_pending_params,
        gpt_live_context_image_ready_params, handle_codex_context_image_response,
        handle_codex_handoff_message, handle_codex_live_message, handle_codex_text_message,
        handle_context_image_server_value, handle_server_event, input_image_content,
        local_tool_fast_speech, mcp_tool_fast_speech, openai_context_response_blockers,
        openai_deferred_response_is_ready, openai_function_output, response_total_tokens,
        shared_system_prompt, take_latest_ready_codex_context_image, voice_tools,
    };
    use crate::media::{Attachment, ScreenInfo, jpeg_upload_probe_attachment};
    use base64::{Engine, engine::general_purpose::STANDARD};
    use image::ImageFormat;
    use serde_json::{Value, json};

    #[test]
    fn native_input_audio_capture_flushes_bounded_preroll_and_active_audio() {
        let (events, received) = std::sync::mpsc::channel();
        let mut capture = NativeInputAudioCapture::default();
        capture.push(vec![1; NATIVE_INPUT_PRE_ROLL_SAMPLES / 2], false, &events);
        capture.push(vec![2; NATIVE_INPUT_PRE_ROLL_SAMPLES], false, &events);
        assert!(received.try_recv().is_err());

        capture.sync(true, &events);
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputAudio { samples }
                if samples == vec![2; NATIVE_INPUT_PRE_ROLL_SAMPLES]
        ));

        capture.push(vec![3; 240], true, &events);
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputAudio { samples } if samples == vec![3; 240]
        ));
        capture.sync(false, &events);
        capture.push(vec![4; 240], false, &events);
        assert!(received.try_recv().is_err());
    }

    #[test]
    fn native_remote_audio_gate_ignores_comfort_noise_and_keeps_one_tail() {
        let mut gate = NativeRemoteAudioGate::default();
        assert!(gate.push(vec![0; 240]).is_empty());

        let started = gate.push(vec![100; 240]);
        assert_eq!(started, vec![vec![0; 240], vec![100; 240]]);

        assert_eq!(
            gate.push(vec![0; NATIVE_REMOTE_TAIL_SAMPLES]),
            vec![vec![0; NATIVE_REMOTE_TAIL_SAMPLES]]
        );
        assert!(gate.push(vec![0; 240]).is_empty());
    }

    #[test]
    fn preconnect_audio_buffer_preserves_chunk_order() {
        let mut audio = PendingAudioBuffer::default();
        audio.push(vec![1, 2]);
        audio.push(vec![3, 4, 5]);

        assert_eq!(audio.sample_count(), 5);
        assert_eq!(audio.pop_front(), Some(vec![1, 2]));
        assert_eq!(audio.pop_front(), Some(vec![3, 4, 5]));
        assert_eq!(audio.pop_front(), None);
        assert_eq!(audio.sample_count(), 0);
    }

    #[test]
    fn response_usage_supports_openai_and_codex_shapes() {
        assert_eq!(
            response_total_tokens(&json!({
                "response": {"usage": {"total_tokens": 42}}
            })),
            Some(42)
        );
        assert_eq!(
            response_total_tokens(&json!({
                "params": {"turn": {"usage": {"inputTokens": 10, "outputTokens": 7}}}
            })),
            Some(17)
        );
    }

    #[test]
    fn preconnect_audio_buffer_keeps_only_latest_minute() {
        let mut audio = PendingAudioBuffer::default();
        let samples = (0..CONNECT_AUDIO_BUFFER_MAX_SAMPLES + 3)
            .map(|index| (index % i16::MAX as usize) as i16)
            .collect::<Vec<_>>();
        let expected = samples[3..].to_vec();
        audio.push(samples);

        assert_eq!(audio.sample_count(), CONNECT_AUDIO_BUFFER_MAX_SAMPLES);
        assert_eq!(audio.pop_front(), Some(expected));
    }

    #[test]
    fn openai_voice_end_detection_is_low_latency() {
        assert_eq!(OPENAI_VAD_SILENCE_MS, 300);
    }

    #[test]
    fn extracts_all_completed_function_calls() {
        let event = json!({
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "name": "click_screen",
                        "call_id": "call_1",
                        "arguments": "{\"x\":12,\"y\":34}"
                    },
                    {
                        "type": "message",
                        "content": []
                    },
                    {
                        "type": "function_call",
                        "name": "insert_text",
                        "call_id": "call_2",
                        "arguments": "{\"text\":\"hello\"}"
                    }
                ]
            }
        });

        let calls = extract_function_calls(&event);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "click_screen");
        assert_eq!(calls[0].arguments, "{\"x\":12,\"y\":34}");
        assert_eq!(calls[1].call_id, "call_2");
        assert_eq!(calls[1].name, "insert_text");
        assert_eq!(calls[1].arguments, "{\"text\":\"hello\"}");
    }

    #[test]
    fn extracts_streamed_function_call_as_soon_as_arguments_finish() {
        let event = json!({
            "type": "response.function_call_arguments.done",
            "name": "click_screen",
            "call_id": "call_fast",
            "arguments": "{\"x\":500,\"y\":300}"
        });

        let call = extract_function_call_event(&event).unwrap();
        assert_eq!(call.call_id, "call_fast");
        assert_eq!(call.name, "click_screen");
        assert_eq!(call.arguments, "{\"x\":500,\"y\":300}");
    }

    #[test]
    fn shared_prompt_uses_the_simple_direct_tool_rules() {
        let prompt = shared_system_prompt(
            "Call me Ecoo.",
            ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        );

        assert!(prompt.starts_with(
            "- When this is a voice or realtime session and the user asks to click on the screen,call the tool at start of speak by yourself, do not ask text model"
        ));
        assert!(prompt.contains("call create_image immediately"));
        assert!(prompt.contains("reply exactly: note saved"));
        assert!(prompt.contains("configured image model and resolution"));
        assert!(!prompt.contains("Call ask_text_model as your first output"));
        assert!(!prompt.contains("GPT-Live visual-context rules"));
        assert!(prompt.contains(
            "Additional user-configured instructions:
Call me Ecoo."
        ));
    }

    #[test]
    fn codex_thread_uses_the_shared_prompt_verbatim() {
        let prompt = "exact shared prompt".to_owned();
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexGptLive,
            api_key: "oauth-secret".to_owned(),
            chatgpt_account_id: Some("account-123".to_owned()),
            model: "unused".to_owned(),
            voice: "cove".to_owned(),
            thinking_level: "low".to_owned(),
            system_prompt: prompt.clone(),
            screen_info: ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        };
        let params = codex_live_thread_start_params(&options, prompt.clone(), "/tmp".to_owned());
        assert_eq!(params["baseInstructions"], prompt);
    }

    #[test]
    fn image_inputs_always_request_high_detail() {
        let content = input_image_content("data:image/png;base64,abc".to_owned());
        assert_eq!(content["type"], "input_image");
        assert_eq!(content["detail"], "high");
    }

    #[test]
    fn codex_live_audio_is_downmixed_and_resampled_to_24k() {
        let stereo_48k = [1_000, 3_000, 2_000, 4_000, 3_000, 5_000, 4_000, 6_000];
        let decoded = decode_audio_to_24k_mono(&encode_pcm(&stereo_48k), 48_000, 2).unwrap();
        assert_eq!(decoded, vec![2_000, 4_000]);
    }

    #[test]
    fn codex_oauth_uses_app_server_managed_auth_for_webrtc() {
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexGptLive,
            api_key: "oauth-secret".to_owned(),
            chatgpt_account_id: Some("account-123".to_owned()),
            model: "unused".to_owned(),
            voice: "cove".to_owned(),
            thinking_level: "low".to_owned(),
            system_prompt: "shared prompt".to_owned(),
            screen_info: ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        };
        let params =
            codex_live_thread_start_params(&options, "instructions".to_owned(), "/tmp".to_owned());

        assert_eq!(params["ephemeral"], true);
        assert!(params.get("modelProvider").is_none());
        assert_eq!(params["config"]["features.realtime_conversation"], true);
        assert_eq!(params["config"]["suppress_unstable_features_warning"], true);
        assert_eq!(params["baseInstructions"], "instructions");
    }

    #[test]
    fn codex_live_thread_registers_local_dynamic_tools() {
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexGptLive,
            api_key: "oauth-secret".to_owned(),
            chatgpt_account_id: Some("account-123".to_owned()),
            model: "unused".to_owned(),
            voice: "ember".to_owned(),
            thinking_level: "low".to_owned(),
            system_prompt: "shared prompt".to_owned(),
            screen_info: ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        };
        let params =
            codex_live_thread_start_params(&options, "instructions".to_owned(), "/tmp".to_owned());
        let tools = params["dynamicTools"].as_array().unwrap();
        assert_eq!(tools.len(), 10);
        assert!(tools.iter().any(|tool| tool["name"] == "move_pointer"));
        assert!(tools.iter().any(|tool| tool["name"] == "click_screen"));
        assert!(tools.iter().any(|tool| tool["name"] == "run_bash"));
        assert!(tools.iter().any(|tool| tool["name"] == "insert_text"));
        assert!(tools.iter().any(|tool| tool["name"] == "ask_text_model"));
        assert!(tools.iter().any(|tool| tool["name"] == "create_image"));
        for name in [
            "replace_note_text",
            "remove_note_text",
            "rename_note",
            "create_note",
        ] {
            assert!(tools.iter().any(|tool| tool["name"] == name));
        }
        assert!(tools.iter().all(|tool| tool.get("inputSchema").is_some()));
    }

    #[test]
    fn voice_and_text_tool_sets_have_the_expected_direct_click_boundary() {
        let screen = ScreenInfo {
            origin_x: 0,
            origin_y: 0,
            logical_width: 1408,
            logical_height: 881,
            backing_width: 2816,
            backing_height: 1762,
            scale_factor: 2.0,
        };
        let voice = voice_tools(screen).as_array().unwrap().clone();
        let text = codex_dynamic_tools(screen).as_array().unwrap().clone();
        assert!(voice.iter().any(|tool| tool["name"] == "ask_text_model"));
        assert!(voice.iter().any(|tool| tool["name"] == "click_screen"));
        assert!(text.iter().any(|tool| tool["name"] == "click_screen"));
        assert!(!text.iter().any(|tool| tool["name"] == "ask_text_model"));
        assert!(voice.iter().any(|tool| tool["name"] == "create_image"));
        assert!(text.iter().any(|tool| tool["name"] == "create_image"));

        let ask = voice
            .iter()
            .find(|tool| tool["name"] == "ask_text_model")
            .unwrap();
        assert_eq!(ask["parameters"]["required"], json!(["prompt"]));
        assert_eq!(
            ask["parameters"]["properties"]["thinking_level"]["enum"],
            json!(["minimal", "low", "medium", "high", "xhigh", "ultra"])
        );
        assert_eq!(ask["parameters"]["properties"]["image"]["type"], "string");
        assert_eq!(
            ask["parameters"]["properties"]["include_screenshot"]["type"],
            "boolean"
        );
        assert!(
            ask["parameters"]["properties"]["include_screenshot"]["description"]
                .as_str()
                .unwrap()
                .contains("fresh current-screen screenshot")
        );
    }

    #[test]
    fn openai_context_image_ack_confirms_matching_upload() {
        let (events, received) = std::sync::mpsc::channel();
        let mut handled = HashSet::new();
        let item_id = context_image_item_id(42);
        let mut pending = HashMap::from([(
            item_id.clone(),
            PendingOpenAiContextUpload {
                upload_id: 42,
                deadline: Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT,
            },
        )]);
        let mut transcripts = HashMap::new();
        let signal = handle_server_event(
            &json!({
                "type": "conversation.item.created",
                "item": {"id": item_id}
            })
            .to_string(),
            &events,
            &mut handled,
            &mut pending,
            &mut transcripts,
        )
        .unwrap();

        assert_eq!(signal, ServerSignal::None);
        assert!(pending.is_empty());
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploaded { upload_id: 42 }
        ));
    }

    #[test]
    fn context_image_wire_event_contains_decodable_jpeg() {
        let image = jpeg_upload_probe_attachment().unwrap();
        let item_id = context_image_item_id(77);
        let event = context_image_item_event(77, &image).unwrap();

        assert_eq!(event["type"], "conversation.item.create");
        assert_eq!(event["event_id"], item_id);
        assert_eq!(event["item"]["id"], item_id);
        assert_eq!(event["item"]["content"][0]["type"], "input_text");
        let label = event["item"]["content"][0]["text"].as_str().unwrap();
        assert!(label.contains("#77"));
        assert!(label.contains("supersedes"));
        assert!(label.contains("ask_text_model"));
        assert!(label.contains("include_screenshot=true"));
        assert!(label.contains("Do not estimate coordinates or call click_screen directly"));
        assert_eq!(event["item"]["content"][1]["type"], "input_image");
        assert_eq!(event["item"]["content"][1]["detail"], "high");
        let data_url = event["item"]["content"][1]["image_url"].as_str().unwrap();
        let bytes = STANDARD
            .decode(data_url.strip_prefix("data:image/jpeg;base64,").unwrap())
            .unwrap();
        assert!(bytes.len() > 65_536);
        assert_eq!(image::guess_format(&bytes).unwrap(), ImageFormat::Jpeg);
    }

    #[test]
    fn codex_context_injection_contains_decodable_jpeg() {
        let image = jpeg_upload_probe_attachment().unwrap();
        let params = codex_context_image_inject_params("thread-1", 78, &image).unwrap();

        assert_eq!(params["threadId"], "thread-1");
        assert_eq!(params["items"][0]["type"], "message");
        assert_eq!(params["items"][0]["role"], "user");
        assert_eq!(params["items"][0]["content"][0]["type"], "input_text");
        let label = params["items"][0]["content"][0]["text"].as_str().unwrap();
        assert!(label.contains("#78"));
        assert!(label.contains("call click_screen directly"));
        assert!(!label.contains("ask_text_model"));
        assert_eq!(params["items"][0]["content"][1]["type"], "input_image");
        assert_eq!(params["items"][0]["content"][1]["detail"], "high");
        let data_url = params["items"][0]["content"][1]["image_url"]
            .as_str()
            .unwrap();
        let bytes = STANDARD
            .decode(data_url.strip_prefix("data:image/jpeg;base64,").unwrap())
            .unwrap();
        assert!(bytes.len() > 65_536);
        assert_eq!(image::guess_format(&bytes).unwrap(), ImageFormat::Jpeg);
    }

    #[test]
    fn gpt_live_image_order_notices_stay_small_and_text_only() {
        for params in [
            gpt_live_context_image_pending_params("thread-1", 79),
            gpt_live_context_image_ready_params("thread-1", 79),
            gpt_live_context_image_failed_params("thread-1", 79),
        ] {
            assert_eq!(params["threadId"], "thread-1");
            assert_eq!(params["role"], "developer");
            assert!(params["text"].as_str().unwrap().contains("#79"));
            let encoded = serde_json::to_vec(&params).unwrap();
            assert!(encoded.len() < 512);
            assert!(
                !encoded
                    .windows(b"base64".len())
                    .any(|part| part == b"base64")
            );
            assert!(params.get("items").is_none());
        }
    }

    #[test]
    fn both_realtime_ack_variants_confirm_jpeg_uploads() {
        for (upload_id, kind) in [
            (81, "conversation.item.created"),
            (82, "conversation.item.added"),
        ] {
            let (events, received) = std::sync::mpsc::channel();
            let item_id = context_image_item_id(upload_id);
            let mut pending = HashMap::from([(
                item_id.clone(),
                PendingOpenAiContextUpload {
                    upload_id,
                    deadline: Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT,
                },
            )]);
            assert!(handle_context_image_server_value(
                &json!({"type": kind, "item": {"id": item_id}}),
                &mut pending,
                &events,
            ));
            assert!(pending.is_empty());
            assert!(matches!(
                received.recv().unwrap(),
                Event::ContextImageUploaded { upload_id: confirmed }
                    if confirmed == upload_id
            ));
        }
    }

    #[test]
    fn codex_sideband_item_added_confirms_matching_jpeg_upload() {
        let (events, received) = std::sync::mpsc::channel();
        let item_id = context_image_item_id(83);
        let mut pending = HashMap::from([(
            item_id.clone(),
            PendingOpenAiContextUpload {
                upload_id: 83,
                deadline: Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT,
            },
        )]);

        assert!(handle_context_image_server_value(
            &json!({
                "method": "thread/realtime/itemAdded",
                "params": {"item": {"id": item_id}}
            }),
            &mut pending,
            &events,
        ));
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploaded { upload_id: 83 }
        ));
    }

    #[test]
    fn openai_context_image_error_fails_matching_upload() {
        let (events, received) = std::sync::mpsc::channel();
        let mut handled = HashSet::new();
        let event_id = context_image_item_id(9);
        let mut pending = HashMap::from([(
            event_id.clone(),
            PendingOpenAiContextUpload {
                upload_id: 9,
                deadline: Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT,
            },
        )]);
        let mut transcripts = HashMap::new();
        handle_server_event(
            &json!({
                "type": "error",
                "error": {
                    "event_id": event_id,
                    "message": "image rejected"
                }
            })
            .to_string(),
            &events,
            &mut handled,
            &mut pending,
            &mut transcripts,
        )
        .unwrap();

        assert!(pending.is_empty());
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploadFailed { upload_id: 9, detail }
                if detail == "image rejected"
        ));
    }

    #[test]
    fn openai_context_image_timeout_ignores_late_acknowledgement() {
        let (events, received) = std::sync::mpsc::channel();
        let now = Instant::now();
        let item_id = context_image_item_id(10);
        let mut pending = HashMap::from([(
            item_id.clone(),
            PendingOpenAiContextUpload {
                upload_id: 10,
                deadline: now,
            },
        )]);

        assert_eq!(expire_openai_context_uploads(&mut pending, now, &events), 1);
        assert!(pending.is_empty());
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploadFailed { upload_id: 10, detail }
                if detail.contains("10 seconds")
        ));

        let mut handled = HashSet::new();
        let mut transcripts = HashMap::new();
        handle_server_event(
            &json!({
                "type": "conversation.item.created",
                "item": {"id": item_id}
            })
            .to_string(),
            &events,
            &mut handled,
            &mut pending,
            &mut transcripts,
        )
        .unwrap();
        assert!(received.try_recv().is_err());
    }

    #[test]
    fn openai_response_waits_for_every_screenshot_that_was_pending_when_requested() {
        let deadline = Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT;
        let mut pending = HashMap::from([
            (
                context_image_item_id(1),
                PendingOpenAiContextUpload {
                    upload_id: 1,
                    deadline,
                },
            ),
            (
                context_image_item_id(2),
                PendingOpenAiContextUpload {
                    upload_id: 2,
                    deadline,
                },
            ),
        ]);
        let blockers = openai_context_response_blockers(&pending);

        assert_eq!(blockers.len(), 2);
        assert!(!openai_deferred_response_is_ready(
            Some(&blockers),
            &pending
        ));
        pending.remove(&context_image_item_id(2));
        assert!(!openai_deferred_response_is_ready(
            Some(&blockers),
            &pending
        ));
        pending.remove(&context_image_item_id(1));
        assert!(openai_deferred_response_is_ready(Some(&blockers), &pending));

        // A screenshot queued after response.create belongs to a later turn and
        // must not retroactively block the already deferred response.
        pending.insert(
            context_image_item_id(3),
            PendingOpenAiContextUpload {
                upload_id: 3,
                deadline,
            },
        );
        assert!(openai_deferred_response_is_ready(Some(&blockers), &pending));
    }

    #[test]
    fn codex_live_out_of_order_acks_announce_only_the_latest_screenshot() {
        let (events, received) = std::sync::mpsc::channel();
        let deadline = Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT;
        let upload = |upload_id| InFlightContextImage {
            upload_id,
            name: format!("screen-{upload_id}.jpg"),
            width: 1408,
            height: 881,
            byte_size: 3,
            turn_id: "turn-1".to_owned(),
            deadline,
        };
        let mut in_flight = HashMap::from([(71, upload(1)), (72, upload(2))]);

        assert_eq!(
            handle_codex_context_image_response(
                &json!({"jsonrpc": "2.0", "id": 72, "result": {}}),
                &mut in_flight,
                &events,
            ),
            CodexContextImageResponse::Uploaded(2)
        );
        let mut latest_ready = Some(2);
        assert!(in_flight.contains_key(&71));
        assert!(!in_flight.contains_key(&72));
        assert_eq!(
            take_latest_ready_codex_context_image(&mut latest_ready, 2, &in_flight),
            None
        );
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploaded { upload_id: 2 }
        ));

        assert_eq!(
            handle_codex_context_image_response(
                &json!({"jsonrpc": "2.0", "id": 71, "result": {}}),
                &mut in_flight,
                &events,
            ),
            CodexContextImageResponse::Uploaded(1)
        );
        assert!(in_flight.is_empty());
        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploaded { upload_id: 1 }
        ));
        assert_eq!(
            take_latest_ready_codex_context_image(&mut latest_ready, 2, &in_flight),
            Some(2)
        );
        assert_eq!(latest_ready, None);
    }

    #[test]
    fn codex_live_context_image_times_out_in_flight() {
        let (events, received) = std::sync::mpsc::channel();
        let now = Instant::now();
        let mut in_flight = HashMap::from([(
            71,
            InFlightContextImage {
                upload_id: 1,
                name: "in-flight.jpg".to_owned(),
                width: 1408,
                height: 881,
                byte_size: 3,
                turn_id: "turn-1".to_owned(),
                deadline: now,
            },
        )]);

        assert_eq!(
            expire_codex_context_images(&mut in_flight, now, &events),
            vec![1]
        );
        assert!(in_flight.is_empty());

        assert!(matches!(
            received.recv().unwrap(),
            Event::ContextImageUploadFailed { upload_id: 1, detail }
                if detail.contains("10 seconds")
        ));
    }

    #[test]
    fn openai_transcription_delta_is_visible_before_turn_completion() {
        let (events, received) = std::sync::mpsc::channel();
        let mut handled = HashSet::new();
        let mut pending = HashMap::new();
        let mut transcripts = HashMap::new();

        for delta in ["hello", " world"] {
            handle_server_event(
                &json!({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "voice-1",
                    "delta": delta
                })
                .to_string(),
                &events,
                &mut handled,
                &mut pending,
                &mut transcripts,
            )
            .unwrap();
        }

        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == "voice-1" && text == "hello"
        ));
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == "voice-1" && text == "hello world"
        ));
    }

    #[test]
    fn context_image_upload_item_ids_round_trip() {
        let item_id = context_image_item_id(1234);
        assert_eq!(context_image_upload_id(&item_id), Some(1234));
        assert_eq!(context_image_upload_id("unrelated"), None);
    }

    #[test]
    fn codex_live_image_turn_contains_text_and_image() {
        let attachments = vec![Attachment::Image {
            name: "screen.jpg".to_owned(),
            data_url: "data:image/jpeg;base64,abc".to_owned(),
            thumbnail: vec![],
            width: 1408,
            height: 881,
            byte_size: 3,
        }];
        let input = codex_turn_input("What is on screen?", &attachments);
        assert_eq!(input[0]["type"], "text");
        assert_eq!(input[0]["text"], "What is on screen?");
        assert_eq!(input[0]["text_elements"], json!([]));
        assert_eq!(input[1]["type"], "image");
        assert_eq!(input[1]["url"], "data:image/jpeg;base64,abc");
    }

    #[test]
    fn parses_codex_dynamic_tool_server_request() {
        let message = json!({
            "id": 77,
            "method": "item/tool/call",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "callId": "call-77",
                "namespace": null,
                "tool": "run_bash",
                "arguments": {"command": "pwd"}
            }
        });
        let (request_id, call) = dynamic_tool_request(&message).unwrap();
        assert_eq!(request_id, json!(77));
        assert_eq!(call.call_id, "call-77");
        assert_eq!(call.name, "run_bash");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap()["command"],
            "pwd"
        );
    }

    #[test]
    fn image_tool_results_return_text_and_image_content_items() {
        let output = json!({
            "ok": true,
            "model": "gpt-image-2",
            "resolution": "1024x1024",
            "image_url": "data:image/png;base64,aGVsbG8="
        })
        .to_string();
        let (content_items, success) = dynamic_tool_content_items(&output);
        assert!(success);
        assert_eq!(content_items.as_array().unwrap().len(), 2);
        assert_eq!(content_items[0]["type"], "inputText");
        assert!(
            !content_items[0]["text"]
                .as_str()
                .unwrap()
                .contains("aGVsbG8=")
        );
        assert_eq!(content_items[1]["type"], "inputImage");
        assert_eq!(
            content_items[1]["imageUrl"],
            "data:image/png;base64,aGVsbG8="
        );
    }

    #[test]
    fn openai_voice_image_results_are_compact() {
        let output = json!({
            "ok": true,
            "model": "gpt-image-2",
            "resolution": "1024x1024",
            "image_url": format!("data:image/png;base64,{}", "x".repeat(1000))
        })
        .to_string();
        let compact = openai_function_output(&output);
        assert!(compact.len() < output.len());
        let value: Value = serde_json::from_str(&compact).unwrap();
        assert_eq!(value["image_attached"], true);
        assert!(value.get("image_url").is_none());
        assert_eq!(
            value["message"],
            "The generated image is attached in the app."
        );
    }

    #[test]
    fn codex_live_user_transcript_delta_is_visible_immediately() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState {
            input_item_id: Some("voice-1".to_owned()),
            ..Default::default()
        };
        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "user", "delta": "hello"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == "voice-1" && text == "hello"
        ));
    }

    #[test]
    fn codex_live_consecutive_user_utterances_get_unique_turn_ids() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "user", "delta": "first"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(received.recv().unwrap(), Event::SpeechStarted));
        let first_id = match received.recv().unwrap() {
            Event::InputCommitted { item_id } => item_id,
            other => panic!("expected first InputCommitted, got {other:?}"),
        };
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == first_id && text == "first"
        ));

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/done",
                "params": {"role": "user", "text": "first"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == first_id && text == "first"
        ));
        assert!(matches!(received.recv().unwrap(), Event::SpeechStopped));

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "user", "delta": "second"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(received.recv().unwrap(), Event::SpeechStarted));
        let second_id = match received.recv().unwrap() {
            Event::InputCommitted { item_id } => item_id,
            other => panic!("expected second InputCommitted, got {other:?}"),
        };
        assert_ne!(first_id, second_id);
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputTranscript { item_id, text }
                if item_id == second_id && text == "second"
        ));
    }

    #[test]
    fn codex_live_user_speech_does_not_finish_overlapping_assistant_response() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();
        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "assistant", "delta": "translating"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        let response_id = match received.recv().unwrap() {
            Event::AssistantResponseStarted { response_id } => response_id,
            other => panic!("expected AssistantResponseStarted, got {other:?}"),
        };
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id: id, delta }
                if id == response_id && delta == "translating"
        ));

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/itemAdded",
                "params": {"item": {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "overlap-user-1"
                }}
            }),
            &events,
            &mut state,
        )
        .unwrap();

        assert!(matches!(received.recv().unwrap(), Event::SpeechStarted));
        assert!(matches!(
            received.recv().unwrap(),
            Event::InputCommitted { item_id } if item_id == "overlap-user-1"
        ));
        assert_eq!(
            state.active_response_id.as_deref(),
            Some(response_id.as_str())
        );
        assert!(received.try_recv().is_err());
    }

    #[test]
    fn codex_live_transcript_done_keeps_continuous_response_open() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();
        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "assistant", "delta": "first segment"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        let response_id = match received.recv().unwrap() {
            Event::AssistantResponseStarted { response_id } => response_id,
            other => panic!("expected AssistantResponseStarted, got {other:?}"),
        };
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id: id, delta }
                if id == response_id && delta == "first segment"
        ));

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/done",
                "params": {"role": "assistant", "text": "first segment"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert_eq!(
            state.active_response_id.as_deref(),
            Some(response_id.as_str())
        );
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantSegmentDone { response_id: id } if id == response_id
        ));
        assert!(state.response_finish_deadline.is_some());
        assert!(received.try_recv().is_err());

        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "assistant", "delta": " second segment"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id: id, delta }
                if id == response_id && delta == " second segment"
        ));
        assert!(state.response_finish_deadline.is_none());
        assert!(received.try_recv().is_err());
    }

    #[test]
    fn codex_live_completed_response_finishes_after_quiet_tail() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();
        let response_id = state.ensure_response(&events);
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantResponseStarted { response_id: id } if id == response_id
        ));
        state.response_finish_deadline = Some(Instant::now() - Duration::from_millis(1));
        assert!(state.finish_response_if_due(Instant::now(), &events));
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantDone { response_id: id } if id == response_id
        ));
        assert!(state.active_response_id.is_none());
        assert!(state.response_finish_deadline.is_none());
    }

    #[test]
    fn codex_live_audio_starts_reply_before_sideband_transcript() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();

        assert!(emit_codex_live_remote_audio(
            &mut state,
            &events,
            vec![1; 480]
        ));
        let response_id = match received.recv().unwrap() {
            Event::AssistantResponseStarted { response_id } => response_id,
            other => panic!("unexpected event: {other:?}"),
        };
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantAudio {
                response_id: id,
                samples
            } if id == response_id && samples.len() == 480
        ));
        assert_eq!(
            state.active_response_id.as_deref(),
            Some(response_id.as_str())
        );
    }

    #[test]
    fn codex_live_sideband_identifies_assistant_transcript() {
        assert!(codex_message_is_assistant_transcript(&json!({
            "method": "thread/realtime/transcript/delta",
            "params": {"role": "assistant", "delta": "hello"}
        })));
        assert!(!codex_message_is_assistant_transcript(&json!({
            "method": "thread/realtime/transcript/delta",
            "params": {"role": "user", "delta": "hello"}
        })));
    }

    #[test]
    fn codex_text_stream_emits_standard_app_server_assistant_events() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexTextState::default();

        handle_codex_text_message(
            &json!({
                "method": "turn/started",
                "params": {"turn": {"id": "turn-1"}}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantResponseStarted { response_id } if response_id == "turn-1"
        ));

        handle_codex_text_message(
            &json!({
                "method": "item/agentMessage/delta",
                "params": {"delta": "hello"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id, delta }
                if response_id == "turn-1" && delta == "hello"
        ));

        handle_codex_text_message(
            &json!({
                "method": "item/agentMessage/delta",
                "params": {"delta": " world"}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id, delta }
                if response_id == "turn-1" && delta == " world"
        ));

        handle_codex_text_message(
            &json!({
                "method": "item/completed",
                "params": {
                    "item": {"type": "agentMessage", "text": "hello world"}
                }
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(received.try_recv().is_err());

        handle_codex_text_message(
            &json!({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed"}}
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantDone { response_id } if response_id == "turn-1"
        ));
        assert!(state.active_response_id.is_none());
    }

    #[test]
    fn codex_text_thread_uses_selected_model_and_local_tools() {
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexText,
            api_key: "oauth-secret".to_owned(),
            chatgpt_account_id: Some("account-123".to_owned()),
            model: "gpt-5.6-sol".to_owned(),
            voice: "unused".to_owned(),
            thinking_level: "low".to_owned(),
            system_prompt: "shared prompt".to_owned(),
            screen_info: ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        };
        let params =
            codex_text_thread_start_params(&options, "instructions".to_owned(), "/tmp".to_owned());

        assert_eq!(params["model"], "gpt-5.6-sol");
        assert_eq!(params["baseInstructions"], "instructions");
        assert_eq!(params["reasoningEffort"], "low");
        let tools = params["dynamicTools"].as_array().unwrap();
        assert_eq!(tools.len(), 9);
        assert!(tools.iter().any(|tool| tool["name"] == "click_screen"));
        assert!(tools.iter().any(|tool| tool["name"] == "run_bash"));
        assert!(!tools.iter().any(|tool| tool["name"] == "ask_text_model"));
        assert!(tools.iter().any(|tool| tool["name"] == "create_image"));
        for name in [
            "replace_note_text",
            "remove_note_text",
            "rename_note",
            "create_note",
        ] {
            assert!(tools.iter().any(|tool| tool["name"] == name));
        }
    }

    #[test]
    fn text_turn_carries_the_selected_thinking_effort() {
        let params = codex_text_turn_start_params("thread-1", "click the corner", &[], "high");
        assert_eq!(params["threadId"], "thread-1");
        assert_eq!(params["effort"], "high");
        assert_eq!(params["input"][0]["text"], "click the corner");
    }

    #[test]
    fn completed_calendar_mcp_result_uses_fast_speech_lane() {
        let fast = mcp_tool_fast_speech(&json!({
            "method": "item/completed",
            "params": {
                "turnId": "turn-calendar",
                "item": {
                    "type": "mcpToolCall",
                    "status": "completed",
                    "server": "codex_apps",
                    "tool": "calendar_search",
                    "appContext": {"appName": "Google Calendar"},
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "Team sync is at 3 PM."
                        }],
                        "structuredContent": null
                    },
                    "error": null
                }
            }
        }))
        .expect("fast Calendar result");
        assert_eq!(fast.turn_id, "turn-calendar");
        assert_eq!(fast.label, "Google Calendar");
        assert!(fast.text.contains("Team sync is at 3 PM."));
        assert_eq!(fast.text, "Team sync is at 3 PM.");
    }

    #[test]
    fn failed_or_large_mcp_result_does_not_use_fast_speech_lane() {
        assert!(
            mcp_tool_fast_speech(&json!({
                "method": "item/completed",
                "params": {
                    "turnId": "turn-failed",
                    "item": {
                        "type": "mcpToolCall",
                        "status": "failed",
                        "server": "codex_apps",
                        "tool": "calendar_search",
                        "result": null,
                        "error": {"message": "auth failed"}
                    }
                }
            }))
            .is_none()
        );
        let huge = "x".repeat(2_001);
        assert!(
            mcp_tool_fast_speech(&json!({
                "method": "item/completed",
                "params": {
                    "turnId": "turn-large",
                    "item": {
                        "type": "mcpToolCall",
                        "status": "completed",
                        "server": "codex_apps",
                        "tool": "calendar_search",
                        "result": {"content": [{"type": "text", "text": huge}]},
                        "error": null
                    }
                }
            }))
            .is_none()
        );
    }

    #[test]
    fn pointer_and_insert_results_use_exact_fast_speech() {
        assert_eq!(
            local_tool_fast_speech(
                "click_screen",
                r#"{"ok":true,"assistant_reply":"Done","assistant_reply_exact":true}"#,
                true,
            ),
            Some("Done".to_owned())
        );
        assert_eq!(
            local_tool_fast_speech("insert_text", r#"{"ok":true}"#, true),
            Some("Done.".to_owned())
        );
        assert!(local_tool_fast_speech("insert_text", r#"{"ok":false}"#, false).is_none());
    }

    #[test]
    fn gpt_live_uses_streaming_handoffs_and_no_reasoning() {
        let params = codex_live_realtime_start_params(
            "thread-1",
            "v=0\r\n".to_owned(),
            "ember",
            "prompt".to_owned(),
        );
        assert_eq!(params["clientManagedHandoffs"], false);
        assert_eq!(params["codexResponsesAsItems"], false);
        assert_eq!(params["delegationAckFiller"], false);
        assert_eq!(params["codexResponseHandoffMode"], "thinking");

        let options = ConnectOptions {
            backend: RealtimeBackend::CodexGptLive,
            api_key: "secret".to_owned(),
            chatgpt_account_id: Some("account".to_owned()),
            model: "unused".to_owned(),
            voice: "ember".to_owned(),
            thinking_level: "high".to_owned(),
            system_prompt: "prompt".to_owned(),
            screen_info: ScreenInfo {
                origin_x: 0,
                origin_y: 0,
                logical_width: 1408,
                logical_height: 881,
                backing_width: 2816,
                backing_height: 1762,
                scale_factor: 2.0,
            },
        };
        let thread =
            codex_live_thread_start_params(&options, "prompt".to_owned(), "/tmp".to_owned());
        assert_eq!(thread["reasoningEffort"], "none");
    }

    #[test]
    fn spoken_output_cancels_completed_handoff_fallback() {
        let mut state = CodexHandoffState::default();
        state.schedule_fallback("result".to_owned());
        state.note_spoken_output();
        assert_eq!(
            state.take_fallback_if_due(
                Instant::now() + CODEX_HANDOFF_FALLBACK_DELAY + Duration::from_secs(1)
            ),
            None
        );
    }

    #[test]
    fn completed_handoff_schedules_exact_fallback_result() {
        let (events, _received) = std::sync::mpsc::channel();
        let mut state = CodexHandoffState::default();

        assert_eq!(
            handle_codex_handoff_message(
                &json!({
                    "method": "turn/started",
                    "params": {"turn": {"id": "visual-turn-2"}}
                }),
                &events,
                &mut state,
            )
            .unwrap(),
            CodexHandoffAction::Handled
        );
        assert_eq!(
            handle_codex_handoff_message(
                &json!({
                    "method": "item/agentMessage/delta",
                    "params": {"delta": "dog"}
                }),
                &events,
                &mut state,
            )
            .unwrap(),
            CodexHandoffAction::Handled
        );
        assert_eq!(
            handle_codex_handoff_message(
                &json!({
                    "method": "turn/completed",
                    "params": {"turn": {"id": "visual-turn-2", "status": "completed"}}
                }),
                &events,
                &mut state,
            )
            .unwrap(),
            CodexHandoffAction::Handled
        );
        assert!(state.active_turn_id.is_none());
        assert!(state.response_text.is_empty());
        assert_eq!(
            state.take_fallback_if_due(
                Instant::now() + CODEX_HANDOFF_FALLBACK_DELAY + Duration::from_millis(1)
            ),
            Some("dog".to_owned())
        );
    }

    #[test]
    fn codex_live_webrtc_opts_out_of_duplicate_sideband_audio() {
        let capabilities = codex_live_initialize_capabilities();
        assert_eq!(capabilities["experimentalApi"], true);
        assert_eq!(
            capabilities["optOutNotificationMethods"],
            json!(["thread/realtime/outputAudio/delta"])
        );
    }

    #[test]
    fn codex_live_sideband_audio_is_ignored_for_webrtc() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();
        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/outputAudio/delta",
                "params": {
                    "audio": {
                        "data": encode_pcm(&[1_000, -1_000]),
                        "sampleRate": 24_000,
                        "numChannels": 1
                    }
                }
            }),
            &events,
            &mut state,
        )
        .unwrap();
        assert!(received.try_recv().is_err());
        assert!(state.active_response_id.is_none());
        assert!(!codex_message_starts_reply(&json!({
            "method": "thread/realtime/outputAudio/delta"
        })));
    }

    #[test]
    fn codex_live_reply_watchdog_clears_on_handoff_and_assistant_output() {
        assert!(codex_message_starts_reply(&json!({
            "method": "thread/realtime/itemAdded",
            "params": {"item": {"type": "handoff_request"}}
        })));
        assert!(codex_message_starts_reply(&json!({
            "method": "thread/realtime/transcript/delta",
            "params": {"role": "assistant", "delta": "hello"}
        })));
        assert!(!codex_message_starts_reply(&json!({
            "method": "thread/realtime/transcript/delta",
            "params": {"role": "user", "delta": "hello"}
        })));
    }

    #[test]
    fn codex_live_access_denial_has_actionable_error() {
        let message = codex_live_start_error("Voice session access denied.");
        assert!(message.contains("not enabled"));
        assert!(message.contains("OpenAI Realtime"));
    }

    #[test]
    fn codex_live_transcript_starts_a_ui_response() {
        let (events, received) = std::sync::mpsc::channel();
        let mut state = CodexLiveState::default();
        handle_codex_live_message(
            &json!({
                "method": "thread/realtime/transcript/delta",
                "params": {"role": "assistant", "delta": "Hello"}
            }),
            &events,
            &mut state,
        )
        .unwrap();

        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantResponseStarted { response_id }
                if response_id == "codex-live-1"
        ));
        assert!(matches!(
            received.recv().unwrap(),
            Event::AssistantTranscriptDelta { response_id, delta }
                if response_id == "codex-live-1" && delta == "Hello"
        ));
    }
}

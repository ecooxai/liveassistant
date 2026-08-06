use crate::{
    auth::{self, CodexCredentials},
    image_generation,
    media::{self, ScreenInfo},
    realtime::{self, Command, ConnectOptions, Event, RealtimeBackend, RealtimeClient, ToolOutput},
    tools,
};
use anyhow::Context as _;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message as WsMessage, WebSocket},
    },
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use futures_util::{Sink, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, RwLock, mpsc},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_CSS: &str = include_str!("../web/app.css");
const APP_JS: &str = include_str!("../web/app.js");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ConnectionState {
    #[default]
    Offline,
    Connecting,
    Live,
    Reconnecting,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AuthMode {
    #[default]
    ApiKey,
    Codex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct PublicSettings {
    backend: RealtimeBackend,
    auth_mode: AuthMode,
    model: String,
    voice: String,
    thinking_level: String,
    system_prompt: String,
    send_screenshot: bool,
    image_model: String,
    image_resolution: String,
}

impl Default for PublicSettings {
    fn default() -> Self {
        Self {
            backend: RealtimeBackend::CodexGptLive,
            auth_mode: AuthMode::Codex,
            model: "gpt-realtime-2.1".to_owned(),
            voice: "ember".to_owned(),
            thinking_level: "low".to_owned(),
            system_prompt: realtime::default_system_prompt(primary_screen_info()),
            send_screenshot: true,
            image_model: "gpt-image-2".to_owned(),
            image_resolution: "1024x1024".to_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, Serialize)]
struct ToolView {
    call_id: String,
    name: String,
    arguments: String,
    status: String,
    output: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatMessage {
    id: u64,
    role: MessageRole,
    text: String,
    created_at: u64,
    streaming: bool,
    tool_calls: Vec<ToolView>,
    image_url: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AppState {
    revision: u64,
    connection: ConnectionState,
    status: String,
    error: Option<String>,
    has_credentials: bool,
    settings: PublicSettings,
    messages: Vec<ChatMessage>,
    last_activity_at: u64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            revision: 0,
            connection: ConnectionState::Offline,
            status: "Ready".to_owned(),
            error: None,
            has_credentials: std::env::var("OPENAI_API_KEY")
                .ok()
                .is_some_and(|value| !value.trim().is_empty()),
            settings: PublicSettings::default(),
            messages: vec![ChatMessage {
                id: 1,
                role: MessageRole::System,
                text: "Rust backend ready. Connect a model to begin.".to_owned(),
                created_at: unix_seconds(),
                streaming: false,
                tool_calls: Vec::new(),
                image_url: None,
            }],
            last_activity_at: unix_seconds(),
        }
    }
}

#[derive(Clone)]
struct Backend {
    state: Arc<RwLock<AppState>>,
    actions: mpsc::Sender<Action>,
    push: broadcast::Sender<PushEvent>,
}

#[derive(Clone, Debug)]
enum PushEvent {
    StateChanged,
    Audio(Vec<i16>),
}

enum Action {
    Connect { api_key: Option<String> },
    Disconnect,
    SendText(String),
    Audio(Vec<i16>),
    Clear,
    UpdateSettings(PublicSettings),
    ToolFinished {
        call_id: String,
        output: String,
        image_url: Option<String>,
    },
    Shutdown,
}

#[derive(Deserialize)]
struct ConnectRequest {
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct MessageRequest {
    text: String,
}

pub async fn run(port: u16, open_browser: bool) -> anyhow::Result<()> {
    let backend = Backend::spawn();
    let app = Router::new()
        .route("/", get(index))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/health", get(health))
        .route("/api/state", get(api_state))
        .route("/api/connect", post(api_connect))
        .route("/api/disconnect", post(api_disconnect))
        .route("/api/message", post(api_message))
        .route("/api/settings", post(api_settings))
        .route("/api/clear", post(api_clear))
        .route("/api/audio", post(api_audio))
        .route("/ws", get(ws_handler))
        .with_state(backend.clone());

    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| format!("Could not bind Live Assistant to http://{address}"))?;
    let url = format!("http://127.0.0.1:{port}");
    println!("Live Assistant is running at {url}");
    if open_browser {
        open_url(&url);
    }

    let shutdown_backend = backend.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            let _ = shutdown_backend.actions.send(Action::Shutdown);
        })
        .await
        .context("Live Assistant server stopped unexpectedly")?;
    Ok(())
}

impl Backend {
    fn spawn() -> Self {
        let state = Arc::new(RwLock::new(AppState::default()));
        let (actions, action_rx) = mpsc::channel();
        let (push, _) = broadcast::channel(128);
        let backend = Self {
            state: state.clone(),
            actions: actions.clone(),
            push: push.clone(),
        };
        thread::spawn(move || worker_loop(state, action_rx, actions, push));
        backend
    }

    fn snapshot(&self) -> AppState {
        self.state.read().expect("app state read lock").clone()
    }
}

fn worker_loop(
    state: Arc<RwLock<AppState>>,
    action_rx: mpsc::Receiver<Action>,
    action_tx: mpsc::Sender<Action>,
    push: broadcast::Sender<PushEvent>,
) {
    let realtime = RealtimeClient::spawn();
    let mut active_assistant: Option<usize> = None;
    let mut input_messages: HashMap<String, usize> = HashMap::new();
    let mut credentials: Option<CodexCredentials> = None;
    let mut next_message_id = 2_u64;

    'worker: loop {
        while let Ok(action) = action_rx.try_recv() {
            match action {
                Action::Connect { api_key } => {
                    let settings = state.read().expect("app state read lock").settings.clone();
                    match resolve_credentials(settings.auth_mode, api_key) {
                        Ok(resolved) => {
                            credentials = Some(resolved.clone());
                            let screen_info = primary_screen_info();
                            let system_prompt = if settings.system_prompt.trim().is_empty() {
                                realtime::default_system_prompt(screen_info)
                            } else {
                                settings.system_prompt.clone()
                            };
                            let options = ConnectOptions {
                                backend: settings.backend,
                                api_key: resolved.bearer_token,
                                chatgpt_account_id: resolved.chatgpt_account_id,
                                model: settings.model,
                                voice: settings.voice,
                                thinking_level: settings.thinking_level,
                                system_prompt,
                                screen_info,
                            };
                            if realtime.commands.send(Command::Connect(options)).is_err() {
                                set_error(&state, &push, "Could not start the realtime transport");
                            } else {
                                mutate_state(&state, &push, |app| {
                                    app.connection = ConnectionState::Connecting;
                                    app.status = "Connecting".to_owned();
                                    app.error = None;
                                    app.has_credentials = true;
                                });
                            }
                        }
                        Err(error) => set_error(&state, &push, &format!("{error:#}")),
                    }
                }
                Action::Disconnect => {
                    let _ = realtime.commands.send(Command::Disconnect);
                    active_assistant = None;
                    input_messages.clear();
                    mutate_state(&state, &push, |app| {
                        app.connection = ConnectionState::Offline;
                        app.status = "Ready".to_owned();
                    });
                }
                Action::SendText(text) => {
                    let text = text.trim().to_owned();
                    if text.is_empty() {
                        continue;
                    }
                    let is_live = state
                        .read()
                        .expect("app state read lock")
                        .connection
                        == ConnectionState::Live;
                    if !is_live {
                        set_error(&state, &push, "Connect a model before sending a message");
                        continue;
                    }
                    let settings = state.read().expect("app state read lock").settings.clone();
                    let message_id = next_message_id;
                    next_message_id = next_message_id.saturating_add(1);
                    mutate_state(&state, &push, |app| {
                        app.messages.push(ChatMessage {
                            id: message_id,
                            role: MessageRole::User,
                            text: text.clone(),
                            created_at: unix_seconds(),
                            streaming: false,
                            tool_calls: Vec::new(),
                            image_url: None,
                        });
                        app.status = "Thinking".to_owned();
                        app.error = None;
                    });
                    let attachments = if settings.send_screenshot {
                        let screen = primary_screen_info();
                        media::capture_screenshot(
                            screen.logical_width,
                            screen.logical_height,
                            false,
                            None,
                        )
                        .ok()
                        .into_iter()
                        .collect()
                    } else {
                        Vec::new()
                    };
                    if realtime
                        .commands
                        .send(Command::SendTurn {
                            text,
                            attachments,
                            thinking_level: settings.thinking_level,
                        })
                        .is_err()
                    {
                        set_error(&state, &push, "Could not send the message to the backend");
                    }
                }
                Action::Audio(samples) => {
                    if !samples.is_empty() {
                        let _ = realtime.commands.send(Command::AudioChunk(samples));
                    }
                }
                Action::Clear => {
                    active_assistant = None;
                    input_messages.clear();
                    mutate_state(&state, &push, |app| {
                        app.messages.clear();
                        app.status = if app.connection == ConnectionState::Live {
                            "Listening".to_owned()
                        } else {
                            "Ready".to_owned()
                        };
                        app.error = None;
                    });
                }
                Action::UpdateSettings(settings) => mutate_state(&state, &push, |app| {
                    app.settings = settings;
                    app.error = None;
                }),
                Action::ToolFinished {
                    call_id,
                    output,
                    image_url,
                } => {
                    mutate_state(&state, &push, |app| {
                        for message in app.messages.iter_mut().rev() {
                            if let Some(tool) = message
                                .tool_calls
                                .iter_mut()
                                .find(|tool| tool.call_id == call_id)
                            {
                                tool.status = "done".to_owned();
                                tool.output = Some(compact_output(&output));
                                if image_url.is_some() {
                                    message.image_url = image_url.clone();
                                }
                                break;
                            }
                        }
                        app.status = "Thinking".to_owned();
                    });
                    let _ = realtime.commands.send(Command::ToolOutputs(vec![ToolOutput {
                        call_id,
                        output,
                    }]));
                }
                Action::Shutdown => {
                    let _ = realtime.commands.send(Command::Shutdown);
                    break 'worker;
                }
            }
        }

        match realtime.events.recv_timeout(Duration::from_millis(25)) {
            Ok(event) => handle_realtime_event(
                event,
                &state,
                &push,
                &action_tx,
                &credentials,
                &mut active_assistant,
                &mut input_messages,
                &mut next_message_id,
            ),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_realtime_event(
    event: Event,
    state: &Arc<RwLock<AppState>>,
    push: &broadcast::Sender<PushEvent>,
    action_tx: &mpsc::Sender<Action>,
    credentials: &Option<CodexCredentials>,
    active_assistant: &mut Option<usize>,
    input_messages: &mut HashMap<String, usize>,
    next_message_id: &mut u64,
) {
    match event {
        Event::Connecting => mutate_state(state, push, |app| {
            app.connection = ConnectionState::Connecting;
            app.status = "Connecting".to_owned();
            app.error = None;
        }),
        Event::Reconnecting { attempt, reason } => mutate_state(state, push, |app| {
            app.connection = ConnectionState::Reconnecting;
            app.status = format!("Reconnecting · attempt {attempt}");
            app.error = Some(reason);
        }),
        Event::Connected => mutate_state(state, push, |app| {
            app.connection = ConnectionState::Live;
            app.status = if app.settings.backend == RealtimeBackend::CodexText {
                "Ready".to_owned()
            } else {
                "Listening".to_owned()
            };
            app.error = None;
        }),
        Event::Disconnected => mutate_state(state, push, |app| {
            app.connection = ConnectionState::Offline;
            app.status = "Ready".to_owned();
        }),
        Event::SpeechStarted => {
            let message_id = take_message_id(next_message_id);
            mutate_state(state, push, |app| {
                app.status = "Listening".to_owned();
                app.messages.push(ChatMessage {
                    id: message_id,
                    role: MessageRole::User,
                    text: "Listening…".to_owned(),
                    created_at: unix_seconds(),
                    streaming: true,
                    tool_calls: Vec::new(),
                    image_url: None,
                });
            });
        }
        Event::SpeechStopped => mutate_state(state, push, |app| {
            app.status = "Thinking".to_owned();
            if let Some(message) = app
                .messages
                .iter_mut()
                .rev()
                .find(|message| message.role == MessageRole::User && message.streaming)
            {
                message.streaming = false;
            }
        }),
        Event::InputCommitted { item_id } => {
            let index = state
                .read()
                .expect("app state read lock")
                .messages
                .iter()
                .rposition(|message| message.role == MessageRole::User);
            if let Some(index) = index {
                input_messages.insert(item_id, index);
            }
        }
        Event::InputTranscript { item_id, text } => mutate_state(state, push, |app| {
            let index = input_messages.get(&item_id).copied().or_else(|| {
                app.messages
                    .iter()
                    .rposition(|message| message.role == MessageRole::User)
            });
            if let Some(index) = index
                && let Some(message) = app.messages.get_mut(index)
            {
                message.text = text;
                message.streaming = false;
            }
        }),
        Event::AssistantResponseStarted { .. } => {
            let message_id = take_message_id(next_message_id);
            mutate_state(state, push, |app| {
                app.messages.push(ChatMessage {
                    id: message_id,
                    role: MessageRole::Assistant,
                    text: String::new(),
                    created_at: unix_seconds(),
                    streaming: true,
                    tool_calls: Vec::new(),
                    image_url: None,
                });
                *active_assistant = Some(app.messages.len() - 1);
                app.status = "Responding".to_owned();
            });
        }
        Event::AssistantTranscriptDelta { delta, .. } => mutate_state(state, push, |app| {
            let index = ensure_assistant_message(app, active_assistant, next_message_id);
            if let Some(message) = app.messages.get_mut(index) {
                message.text.push_str(&delta);
            }
        }),
        Event::AssistantAudio { samples, .. } => {
            let _ = push.send(PushEvent::Audio(samples));
        }
        Event::AssistantDone { .. } => {
            mutate_state(state, push, |app| {
                if let Some(index) = *active_assistant
                    && let Some(message) = app.messages.get_mut(index)
                {
                    message.streaming = false;
                }
                app.status = if app.settings.backend == RealtimeBackend::CodexText {
                    "Ready".to_owned()
                } else {
                    "Listening".to_owned()
                };
            });
            *active_assistant = None;
        }
        Event::ToolCalls(calls) => {
            let settings = state.read().expect("app state read lock").settings.clone();
            let screen = primary_screen_info();
            mutate_state(state, push, |app| {
                let index = ensure_assistant_message(app, active_assistant, next_message_id);
                if let Some(message) = app.messages.get_mut(index) {
                    message.tool_calls.extend(calls.iter().map(|call| ToolView {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: pretty_json(&call.arguments),
                        status: "running".to_owned(),
                        output: None,
                    }));
                }
                app.status = format!(
                    "Running {} tool{}",
                    calls.len(),
                    if calls.len() == 1 { "" } else { "s" }
                );
            });
            for call in calls {
                let action_tx = action_tx.clone();
                let credentials = credentials.clone();
                let settings = settings.clone();
                thread::spawn(move || {
                    let (output, image_url) = if call.name == "create_image" {
                        run_image_tool(&call.call_id, &call.arguments, &settings, credentials)
                    } else {
                        let context = tools::ScreenContext {
                            screenshot_width: screen.logical_width,
                            screenshot_height: screen.logical_height,
                        };
                        (
                            tools::execute_with_context(&call.name, &call.arguments, context),
                            None,
                        )
                    };
                    let _ = action_tx.send(Action::ToolFinished {
                        call_id: call.call_id,
                        output,
                        image_url,
                    });
                });
            }
        }
        Event::ToolOutputsSubmitted { .. } => mutate_state(state, push, |app| {
            app.status = "Thinking".to_owned();
        }),
        Event::Error(error) => set_error(state, push, &error),
        Event::ContextImageAccepted { .. }
        | Event::ContextImageUploaded { .. }
        | Event::ContextImageUploadFailed { .. }
        | Event::AssistantItem { .. }
        | Event::AssistantSegmentDone { .. }
        | Event::AssistantUsage { .. } => {}
    }
}

fn ensure_assistant_message(
    app: &mut AppState,
    active_assistant: &mut Option<usize>,
    next_message_id: &mut u64,
) -> usize {
    if let Some(index) = *active_assistant
        && app.messages.get(index).is_some()
    {
        return index;
    }
    let message_id = take_message_id(next_message_id);
    app.messages.push(ChatMessage {
        id: message_id,
        role: MessageRole::Assistant,
        text: String::new(),
        created_at: unix_seconds(),
        streaming: true,
        tool_calls: Vec::new(),
        image_url: None,
    });
    let index = app.messages.len() - 1;
    *active_assistant = Some(index);
    index
}

fn take_message_id(next_message_id: &mut u64) -> u64 {
    let id = *next_message_id;
    *next_message_id = next_message_id.saturating_add(1);
    id
}

fn resolve_credentials(auth_mode: AuthMode, api_key: Option<String>) -> anyhow::Result<CodexCredentials> {
    if auth_mode == AuthMode::Codex {
        return auth::codex_credentials();
    }
    let key = api_key
        .filter(|value| !value.trim().is_empty())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .context("Enter an OpenAI API key or set OPENAI_API_KEY")?;
    Ok(CodexCredentials {
        bearer_token: key,
        chatgpt_account_id: None,
    })
}

fn run_image_tool(
    call_id: &str,
    arguments: &str,
    settings: &PublicSettings,
    credentials: Option<CodexCredentials>,
) -> (String, Option<String>) {
    let result = credentials
        .context("No active credentials are available for image generation")
        .and_then(|credentials| {
            image_generation::request_from_tool_arguments(
                arguments,
                &settings.image_model,
                &settings.image_resolution,
                call_id,
            )
            .and_then(|request| image_generation::generate(request, &credentials))
        });
    match result {
        Ok(result) => {
            let image_url = result.data_url.clone();
            (
                json!({
                    "ok": true,
                    "model": result.model,
                    "resolution": result.resolution,
                    "image_url": result.data_url,
                })
                .to_string(),
                Some(image_url),
            )
        }
        Err(error) => (
            json!({"ok": false, "error": format!("{error:#}")}).to_string(),
            None,
        ),
    }
}

fn mutate_state(
    state: &Arc<RwLock<AppState>>,
    push: &broadcast::Sender<PushEvent>,
    mutate: impl FnOnce(&mut AppState),
) {
    {
        let mut app = state.write().expect("app state write lock");
        mutate(&mut app);
        app.revision = app.revision.saturating_add(1);
        app.last_activity_at = unix_seconds();
    }
    let _ = push.send(PushEvent::StateChanged);
}

fn set_error(state: &Arc<RwLock<AppState>>, push: &broadcast::Sender<PushEvent>, error: &str) {
    mutate_state(state, push, |app| {
        app.error = Some(error.to_owned());
        app.status = "Needs attention".to_owned();
        if app.connection != ConnectionState::Live {
            app.connection = ConnectionState::Offline;
        }
    });
}

fn primary_screen_info() -> ScreenInfo {
    media::primary_screen_info().unwrap_or(ScreenInfo {
        origin_x: 0,
        origin_y: 0,
        logical_width: 1440,
        logical_height: 900,
        backing_width: 1440,
        backing_height: 900,
        scale_factor: 1.0,
    })
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn pretty_json(value: &str) -> String {
    serde_json::from_str::<Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| value.to_owned())
}

fn compact_output(value: &str) -> String {
    let mut output = value.to_owned();
    if output.len() > 1_500 {
        output.truncate(1_500);
        output.push('…');
    }
    output
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn css() -> Response {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS).into_response()
}

async fn js() -> Response {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], APP_JS).into_response()
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true, "service": "live-assistant"}))
}

async fn api_state(State(backend): State<Backend>) -> Json<AppState> {
    Json(backend.snapshot())
}

async fn api_connect(
    State(backend): State<Backend>,
    Json(request): Json<ConnectRequest>,
) -> impl IntoResponse {
    send_action(&backend, Action::Connect { api_key: request.api_key })
}

async fn api_disconnect(State(backend): State<Backend>) -> impl IntoResponse {
    send_action(&backend, Action::Disconnect)
}

async fn api_message(
    State(backend): State<Backend>,
    Json(request): Json<MessageRequest>,
) -> impl IntoResponse {
    send_action(&backend, Action::SendText(request.text))
}

async fn api_settings(
    State(backend): State<Backend>,
    Json(settings): Json<PublicSettings>,
) -> impl IntoResponse {
    send_action(&backend, Action::UpdateSettings(settings))
}

async fn api_clear(State(backend): State<Backend>) -> impl IntoResponse {
    send_action(&backend, Action::Clear)
}

async fn api_audio(State(backend): State<Backend>, body: Bytes) -> impl IntoResponse {
    send_action(&backend, Action::Audio(pcm16_from_bytes(&body)))
}

fn send_action(backend: &Backend, action: Action) -> (StatusCode, Json<Value>) {
    match backend.actions.send(action) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"ok": true}))),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "error": "Backend worker is unavailable"})),
        ),
    }
}

async fn ws_handler(ws: WebSocketUpgrade, State(backend): State<Backend>) -> Response {
    ws.on_upgrade(move |socket| ws_session(socket, backend))
}

async fn ws_session(socket: WebSocket, backend: Backend) {
    let (mut sender, mut receiver) = socket.split();
    if send_state_ws(&mut sender, &backend.snapshot()).await.is_err() {
        return;
    }
    let mut push = backend.push.subscribe();
    loop {
        tokio::select! {
            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(WsMessage::Binary(bytes))) => {
                        let _ = backend.actions.send(Action::Audio(pcm16_from_bytes(&bytes)));
                    }
                    Some(Ok(WsMessage::Close(_))) | Some(Err(_)) | None => break,
                    _ => {}
                }
            }
            event = push.recv() => {
                match event {
                    Ok(PushEvent::StateChanged) => {
                        if send_state_ws(&mut sender, &backend.snapshot()).await.is_err() {
                            break;
                        }
                    }
                    Ok(PushEvent::Audio(samples)) => {
                        if sender
                            .send(WsMessage::Binary(pcm16_to_bytes(&samples).into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        if send_state_ws(&mut sender, &backend.snapshot()).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

async fn send_state_ws<S>(sender: &mut S, state: &AppState) -> Result<(), axum::Error>
where
    S: Sink<WsMessage, Error = axum::Error> + Unpin,
{
    let payload = serde_json::to_string(&json!({"type": "state", "state": state}))
        .unwrap_or_else(|_| "{\"type\":\"error\"}".to_owned());
    sender.send(WsMessage::Text(payload.into())).await
}

fn pcm16_from_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn pcm16_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[cfg(target_os = "macos")]
fn open_url(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}

#[cfg(target_os = "linux")]
fn open_url(url: &str) {
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

#[cfg(target_os = "windows")]
fn open_url(url: &str) {
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn open_url(_url: &str) {}

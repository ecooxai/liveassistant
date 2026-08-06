use crate::{
    auth::{self, CodexCredentials},
    codex_account, image_generation,
    media::{self, Attachment, ScreenInfo},
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
    collections::{HashMap, HashSet},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, RwLock, mpsc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_CSS: &str = include_str!("../web/app.css");
const APP_JS: &str = include_str!("../web/app.js");
const CONTEXT_UPLOAD_TIMEOUT: Duration = Duration::from_secs(10);

const FALLBACK_REALTIME_MODELS: &[(&str, &str, &str)] = &[
    (
        "gpt-realtime-2.1",
        "GPT-Realtime 2.1",
        "Reasoning voice model with improved interruption, noise handling, and tool use.",
    ),
    (
        "gpt-realtime-2.1-mini",
        "GPT-Realtime 2.1 mini",
        "Smaller Realtime 2.1 model for lower-latency voice sessions.",
    ),
    (
        "gpt-realtime-2",
        "GPT-Realtime 2",
        "Reasoning speech-to-speech model with tool use.",
    ),
    (
        "gpt-realtime-1.5",
        "GPT-Realtime 1.5",
        "General voice-agent model for audio input and output.",
    ),
    (
        "gpt-realtime",
        "GPT-Realtime",
        "General-availability realtime audio and text model.",
    ),
];
const FALLBACK_TEXT_MODELS: &[(&str, &str, &str)] = &[
    (
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "Frontier model for complex reasoning and coding.",
    ),
    (
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "Balanced intelligence, latency, and cost.",
    ),
    (
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "Cost-sensitive model for high-volume text work.",
    ),
    ("gpt-5.6", "GPT-5.6", "Latest GPT-5.6 alias."),
    (
        "chat-latest",
        "Chat Latest",
        "Latest Instant model used by ChatGPT.",
    ),
];
const FALLBACK_REALTIME_VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar",
];
const FALLBACK_GPT_LIVE_VOICES: &[&str] = &[
    "juniper", "maple", "spruce", "ember", "vale", "breeze", "arbor", "sol", "cove",
];

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

impl PublicSettings {
    fn for_backend(backend: RealtimeBackend, model: Option<String>) -> Self {
        let mut settings = Self::default();
        settings.backend = backend;
        match backend {
            RealtimeBackend::CodexGptLive => {
                settings.auth_mode = AuthMode::Codex;
                settings.model = model.unwrap_or_else(|| "gpt-realtime-2.1".to_owned());
                settings.voice = "ember".to_owned();
            }
            RealtimeBackend::OpenAiRealtime => {
                settings.auth_mode = AuthMode::ApiKey;
                settings.model = model.unwrap_or_else(|| "gpt-realtime-2.1".to_owned());
                settings.voice = "marin".to_owned();
            }
            RealtimeBackend::CodexText => {
                settings.auth_mode = AuthMode::Codex;
                settings.model = model.unwrap_or_else(|| "gpt-5.6-luna".to_owned());
                settings.voice.clear();
                settings.send_screenshot = false;
            }
        }
        settings
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
struct AttachmentView {
    id: u64,
    name: String,
    kind: String,
    data_url: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    byte_size: usize,
    seconds: Option<f32>,
    status: String,
    included_screen: bool,
    upload_id: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatMessage {
    id: u64,
    role: MessageRole,
    text: String,
    started_at: u64,
    finished_at: Option<u64>,
    elapsed_ms: Option<u64>,
    streaming: bool,
    tool_calls: Vec<ToolView>,
    attachments: Vec<AttachmentView>,
    generated_image_url: Option<String>,
    response_id: Option<String>,
    token_count: Option<u64>,
    token_count_is_estimate: bool,
    tokens_per_second: Option<f64>,
}

impl ChatMessage {
    fn new(id: u64, role: MessageRole, text: String, streaming: bool) -> Self {
        let mut message = Self {
            id,
            role,
            text,
            started_at: unix_millis(),
            finished_at: None,
            elapsed_ms: None,
            streaming,
            tool_calls: Vec::new(),
            attachments: Vec::new(),
            generated_image_url: None,
            response_id: None,
            token_count: None,
            token_count_is_estimate: false,
            tokens_per_second: None,
        };
        message.refresh_token_estimate();
        message
    }

    fn finish(&mut self) {
        if self.finished_at.is_none() {
            let finished = unix_millis();
            self.finished_at = Some(finished);
            self.elapsed_ms = Some(finished.saturating_sub(self.started_at));
        }
        self.streaming = false;
        if !self.token_count_is_estimate && self.token_count.is_some() {
            self.recompute_tokens_per_second();
        } else {
            self.refresh_token_estimate();
        }
    }

    fn refresh_token_estimate(&mut self) {
        if !self.token_count_is_estimate && self.token_count.is_some() {
            return;
        }
        self.token_count = Some(estimated_message_tokens(self));
        self.token_count_is_estimate = true;
        self.recompute_tokens_per_second();
    }

    fn set_actual_usage(&mut self, total_tokens: u64) {
        self.token_count = Some(total_tokens);
        self.token_count_is_estimate = false;
        self.recompute_tokens_per_second();
    }

    fn recompute_tokens_per_second(&mut self) {
        self.tokens_per_second = self.elapsed_ms.and_then(|elapsed| {
            let seconds = elapsed as f64 / 1_000.0;
            (seconds > 0.0).then(|| self.token_count.unwrap_or_default() as f64 / seconds)
        });
    }
}

#[derive(Clone, Debug, Serialize)]
struct ModelOption {
    id: String,
    label: String,
    description: String,
    source: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelCatalog {
    loading: bool,
    error: Option<String>,
    refreshed_at: Option<u64>,
    gpt_live_models: Vec<ModelOption>,
    realtime_models: Vec<ModelOption>,
    text_models: Vec<ModelOption>,
    realtime_voices: Vec<String>,
    gpt_live_voices: Vec<String>,
    image_models: Vec<ModelOption>,
    warnings: Vec<String>,
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self {
            loading: true,
            error: None,
            refreshed_at: None,
            gpt_live_models: fallback_realtime_models("official"),
            realtime_models: fallback_realtime_models("official"),
            text_models: fallback_text_models("official"),
            realtime_voices: FALLBACK_REALTIME_VOICES
                .iter()
                .map(|voice| (*voice).to_owned())
                .collect(),
            gpt_live_voices: FALLBACK_GPT_LIVE_VOICES
                .iter()
                .map(|voice| (*voice).to_owned())
                .collect(),
            image_models: vec![
                model_option(
                    "gpt-image-2",
                    "GPT Image 2",
                    "Latest image generation model.",
                    "official",
                ),
                model_option(
                    "gpt-image-1.5",
                    "GPT Image 1.5",
                    "Previous image generation model.",
                    "official",
                ),
                model_option(
                    "gpt-image-1",
                    "GPT Image 1",
                    "Legacy image generation model.",
                    "official",
                ),
                model_option(
                    "gpt-image-1-mini",
                    "GPT Image 1 mini",
                    "Smaller image generation model.",
                    "official",
                ),
            ],
            warnings: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct TabView {
    id: u64,
    title: String,
    connection: ConnectionState,
    status: String,
    error: Option<String>,
    settings: PublicSettings,
    messages: Vec<ChatMessage>,
    pending_attachments: Vec<AttachmentView>,
}

impl TabView {
    fn new(id: u64, settings: PublicSettings) -> Self {
        Self {
            id,
            title: tab_title(&settings),
            connection: ConnectionState::Offline,
            status: "Ready".to_owned(),
            error: None,
            settings,
            messages: Vec::new(),
            pending_attachments: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct AppState {
    revision: u64,
    active_tab_id: u64,
    tabs: Vec<TabView>,
    catalog: ModelCatalog,
    has_credentials: bool,
    last_activity_at: u64,
}

impl Default for AppState {
    fn default() -> Self {
        let gpt_live = TabView::new(
            1,
            PublicSettings::for_backend(RealtimeBackend::CodexGptLive, None),
        );
        let realtime = TabView::new(
            2,
            PublicSettings::for_backend(RealtimeBackend::OpenAiRealtime, None),
        );
        Self {
            revision: 0,
            active_tab_id: 1,
            tabs: vec![gpt_live, realtime],
            catalog: ModelCatalog::default(),
            has_credentials: std::env::var("OPENAI_API_KEY")
                .ok()
                .is_some_and(|value| !value.trim().is_empty()),
            last_activity_at: unix_millis(),
        }
    }
}

struct PendingAttachment {
    view: AttachmentView,
    attachment: Attachment,
}

struct RuntimeTab {
    client: RealtimeClient,
    credentials: Option<CodexCredentials>,
    active_assistant: Option<usize>,
    input_messages: HashMap<String, usize>,
    pending: Vec<PendingAttachment>,
    active_voice_message: Option<usize>,
    next_upload_id: u64,
}

impl RuntimeTab {
    fn new() -> Self {
        Self {
            client: RealtimeClient::spawn(),
            credentials: None,
            active_assistant: None,
            input_messages: HashMap::new(),
            pending: Vec::new(),
            active_voice_message: None,
            next_upload_id: 0,
        }
    }

    fn allocate_upload_id(&mut self) -> u64 {
        self.next_upload_id = self.next_upload_id.wrapping_add(1).max(1);
        self.next_upload_id
    }
}

struct WorkerState {
    app: AppState,
    runtimes: HashMap<u64, RuntimeTab>,
    platform_api_key: Option<String>,
    next_tab_id: u64,
    next_message_id: u64,
    next_attachment_id: u64,
}

impl WorkerState {
    fn new() -> Self {
        let app = AppState::default();
        let runtimes = app
            .tabs
            .iter()
            .map(|tab| (tab.id, RuntimeTab::new()))
            .collect();
        Self {
            app,
            runtimes,
            platform_api_key: std::env::var("OPENAI_API_KEY")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            next_tab_id: 3,
            next_message_id: 1,
            next_attachment_id: 1,
        }
    }

    fn tab(&self, id: u64) -> Option<&TabView> {
        self.app.tabs.iter().find(|tab| tab.id == id)
    }

    fn tab_mut(&mut self, id: u64) -> Option<&mut TabView> {
        self.app.tabs.iter_mut().find(|tab| tab.id == id)
    }

    fn active_tab_id(&self) -> u64 {
        self.app.active_tab_id
    }

    fn take_message_id(&mut self) -> u64 {
        let id = self.next_message_id;
        self.next_message_id = self.next_message_id.saturating_add(1);
        id
    }

    fn take_attachment_id(&mut self) -> u64 {
        let id = self.next_attachment_id;
        self.next_attachment_id = self.next_attachment_id.saturating_add(1);
        id
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
    Connect {
        tab_id: Option<u64>,
        api_key: Option<String>,
    },
    Disconnect {
        tab_id: Option<u64>,
    },
    SendText {
        tab_id: Option<u64>,
        text: String,
    },
    Audio {
        tab_id: Option<u64>,
        samples: Vec<i16>,
    },
    Clear {
        tab_id: Option<u64>,
    },
    UpdateSettings {
        tab_id: Option<u64>,
        settings: PublicSettings,
    },
    AddTab {
        backend: RealtimeBackend,
        model: String,
        voice: Option<String>,
    },
    SwitchTab {
        tab_id: u64,
    },
    CloseTab {
        tab_id: u64,
    },
    AddImage {
        tab_id: Option<u64>,
        name: String,
        data_url: String,
    },
    CaptureScreen {
        tab_id: Option<u64>,
    },
    RemoveAttachment {
        tab_id: Option<u64>,
        attachment_id: u64,
    },
    AttachmentReady {
        tab_id: u64,
        result: Result<Attachment, String>,
        included_screen: bool,
    },
    AutoScreenshotReady {
        tab_id: u64,
        message_id: u64,
        upload_id: u64,
        result: Result<Attachment, String>,
    },
    RefreshCatalog {
        api_key: Option<String>,
    },
    CatalogReady(Result<ModelCatalog, String>),
    ToolFinished {
        tab_id: u64,
        call_id: String,
        output: String,
        image_url: Option<String>,
    },
    Shutdown,
}

#[derive(Deserialize)]
struct TabRequest {
    tab_id: Option<u64>,
}

#[derive(Deserialize)]
struct ConnectRequest {
    tab_id: Option<u64>,
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct MessageRequest {
    tab_id: Option<u64>,
    text: String,
}

#[derive(Deserialize)]
struct SettingsRequest {
    tab_id: Option<u64>,
    settings: PublicSettings,
}

#[derive(Deserialize)]
struct AddTabRequest {
    backend: RealtimeBackend,
    model: String,
    voice: Option<String>,
}

#[derive(Deserialize)]
struct SwitchTabRequest {
    tab_id: u64,
}

#[derive(Deserialize)]
struct UploadRequest {
    tab_id: Option<u64>,
    name: String,
    data_url: String,
}

#[derive(Deserialize)]
struct RemoveAttachmentRequest {
    tab_id: Option<u64>,
    attachment_id: u64,
}

#[derive(Deserialize)]
struct RefreshCatalogRequest {
    api_key: Option<String>,
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
        .route("/api/tabs/add", post(api_add_tab))
        .route("/api/tabs/switch", post(api_switch_tab))
        .route("/api/tabs/close", post(api_close_tab))
        .route("/api/upload", post(api_upload))
        .route("/api/capture-screen", post(api_capture_screen))
        .route("/api/attachments/remove", post(api_remove_attachment))
        .route("/api/catalog/refresh", post(api_refresh_catalog))
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
        let worker = WorkerState::new();
        let state = Arc::new(RwLock::new(worker.app.clone()));
        let (actions, action_rx) = mpsc::channel();
        let (push, _) = broadcast::channel(128);
        let backend = Self {
            state: state.clone(),
            actions: actions.clone(),
            push: push.clone(),
        };
        let startup_actions = actions.clone();
        thread::spawn(move || worker_loop(worker, state, action_rx, actions, push));
        let _ = startup_actions.send(Action::RefreshCatalog { api_key: None });
        backend
    }

    fn snapshot(&self) -> AppState {
        self.state.read().expect("app state read lock").clone()
    }
}

fn worker_loop(
    mut worker: WorkerState,
    shared: Arc<RwLock<AppState>>,
    action_rx: mpsc::Receiver<Action>,
    action_tx: mpsc::Sender<Action>,
    push: broadcast::Sender<PushEvent>,
) {
    publish(&mut worker, &shared, &push);
    'worker: loop {
        match action_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(action) => {
                if handle_action(action, &mut worker, &action_tx, &push) {
                    break 'worker;
                }
                while let Ok(action) = action_rx.try_recv() {
                    if handle_action(action, &mut worker, &action_tx, &push) {
                        break 'worker;
                    }
                }
                publish(&mut worker, &shared, &push);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let tab_ids = worker.runtimes.keys().copied().collect::<Vec<_>>();
        let mut changed = false;
        for tab_id in tab_ids {
            loop {
                let event = worker
                    .runtimes
                    .get(&tab_id)
                    .and_then(|runtime| runtime.client.events.try_recv().ok());
                let Some(event) = event else { break };
                handle_realtime_event(event, tab_id, &mut worker, &action_tx, &push);
                changed = true;
            }
        }
        if changed {
            publish(&mut worker, &shared, &push);
        }
    }

    for runtime in worker.runtimes.values() {
        let _ = runtime.client.commands.send(Command::Shutdown);
    }
}

fn handle_action(
    action: Action,
    worker: &mut WorkerState,
    action_tx: &mpsc::Sender<Action>,
    push: &broadcast::Sender<PushEvent>,
) -> bool {
    match action {
        Action::Connect { tab_id, api_key } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            if let Some(key) = api_key
                .as_deref()
                .map(str::trim)
                .filter(|key| !key.is_empty())
            {
                worker.platform_api_key = Some(key.to_owned());
                let _ = action_tx.send(Action::RefreshCatalog {
                    api_key: Some(key.to_owned()),
                });
            }
            connect_tab(worker, tab_id, api_key);
        }
        Action::Disconnect { tab_id } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                let _ = runtime.client.commands.send(Command::Disconnect);
                runtime.active_assistant = None;
                runtime.active_voice_message = None;
                runtime.input_messages.clear();
            }
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.connection = ConnectionState::Offline;
                tab.status = "Ready".to_owned();
                tab.error = None;
            }
        }
        Action::SendText { tab_id, text } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            send_text(worker, tab_id, text);
        }
        Action::Audio { tab_id, samples } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            if !samples.is_empty()
                && let Some(runtime) = worker.runtimes.get(&tab_id)
            {
                let _ = runtime.client.commands.send(Command::AudioChunk(samples));
            }
        }
        Action::Clear { tab_id } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.active_assistant = None;
                runtime.active_voice_message = None;
                runtime.input_messages.clear();
            }
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.messages.clear();
                tab.error = None;
                tab.status = if tab.connection == ConnectionState::Live
                    && tab.settings.backend != RealtimeBackend::CodexText
                {
                    "Listening".to_owned()
                } else {
                    "Ready".to_owned()
                };
            }
        }
        Action::UpdateSettings { tab_id, settings } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            let was_live = worker
                .tab(tab_id)
                .is_some_and(|tab| tab.connection != ConnectionState::Offline);
            if was_live && let Some(runtime) = worker.runtimes.get(&tab_id) {
                let _ = runtime.client.commands.send(Command::Disconnect);
            }
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.title = tab_title(&settings);
                tab.settings = settings;
                tab.connection = ConnectionState::Offline;
                tab.status = if was_live {
                    "Settings changed · reconnect".to_owned()
                } else {
                    "Ready".to_owned()
                };
                tab.error = None;
            }
        }
        Action::AddTab {
            backend,
            model,
            voice,
        } => {
            let id = worker.next_tab_id;
            worker.next_tab_id = worker.next_tab_id.saturating_add(1);
            let mut settings = PublicSettings::for_backend(
                backend,
                (!model.trim().is_empty()).then(|| model.trim().to_owned()),
            );
            if let Some(voice) = voice.filter(|voice| !voice.trim().is_empty()) {
                settings.voice = voice;
            }
            worker.app.tabs.push(TabView::new(id, settings));
            worker.runtimes.insert(id, RuntimeTab::new());
            worker.app.active_tab_id = id;
        }
        Action::SwitchTab { tab_id } => {
            if worker.tab(tab_id).is_some() {
                worker.app.active_tab_id = tab_id;
            }
        }
        Action::CloseTab { tab_id } => {
            if worker.app.tabs.len() <= 1 {
                if let Some(tab) = worker.tab_mut(tab_id) {
                    tab.error = Some("At least one tab must remain open".to_owned());
                }
            } else if let Some(index) = worker.app.tabs.iter().position(|tab| tab.id == tab_id) {
                if let Some(runtime) = worker.runtimes.remove(&tab_id) {
                    let _ = runtime.client.commands.send(Command::Shutdown);
                }
                worker.app.tabs.remove(index);
                if worker.app.active_tab_id == tab_id {
                    let next = index.min(worker.app.tabs.len().saturating_sub(1));
                    worker.app.active_tab_id = worker.app.tabs[next].id;
                }
            }
        }
        Action::AddImage {
            tab_id,
            name,
            data_url,
        } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            let sender = action_tx.clone();
            thread::spawn(move || {
                let result = media::image_from_data_url(name, &data_url)
                    .map_err(|error| format!("{error:#}"));
                let _ = sender.send(Action::AttachmentReady {
                    tab_id,
                    result,
                    included_screen: false,
                });
            });
            set_tab_status(worker, tab_id, "Preparing image", None);
        }
        Action::CaptureScreen { tab_id } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            let sender = action_tx.clone();
            thread::spawn(move || {
                let screen = primary_screen_info();
                let result = media::capture_screenshot(
                    screen.logical_width,
                    screen.logical_height,
                    false,
                    None,
                )
                .map_err(|error| format!("{error:#}"));
                let _ = sender.send(Action::AttachmentReady {
                    tab_id,
                    result,
                    included_screen: true,
                });
            });
            set_tab_status(worker, tab_id, "Capturing screen", None);
        }
        Action::RemoveAttachment {
            tab_id,
            attachment_id,
        } => {
            let tab_id = tab_id.unwrap_or_else(|| worker.active_tab_id());
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.pending.retain(|item| item.view.id != attachment_id);
                sync_pending_views(worker, tab_id);
            }
        }
        Action::AttachmentReady {
            tab_id,
            result,
            included_screen,
        } => match result {
            Ok(attachment) => {
                let id = worker.take_attachment_id();
                let view = attachment_view(id, &attachment, included_screen, "ready", None);
                if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                    runtime.pending.push(PendingAttachment { view, attachment });
                    sync_pending_views(worker, tab_id);
                    set_tab_status(worker, tab_id, "Attachment ready", None);
                }
            }
            Err(error) => set_tab_status(worker, tab_id, "Attachment failed", Some(error)),
        },
        Action::AutoScreenshotReady {
            tab_id,
            message_id,
            upload_id,
            result,
        } => match result {
            Ok(attachment) => {
                let id = worker.take_attachment_id();
                let view = attachment_view(id, &attachment, true, "uploading", Some(upload_id));
                if let Some(tab) = worker.tab_mut(tab_id)
                    && let Some(message) = tab
                        .messages
                        .iter_mut()
                        .find(|message| message.id == message_id)
                {
                    message.attachments.push(view);
                    message.refresh_token_estimate();
                    tab.status = "Thinking · screen uploading".to_owned();
                }
                if let Some(runtime) = worker.runtimes.get(&tab_id) {
                    let _ = runtime.client.commands.send(Command::SendContextImage {
                        upload_id,
                        image: attachment,
                        deadline: Instant::now() + CONTEXT_UPLOAD_TIMEOUT,
                    });
                }
            }
            Err(error) => set_tab_status(worker, tab_id, "Screen capture failed", Some(error)),
        },
        Action::RefreshCatalog { api_key } => {
            if let Some(key) = api_key
                .as_deref()
                .map(str::trim)
                .filter(|key| !key.is_empty())
            {
                worker.platform_api_key = Some(key.to_owned());
                worker.app.has_credentials = true;
            }
            worker.app.catalog.loading = true;
            worker.app.catalog.error = None;
            let sender = action_tx.clone();
            let key = worker.platform_api_key.clone();
            thread::spawn(move || {
                let result =
                    load_model_catalog(key.as_deref()).map_err(|error| format!("{error:#}"));
                let _ = sender.send(Action::CatalogReady(result));
            });
        }
        Action::CatalogReady(result) => match result {
            Ok(catalog) => worker.app.catalog = catalog,
            Err(error) => {
                worker.app.catalog.loading = false;
                worker.app.catalog.error = Some(error);
            }
        },
        Action::ToolFinished {
            tab_id,
            call_id,
            output,
            image_url,
        } => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                for message in tab.messages.iter_mut().rev() {
                    if let Some(tool) = message
                        .tool_calls
                        .iter_mut()
                        .find(|tool| tool.call_id == call_id)
                    {
                        tool.status = "done".to_owned();
                        tool.output = Some(compact_output(&output));
                        if image_url.is_some() {
                            message.generated_image_url = image_url.clone();
                        }
                        message.refresh_token_estimate();
                        break;
                    }
                }
                tab.status = "Thinking".to_owned();
            }
            if let Some(runtime) = worker.runtimes.get(&tab_id) {
                let _ = runtime
                    .client
                    .commands
                    .send(Command::ToolOutputs(vec![ToolOutput { call_id, output }]));
            }
        }
        Action::Shutdown => return true,
    }
    let _ = push;
    false
}

fn connect_tab(worker: &mut WorkerState, tab_id: u64, api_key: Option<String>) {
    let Some(settings) = worker.tab(tab_id).map(|tab| tab.settings.clone()) else {
        return;
    };
    let key = api_key.or_else(|| worker.platform_api_key.clone());
    match resolve_credentials(settings.auth_mode, key) {
        Ok(credentials) => {
            worker.app.has_credentials = true;
            let screen_info = primary_screen_info();
            let system_prompt = if settings.system_prompt.trim().is_empty() {
                realtime::default_system_prompt(screen_info)
            } else {
                settings.system_prompt.clone()
            };
            let options = ConnectOptions {
                backend: settings.backend,
                api_key: credentials.bearer_token.clone(),
                chatgpt_account_id: credentials.chatgpt_account_id.clone(),
                model: settings.model.clone(),
                voice: settings.voice.clone(),
                thinking_level: settings.thinking_level.clone(),
                system_prompt,
                screen_info,
            };
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.credentials = Some(credentials);
                if runtime
                    .client
                    .commands
                    .send(Command::Connect(options))
                    .is_err()
                {
                    set_tab_status(
                        worker,
                        tab_id,
                        "Connection failed",
                        Some("Could not start the model transport".to_owned()),
                    );
                } else if let Some(tab) = worker.tab_mut(tab_id) {
                    tab.connection = ConnectionState::Connecting;
                    tab.status = "Connecting".to_owned();
                    tab.error = None;
                }
            }
        }
        Err(error) => set_tab_status(
            worker,
            tab_id,
            "Needs attention",
            Some(format!("{error:#}")),
        ),
    }
}

fn send_text(worker: &mut WorkerState, tab_id: u64, text: String) {
    let text = text.trim().to_owned();
    if text.is_empty() {
        return;
    }
    let Some(settings) = worker.tab(tab_id).map(|tab| tab.settings.clone()) else {
        return;
    };
    if worker
        .tab(tab_id)
        .is_none_or(|tab| tab.connection != ConnectionState::Live)
    {
        set_tab_status(
            worker,
            tab_id,
            "Needs attention",
            Some("Connect this tab before sending a message".to_owned()),
        );
        return;
    }

    let mut attachments = worker
        .runtimes
        .get_mut(&tab_id)
        .map(|runtime| std::mem::take(&mut runtime.pending))
        .unwrap_or_default();

    if settings.send_screenshot
        && let Ok(screen_attachment) = capture_screen_attachment()
    {
        let id = worker.take_attachment_id();
        let view = attachment_view(id, &screen_attachment, true, "ready", None);
        attachments.push(PendingAttachment {
            view,
            attachment: screen_attachment,
        });
    }

    let message_id = worker.take_message_id();
    let mut message = ChatMessage::new(message_id, MessageRole::User, text.clone(), false);
    message.attachments = attachments.iter().map(|item| item.view.clone()).collect();
    message.finish();
    if let Some(tab) = worker.tab_mut(tab_id) {
        tab.messages.push(message);
        tab.pending_attachments.clear();
        tab.status = "Thinking".to_owned();
        tab.error = None;
    }
    let wire_attachments = attachments
        .into_iter()
        .map(|item| item.attachment)
        .collect::<Vec<_>>();
    if let Some(runtime) = worker.runtimes.get(&tab_id)
        && runtime
            .client
            .commands
            .send(Command::SendTurn {
                text,
                attachments: wire_attachments,
                thinking_level: settings.thinking_level,
            })
            .is_err()
    {
        set_tab_status(
            worker,
            tab_id,
            "Send failed",
            Some("Could not send the message to the backend".to_owned()),
        );
    }
}

fn handle_realtime_event(
    event: Event,
    tab_id: u64,
    worker: &mut WorkerState,
    action_tx: &mpsc::Sender<Action>,
    push: &broadcast::Sender<PushEvent>,
) {
    match event {
        Event::Connecting => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.connection = ConnectionState::Connecting;
                tab.status = "Connecting".to_owned();
                tab.error = None;
            }
        }
        Event::Reconnecting { attempt, reason } => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.connection = ConnectionState::Reconnecting;
                tab.status = format!("Reconnecting · attempt {attempt}");
                tab.error = Some(reason);
            }
        }
        Event::Connected => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.connection = ConnectionState::Live;
                tab.status = if tab.settings.backend == RealtimeBackend::CodexText {
                    "Ready".to_owned()
                } else {
                    "Listening".to_owned()
                };
                tab.error = None;
            }
        }
        Event::Disconnected => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.connection = ConnectionState::Offline;
                tab.status = "Ready".to_owned();
            }
        }
        Event::SpeechStarted => {
            let message_id = worker.take_message_id();
            let settings = worker.tab(tab_id).map(|tab| tab.settings.clone());
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.status = "Listening".to_owned();
                tab.messages.push(ChatMessage::new(
                    message_id,
                    MessageRole::User,
                    "Listening…".to_owned(),
                    true,
                ));
            }
            let voice_message_index = worker
                .tab(tab_id)
                .and_then(|tab| tab.messages.len().checked_sub(1));
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.active_voice_message = voice_message_index;
                if settings.is_some_and(|settings| settings.send_screenshot) {
                    let upload_id = runtime.allocate_upload_id();
                    let sender = action_tx.clone();
                    thread::spawn(move || {
                        let result =
                            capture_screen_attachment().map_err(|error| format!("{error:#}"));
                        let _ = sender.send(Action::AutoScreenshotReady {
                            tab_id,
                            message_id,
                            upload_id,
                            result,
                        });
                    });
                }
            }
        }
        Event::SpeechStopped => {
            if let Some(runtime) = worker.runtimes.get(&tab_id)
                && let Some(index) = runtime.active_voice_message
                && let Some(tab) = worker.tab_mut(tab_id)
                && let Some(message) = tab.messages.get_mut(index)
            {
                message.finish();
                tab.status = "Thinking".to_owned();
            }
        }
        Event::InputCommitted { item_id } => {
            let index = worker.tab(tab_id).and_then(|tab| {
                tab.messages
                    .iter()
                    .rposition(|message| message.role == MessageRole::User)
            });
            if let Some(index) = index
                && let Some(runtime) = worker.runtimes.get_mut(&tab_id)
            {
                runtime.input_messages.insert(item_id, index);
            }
        }
        Event::InputTranscript { item_id, text } => {
            let index = worker
                .runtimes
                .get(&tab_id)
                .and_then(|runtime| runtime.input_messages.get(&item_id).copied())
                .or_else(|| {
                    worker.tab(tab_id).and_then(|tab| {
                        tab.messages
                            .iter()
                            .rposition(|message| message.role == MessageRole::User)
                    })
                });
            if let Some(index) = index
                && let Some(tab) = worker.tab_mut(tab_id)
                && let Some(message) = tab.messages.get_mut(index)
            {
                message.text = text;
                message.refresh_token_estimate();
            }
        }
        Event::ContextImageAccepted { upload_id } => {
            update_upload_status(worker, tab_id, upload_id, "uploading", None);
        }
        Event::ContextImageUploaded { upload_id } => {
            update_upload_status(worker, tab_id, upload_id, "uploaded", None);
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.status = "Thinking · screen uploaded".to_owned();
            }
        }
        Event::ContextImageUploadFailed { upload_id, detail } => {
            update_upload_status(worker, tab_id, upload_id, "failed", Some(detail.clone()));
            set_tab_status(worker, tab_id, "Screen upload failed", Some(detail));
        }
        Event::AssistantResponseStarted { response_id } => {
            let message_id = worker.take_message_id();
            let mut message =
                ChatMessage::new(message_id, MessageRole::Assistant, String::new(), true);
            message.response_id = Some(response_id);
            let assistant_index = if let Some(tab) = worker.tab_mut(tab_id) {
                tab.messages.push(message);
                tab.status = "Responding".to_owned();
                tab.messages.len().checked_sub(1)
            } else {
                None
            };
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.active_assistant = assistant_index;
            }
        }
        Event::AssistantTranscriptDelta { response_id, delta } => {
            let index = ensure_assistant_message(worker, tab_id, Some(response_id));
            if let Some(tab) = worker.tab_mut(tab_id)
                && let Some(message) = tab.messages.get_mut(index)
            {
                message.text.push_str(&delta);
                message.refresh_token_estimate();
            }
        }
        Event::AssistantAudio { samples, .. } => {
            if worker.active_tab_id() == tab_id {
                let _ = push.send(PushEvent::Audio(samples));
            }
        }
        Event::AssistantDone { response_id } => {
            let index = worker
                .runtimes
                .get(&tab_id)
                .and_then(|runtime| runtime.active_assistant)
                .or_else(|| {
                    worker.tab(tab_id).and_then(|tab| {
                        tab.messages.iter().rposition(|message| {
                            message.role == MessageRole::Assistant
                                && message.response_id.as_deref() == Some(response_id.as_str())
                        })
                    })
                });
            if let Some(index) = index
                && let Some(tab) = worker.tab_mut(tab_id)
                && let Some(message) = tab.messages.get_mut(index)
            {
                message.finish();
                tab.status = if tab.settings.backend == RealtimeBackend::CodexText {
                    "Ready".to_owned()
                } else {
                    "Listening".to_owned()
                };
            }
            if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
                runtime.active_assistant = None;
            }
        }
        Event::AssistantUsage {
            response_id,
            total_tokens,
        } => {
            if let Some(tab) = worker.tab_mut(tab_id)
                && let Some(message) = tab.messages.iter_mut().rev().find(|message| {
                    message.role == MessageRole::Assistant
                        && message.response_id.as_deref() == Some(response_id.as_str())
                })
            {
                message.set_actual_usage(total_tokens);
            }
        }
        Event::ToolCalls(calls) => {
            let settings = worker
                .tab(tab_id)
                .map(|tab| tab.settings.clone())
                .unwrap_or_default();
            let credentials = worker
                .runtimes
                .get(&tab_id)
                .and_then(|runtime| runtime.credentials.clone());
            let index = ensure_assistant_message(worker, tab_id, None);
            if let Some(tab) = worker.tab_mut(tab_id) {
                if let Some(message) = tab.messages.get_mut(index) {
                    message.tool_calls.extend(calls.iter().map(|call| ToolView {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: pretty_json(&call.arguments),
                        status: "running".to_owned(),
                        output: None,
                    }));
                    message.refresh_token_estimate();
                }
                tab.status = format!(
                    "Running {} tool{}",
                    calls.len(),
                    if calls.len() == 1 { "" } else { "s" }
                );
            }
            let screen = primary_screen_info();
            for call in calls {
                let sender = action_tx.clone();
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
                    let _ = sender.send(Action::ToolFinished {
                        tab_id,
                        call_id: call.call_id,
                        output,
                        image_url,
                    });
                });
            }
        }
        Event::ToolOutputsSubmitted { .. } => {
            if let Some(tab) = worker.tab_mut(tab_id) {
                tab.status = "Thinking".to_owned();
            }
        }
        Event::Error(error) => set_tab_status(worker, tab_id, "Needs attention", Some(error)),
        Event::AssistantItem { .. } | Event::AssistantSegmentDone { .. } => {}
    }
}

fn ensure_assistant_message(
    worker: &mut WorkerState,
    tab_id: u64,
    response_id: Option<String>,
) -> usize {
    if let Some(index) = worker
        .runtimes
        .get(&tab_id)
        .and_then(|runtime| runtime.active_assistant)
        && worker
            .tab(tab_id)
            .and_then(|tab| tab.messages.get(index))
            .is_some()
    {
        return index;
    }
    let message_id = worker.take_message_id();
    let mut message = ChatMessage::new(message_id, MessageRole::Assistant, String::new(), true);
    message.response_id = response_id;
    let index = if let Some(tab) = worker.tab_mut(tab_id) {
        tab.messages.push(message);
        tab.messages.len() - 1
    } else {
        0
    };
    if let Some(runtime) = worker.runtimes.get_mut(&tab_id) {
        runtime.active_assistant = Some(index);
    }
    index
}

fn update_upload_status(
    worker: &mut WorkerState,
    tab_id: u64,
    upload_id: u64,
    status: &str,
    error: Option<String>,
) {
    if let Some(tab) = worker.tab_mut(tab_id) {
        for message in tab.messages.iter_mut().rev() {
            if let Some(attachment) = message
                .attachments
                .iter_mut()
                .find(|attachment| attachment.upload_id == Some(upload_id))
            {
                attachment.status = status.to_owned();
                break;
            }
        }
        if let Some(error) = error {
            tab.error = Some(error);
        }
    }
}

fn sync_pending_views(worker: &mut WorkerState, tab_id: u64) {
    let views = worker
        .runtimes
        .get(&tab_id)
        .map(|runtime| {
            runtime
                .pending
                .iter()
                .map(|item| item.view.clone())
                .collect()
        })
        .unwrap_or_default();
    if let Some(tab) = worker.tab_mut(tab_id) {
        tab.pending_attachments = views;
    }
}

fn set_tab_status(worker: &mut WorkerState, tab_id: u64, status: &str, error: Option<String>) {
    if let Some(tab) = worker.tab_mut(tab_id) {
        tab.status = status.to_owned();
        tab.error = error;
        if tab.connection != ConnectionState::Live && status == "Needs attention" {
            tab.connection = ConnectionState::Offline;
        }
    }
}

fn publish(
    worker: &mut WorkerState,
    shared: &Arc<RwLock<AppState>>,
    push: &broadcast::Sender<PushEvent>,
) {
    worker.app.revision = worker.app.revision.saturating_add(1);
    worker.app.last_activity_at = unix_millis();
    *shared.write().expect("app state write lock") = worker.app.clone();
    let _ = push.send(PushEvent::StateChanged);
}

fn resolve_credentials(
    auth_mode: AuthMode,
    api_key: Option<String>,
) -> anyhow::Result<CodexCredentials> {
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

fn capture_screen_attachment() -> anyhow::Result<Attachment> {
    let screen = primary_screen_info();
    media::capture_screenshot(screen.logical_width, screen.logical_height, false, None)
}

fn attachment_view(
    id: u64,
    attachment: &Attachment,
    included_screen: bool,
    status: &str,
    upload_id: Option<u64>,
) -> AttachmentView {
    match attachment {
        Attachment::Image {
            name,
            data_url,
            width,
            height,
            byte_size,
            ..
        } => AttachmentView {
            id,
            name: name.clone(),
            kind: "image".to_owned(),
            data_url: Some(data_url.clone()),
            width: Some(*width),
            height: Some(*height),
            byte_size: *byte_size,
            seconds: None,
            status: status.to_owned(),
            included_screen,
            upload_id,
        },
        Attachment::Audio {
            name,
            pcm24k,
            seconds,
        } => AttachmentView {
            id,
            name: name.clone(),
            kind: "audio".to_owned(),
            data_url: None,
            width: None,
            height: None,
            byte_size: pcm24k.len() * 2,
            seconds: Some(*seconds),
            status: status.to_owned(),
            included_screen,
            upload_id,
        },
    }
}

fn estimated_text_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    let words = text.split_whitespace().count() as u64;
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4).max(words)
    }
}

fn estimated_message_tokens(message: &ChatMessage) -> u64 {
    let mut total = estimated_text_tokens(&message.text);
    for tool in &message.tool_calls {
        total = total
            .saturating_add(estimated_text_tokens(&tool.name))
            .saturating_add(estimated_text_tokens(&tool.arguments));
        if let Some(output) = &tool.output {
            total = total.saturating_add(estimated_text_tokens(output));
        }
    }
    for attachment in &message.attachments {
        total = total.saturating_add(estimated_text_tokens(&attachment.name));
        if attachment.kind == "image" {
            total = total.saturating_add(256);
        } else if let Some(seconds) = attachment.seconds {
            total = total.saturating_add((seconds * 10.0).ceil() as u64);
        }
    }
    total.max(u64::from(
        !message.text.is_empty()
            || !message.tool_calls.is_empty()
            || !message.attachments.is_empty(),
    ))
}

fn load_model_catalog(api_key: Option<&str>) -> anyhow::Result<ModelCatalog> {
    let mut catalog = ModelCatalog::default();
    catalog.loading = false;
    catalog.refreshed_at = Some(unix_millis());
    let mut warnings = Vec::new();

    match codex_account::load() {
        Ok(info) => {
            let mut text = Vec::new();
            let mut seen = HashSet::new();
            for model in info.models {
                if !model.hidden && seen.insert(model.name.clone()) {
                    text.push(model_option(
                        &model.name,
                        &model.display_name,
                        "Available through the connected Codex account.",
                        "codex_account",
                    ));
                }
            }
            merge_models(&mut text, fallback_text_models("official"));
            catalog.text_models = text;
            if !info.realtime_voices.v1.is_empty() {
                catalog.realtime_voices = info.realtime_voices.v1;
            }
            if !info.realtime_voices.v2.is_empty() {
                catalog.gpt_live_voices = info.realtime_voices.v2;
            }
            warnings.extend(info.warnings);
        }
        Err(error) => warnings.push(format!("Codex catalog unavailable: {error:#}")),
    }

    if let Some(api_key) = api_key.filter(|key| !key.trim().is_empty()) {
        match list_platform_models(api_key) {
            Ok(models) => {
                let mut realtime = models
                    .iter()
                    .filter(|id| is_realtime_model(id))
                    .map(|id| {
                        model_option(
                            id,
                            &model_label(id),
                            "Available to this API key.",
                            "openai_api",
                        )
                    })
                    .collect::<Vec<_>>();
                merge_models(&mut realtime, fallback_realtime_models("official"));
                catalog.realtime_models = realtime.clone();
                catalog.gpt_live_models = realtime;

                let mut text = models
                    .iter()
                    .filter(|id| is_text_model(id))
                    .map(|id| {
                        model_option(
                            id,
                            &model_label(id),
                            "Available to this API key.",
                            "openai_api",
                        )
                    })
                    .collect::<Vec<_>>();
                merge_models(&mut text, catalog.text_models);
                catalog.text_models = text;
            }
            Err(error) => warnings.push(format!("OpenAI model list unavailable: {error:#}")),
        }
    }

    catalog.warnings = warnings;
    Ok(catalog)
}

fn list_platform_models(api_key: &str) -> anyhow::Result<Vec<String>> {
    let response = ureq::get("https://api.openai.com/v1/models")
        .set("Authorization", &format!("Bearer {}", api_key.trim()))
        .call();
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            anyhow::bail!("HTTP {code}: {detail}");
        }
        Err(error) => return Err(error).context("Network error while listing OpenAI models"),
    };
    let value: Value = response
        .into_json()
        .context("OpenAI model list returned invalid JSON")?;
    let mut models = value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    Ok(models)
}

fn is_realtime_model(id: &str) -> bool {
    id.starts_with("gpt-realtime")
        && !id.contains("whisper")
        && !id.contains("translate")
        && !id.contains("transcription")
}

fn is_text_model(id: &str) -> bool {
    let general_prefix = id.starts_with("gpt-") || id.starts_with('o') || id.starts_with("chat-");
    general_prefix
        && !id.contains("realtime")
        && !id.contains("audio")
        && !id.contains("transcribe")
        && !id.contains("tts")
        && !id.contains("image")
        && !id.contains("embedding")
        && !id.contains("moderation")
        && !id.contains("search")
        && !id.contains("sora")
}

fn fallback_realtime_models(source: &str) -> Vec<ModelOption> {
    FALLBACK_REALTIME_MODELS
        .iter()
        .map(|(id, label, description)| model_option(id, label, description, source))
        .collect()
}

fn fallback_text_models(source: &str) -> Vec<ModelOption> {
    FALLBACK_TEXT_MODELS
        .iter()
        .map(|(id, label, description)| model_option(id, label, description, source))
        .collect()
}

fn model_option(id: &str, label: &str, description: &str, source: &str) -> ModelOption {
    ModelOption {
        id: id.to_owned(),
        label: label.to_owned(),
        description: description.to_owned(),
        source: source.to_owned(),
    }
}

fn merge_models(target: &mut Vec<ModelOption>, fallback: Vec<ModelOption>) {
    let mut seen = target
        .iter()
        .map(|model| model.id.clone())
        .collect::<HashSet<_>>();
    for model in fallback {
        if seen.insert(model.id.clone()) {
            target.push(model);
        }
    }
}

fn model_label(id: &str) -> String {
    id.split('-')
        .map(|part| {
            if part.eq_ignore_ascii_case("gpt") {
                "GPT".to_owned()
            } else if part.len() <= 2 && part.chars().all(|character| character.is_ascii_digit()) {
                part.to_owned()
            } else {
                let mut chars = part.chars();
                chars
                    .next()
                    .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join("-")
}

fn tab_title(settings: &PublicSettings) -> String {
    match settings.backend {
        RealtimeBackend::CodexGptLive => {
            format!("GPT-Live · {}", short_model_name(&settings.model))
        }
        RealtimeBackend::OpenAiRealtime => {
            format!("Realtime · {}", short_model_name(&settings.model))
        }
        RealtimeBackend::CodexText => short_model_name(&settings.model),
    }
}

fn short_model_name(model: &str) -> String {
    model
        .strip_prefix("gpt-")
        .unwrap_or(model)
        .replace("realtime-", "RT ")
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

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        APP_JS,
    )
        .into_response()
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
    send_action(
        &backend,
        Action::Connect {
            tab_id: request.tab_id,
            api_key: request.api_key,
        },
    )
}

async fn api_disconnect(
    State(backend): State<Backend>,
    Json(request): Json<TabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::Disconnect {
            tab_id: request.tab_id,
        },
    )
}

async fn api_message(
    State(backend): State<Backend>,
    Json(request): Json<MessageRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::SendText {
            tab_id: request.tab_id,
            text: request.text,
        },
    )
}

async fn api_settings(
    State(backend): State<Backend>,
    Json(request): Json<SettingsRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::UpdateSettings {
            tab_id: request.tab_id,
            settings: request.settings,
        },
    )
}

async fn api_clear(
    State(backend): State<Backend>,
    Json(request): Json<TabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::Clear {
            tab_id: request.tab_id,
        },
    )
}

async fn api_audio(State(backend): State<Backend>, body: Bytes) -> impl IntoResponse {
    send_action(
        &backend,
        Action::Audio {
            tab_id: None,
            samples: pcm16_from_bytes(&body),
        },
    )
}

async fn api_add_tab(
    State(backend): State<Backend>,
    Json(request): Json<AddTabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::AddTab {
            backend: request.backend,
            model: request.model,
            voice: request.voice,
        },
    )
}

async fn api_switch_tab(
    State(backend): State<Backend>,
    Json(request): Json<SwitchTabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::SwitchTab {
            tab_id: request.tab_id,
        },
    )
}

async fn api_close_tab(
    State(backend): State<Backend>,
    Json(request): Json<SwitchTabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::CloseTab {
            tab_id: request.tab_id,
        },
    )
}

async fn api_upload(
    State(backend): State<Backend>,
    Json(request): Json<UploadRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::AddImage {
            tab_id: request.tab_id,
            name: request.name,
            data_url: request.data_url,
        },
    )
}

async fn api_capture_screen(
    State(backend): State<Backend>,
    Json(request): Json<TabRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::CaptureScreen {
            tab_id: request.tab_id,
        },
    )
}

async fn api_remove_attachment(
    State(backend): State<Backend>,
    Json(request): Json<RemoveAttachmentRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::RemoveAttachment {
            tab_id: request.tab_id,
            attachment_id: request.attachment_id,
        },
    )
}

async fn api_refresh_catalog(
    State(backend): State<Backend>,
    Json(request): Json<RefreshCatalogRequest>,
) -> impl IntoResponse {
    send_action(
        &backend,
        Action::RefreshCatalog {
            api_key: request.api_key,
        },
    )
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
    if send_state_ws(&mut sender, &backend.snapshot())
        .await
        .is_err()
    {
        return;
    }
    let mut push = backend.push.subscribe();
    loop {
        tokio::select! {
            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(WsMessage::Binary(bytes))) => {
                        let _ = backend.actions.send(Action::Audio {
                            tab_id: None,
                            samples: pcm16_from_bytes(&bytes),
                        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_independent_voice_tabs() {
        let state = AppState::default();
        assert_eq!(state.tabs.len(), 2);
        assert_eq!(state.active_tab_id, state.tabs[0].id);
        assert_eq!(
            state.tabs[0].settings.backend,
            RealtimeBackend::CodexGptLive
        );
        assert_eq!(
            state.tabs[1].settings.backend,
            RealtimeBackend::OpenAiRealtime
        );
        assert_ne!(state.tabs[0].id, state.tabs[1].id);
    }

    #[test]
    fn message_metrics_use_actual_usage_when_available() {
        let mut message =
            ChatMessage::new(1, MessageRole::Assistant, "hello world".to_owned(), true);
        message.started_at = unix_millis().saturating_sub(2_000);
        message.set_actual_usage(100);
        message.finish();
        assert_eq!(message.token_count, Some(100));
        assert!(!message.token_count_is_estimate);
        assert!(message.elapsed_ms.is_some_and(|elapsed| elapsed >= 2_000));
        assert!(
            message
                .tokens_per_second
                .is_some_and(|rate| rate > 40.0 && rate <= 50.0)
        );
    }

    #[test]
    fn image_attachments_contribute_to_local_token_estimate() {
        let mut message = ChatMessage::new(1, MessageRole::User, "look".to_owned(), false);
        let before = message.token_count.unwrap_or_default();
        message.attachments.push(AttachmentView {
            id: 1,
            name: "screen.jpg".to_owned(),
            kind: "image".to_owned(),
            data_url: None,
            width: Some(1280),
            height: Some(800),
            byte_size: 1000,
            seconds: None,
            status: "uploaded".to_owned(),
            included_screen: true,
            upload_id: Some(1),
        });
        message.refresh_token_estimate();
        assert!(message.token_count.unwrap_or_default() >= before + 256);
        assert!(message.token_count_is_estimate);
    }

    #[test]
    fn platform_model_filters_keep_realtime_and_text_separate() {
        assert!(is_realtime_model("gpt-realtime-2.1"));
        assert!(!is_realtime_model("gpt-5.6-sol"));
        assert!(is_text_model("gpt-5.6-sol"));
        assert!(is_text_model("o4-mini"));
        assert!(!is_text_model("gpt-realtime-2.1"));
        assert!(!is_text_model("gpt-image-2"));
    }

    #[test]
    fn model_merge_preserves_unique_ids() {
        let mut models = vec![model_option("gpt-5.6", "GPT-5.6", "", "account")];
        merge_models(
            &mut models,
            vec![
                model_option("gpt-5.6", "duplicate", "", "official"),
                model_option("gpt-5.6-luna", "GPT-5.6 Luna", "", "official"),
            ],
        );
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].label, "GPT-5.6");
    }
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

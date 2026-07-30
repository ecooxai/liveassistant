use crate::{
    audio::{Microphone, Speaker},
    auth,
    codex_account::{self, CodexAccountInfo, CodexUsageInfo, RateLimitWindow},
    live_pointer,
    media::{self, Attachment},
    realtime::{Command, ConnectOptions, Event, RealtimeBackend, RealtimeClient, ToolOutput},
    tools,
};
use anyhow::Context as _;
use base64::{Engine, engine::general_purpose::STANDARD};
use eframe::egui::{self, Color32, RichText, Stroke};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};

const SETTINGS_KEY: &str = "live_assistant.settings";
const SPEECH_SCREENSHOT_SAMPLE_TARGET: usize = 24_000;
const CODEX_USAGE_REFRESH_INTERVAL: Duration = Duration::from_secs(30);
const ASSISTANT_WAV_SPLIT_MIN_SAMPLES: usize = 24_000 * 5;
const MESSAGE_CONTINUATION_WINDOW: Duration = Duration::from_secs(5);
const LEGACY_CLICK_INSTRUCTION: &str = " Do not claim to click or change anything on the computer.";
const LEGACY_DEFAULT_INSTRUCTIONS: &str = "You are a concise, helpful desktop voice assistant. \
    Stay silent until the user has finished speaking; never greet or speak just because the \
    session started. Use the current screen when it is relevant.";
const LEGACY_SHORT_INSTRUCTIONS: &str = "You are a concise, helpful desktop voice assistant. Use \
    the current screen when it is relevant.";
const FALLBACK_REALTIME_VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar",
];
const GPT_LIVE_VOICES: &[&str] = &[
    "juniper", "maple", "spruce", "ember", "vale", "breeze", "arbor", "sol", "cove",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum AuthMode {
    #[default]
    ApiKey,
    CodexApiKey,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    backend: RealtimeBackend,
    model: String,
    voice: String,
    instructions: String,
    auth_mode: AuthMode,
    send_screenshot: bool,
    screenshot_width: u32,
    screenshot_height: u32,
    show_live_pointer: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: RealtimeBackend::OpenAiRealtime,
            model: "gpt-realtime-2.1".to_owned(),
            voice: "marin".to_owned(),
            instructions: String::new(),
            auth_mode: AuthMode::ApiKey,
            send_screenshot: true,
            screenshot_width: 1440,
            screenshot_height: 900,
            show_live_pointer: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
}

struct ChatImage {
    name: String,
    thumbnail: Option<egui::ColorImage>,
    sent_image: Vec<u8>,
    width: u32,
    height: u32,
    byte_size: usize,
    thumbnail_texture: Option<egui::TextureHandle>,
    full_texture: Option<egui::TextureHandle>,
    thumbnail_load_attempted: bool,
    full_load_attempted: bool,
}

struct PreparedChatImage {
    name: String,
    thumbnail: egui::ColorImage,
    sent_image: Vec<u8>,
    width: u32,
    height: u32,
    byte_size: usize,
}

impl PreparedChatImage {
    fn from_attachment(attachment: &Attachment) -> anyhow::Result<Option<Self>> {
        match attachment {
            Attachment::Image {
                name,
                data_url,
                thumbnail,
                width,
                height,
                byte_size,
            } => {
                let sent_image = data_url
                    .split_once(',')
                    .map(|(_, encoded)| STANDARD.decode(encoded))
                    .transpose()?
                    .unwrap_or_default();
                anyhow::ensure!(!sent_image.is_empty(), "sent image data is unavailable");
                Ok(Some(Self {
                    name: name.clone(),
                    thumbnail: decode_color_image(thumbnail)?,
                    sent_image,
                    width: *width,
                    height: *height,
                    byte_size: *byte_size,
                }))
            }
            Attachment::Audio { .. } => Ok(None),
        }
    }

    fn into_chat_image(self) -> ChatImage {
        ChatImage {
            name: self.name,
            thumbnail: Some(self.thumbnail),
            sent_image: self.sent_image,
            width: self.width,
            height: self.height,
            byte_size: self.byte_size,
            thumbnail_texture: None,
            full_texture: None,
            thumbnail_load_attempted: false,
            full_load_attempted: false,
        }
    }
}

impl ChatImage {
    fn from_attachment(attachment: &Attachment) -> Option<Self> {
        PreparedChatImage::from_attachment(attachment)
            .ok()
            .flatten()
            .map(PreparedChatImage::into_chat_image)
    }

    fn ensure_thumbnail_texture(&mut self, ctx: &egui::Context, id: String) -> anyhow::Result<()> {
        if self.thumbnail_texture.is_some() || self.thumbnail_load_attempted {
            return Ok(());
        }
        self.thumbnail_load_attempted = true;
        let thumbnail = self
            .thumbnail
            .take()
            .context("thumbnail pixels are unavailable")?;
        self.thumbnail_texture =
            Some(ctx.load_texture(id, thumbnail, egui::TextureOptions::LINEAR));
        Ok(())
    }

    fn ensure_full_texture(&mut self, ctx: &egui::Context, id: String) -> anyhow::Result<()> {
        if self.full_texture.is_some() || self.full_load_attempted {
            return Ok(());
        }
        self.full_load_attempted = true;
        anyhow::ensure!(
            !self.sent_image.is_empty(),
            "sent image data is unavailable"
        );
        self.full_texture = Some(load_texture(ctx, id, &self.sent_image)?);
        Ok(())
    }
}

fn load_texture(
    ctx: &egui::Context,
    id: String,
    encoded_image: &[u8],
) -> anyhow::Result<egui::TextureHandle> {
    let color_image = decode_color_image(encoded_image)?;
    Ok(ctx.load_texture(id, color_image, egui::TextureOptions::LINEAR))
}

fn decode_color_image(encoded_image: &[u8]) -> anyhow::Result<egui::ColorImage> {
    let rgba = image::load_from_memory(encoded_image)?.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        rgba.as_raw(),
    ))
}

struct VoiceTranscriptSegment {
    item_id: String,
    text: String,
}

struct ChatMessage {
    role: Role,
    text: String,
    audio: Vec<i16>,
    tool_calls: Vec<ToolInvocation>,
    attachment_names: Vec<String>,
    images: Vec<ChatImage>,
    included_screen: bool,
    voice_turn: bool,
    server_item_id: Option<String>,
    voice_transcript_segments: Vec<VoiceTranscriptSegment>,
    voice_last_activity_at: Option<Instant>,
}

struct ToolInvocation {
    call_id: String,
    name: String,
    arguments: String,
    output: Option<String>,
}

impl ChatMessage {
    fn user_text(text: String, attachments: &[Attachment]) -> Self {
        let audio = attachments
            .iter()
            .find_map(|attachment| match attachment {
                Attachment::Audio { pcm24k, .. } => Some(pcm24k.clone()),
                Attachment::Image { .. } => None,
            })
            .unwrap_or_default();
        Self {
            role: Role::User,
            text,
            audio,
            tool_calls: Vec::new(),
            attachment_names: audio_attachment_names(attachments),
            images: attachments
                .iter()
                .filter_map(ChatImage::from_attachment)
                .collect(),
            included_screen: false,
            voice_turn: false,
            server_item_id: None,
            voice_transcript_segments: Vec::new(),
            voice_last_activity_at: None,
        }
    }

    fn user_voice(audio: Vec<i16>, screen: Option<ChatImage>) -> Self {
        let included_screen = screen.is_some();
        Self {
            role: Role::User,
            text: String::new(),
            audio,
            tool_calls: Vec::new(),
            attachment_names: Vec::new(),
            images: screen.into_iter().collect(),
            included_screen,
            voice_turn: true,
            server_item_id: None,
            voice_transcript_segments: Vec::new(),
            voice_last_activity_at: None,
        }
    }

    fn assistant() -> Self {
        Self {
            role: Role::Assistant,
            text: String::new(),
            audio: Vec::new(),
            tool_calls: Vec::new(),
            attachment_names: Vec::new(),
            images: Vec::new(),
            included_screen: false,
            voice_turn: false,
            server_item_id: None,
            voice_transcript_segments: Vec::new(),
            voice_last_activity_at: None,
        }
    }
}

fn recent_voice_continuation_index(messages: &[ChatMessage], now: Instant) -> Option<usize> {
    // Only the newest user message can be continued. Assistant replies may sit
    // below it, but a newer typed/user message must start a separate group.
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role != Role::User {
            continue;
        }
        if !message.voice_turn {
            return None;
        }
        let last_activity = message.voice_last_activity_at?;
        let elapsed = now.checked_duration_since(last_activity)?;
        return (elapsed <= MESSAGE_CONTINUATION_WINDOW).then_some(index);
    }
    None
}

fn seed_existing_voice_transcript(message: &mut ChatMessage) {
    if message.voice_transcript_segments.is_empty() && !message.text.trim().is_empty() {
        message
            .voice_transcript_segments
            .push(VoiceTranscriptSegment {
                item_id: String::new(),
                text: message.text.trim().to_owned(),
            });
    }
}

fn prepare_voice_continuation(message: &mut ChatMessage, now: Instant) {
    seed_existing_voice_transcript(message);
    message.server_item_id = None;
    message.voice_last_activity_at = Some(now);
}

fn merge_voice_transcript(prefix: &str, current: &str) -> String {
    let prefix = prefix.trim();
    let current = current.trim();
    if prefix.is_empty() {
        return current.to_owned();
    }
    if current.is_empty() || prefix.eq_ignore_ascii_case(current) {
        return prefix.to_owned();
    }
    if current.len() >= prefix.len()
        && current
            .get(..prefix.len())
            .is_some_and(|start| start.eq_ignore_ascii_case(prefix))
    {
        return current.to_owned();
    }
    if prefix.len() >= current.len()
        && prefix
            .get(prefix.len() - current.len()..)
            .is_some_and(|end| end.eq_ignore_ascii_case(current))
    {
        return prefix.to_owned();
    }

    // Avoid repeated boundary words when the backend carries a little context
    // into the next VAD segment (for example "good luck" + "luck today").
    let prefix_words = prefix.split_whitespace().collect::<Vec<_>>();
    let current_words = current.split_whitespace().collect::<Vec<_>>();
    let max_overlap = prefix_words.len().min(current_words.len());
    for overlap in (1..=max_overlap).rev() {
        let suffix = &prefix_words[prefix_words.len() - overlap..];
        let start = &current_words[..overlap];
        if suffix
            .iter()
            .zip(start)
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
        {
            let remainder = current_words[overlap..].join(" ");
            return if remainder.is_empty() {
                prefix.to_owned()
            } else {
                format!("{prefix} {remainder}")
            };
        }
    }
    format!("{prefix} {current}")
}

fn message_owns_voice_item(message: &ChatMessage, item_id: &str) -> bool {
    message.server_item_id.as_deref() == Some(item_id)
        || message
            .voice_transcript_segments
            .iter()
            .any(|segment| segment.item_id == item_id)
}

fn register_voice_item(message: &mut ChatMessage, item_id: String, now: Instant) {
    seed_existing_voice_transcript(message);
    if !message_owns_voice_item(message, &item_id) {
        message
            .voice_transcript_segments
            .push(VoiceTranscriptSegment {
                item_id: item_id.clone(),
                text: String::new(),
            });
    }
    message.server_item_id = Some(item_id);
    message.voice_last_activity_at = Some(now);
}

fn update_voice_transcript(message: &mut ChatMessage, item_id: String, text: String, now: Instant) {
    register_voice_item(message, item_id.clone(), now);
    if let Some(segment) = message
        .voice_transcript_segments
        .iter_mut()
        .find(|segment| segment.item_id == item_id)
    {
        segment.text = text;
    }
    message.text = message
        .voice_transcript_segments
        .iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .fold(String::new(), |combined, segment| {
            merge_voice_transcript(&combined, &segment.text)
        });
    message.voice_last_activity_at = Some(now);
}

fn append_voice_audio_fragment(message: &mut ChatMessage, audio: &[i16], now: Instant) {
    if !audio.is_empty() {
        message.audio.extend_from_slice(audio);
    }
    message.voice_last_activity_at = Some(now);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnectionState {
    Offline,
    Connecting,
    Live,
}

#[derive(Default)]
struct SpeechScreenshotGate {
    speech_active: bool,
    sent: bool,
}

impl SpeechScreenshotGate {
    fn begin(&mut self) {
        self.speech_active = true;
        self.sent = false;
    }

    fn end(&mut self) {
        self.speech_active = false;
        self.sent = false;
    }

    fn should_capture(&self, loud_speech_samples: usize, speech_is_active: bool) -> bool {
        if self.sent || !self.speech_active || !speech_is_active {
            return false;
        }
        loud_speech_samples >= SPEECH_SCREENSHOT_SAMPLE_TARGET
    }

    fn mark_sent(&mut self) {
        self.sent = true;
    }
}

struct SpeechScreenshotResult {
    turn_id: u64,
    result: Result<(Attachment, PreparedChatImage), String>,
}

pub struct LiveAssistantApp {
    realtime: RealtimeClient,
    microphone: Option<Microphone>,
    speaker: Option<Speaker>,
    settings: Settings,
    api_key: String,
    show_settings: bool,
    state: ConnectionState,
    status: String,
    error: Option<String>,
    composer: String,
    pending: Vec<Attachment>,
    messages: Vec<ChatMessage>,
    active_assistant_message: Option<usize>,
    assistant_group_deadline: Option<Instant>,
    assistant_text_needs_separator: bool,
    active_response_id: Option<String>,
    last_assistant_item_id: Option<String>,
    speech_screenshot_gate: SpeechScreenshotGate,
    speech_turn_id: u64,
    screenshot_capture_in_flight: Option<u64>,
    screenshot_message_index: Option<usize>,
    deferred_voice_response: bool,
    screenshot_result_tx: Sender<SpeechScreenshotResult>,
    screenshot_result_rx: Receiver<SpeechScreenshotResult>,
    active_voice_message: Option<usize>,
    image_viewer: Option<(usize, usize)>,
    should_scroll: bool,
    tool_calls_running: usize,
    pending_tool_reply: bool,
    tool_result_tx: Sender<(String, String)>,
    tool_result_rx: Receiver<(String, String)>,
    codex_info: Option<CodexAccountInfo>,
    codex_info_loading: bool,
    codex_info_error: Option<String>,
    codex_info_tx: Sender<Result<CodexAccountInfo, String>>,
    codex_info_rx: Receiver<Result<CodexAccountInfo, String>>,
    codex_usage: Option<CodexUsageInfo>,
    codex_usage_loading: bool,
    codex_usage_error: Option<String>,
    codex_usage_refreshed_at: Option<Instant>,
    codex_usage_tx: Sender<Result<CodexUsageInfo, String>>,
    codex_usage_rx: Receiver<Result<CodexUsageInfo, String>>,
    pointer_overlay: live_pointer::OverlayState,
}

fn ensure_assistant_message_index(
    messages: &mut Vec<ChatMessage>,
    active: &mut Option<usize>,
) -> usize {
    if let Some(index) = *active
        && messages
            .get(index)
            .is_some_and(|message| message.role == Role::Assistant)
    {
        return index;
    }
    messages.push(ChatMessage::assistant());
    let index = messages.len() - 1;
    *active = Some(index);
    index
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UserMessagePlacement {
    user_index: usize,
    moved_assistant: Option<(usize, usize)>,
}

/// Append a user message while keeping an in-progress assistant reply directly
/// below the newest user turn. This preserves one transcript/WAV target for the
/// current reply without rendering that reply above overlapping user speech.
fn append_user_message_before_active_assistant(
    messages: &mut Vec<ChatMessage>,
    active_assistant: &mut Option<usize>,
    user_message: ChatMessage,
) -> UserMessagePlacement {
    let active_index = active_assistant.and_then(|index| {
        messages
            .get(index)
            .is_some_and(|message| message.role == Role::Assistant)
            .then_some(index)
    });

    if let Some(from_index) = active_index {
        let assistant = messages.remove(from_index);
        messages.push(user_message);
        let user_index = messages.len() - 1;
        messages.push(assistant);
        let to_index = messages.len() - 1;
        *active_assistant = Some(to_index);
        UserMessagePlacement {
            user_index,
            moved_assistant: Some((from_index, to_index)),
        }
    } else {
        messages.push(user_message);
        *active_assistant = None;
        UserMessagePlacement {
            user_index: messages.len() - 1,
            moved_assistant: None,
        }
    }
}

fn remap_message_index(index: usize, placement: UserMessagePlacement) -> usize {
    let Some((from_index, to_index)) = placement.moved_assistant else {
        return index;
    };
    if index == from_index {
        to_index
    } else if index > from_index {
        index - 1
    } else {
        index
    }
}

impl LiveAssistantApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let mut settings: Settings = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, SETTINGS_KEY))
            .unwrap_or_default();
        settings.instructions = settings.instructions.replace(LEGACY_CLICK_INSTRUCTION, "");
        if [LEGACY_DEFAULT_INSTRUCTIONS, LEGACY_SHORT_INSTRUCTIONS]
            .contains(&settings.instructions.trim())
        {
            settings.instructions.clear();
        }
        if let Ok((width, height)) = media::primary_screen_resolution() {
            settings.screenshot_width = width;
            settings.screenshot_height = height;
        }
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let speaker = Speaker::new().ok();
        let (tool_result_tx, tool_result_rx) = mpsc::channel();
        let (codex_info_tx, codex_info_rx) = mpsc::channel();
        let (codex_usage_tx, codex_usage_rx) = mpsc::channel();
        let (screenshot_result_tx, screenshot_result_rx) = mpsc::channel();
        Self {
            realtime: RealtimeClient::spawn(),
            microphone: None,
            speaker,
            settings,
            api_key,
            show_settings: false,
            state: ConnectionState::Offline,
            status: "Ready".to_owned(),
            error: None,
            composer: String::new(),
            pending: Vec::new(),
            messages: Vec::new(),
            active_assistant_message: None,
            assistant_group_deadline: None,
            assistant_text_needs_separator: false,
            active_response_id: None,
            last_assistant_item_id: None,
            speech_screenshot_gate: SpeechScreenshotGate::default(),
            speech_turn_id: 0,
            screenshot_capture_in_flight: None,
            screenshot_message_index: None,
            deferred_voice_response: false,
            screenshot_result_tx,
            screenshot_result_rx,
            active_voice_message: None,
            image_viewer: None,
            should_scroll: false,
            tool_calls_running: 0,
            pending_tool_reply: false,
            tool_result_tx,
            tool_result_rx,
            codex_info: None,
            codex_info_loading: false,
            codex_info_error: None,
            codex_info_tx,
            codex_info_rx,
            codex_usage: None,
            codex_usage_loading: false,
            codex_usage_error: None,
            codex_usage_refreshed_at: None,
            codex_usage_tx,
            codex_usage_rx,
            pointer_overlay: live_pointer::OverlayState::new(),
        }
    }

    fn refresh_codex_info(&mut self) {
        if self.codex_info_loading {
            return;
        }
        self.codex_info_loading = true;
        self.codex_info_error = None;
        let sender = self.codex_info_tx.clone();
        thread::spawn(move || {
            let result = codex_account::load().map_err(|error| format!("{error:#}"));
            let _ = sender.send(result);
        });
    }

    fn refresh_codex_usage(&mut self) {
        if self.codex_usage_loading {
            return;
        }
        self.codex_usage_loading = true;
        let sender = self.codex_usage_tx.clone();
        thread::spawn(move || {
            let result = codex_account::load_usage().map_err(|error| format!("{error:#}"));
            let _ = sender.send(result);
        });
    }

    fn maybe_refresh_codex_usage(&mut self) {
        if self.settings.backend != RealtimeBackend::CodexGptLive || self.codex_usage_loading {
            return;
        }
        let due = self
            .codex_usage_refreshed_at
            .is_none_or(|last| last.elapsed() >= CODEX_USAGE_REFRESH_INTERVAL);
        if due {
            self.refresh_codex_usage();
        }
    }

    fn resolve_credentials(&self) -> anyhow::Result<(String, Option<String>)> {
        match self.settings.auth_mode {
            AuthMode::ApiKey => {
                let key = self.api_key.trim();
                if key.is_empty() {
                    anyhow::bail!("Enter an OpenAI Platform API key in Settings.")
                }
                Ok((key.to_owned(), None))
            }
            AuthMode::CodexApiKey => {
                let creds = auth::codex_credentials()?;
                Ok((creds.bearer_token, creds.chatgpt_account_id))
            }
        }
    }

    fn start(&mut self) {
        self.error = None;
        let screen_info = match media::primary_screen_info() {
            Ok(screen) => {
                self.settings.screenshot_width = screen.logical_width;
                self.settings.screenshot_height = screen.logical_height;
                screen
            }
            Err(error) => {
                self.error = Some(format!(
                    "Could not read the primary display resolution: {error:#}"
                ));
                return;
            }
        };
        match self.resolve_credentials() {
            Ok((api_key, chatgpt_account_id)) => {
                let options = ConnectOptions {
                    backend: self.settings.backend,
                    api_key,
                    chatgpt_account_id,
                    model: self.settings.model.clone(),
                    voice: self.settings.voice.clone(),
                    instructions: self.settings.instructions.clone(),
                    screen_info,
                };
                let _ = self.realtime.commands.send(Command::Connect(options));
                self.state = ConnectionState::Connecting;
                self.status = "Connecting…".to_owned();
            }
            Err(error) => {
                self.show_settings = true;
                self.error = Some(error.to_string());
            }
        }
    }

    fn stop(&mut self) {
        let _ = self.realtime.commands.send(Command::Disconnect);
        self.microphone = None;
        if let Some(speaker) = &mut self.speaker {
            let _ = speaker.clear();
        }
        self.state = ConnectionState::Offline;
        self.active_response_id = None;
        self.last_assistant_item_id = None;
        self.active_assistant_message = None;
        self.assistant_group_deadline = None;
        self.assistant_text_needs_separator = false;
        self.speech_screenshot_gate.end();
        self.cancel_speech_screenshot();
        self.active_voice_message = None;
        self.tool_calls_running = 0;
        self.pending_tool_reply = false;
        self.status = "Stopped".to_owned();
    }

    fn process_events(&mut self, ctx: &egui::Context) {
        self.process_screenshot_results(ctx);

        while let Ok(result) = self.codex_info_rx.try_recv() {
            self.codex_info_loading = false;
            match result {
                Ok(info) => {
                    self.codex_info = Some(info);
                    self.codex_info_error = None;
                }
                Err(error) => {
                    self.codex_info = None;
                    self.codex_info_error = Some(error);
                }
            }
        }

        while let Ok(result) = self.codex_usage_rx.try_recv() {
            self.codex_usage_loading = false;
            self.codex_usage_refreshed_at = Some(Instant::now());
            match result {
                Ok(usage) => {
                    self.codex_usage = Some(usage);
                    self.codex_usage_error = None;
                }
                Err(error) => {
                    self.codex_usage_error = Some(error);
                }
            }
        }

        while let Ok((call_id, output)) = self.tool_result_rx.try_recv() {
            if let Some(tool) = self
                .messages
                .iter_mut()
                .flat_map(|message| message.tool_calls.iter_mut())
                .find(|tool| tool.call_id == call_id)
            {
                tool.output = Some(pretty_tool_arguments(&output));
                self.should_scroll = true;
            }
        }

        while let Ok(event) = self.realtime.events.try_recv() {
            match event {
                Event::Connecting => {
                    self.state = ConnectionState::Connecting;
                    self.status = "Connecting…".to_owned();
                }
                Event::Reconnecting { attempt, reason } => {
                    self.state = ConnectionState::Connecting;
                    self.status = format!("Reconnecting GPT-Live… attempt {attempt}");
                    self.error = None;
                    eprintln!(
                        "[live-assistant reconnect-ui] attempt={} reason={}",
                        attempt, reason
                    );
                }
                Event::Connected if self.microphone.is_some() && self.speaker.is_some() => {
                    self.state = ConnectionState::Live;
                    self.status = "Mic on · Listening".to_owned();
                    self.error = None;
                }
                Event::Connected => match Microphone::start(self.realtime.commands.clone()) {
                    Ok(mic) if self.speaker.is_some() => {
                        self.microphone = Some(mic);
                        self.state = ConnectionState::Live;
                        self.status = "Mic on · AEC listening".to_owned();
                        self.error = None;
                    }
                    Ok(_) => {
                        self.error = Some("No audio output device is available".to_owned());
                        self.stop();
                    }
                    Err(error) => {
                        self.error = Some(format!("{error:#}"));
                        self.stop();
                    }
                },
                Event::Disconnected => {
                    self.microphone = None;
                    if let Some(speaker) = &mut self.speaker {
                        let _ = speaker.clear();
                    }
                    self.state = ConnectionState::Offline;
                    self.active_response_id = None;
                    self.last_assistant_item_id = None;
                    self.active_assistant_message = None;
                    self.speech_screenshot_gate.end();
                    self.cancel_speech_screenshot();
                    self.active_voice_message = None;
                    self.tool_calls_running = 0;
                    self.pending_tool_reply = false;
                    self.status = "Offline".to_owned();
                }
                Event::SpeechStarted => {
                    if self.settings.backend == RealtimeBackend::CodexGptLive {
                        // GPT-Live is backend-controlled full duplex. Never clear,
                        // truncate, pause, or close assistant playback in the frontend,
                        // including during a temporary quiet gap in the response.
                        self.status = if self.active_response_id.is_some()
                            || self
                                .speaker
                                .as_ref()
                                .map(Speaker::assistant_is_playing)
                                .unwrap_or(false)
                        {
                            "Mic on · Speaking + hearing you…".to_owned()
                        } else {
                            "Mic on · Hearing you…".to_owned()
                        };
                        self.ensure_active_voice_message();
                        self.should_scroll = true;
                        continue;
                    }
                    // OpenAI Realtime uses interruption/truncation for barge-in.
                    self.status = "Mic on · Hearing you…".to_owned();
                    let played_ms = if let Some(speaker) = &mut self.speaker {
                        match speaker.interrupt_assistant() {
                            Ok(played_ms) => played_ms,
                            Err(error) => {
                                self.error = Some(error.to_string());
                                None
                            }
                        }
                    } else {
                        None
                    };
                    if let (Some(item_id), Some(audio_end_ms)) =
                        (self.last_assistant_item_id.take(), played_ms)
                    {
                        if let Some(message) = self
                            .messages
                            .iter_mut()
                            .rev()
                            .find(|message| message.role == Role::Assistant)
                        {
                            let heard_samples = audio_end_ms as usize * 24_000 / 1_000;
                            message.audio.truncate(heard_samples);
                        }
                        let _ = self.realtime.commands.send(Command::TruncateAssistant {
                            item_id,
                            audio_end_ms,
                        });
                    }
                    // Ignore any already-in-flight deltas from the interrupted response,
                    // but keep the assistant transcript/WAV group open for five seconds.
                    self.active_response_id = None;
                    self.touch_assistant_group(Instant::now());
                    self.ensure_active_voice_message();
                    self.should_scroll = true;
                }
                Event::SpeechStopped => {
                    // Server VAD has ended the utterance; only reply on real speech.
                    self.speech_screenshot_gate.end();
                    if !self
                        .microphone
                        .as_ref()
                        .map(Microphone::in_speech)
                        .unwrap_or(false)
                    {
                        if let Some(mic) = &self.microphone {
                            mic.reset_turn();
                        }
                        self.discard_active_voice_message_if_empty();
                        continue;
                    }
                    const MIN_VOICE_SAMPLES: usize = 24_000 / 2; // ~0.5s at 24 kHz
                    let audio = self
                        .microphone
                        .as_ref()
                        .map(Microphone::finish_turn)
                        .unwrap_or_default();
                    if audio.len() < MIN_VOICE_SAMPLES {
                        let has_visible_content = self
                            .active_voice_message
                            .and_then(|index| self.messages.get(index))
                            .is_some_and(|message| {
                                !message.text.trim().is_empty() || !message.images.is_empty()
                            });
                        if !has_visible_content {
                            self.discard_active_voice_message_if_empty();
                            if self.state == ConnectionState::Live {
                                self.status = "Mic on · Listening".to_owned();
                            }
                            continue;
                        }
                    }
                    self.status = "Thinking…".to_owned();
                    self.finish_voice_message(audio);
                    self.should_scroll = true;
                    if self.screenshot_capture_in_flight.is_some() {
                        self.deferred_voice_response = true;
                        self.status = "Finishing screen capture…".to_owned();
                    } else {
                        let _ = self.realtime.commands.send(Command::CreateResponse);
                    }
                }
                Event::InputCommitted { item_id } => {
                    let now = Instant::now();
                    let active = self.active_voice_message.filter(|index| {
                        self.messages
                            .get(*index)
                            .is_some_and(|message| message.role == Role::User && message.voice_turn)
                    });
                    let index = active.or_else(|| {
                        recent_voice_continuation_index(&self.messages, now)
                            .filter(|index| self.messages[*index].server_item_id.is_none())
                    });
                    if let Some(index) = index {
                        register_voice_item(&mut self.messages[index], item_id, now);
                    }
                }
                Event::InputTranscript { item_id, text } => {
                    let now = Instant::now();
                    // First route late transcription events back to the card that owns
                    // their committed server item. SpeechStopped may precede transcript
                    // completion, so using only active_voice_message creates duplicates.
                    let matching_item = self.messages.iter().position(|message| {
                        message.role == Role::User
                            && message.voice_turn
                            && message_owns_voice_item(message, &item_id)
                    });
                    let active = self.active_voice_message.filter(|index| {
                        self.messages
                            .get(*index)
                            .is_some_and(|message| message.role == Role::User && message.voice_turn)
                    });
                    let index = matching_item
                        .or(active)
                        .or_else(|| recent_voice_continuation_index(&self.messages, now))
                        .unwrap_or_else(|| {
                            let index =
                                self.append_user_message(ChatMessage::user_voice(Vec::new(), None));
                            self.messages[index].voice_last_activity_at = Some(now);
                            index
                        });
                    let continuing_existing_item =
                        !message_owns_voice_item(&self.messages[index], &item_id)
                            && !self.messages[index].text.trim().is_empty();
                    update_voice_transcript(&mut self.messages[index], item_id, text, now);
                    if continuing_existing_item {
                        eprintln!(
                            "[live-assistant user-group] merged transcript continuation into message={} window_seconds={}",
                            index,
                            MESSAGE_CONTINUATION_WINDOW.as_secs()
                        );
                    }
                    self.status = "Mic on · Hearing you…".to_owned();
                    self.should_scroll = true;
                }
                Event::AssistantResponseStarted { response_id } => {
                    let now = Instant::now();
                    self.finalize_expired_assistant_group(now);
                    let continue_group = self.assistant_group_is_open(now);
                    if !continue_group {
                        self.active_assistant_message = None;
                        self.assistant_group_deadline = None;
                    }
                    self.assistant_text_needs_separator = continue_group
                        && self.active_assistant_message.is_some_and(|index| {
                            self.messages
                                .get(index)
                                .is_some_and(|message| !message.text.trim().is_empty())
                        });
                    if let Some(speaker) = &mut self.speaker {
                        speaker.begin_assistant_response();
                    }
                    self.active_response_id = Some(response_id);
                    self.last_assistant_item_id = None;
                    self.pending_tool_reply = false;
                    self.touch_assistant_group(now);
                    if continue_group {
                        eprintln!(
                            "[live-assistant assistant-group] continuing message={:?} window_seconds={}",
                            self.active_assistant_message,
                            MESSAGE_CONTINUATION_WINDOW.as_secs()
                        );
                    }
                }
                Event::AssistantItem {
                    response_id,
                    item_id,
                } => {
                    if self.response_is_active(&response_id) {
                        self.last_assistant_item_id = Some(item_id.clone());
                        self.current_assistant().server_item_id = Some(item_id);
                    }
                }
                Event::AssistantTranscriptDelta { response_id, delta } => {
                    if !self.response_is_active(&response_id) {
                        continue;
                    }
                    {
                        let needs_separator = self.assistant_text_needs_separator;
                        let message = self.current_assistant();
                        if needs_separator
                            && !message.text.is_empty()
                            && !message.text.ends_with(char::is_whitespace)
                            && !delta.trim().is_empty()
                        {
                            message.text.push('\n');
                        }
                        message.text.push_str(&delta);
                    }
                    self.assistant_text_needs_separator = false;
                    self.touch_assistant_group(Instant::now());
                    self.should_scroll = true;
                }
                Event::AssistantAudio {
                    response_id,
                    samples,
                } => {
                    if !self.response_is_active(&response_id) {
                        continue;
                    }
                    if let Some(speaker) = &mut self.speaker {
                        speaker.append_assistant(samples.clone());
                    }
                    self.current_assistant().audio.extend_from_slice(&samples);
                    self.touch_assistant_group(Instant::now());
                    self.status = "Mic on · Speaking…".to_owned();
                    self.should_scroll = true;
                }
                Event::AssistantSegmentDone { response_id } => {
                    if self.response_is_active(&response_id) {
                        let long_segment = self
                            .active_assistant_message
                            .and_then(|index| self.messages.get(index))
                            .is_some_and(|message| {
                                message.role == Role::Assistant
                                    && should_split_assistant_wav(message.audio.len())
                            });
                        self.touch_assistant_group(Instant::now());
                        if long_segment {
                            eprintln!(
                                "[live-assistant assistant-group] long WAV split deferred for {}s",
                                MESSAGE_CONTINUATION_WINDOW.as_secs()
                            );
                        }
                    }
                }
                Event::AssistantDone { response_id } => {
                    if self.response_is_active(&response_id) {
                        self.active_response_id = None;
                        self.touch_assistant_group(Instant::now());
                        if self.tool_calls_running > 0 {
                            self.status = format!(
                                "Running {} computer action{}…",
                                self.tool_calls_running,
                                if self.tool_calls_running == 1 {
                                    ""
                                } else {
                                    "s"
                                }
                            );
                        } else if self.state == ConnectionState::Live
                            && !self
                                .speaker
                                .as_ref()
                                .map(Speaker::assistant_is_playing)
                                .unwrap_or(false)
                        {
                            self.status = "Mic on · Listening".to_owned();
                        }
                    }
                }
                Event::ToolCalls(calls) => {
                    let count = calls.len();
                    self.tool_calls_running = self.tool_calls_running.saturating_add(count);
                    self.pending_tool_reply = true;
                    self.touch_assistant_group(Instant::now());
                    {
                        let message = self.current_assistant();
                        message
                            .tool_calls
                            .extend(calls.iter().map(|call| ToolInvocation {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                arguments: pretty_tool_arguments(&call.arguments),
                                output: None,
                            }));
                    }
                    self.should_scroll = true;
                    self.status = format!(
                        "Running {count} computer action{}…",
                        if count == 1 { "" } else { "s" }
                    );
                    let commands = self.realtime.commands.clone();
                    let screenshot_width = self.settings.screenshot_width;
                    let screenshot_height = self.settings.screenshot_height;
                    let tool_result_tx = self.tool_result_tx.clone();
                    thread::spawn(move || {
                        let screen_context = tools::ScreenContext {
                            screenshot_width,
                            screenshot_height,
                        };
                        let outputs: Vec<_> = calls
                            .into_iter()
                            .map(|call| ToolOutput {
                                call_id: call.call_id,
                                output: tools::execute_with_context(
                                    &call.name,
                                    &call.arguments,
                                    screen_context,
                                ),
                            })
                            .collect();
                        for output in &outputs {
                            let _ = tool_result_tx
                                .send((output.call_id.clone(), output.output.clone()));
                        }
                        let _ = commands.send(Command::ToolOutputs(outputs));
                    });
                }
                Event::ToolOutputsSubmitted { count } => {
                    self.tool_calls_running = self.tool_calls_running.saturating_sub(count);
                    self.status = "Thinking…".to_owned();
                }
                Event::Error(message) => {
                    self.error = Some(message);
                    if self.state != ConnectionState::Live {
                        self.state = ConnectionState::Offline;
                    }
                }
            }
        }

        self.finalize_expired_assistant_group(Instant::now());

        if self.state == ConnectionState::Live
            && self.active_response_id.is_none()
            && !self
                .speaker
                .as_ref()
                .map(Speaker::assistant_is_playing)
                .unwrap_or(false)
        {
            if self.status.contains("Speaking") {
                self.status = "Mic on · Listening".to_owned();
            }
        }
    }

    fn assistant_group_is_open(&self, now: Instant) -> bool {
        self.active_assistant_message.is_some_and(|index| {
            self.messages
                .get(index)
                .is_some_and(|message| message.role == Role::Assistant)
        }) && (self.pending_tool_reply
            || self
                .assistant_group_deadline
                .is_none_or(|deadline| now <= deadline))
    }

    fn touch_assistant_group(&mut self, now: Instant) {
        if self.active_assistant_message.is_some_and(|index| {
            self.messages
                .get(index)
                .is_some_and(|message| message.role == Role::Assistant)
        }) {
            self.assistant_group_deadline = Some(now + MESSAGE_CONTINUATION_WINDOW);
        }
    }

    fn finalize_expired_assistant_group(&mut self, now: Instant) {
        let expired = self
            .assistant_group_deadline
            .is_some_and(|deadline| now > deadline);
        if !expired
            || self.active_response_id.is_some()
            || self.pending_tool_reply
            || self.tool_calls_running > 0
        {
            return;
        }
        if let Some(index) = self.active_assistant_message.take()
            && let Some(message) = self.messages.get(index)
        {
            eprintln!(
                "[live-assistant assistant-group] finalized message={} quiet_seconds={} audio_seconds={:.2}",
                index,
                MESSAGE_CONTINUATION_WINDOW.as_secs(),
                message.audio.len() as f64 / 24_000.0
            );
        }
        self.assistant_group_deadline = None;
        self.assistant_text_needs_separator = false;
    }

    fn append_user_message(&mut self, message: ChatMessage) -> usize {
        // A user message only moves a still-open assistant group. Once the
        // five-second AI continuation window has expired, keep the completed
        // assistant card in its existing conversation position.
        self.finalize_expired_assistant_group(Instant::now());
        let placement = append_user_message_before_active_assistant(
            &mut self.messages,
            &mut self.active_assistant_message,
            message,
        );
        if let Some(index) = self.active_voice_message {
            self.active_voice_message = Some(remap_message_index(index, placement));
        }
        if let Some(index) = self.screenshot_message_index {
            self.screenshot_message_index = Some(remap_message_index(index, placement));
        }
        if let Some((message_index, image_index)) = self.image_viewer {
            self.image_viewer = Some((remap_message_index(message_index, placement), image_index));
        }
        placement.user_index
    }

    fn remove_message(&mut self, index: usize) {
        if index >= self.messages.len() {
            return;
        }
        self.messages.remove(index);
        let remap = |tracked: &mut Option<usize>| {
            *tracked = match *tracked {
                Some(value) if value == index => None,
                Some(value) if value > index => Some(value - 1),
                value => value,
            };
        };
        remap(&mut self.active_assistant_message);
        remap(&mut self.active_voice_message);
        remap(&mut self.screenshot_message_index);
        if let Some((message_index, image_index)) = self.image_viewer {
            self.image_viewer = if message_index == index {
                None
            } else {
                Some((
                    if message_index > index {
                        message_index - 1
                    } else {
                        message_index
                    },
                    image_index,
                ))
            };
        }
    }

    fn ensure_active_voice_message(&mut self) -> usize {
        if let Some(index) = self.active_voice_message
            && self
                .messages
                .get(index)
                .is_some_and(|message| message.role == Role::User && message.voice_turn)
        {
            return index;
        }
        if let Some(microphone) = &self.microphone
            && !microphone.in_speech()
        {
            microphone.begin_turn();
        }
        self.speech_turn_id = self.speech_turn_id.wrapping_add(1);
        self.speech_screenshot_gate.begin();
        let now = Instant::now();
        let index = if let Some(index) = recent_voice_continuation_index(&self.messages, now) {
            prepare_voice_continuation(&mut self.messages[index], now);
            eprintln!(
                "[live-assistant user-group] continuing message={} window_seconds={} assistant_active={}",
                index,
                MESSAGE_CONTINUATION_WINDOW.as_secs(),
                self.active_response_id.is_some()
            );
            index
        } else {
            let index = self.append_user_message(ChatMessage::user_voice(Vec::new(), None));
            self.messages[index].voice_last_activity_at = Some(now);
            index
        };
        self.active_voice_message = Some(index);
        index
    }

    fn cancel_speech_screenshot(&mut self) {
        self.speech_turn_id = self.speech_turn_id.wrapping_add(1);
        self.screenshot_capture_in_flight = None;
        self.screenshot_message_index = None;
        self.deferred_voice_response = false;
    }

    fn process_screenshot_results(&mut self, ctx: &egui::Context) {
        while let Ok(capture) = self.screenshot_result_rx.try_recv() {
            if self.screenshot_capture_in_flight != Some(capture.turn_id) {
                continue;
            }
            self.screenshot_capture_in_flight = None;
            let screenshot_message_index = self.screenshot_message_index.take();

            match capture.result {
                Ok((image, prepared_image)) => {
                    if self.state != ConnectionState::Live {
                        continue;
                    }
                    let message_index = screenshot_message_index.unwrap_or_else(|| {
                        self.append_user_message(ChatMessage::user_voice(Vec::new(), None))
                    });
                    let image_index = self.messages[message_index].images.len();
                    let mut chat_image = prepared_image.into_chat_image();
                    if let Err(error) = chat_image.ensure_thumbnail_texture(
                        ctx,
                        format!("chat-thumbnail-{message_index}-{image_index}"),
                    ) {
                        self.error = Some(format!(
                            "Screenshot was captured but could not be displayed: {error:#}"
                        ));
                    }
                    let message = &mut self.messages[message_index];
                    message.images.push(chat_image);
                    message.included_screen = true;
                    eprintln!(
                        "[live-assistant image] captured turn={} message={} image={} total_images={}",
                        capture.turn_id,
                        message_index,
                        image_index,
                        message.images.len()
                    );
                    if capture.turn_id == self.speech_turn_id
                        && self
                            .microphone
                            .as_ref()
                            .map(Microphone::in_speech)
                            .unwrap_or(false)
                    {
                        self.active_voice_message = Some(message_index);
                    }
                    self.should_scroll = true;

                    if self
                        .realtime
                        .commands
                        .send(Command::SendContextImage(image))
                        .is_err()
                    {
                        self.error = Some(
                            "Screenshot captured but not sent: Realtime connection closed"
                                .to_owned(),
                        );
                    } else {
                        self.status = "Mic on · Hearing you… · Screen sent".to_owned();
                    }
                }
                Err(error) => {
                    self.error = Some(format!("Screenshot not sent: {error}"));
                }
            }

            if self.deferred_voice_response {
                self.deferred_voice_response = false;
                if self.state == ConnectionState::Live {
                    let _ = self.realtime.commands.send(Command::CreateResponse);
                    self.status = "Thinking…".to_owned();
                }
            }
        }
    }

    fn maybe_send_speech_screenshot(&mut self, ctx: &egui::Context) {
        if self.state != ConnectionState::Live
            || !self.settings.send_screenshot
            || self.screenshot_capture_in_flight.is_some()
        {
            return;
        }
        let (server_speech_is_active, loud_speech_samples) = self
            .microphone
            .as_ref()
            .map(|microphone| (microphone.in_speech(), microphone.loud_speech_samples()))
            .unwrap_or_default();

        // GPT-Live V3 often starts streaming transcript deltas without a dedicated
        // speech_started item. Open the local user turn from the AEC microphone after
        // one continuous second above the loud threshold, then capture immediately.
        if !self.speech_screenshot_gate.speech_active
            && loud_speech_samples >= SPEECH_SCREENSHOT_SAMPLE_TARGET
        {
            self.ensure_active_voice_message();
        }
        let speech_is_active = server_speech_is_active || self.active_voice_message.is_some();
        if !self
            .speech_screenshot_gate
            .should_capture(loud_speech_samples, speech_is_active)
        {
            return;
        }

        self.speech_screenshot_gate.mark_sent();
        let message_index = self.ensure_active_voice_message();
        let turn_id = self.speech_turn_id;
        self.screenshot_capture_in_flight = Some(turn_id);
        self.screenshot_message_index = Some(message_index);
        self.status = "Mic on · Hearing you… · Capturing screen".to_owned();
        self.should_scroll = true;

        let target_width = self.settings.screenshot_width;
        let target_height = self.settings.screenshot_height;
        let show_live_pointer = self.settings.show_live_pointer;
        let pointer_snapshot = self.pointer_overlay.snapshot();
        let result_tx = self.screenshot_result_tx.clone();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = media::capture_screenshot(
                target_width,
                target_height,
                show_live_pointer,
                pointer_snapshot,
            )
            .and_then(|attachment| {
                let prepared = PreparedChatImage::from_attachment(&attachment)?
                    .context("Screenshot capture did not produce an image")?;
                Ok((attachment, prepared))
            })
            .map_err(|error| format!("{error:#}"));
            let _ = result_tx.send(SpeechScreenshotResult { turn_id, result });
            repaint.request_repaint();
        });
    }

    fn discard_active_voice_message_if_empty(&mut self) {
        let Some(index) = self.active_voice_message.take() else {
            return;
        };
        let should_remove = self.messages.get(index).is_some_and(|message| {
            message.role == Role::User
                && message.voice_turn
                && message.text.trim().is_empty()
                && message.audio.is_empty()
                && message.images.is_empty()
        });
        if should_remove {
            self.remove_message(index);
        }
    }

    fn finish_voice_message(&mut self, audio: Vec<i16>) {
        let now = Instant::now();
        if let Some(index) = self.active_voice_message.take()
            && let Some(message) = self.messages.get_mut(index)
            && message.role == Role::User
            && message.voice_turn
        {
            append_voice_audio_fragment(message, &audio, now);
            return;
        }
        let index = self.append_user_message(ChatMessage::user_voice(Vec::new(), None));
        append_voice_audio_fragment(&mut self.messages[index], &audio, now);
        if self.screenshot_capture_in_flight == Some(self.speech_turn_id) {
            self.screenshot_message_index = Some(index);
        }
    }

    fn response_is_active(&self, response_id: &str) -> bool {
        self.active_response_id
            .as_deref()
            .is_some_and(|active| active == response_id)
    }

    fn current_assistant(&mut self) -> &mut ChatMessage {
        let index =
            ensure_assistant_message_index(&mut self.messages, &mut self.active_assistant_message);
        &mut self.messages[index]
    }

    fn send_composer(&mut self) {
        if self.state != ConnectionState::Live {
            self.error = Some("Start the voice session before sending a message.".to_owned());
            return;
        }
        if self.composer.trim().is_empty() && self.pending.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.composer);
        let attachments = std::mem::take(&mut self.pending);
        self.append_user_message(ChatMessage::user_text(text.clone(), &attachments));
        let _ = self
            .realtime
            .commands
            .send(Command::SendTurn { text, attachments });
        self.status = "Thinking…".to_owned();
        self.should_scroll = true;
    }

    fn attach_path(&mut self, path: &Path) {
        let extension = path
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let result = if ["png", "jpg", "jpeg", "gif", "webp", "bmp"].contains(&extension.as_str()) {
            media::load_image(path)
        } else {
            media::load_audio(path)
        };
        match result {
            Ok(attachment) => self.pending.push(attachment),
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn draw_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(RichText::new("Live Assistant").size(22.0));
            ui.add_space(8.0);
            let (dot, label) = match self.state {
                ConnectionState::Offline => (Color32::from_rgb(112, 120, 132), "Offline"),
                ConnectionState::Connecting => (Color32::from_rgb(190, 123, 20), "Connecting"),
                ConnectionState::Live => (Color32::from_rgb(25, 145, 84), "Live"),
            };
            ui.colored_label(dot, "●");
            ui.label(label);
            ui.label(RichText::new(format!("· {}", self.status)).weak());

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⚙ Settings").clicked() {
                    self.show_settings = true;
                }
                let button = if self.state == ConnectionState::Offline {
                    egui::Button::new(RichText::new("● Start voice").color(Color32::WHITE))
                        .fill(Color32::from_rgb(31, 138, 84))
                } else {
                    egui::Button::new(RichText::new("■ Stop").color(Color32::WHITE))
                        .fill(Color32::from_rgb(190, 56, 65))
                };
                if ui.add(button).clicked() {
                    if self.state == ConnectionState::Offline {
                        self.start();
                    } else {
                        self.stop();
                    }
                }
                ui.separator();
                let (usage_label, usage_detail) = codex_usage_header(
                    self.codex_usage.as_ref(),
                    self.codex_usage_loading,
                    self.codex_usage_error.as_deref(),
                );
                ui.label(RichText::new(usage_label).small().strong())
                    .on_hover_text(usage_detail);
            });
        });
    }

    fn draw_empty(&self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label(RichText::new("Talk, type, or share what you see").size(28.0));
            ui.add_space(10.0);
            ui.label(
                RichText::new(
                    "Start voice, then speak naturally. After one second of clear speech, \
                     the current screen is sent while you are still talking.",
                )
                .weak(),
            );
            ui.add_space(24.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("🎙 Semantic turn detection");
                ui.separator();
                ui.label("▣ Logical-resolution screen context");
                ui.separator();
                ui.label("＋ Images & audio");
            });
        });
    }

    fn draw_messages(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_width();
        for index in 0..self.messages.len() {
            let mut image_error = None;
            for (image_index, image) in self.messages[index].images.iter_mut().enumerate() {
                if let Err(error) = image.ensure_thumbnail_texture(
                    ui.ctx(),
                    format!("chat-thumbnail-{index}-{image_index}"),
                ) {
                    image_error = Some(format!("Could not display {}: {error:#}", image.name));
                }
            }
            if image_error.is_some() {
                self.error = image_error;
            }

            let is_user = self.messages[index].role == Role::User;
            let width = available * 0.76;
            let layout = if is_user {
                egui::Layout::right_to_left(egui::Align::Min)
            } else {
                egui::Layout::left_to_right(egui::Align::Min)
            };
            ui.with_layout(layout, |ui| {
                egui::Frame::new()
                    .fill(if is_user {
                        Color32::from_rgb(224, 238, 255)
                    } else {
                        Color32::WHITE
                    })
                    .stroke(Stroke::new(1.0, Color32::from_rgb(215, 221, 230)))
                    .corner_radius(12.0)
                    .inner_margin(egui::Margin::same(12))
                    .show(ui, |ui| {
                        ui.set_max_width(width);
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        ui.vertical(|ui| {
                            ui.set_max_width(width);
                            let message = &self.messages[index];
                            ui.label(
                                RichText::new(if is_user { "YOU" } else { "ASSISTANT" })
                                    .small()
                                    .color(if is_user {
                                        Color32::from_rgb(37, 92, 158)
                                    } else {
                                        Color32::from_rgb(92, 102, 116)
                                    }),
                            );
                            for tool in &message.tool_calls {
                                ui.add_space(7.0);
                                egui::Frame::new()
                                    .fill(Color32::from_rgb(244, 247, 251))
                                    .stroke(Stroke::new(1.0, Color32::from_rgb(205, 214, 225)))
                                    .corner_radius(7.0)
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.set_max_width((width - 16.0).max(120.0));
                                        ui.label(
                                            RichText::new(format!("⚙ {}", tool.name))
                                                .strong()
                                                .color(Color32::from_rgb(52, 83, 124)),
                                        );
                                        ui.label(
                                            RichText::new(&tool.arguments)
                                                .monospace()
                                                .small()
                                                .color(Color32::from_rgb(64, 73, 86)),
                                        );
                                        if let Some(output) = &tool.output {
                                            ui.separator();
                                            ui.label(RichText::new("Result").small().strong());
                                            ui.label(
                                                RichText::new(output)
                                                    .monospace()
                                                    .small()
                                                    .color(Color32::from_rgb(64, 73, 86)),
                                            );
                                        }
                                    });
                            }
                            for (image_index, image) in message.images.iter().enumerate() {
                                ui.add_space(7.0);
                                if let Some(texture) = &image.thumbnail_texture {
                                    let response = ui.add(
                                        egui::Image::from_texture(texture)
                                            .max_size(egui::vec2(width.min(210.0), 140.0))
                                            .corner_radius(8)
                                            .sense(egui::Sense::click()),
                                    );
                                    paint_image_metadata(
                                        ui,
                                        response.rect,
                                        image.width,
                                        image.height,
                                        image.byte_size,
                                    );
                                    if response.clicked() {
                                        self.image_viewer = Some((index, image_index));
                                    }
                                    response
                                        .on_hover_text("Click to view the exact image sent to AI");
                                }
                                let caption = if message.included_screen {
                                    "▣ Current screen"
                                } else {
                                    &image.name
                                };
                                ui.label(RichText::new(caption).small().weak());
                            }
                            if !is_user {
                                ui.add_space(5.0);
                                ui.label(
                                    RichText::new("Transcript")
                                        .small()
                                        .strong()
                                        .color(Color32::from_rgb(70, 80, 94)),
                                );
                            }
                            if !message.text.is_empty() {
                                ui.add_space(2.0);
                                ui.label(&message.text);
                            } else if message.voice_turn {
                                ui.label(RichText::new("Transcribing…").italics().weak());
                            } else if !is_user {
                                ui.label(
                                    RichText::new("Receiving assistant transcript…")
                                        .italics()
                                        .weak(),
                                );
                            }
                            for name in &message.attachment_names {
                                ui.label(format!("📎 {name}"));
                            }
                            // Audio is deliberately rendered last so every assistant card ends
                            // with one complete-reply WAV control row.
                            if !message.audio.is_empty() {
                                let seconds = message.audio.len() as f32 / 24_000.0;
                                let wav_size = media::wav_file_size(message.audio.len());
                                let end_time = format_duration(seconds);
                                let save_name = if is_user {
                                    "voice-turn.wav".to_owned()
                                } else {
                                    assistant_wav_filename(seconds)
                                };
                                ui.add_space(8.0);
                                ui.separator();
                                ui.add_space(3.0);
                                ui.label(
                                    RichText::new(if is_user {
                                        "Voice input"
                                    } else {
                                        "WAV reply"
                                    })
                                    .small()
                                    .strong()
                                    .color(Color32::from_rgb(70, 80, 94)),
                                );
                                ui.horizontal_wrapped(|ui| {
                                    let play_label = if is_user {
                                        "▶ Play input"
                                    } else {
                                        "▶ Play full reply"
                                    };
                                    if ui.small_button(play_label).clicked()
                                        && let Some(speaker) = &mut self.speaker
                                        && let Err(error) = speaker.play_clip(&message.audio)
                                    {
                                        self.error = Some(error.to_string());
                                    }
                                    ui.label(
                                        RichText::new(format!("0:00 → {end_time}")).small().weak(),
                                    );
                                    ui.label(
                                        RichText::new(format!(
                                            "WAV · {}",
                                            format_file_size(wav_size)
                                        ))
                                        .small()
                                        .weak(),
                                    );
                                    let save_label = if is_user {
                                        "Save WAV".to_owned()
                                    } else {
                                        format!("Save WAV · {end_time}")
                                    };
                                    if ui.small_button(save_label).clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_file_name(save_name)
                                            .add_filter("WAV audio", &["wav"])
                                            .save_file()
                                        && let Err(error) = media::save_wav(&path, &message.audio)
                                    {
                                        self.error = Some(format!("{error:#}"));
                                    }
                                });
                            }
                        });
                    });
            });
            ui.add_space(10.0);
        }
    }

    fn draw_composer(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let pasted_image = ctx.input(|input| {
            input.key_pressed(egui::Key::V) && (input.modifiers.command || input.modifiers.ctrl)
        });
        if pasted_image && let Ok(image) = media::image_from_clipboard() {
            self.pending.push(image);
        }

        let dropped: Vec<_> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        for path in dropped {
            self.attach_path(&path);
        }

        if !self.pending.is_empty() {
            ui.horizontal_wrapped(|ui| {
                let mut remove = None;
                for (index, attachment) in self.pending.iter().enumerate() {
                    let (icon, name) = match attachment {
                        Attachment::Image {
                            name, thumbnail, ..
                        } => {
                            let _ = thumbnail;
                            ("▣", name)
                        }
                        Attachment::Audio { name, seconds, .. } => {
                            let _ = seconds;
                            ("♪", name)
                        }
                    };
                    egui::Frame::new()
                        .fill(Color32::from_rgb(235, 240, 246))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(211, 219, 229)))
                        .corner_radius(7.0)
                        .inner_margin(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(format!("{icon} {name}"));
                                if ui.small_button("×").clicked() {
                                    remove = Some(index);
                                }
                            });
                        });
                }
                if let Some(index) = remove {
                    self.pending.remove(index);
                }
            });
            ui.add_space(6.0);
        }

        egui::Frame::new()
            .fill(Color32::WHITE)
            .stroke(Stroke::new(1.0, Color32::from_rgb(205, 214, 225)))
            .corner_radius(12.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                let response = ui.add_sized(
                    [ui.available_width(), 56.0],
                    egui::TextEdit::multiline(&mut self.composer)
                        .hint_text(
                            "Message the assistant…  (Enter to send, Shift+Enter for a new line)",
                        )
                        .frame(false),
                );
                let enter = response.has_focus()
                    && ui.input(|input| {
                        input.key_pressed(egui::Key::Enter) && !input.modifiers.shift
                    });
                ui.horizontal(|ui| {
                    if ui.button("＋ Attach").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Image or audio",
                                &[
                                    "png", "jpg", "jpeg", "gif", "webp", "wav", "mp3", "m4a",
                                    "flac", "ogg",
                                ],
                            )
                            .pick_file()
                    {
                        self.attach_path(&path);
                    }
                    if ui.button("▣ Paste image").clicked() {
                        match media::image_from_clipboard() {
                            Ok(image) => self.pending.push(image),
                            Err(error) => self.error = Some(error.to_string()),
                        }
                    }
                    let level = self
                        .microphone
                        .as_ref()
                        .map(Microphone::level)
                        .unwrap_or(0.0);
                    if self.state == ConnectionState::Live {
                        ui.add(
                            egui::ProgressBar::new((level * 5.0).clamp(0.0, 1.0))
                                .desired_width(70.0),
                        );
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                self.state == ConnectionState::Live,
                                egui::Button::new(RichText::new("Send  ↑").color(Color32::WHITE))
                                    .fill(Color32::from_rgb(47, 105, 185)),
                            )
                            .clicked()
                        {
                            self.send_composer();
                        }
                    });
                });
                if enter {
                    self.send_composer();
                }
            });
    }

    fn draw_settings(&mut self, ctx: &egui::Context) {
        if self.state == ConnectionState::Offline
            && let Ok((width, height)) = media::primary_screen_resolution()
        {
            self.settings.screenshot_width = width;
            self.settings.screenshot_height = height;
        }
        if self.settings.auth_mode == AuthMode::CodexApiKey
            && self.codex_info.is_none()
            && self.codex_info_error.is_none()
            && !self.codex_info_loading
        {
            self.refresh_codex_info();
        }
        let mut open = self.show_settings;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(540.0)
            .default_height(680.0)
            .max_height((ctx.screen_rect().height() - 40.0).max(360.0))
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(255, 255, 255))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(205, 213, 224)))
                    .corner_radius(10.0)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("settings_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.heading("Realtime session");
                        ui.add_space(8.0);
                        egui::Grid::new("settings_grid")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .show(ui, |ui| {
                                ui.label("Voice engine");
                                let previous_backend = self.settings.backend;
                                egui::ComboBox::from_id_salt("voice_engine")
                                    .selected_text(match self.settings.backend {
                                        RealtimeBackend::OpenAiRealtime => {
                                            "OpenAI Realtime API"
                                        }
                                        RealtimeBackend::CodexGptLive => {
                                            "Codex GPT-Live (experimental)"
                                        }
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.settings.backend,
                                            RealtimeBackend::OpenAiRealtime,
                                            "OpenAI Realtime API",
                                        );
                                        ui.selectable_value(
                                            &mut self.settings.backend,
                                            RealtimeBackend::CodexGptLive,
                                            "Codex GPT-Live via app-server",
                                        );
                                    });
                                if previous_backend != self.settings.backend {
                                    self.settings.voice = match self.settings.backend {
                                        RealtimeBackend::OpenAiRealtime => "marin",
                                        RealtimeBackend::CodexGptLive => "ember",
                                    }
                                    .to_owned();
                                }
                                ui.end_row();

                                ui.label("Realtime model");
                                if self.settings.backend == RealtimeBackend::CodexGptLive {
                                    ui.label("gpt-live-1-boulder-alpha · managed by Codex");
                                } else {
                                    egui::ComboBox::from_id_salt("model")
                                        .selected_text(&self.settings.model)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.settings.model,
                                                "gpt-realtime-2.1".to_owned(),
                                                "gpt-realtime-2.1",
                                            );
                                            ui.selectable_value(
                                                &mut self.settings.model,
                                                "gpt-realtime-2".to_owned(),
                                                "gpt-realtime-2 (2.0)",
                                            );
                                        });
                                }
                                ui.end_row();

                                ui.label("Voice");
                                let voices: Vec<String> = match self.settings.backend {
                                    RealtimeBackend::OpenAiRealtime => self
                                        .codex_info
                                        .as_ref()
                                        .map(|info| info.realtime_voices.v2.clone())
                                        .filter(|voices| !voices.is_empty())
                                        .unwrap_or_else(|| {
                                            FALLBACK_REALTIME_VOICES
                                                .iter()
                                                .map(|voice| (*voice).to_owned())
                                                .collect()
                                        }),
                                    RealtimeBackend::CodexGptLive => self
                                        .codex_info
                                        .as_ref()
                                        .map(|info| info.realtime_voices.v1.clone())
                                        .filter(|voices| !voices.is_empty())
                                        .unwrap_or_else(|| {
                                            GPT_LIVE_VOICES
                                                .iter()
                                                .map(|voice| (*voice).to_owned())
                                                .collect()
                                        }),
                                };
                                egui::ComboBox::from_id_salt("voice")
                                    .selected_text(&self.settings.voice)
                                    .show_ui(ui, |ui| {
                                        if !voices.contains(&self.settings.voice) {
                                            let current = self.settings.voice.clone();
                                            ui.selectable_value(
                                                &mut self.settings.voice,
                                                current.clone(),
                                                format!("{current} (saved)"),
                                            );
                                        }
                                        for voice in voices {
                                            let label = voice.clone();
                                            ui.selectable_value(
                                                &mut self.settings.voice,
                                                voice,
                                                label,
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Authentication");
                                let previous_auth_mode = self.settings.auth_mode;
                                ui.vertical(|ui| {
                                    ui.radio_value(
                                        &mut self.settings.auth_mode,
                                        AuthMode::ApiKey,
                                        "OpenAI Platform API key",
                                    );
                                    ui.radio_value(
                                        &mut self.settings.auth_mode,
                                        AuthMode::CodexApiKey,
                                        "Reuse Codex login (~/.codex/auth.json)",
                                    );
                                });
                                if previous_auth_mode != self.settings.auth_mode
                                    && self.settings.auth_mode == AuthMode::CodexApiKey
                                {
                                    self.refresh_codex_info();
                                }
                                ui.end_row();

                                if self.settings.auth_mode == AuthMode::ApiKey {
                                    ui.label("API key");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.api_key)
                                            .password(true)
                                            .desired_width(330.0)
                                            .hint_text("sk-…"),
                                    );
                                    ui.end_row();
                                }
                            });

                        ui.add_space(8.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(245, 249, 255))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(205, 222, 241)))
                            .corner_radius(7.0)
                            .inner_margin(egui::Margin::same(9))
                            .show(ui, |ui| {
                                ui.label(RichText::new("Voice engines").strong());
                                ui.label(
                                    "gpt-realtime-2.1 · current selectable Realtime API model",
                                );
                                ui.label(
                                    "GPT-Live · experimental V3 WebRTC through `codex app-server`; \
                                     supports Codex-managed ChatGPT login",
                                );
                                ui.label(
                                    RichText::new(
                                        "GPT-Live uses Codex handoffs for screenshots/images and the \
                                         app's local click/bash/insert tools. Audio and transcripts \
                                         remain on the low-latency WebRTC path.",
                                    )
                                    .small()
                                    .weak(),
                                );
                            });

                        if self.settings.auth_mode == AuthMode::CodexApiKey {
                            ui.add_space(10.0);
                            self.draw_codex_account(ui);
                        }

                        ui.add_space(8.0);
                        ui.checkbox(
                            &mut self.settings.send_screenshot,
                            "Send the primary display after 1 second of clear speech",
                        );
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label("macOS logical resolution");
                            ui.label(
                                RichText::new(format!(
                                    "{} × {}",
                                    self.settings.screenshot_width, self.settings.screenshot_height
                                ))
                                .strong(),
                            );
                        });
                        ui.label(
                    RichText::new(
                        "Screenshots and tool coordinates use the resolution selected in macOS \
                         Displays, not the Retina panel's physical pixel resolution.",
                    )
                    .small()
                    .weak(),
                );
                        ui.checkbox(
                            &mut self.settings.show_live_pointer,
                            "Show the Live pointer, click feedback, and press coordinates",
                        );
                        ui.label(
                            RichText::new(
                                "The pointer is also composited into screenshots. Global click \
                                 feedback requires Accessibility permission on macOS and X11 on \
                                 Linux.",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_space(12.0);
                        ui.label("Custom system prompt");
                        ui.label(
                    RichText::new(
                        "Appended after Live Assistant's built-in system prompt. Leave blank to \
                         use only the built-in behavior.",
                    )
                    .small()
                    .weak(),
                );
                        ui.add_sized(
                            [ui.available_width(), 96.0],
                            egui::TextEdit::multiline(&mut self.settings.instructions)
                                .hint_text("Add your own instructions…"),
                        );
                        ui.add_space(10.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(238, 246, 255))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(205, 222, 241)))
                            .corner_radius(7.0)
                            .inner_margin(egui::Margin::same(9))
                            .show(ui, |ui| {
                                ui.label(
                            "Reuse Codex reads ~/.codex/auth.json. OpenAI Realtime uses a Platform \
                             API key. GPT-Live V3 uses Codex-managed ChatGPT authentication and a \
                             native WebRTC audio connection. Access and usage limits are enforced by \
                             the selected ChatGPT account.",
                        );
                            });
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Close").clicked() {
                                self.show_settings = false;
                            }
                            ui.label(
                                RichText::new(
                                    "Platform API keys entered here are never saved by this app.",
                                )
                                .small()
                                .weak(),
                            );
                        });
                    });
            });
        self.show_settings = open && self.show_settings;
    }

    fn draw_codex_account(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(Color32::from_rgb(245, 249, 255))
            .stroke(Stroke::new(1.0, Color32::from_rgb(196, 215, 239)))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Codex account").strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(!self.codex_info_loading, egui::Button::new("Refresh"))
                            .clicked()
                        {
                            self.refresh_codex_info();
                        }
                    });
                });

                if self.codex_info_loading {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Reading models and usage from Codex…");
                    });
                    return;
                }
                if let Some(error) = &self.codex_info_error {
                    ui.colored_label(
                        Color32::from_rgb(166, 34, 48),
                        format!("Could not read Codex account: {error}"),
                    );
                    return;
                }
                let Some(info) = &self.codex_info else {
                    ui.label("No Codex account data loaded.");
                    return;
                };

                for limit in &info.rate_limits {
                    let title = match &limit.plan {
                        Some(plan) => format!("{} · {} plan", limit.name, plan),
                        None => limit.name.clone(),
                    };
                    ui.label(RichText::new(title).strong());
                    if let Some(window) = &limit.primary {
                        draw_rate_limit_window(ui, "Usage limit", window);
                    }
                    if let Some(window) = &limit.secondary {
                        draw_rate_limit_window(ui, "Secondary limit", window);
                    }
                    if limit.unlimited_credits {
                        ui.label("Credits: unlimited");
                    } else if let Some(balance) = &limit.credit_balance {
                        ui.label(format!("Credit balance: {balance}"));
                    }
                    if let Some(reason) = &limit.reached_reason {
                        ui.colored_label(
                            Color32::from_rgb(166, 34, 48),
                            format!("Limit reached: {reason}"),
                        );
                    }
                }
                if let Some(count) = info.reset_credits {
                    ui.label(format!("Available full-limit resets: {count}"));
                }

                let usage = &info.token_usage;
                if usage.lifetime_tokens.is_some()
                    || usage.latest_day_tokens.is_some()
                    || usage.peak_daily_tokens.is_some()
                {
                    ui.add_space(7.0);
                    ui.label(RichText::new("Token usage").strong());
                    egui::Grid::new("codex_token_usage")
                        .num_columns(2)
                        .spacing([14.0, 3.0])
                        .show(ui, |ui| {
                            if let Some(tokens) = usage.lifetime_tokens {
                                ui.label("Lifetime");
                                ui.label(format_count(tokens));
                                ui.end_row();
                            }
                            if let Some(tokens) = usage.latest_day_tokens {
                                ui.label(
                                    usage
                                        .latest_day
                                        .as_deref()
                                        .map(|day| format!("Latest day ({day})"))
                                        .unwrap_or_else(|| "Latest day".to_owned()),
                                );
                                ui.label(format_count(tokens));
                                ui.end_row();
                            }
                            if let Some(tokens) = usage.recent_reported_tokens {
                                ui.label("Last 7 reported days");
                                ui.label(format_count(tokens));
                                ui.end_row();
                            }
                            if let Some(tokens) = usage.peak_daily_tokens {
                                ui.label("Peak day");
                                ui.label(format_count(tokens));
                                ui.end_row();
                            }
                        });
                }

                if !info.realtime_voices.v2.is_empty() || !info.realtime_voices.v1.is_empty() {
                    ui.add_space(7.0);
                    ui.label(RichText::new("Available Codex voice personas").strong());
                    if !info.realtime_voices.v2.is_empty() {
                        let default = info
                            .realtime_voices
                            .default_v2
                            .as_deref()
                            .map(|voice| format!(" · default: {voice}"))
                            .unwrap_or_default();
                        ui.label(format!(
                            "OpenAI Realtime / v2{default}: {}",
                            info.realtime_voices.v2.join(", ")
                        ));
                    }
                    if !info.realtime_voices.v1.is_empty() {
                        let default = info
                            .realtime_voices
                            .default_v1
                            .as_deref()
                            .map(|voice| format!(" · default: {voice}"))
                            .unwrap_or_default();
                        ui.label(format!(
                            "GPT-Live / v3 voice set{default}: {}",
                            info.realtime_voices.v1.join(", ")
                        ));
                    }
                }

                ui.add_space(7.0);
                ui.label(
                    RichText::new(format!("Available Codex models ({})", info.models.len()))
                        .strong(),
                );
                ui.label(
                    RichText::new(
                        "These are the models available to the Codex account. The Realtime model \
                         above is separate because voice sessions require a Realtime model ID.",
                    )
                    .small()
                    .weak(),
                );
                egui::ScrollArea::vertical()
                    .id_salt("codex_models")
                    .max_height(130.0)
                    .show(ui, |ui| {
                        for model in &info.models {
                            ui.horizontal(|ui| {
                                ui.label(&model.display_name);
                                ui.label(RichText::new(format!("({})", model.name)).small().weak());
                                if model.hidden {
                                    ui.label(RichText::new("hidden").small().weak());
                                }
                            });
                        }
                    });

                for warning in &info.warnings {
                    ui.label(RichText::new(warning).small().weak());
                }
            });
    }

    fn draw_image_viewer(&mut self, ctx: &egui::Context) {
        let Some((message_index, image_index)) = self.image_viewer else {
            return;
        };
        let Some(image) = self
            .messages
            .get_mut(message_index)
            .and_then(|message| message.images.get_mut(image_index))
        else {
            self.image_viewer = None;
            return;
        };

        let load_result = image.ensure_full_texture(
            ctx,
            format!("chat-full-image-{message_index}-{image_index}"),
        );
        if let Err(error) = load_result {
            self.error = Some(format!("Could not open {}: {error:#}", image.name));
            self.image_viewer = None;
            return;
        }
        let name = image.name.clone();
        let texture = image.full_texture.clone();

        let mut open = true;
        egui::Window::new(name)
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(900.0, 650.0))
            .show(ctx, |ui| {
                if let Some(texture) = &texture {
                    egui::ScrollArea::both().show(ui, |ui| {
                        ui.add(
                            egui::Image::from_texture(texture)
                                .max_size(ui.available_size())
                                .corner_radius(6),
                        );
                    });
                }
            });

        if !open || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.image_viewer = None;
        }
    }

    fn draw_live_pointer(&self, ctx: &egui::Context) {
        if self.state != ConnectionState::Live || !self.settings.show_live_pointer {
            return;
        }
        let Some(pointer) = self.pointer_overlay.snapshot() else {
            return;
        };
        let window_origin =
            pointer.position - egui::vec2(live_pointer::WINDOW_OFFSET, live_pointer::WINDOW_OFFSET);
        let builder = egui::ViewportBuilder::default()
            .with_title(live_pointer::POINTER_WINDOW_TITLE)
            .with_position(window_origin)
            .with_inner_size([live_pointer::WINDOW_SIZE, live_pointer::WINDOW_SIZE])
            .with_min_inner_size([live_pointer::WINDOW_SIZE, live_pointer::WINDOW_SIZE])
            .with_max_inner_size([live_pointer::WINDOW_SIZE, live_pointer::WINDOW_SIZE])
            .with_resizable(false)
            .with_decorations(false)
            .with_transparent(true)
            .with_active(false)
            .with_always_on_top()
            .with_mouse_passthrough(true)
            .with_taskbar(false);

        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("live-pointer-overlay"),
            builder,
            |pointer_ctx, class| {
                if class == egui::ViewportClass::Embedded {
                    return;
                }
                make_viewport_transparent(pointer_ctx);
                live_pointer::harden_native_transparency(live_pointer::POINTER_WINDOW_TITLE);
                let painter = pointer_ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("live-pointer-painter"),
                ));
                live_pointer::paint_egui(
                    &painter,
                    egui::pos2(live_pointer::WINDOW_OFFSET, live_pointer::WINDOW_OFFSET),
                    pointer.appearance,
                );
                pointer_ctx.request_repaint_after(Duration::from_millis(40));
            },
        );

        if pointer.any_button_down()
            && let Some(coordinates) = live_pointer::coordinate_overlay(pointer.position)
        {
            let coordinate_builder = egui::ViewportBuilder::default()
                .with_title(live_pointer::COORDINATE_WINDOW_TITLE)
                .with_position(coordinates.window_origin)
                .with_inner_size([
                    live_pointer::COORDINATE_WINDOW_WIDTH,
                    live_pointer::COORDINATE_WINDOW_HEIGHT,
                ])
                .with_min_inner_size([
                    live_pointer::COORDINATE_WINDOW_WIDTH,
                    live_pointer::COORDINATE_WINDOW_HEIGHT,
                ])
                .with_max_inner_size([
                    live_pointer::COORDINATE_WINDOW_WIDTH,
                    live_pointer::COORDINATE_WINDOW_HEIGHT,
                ])
                .with_resizable(false)
                .with_decorations(false)
                .with_transparent(true)
                .with_active(false)
                .with_always_on_top()
                .with_mouse_passthrough(true)
                .with_taskbar(false);
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("live-pointer-coordinate-overlay"),
                coordinate_builder,
                |coordinate_ctx, class| {
                    if class == egui::ViewportClass::Embedded {
                        return;
                    }
                    make_viewport_transparent(coordinate_ctx);
                    live_pointer::harden_native_transparency(live_pointer::COORDINATE_WINDOW_TITLE);
                    let painter = coordinate_ctx.layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("live-pointer-coordinate-painter"),
                    ));
                    live_pointer::paint_coordinates(&painter, coordinates.x, coordinates.y);
                    coordinate_ctx.request_repaint_after(Duration::from_millis(40));
                },
            );
        }
    }
}

impl eframe::App for LiveAssistantApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // Child pointer viewports need a genuinely transparent swapchain clear.
        // The main viewport remains opaque because its panels paint every pixel.
        [0.0, 0.0, 0.0, 0.0]
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, SETTINGS_KEY, &self.settings);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.state == ConnectionState::Live && self.settings.show_live_pointer {
            self.pointer_overlay.poll();
        }
        self.process_events(ctx);
        self.maybe_refresh_codex_usage();
        self.maybe_send_speech_screenshot(ctx);
        ctx.request_repaint_after(Duration::from_millis(40));

        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(252, 252, 251))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(222, 226, 232)))
                    .inner_margin(egui::Margin::symmetric(18, 12)),
            )
            .show(ctx, |ui| self.draw_header(ui));

        egui::TopBottomPanel::bottom("composer")
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(247, 249, 252))
                    .inner_margin(egui::Margin::symmetric(18, 12)),
            )
            .show(ctx, |ui| {
                if let Some(error) = self.error.clone() {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(255, 235, 237))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(239, 188, 195)))
                        .corner_radius(7.0)
                        .inner_margin(egui::Margin::symmetric(9, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.colored_label(Color32::from_rgb(166, 34, 48), error);
                                if ui.small_button("×").clicked() {
                                    self.error = None;
                                }
                            });
                        });
                    ui.add_space(7.0);
                }
                self.draw_composer(ui, ctx);
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(247, 249, 252))
                    .inner_margin(egui::Margin::symmetric(22, 18)),
            )
            .show(ctx, |ui| {
                if self.messages.is_empty() {
                    self.draw_empty(ui);
                } else {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            self.draw_messages(ui);
                            if self.should_scroll {
                                ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                                self.should_scroll = false;
                            }
                        });
                }
            });

        if self.show_settings {
            self.draw_settings(ctx);
        }
        if self.image_viewer.is_some() {
            self.draw_image_viewer(ctx);
        }
        self.draw_live_pointer(ctx);
    }
}

fn make_viewport_transparent(ctx: &egui::Context) {
    ctx.send_viewport_cmd(egui::ViewportCommand::Transparent(true));
    let mut visuals = ctx.style().visuals.clone();
    visuals.window_fill = Color32::TRANSPARENT;
    visuals.window_stroke = egui::Stroke::NONE;
    visuals.window_shadow = egui::epaint::Shadow::NONE;
    visuals.popup_shadow = egui::epaint::Shadow::NONE;
    visuals.panel_fill = Color32::TRANSPARENT;
    visuals.extreme_bg_color = Color32::TRANSPARENT;
    visuals.faint_bg_color = Color32::TRANSPARENT;
    ctx.set_visuals(visuals);
}

fn audio_attachment_names(attachments: &[Attachment]) -> Vec<String> {
    attachments
        .iter()
        .filter_map(|attachment| match attachment {
            Attachment::Audio { name, .. } => Some(name.clone()),
            Attachment::Image { .. } => None,
        })
        .collect()
}

fn should_split_assistant_wav(sample_count: usize) -> bool {
    sample_count > ASSISTANT_WAV_SPLIT_MIN_SAMPLES
}

fn format_duration(seconds: f32) -> String {
    let total = seconds.round() as u32;
    format!("{}:{:02}", total / 60, total % 60)
}

fn assistant_wav_filename(seconds: f32) -> String {
    let total_ms = (seconds.max(0.0) * 1_000.0).round() as u64;
    let minutes = total_ms / 60_000;
    let seconds = (total_ms / 1_000) % 60;
    let milliseconds = total_ms % 1_000;
    format!("assistant-reply-{minutes:02}m{seconds:02}s-{milliseconds:03}ms.wav")
}

fn codex_usage_header(
    usage: Option<&CodexUsageInfo>,
    loading: bool,
    error: Option<&str>,
) -> (String, String) {
    let Some(usage) = usage else {
        if loading {
            return (
                "Codex usage…".to_owned(),
                "Refreshing Codex usage".to_owned(),
            );
        }
        return (
            "Codex usage unavailable".to_owned(),
            error
                .unwrap_or("No Codex usage snapshot has been loaded yet")
                .to_owned(),
        );
    };
    let primary = usage
        .rate_limits
        .iter()
        .find_map(|limit| limit.primary.as_ref().map(|window| (limit, window)));
    let mut label_parts = vec!["Codex".to_owned()];
    if let Some((_, window)) = primary {
        label_parts.push(format!("{}% used", window.used_percent.clamp(0, 100)));
    }
    if let Some(tokens) = usage.token_usage.latest_day_tokens {
        label_parts.push(format!("{} today", format_compact_count(tokens)));
    }
    if usage.reset_credits.is_some_and(|credits| credits > 0) {
        label_parts.push(format!(
            "{} resets",
            usage.reset_credits.unwrap_or_default()
        ));
    }

    let mut detail = Vec::new();
    for limit in &usage.rate_limits {
        if let Some(window) = &limit.primary {
            let mut line = format!(
                "{}: {}% used · {}% remaining",
                limit.name,
                window.used_percent.clamp(0, 100),
                100 - window.used_percent.clamp(0, 100)
            );
            if let Some(resets_at) = window.resets_at {
                line.push_str(&format!(" · resets {}", format_reset_time(resets_at)));
            }
            detail.push(line);
        }
        if let Some(balance) = &limit.credit_balance {
            detail.push(format!("{} credits: {balance}", limit.name));
        } else if limit.unlimited_credits {
            detail.push(format!("{} credits: unlimited", limit.name));
        }
    }
    if let Some(tokens) = usage.token_usage.latest_day_tokens {
        detail.push(format!("Today: {} tokens", format_count(tokens)));
    }
    if let Some(tokens) = usage.token_usage.lifetime_tokens {
        detail.push(format!("Lifetime: {} tokens", format_count(tokens)));
    }
    if !usage.warnings.is_empty() {
        detail.extend(usage.warnings.iter().cloned());
    }
    if let Some(error) = error {
        detail.push(format!("Last refresh error: {error}"));
    }
    detail.push("Refreshes every 30 seconds".to_owned());
    (
        label_parts.join(" · "),
        detail.join(
            "
",
        ),
    )
}

fn format_compact_count(value: i64) -> String {
    let absolute = value.unsigned_abs() as f64;
    let sign = if value < 0 { "-" } else { "" };
    if absolute >= 1_000_000.0 {
        format!("{sign}{:.1}M", absolute / 1_000_000.0)
    } else if absolute >= 1_000.0 {
        format!("{sign}{:.1}K", absolute / 1_000.0)
    } else {
        format!("{value}")
    }
}

fn draw_rate_limit_window(ui: &mut egui::Ui, label: &str, window: &RateLimitWindow) {
    let used = window.used_percent.clamp(0, 100);
    ui.add(
        egui::ProgressBar::new(used as f32 / 100.0)
            .text(format!("{label}: {used}% used · {}% remaining", 100 - used))
            .desired_width(ui.available_width().min(420.0)),
    );
    let mut detail = Vec::new();
    if let Some(minutes) = window.window_duration_minutes {
        detail.push(format!("{} window", format_minutes(minutes)));
    }
    if let Some(resets_at) = window.resets_at {
        detail.push(format!("resets {}", format_reset_time(resets_at)));
    }
    if !detail.is_empty() {
        ui.label(RichText::new(detail.join(" · ")).small().weak());
    }
}

fn format_reset_time(timestamp: i64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(timestamp);
    let remaining = timestamp.saturating_sub(now);
    if remaining == 0 {
        return "now".to_owned();
    }
    let minutes = (remaining + 59) / 60;
    format!("in {}", format_minutes(minutes))
}

fn format_minutes(minutes: i64) -> String {
    if minutes >= 24 * 60 {
        let days = minutes / (24 * 60);
        let hours = (minutes % (24 * 60)) / 60;
        if hours == 0 {
            format!("{days}d")
        } else {
            format!("{days}d {hours}h")
        }
    } else if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("{minutes}m")
    }
}

fn format_count(value: i64) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(character);
    }
    if negative {
        formatted.insert(0, '-');
    }
    formatted
}

fn paint_image_metadata(
    ui: &egui::Ui,
    image_rect: egui::Rect,
    width: u32,
    height: u32,
    byte_size: usize,
) {
    let text = format!("{width} × {height}  ·  {}", format_file_size(byte_size));
    let font = egui::FontId::proportional(10.5);
    let galley = ui.painter().layout_no_wrap(text, font, Color32::WHITE);
    let padding = egui::vec2(6.0, 3.0);
    let size = galley.size() + padding * 2.0;
    let top_right = image_rect.right_top() + egui::vec2(-5.0, 5.0);
    let overlay_rect = egui::Rect::from_min_size(top_right - egui::vec2(size.x, 0.0), size);
    ui.painter()
        .rect_filled(overlay_rect, 5.0, Color32::from_black_alpha(184));
    ui.painter()
        .galley(overlay_rect.min + padding, galley, Color32::WHITE);
}

fn format_file_size(byte_size: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    if byte_size as f64 >= MIB {
        format!("{:.1} MB", byte_size as f64 / MIB)
    } else if byte_size as f64 >= KIB {
        format!("{:.0} KB", byte_size as f64 / KIB)
    } else {
        format!("{byte_size} B")
    }
}

fn pretty_tool_arguments(arguments: &str) -> String {
    const MAX_VISIBLE_CHARS: usize = 4_000;
    let pretty = serde_json::from_str::<serde_json::Value>(arguments)
        .and_then(|value| serde_json::to_string_pretty(&value))
        .unwrap_or_else(|_| arguments.to_owned());
    let mut visible = pretty.chars().take(MAX_VISIBLE_CHARS).collect::<String>();
    if pretty.chars().count() > MAX_VISIBLE_CHARS {
        visible.push_str("\n… parameters truncated in the UI");
    }
    visible
}

fn install_multilingual_font_fallback(ctx: &egui::Context) -> Option<&'static str> {
    const CANDIDATES: &[(&str, u32)] = &[
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 0),
    ];
    let (path, index, bytes) = CANDIDATES.iter().find_map(|(path, index)| {
        std::fs::read(path)
            .ok()
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| (*path, *index, bytes))
    })?;

    let name = "system_multilingual_fallback".to_owned();
    let mut fonts = egui::FontDefinitions::default();
    let mut data = egui::FontData::from_owned(bytes);
    data.index = index;
    fonts.font_data.insert(name.clone(), Arc::new(data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(names) = fonts.families.get_mut(&family) {
            names.push(name.clone());
        }
    }
    ctx.set_fonts(fonts);
    eprintln!("[live-assistant font] multilingual fallback={path} face_index={index}");
    Some(path)
}

fn configure_style(ctx: &egui::Context) {
    if install_multilingual_font_fallback(ctx).is_none() {
        eprintln!("[live-assistant font] no Japanese/Chinese system font fallback found");
    }
    let mut visuals = egui::Visuals::light();
    visuals.window_fill = Color32::WHITE;
    visuals.panel_fill = Color32::from_rgb(247, 249, 252);
    visuals.extreme_bg_color = Color32::WHITE;
    visuals.faint_bg_color = Color32::from_rgb(241, 244, 248);
    visuals.selection.bg_fill = Color32::from_rgb(70, 125, 199);
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(207, 214, 224));
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = egui::vec2(12.0, 7.0);
    style.spacing.item_spacing = egui::vec2(8.0, 7.0);
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn multilingual_font_fallback_has_japanese_and_chinese_glyphs() {
        let ctx = egui::Context::default();
        assert!(install_multilingual_font_fallback(&ctx).is_some());
        let mut supported = false;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            supported = ctx.fonts(|fonts| {
                fonts.has_glyphs(
                    &egui::FontId::proportional(16.0),
                    "日本語ひらがなカタカナ 中文你好",
                )
            });
        });
        assert!(supported);
    }

    #[test]
    fn codex_usage_header_shows_rate_limit_and_daily_tokens() {
        let usage = CodexUsageInfo {
            rate_limits: vec![codex_account::RateLimit {
                name: "Codex".to_owned(),
                plan: Some("pro".to_owned()),
                primary: Some(RateLimitWindow {
                    used_percent: 42,
                    window_duration_minutes: Some(300),
                    resets_at: None,
                }),
                secondary: None,
                credit_balance: None,
                unlimited_credits: false,
                reached_reason: None,
            }],
            token_usage: codex_account::TokenUsage {
                latest_day_tokens: Some(12_500),
                ..Default::default()
            },
            reset_credits: None,
            warnings: Vec::new(),
        };
        let (label, detail) = codex_usage_header(Some(&usage), false, None);
        assert!(label.contains("42% used"));
        assert!(label.contains("12.5K today"));
        assert!(detail.contains("Refreshes every 30 seconds"));
    }

    #[test]
    fn overlapping_user_message_moves_current_assistant_reply_below_user() {
        let mut messages = vec![ChatMessage::assistant()];
        let mut active_assistant = Some(0);
        messages[0].text = "current reply".to_owned();
        messages[0].audio.extend_from_slice(&[1, 2, 3]);

        let placement = append_user_message_before_active_assistant(
            &mut messages,
            &mut active_assistant,
            ChatMessage::user_voice(vec![9, 9], None),
        );
        let assistant_index = ensure_assistant_message_index(&mut messages, &mut active_assistant);
        messages[assistant_index].audio.extend_from_slice(&[4, 5]);

        assert_eq!(placement.user_index, 0);
        assert_eq!(active_assistant, Some(1));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].audio, vec![9, 9]);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].text, "current reply");
        assert_eq!(messages[1].audio, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn assistant_wav_splits_only_after_more_than_five_seconds() {
        assert!(!should_split_assistant_wav(
            ASSISTANT_WAV_SPLIT_MIN_SAMPLES - 1
        ));
        assert!(!should_split_assistant_wav(ASSISTANT_WAV_SPLIT_MIN_SAMPLES));
        assert!(should_split_assistant_wav(
            ASSISTANT_WAV_SPLIT_MIN_SAMPLES + 1
        ));
    }

    #[test]
    fn assistant_group_window_is_five_seconds() {
        assert_eq!(MESSAGE_CONTINUATION_WINDOW, Duration::from_secs(5));
    }

    #[test]
    fn assistant_text_and_audio_can_append_across_a_user_card() {
        let now = Instant::now();
        let mut assistant = ChatMessage::assistant();
        assistant.text = "first AI part".to_owned();
        assistant.audio.extend_from_slice(&[1, 2]);
        let messages = vec![ChatMessage::user_voice(vec![9], None), assistant];
        let deadline = now + MESSAGE_CONTINUATION_WINDOW;

        assert!(now + Duration::from_secs(4) <= deadline);
        assert!(now + Duration::from_secs(6) > deadline);
        assert_eq!(messages[1].text, "first AI part");
        assert_eq!(messages[1].audio, vec![1, 2]);
    }

    #[test]
    fn assistant_wav_filename_contains_reply_duration() {
        assert_eq!(
            assistant_wav_filename(64.321),
            "assistant-reply-01m04s-321ms.wav"
        );
    }

    #[test]
    fn user_voice_fragments_continue_for_five_seconds_with_assistant_between() {
        let now = Instant::now();
        let mut user = ChatMessage::user_voice(vec![1, 2], None);
        user.text = "I wish we have good".to_owned();
        user.voice_last_activity_at = Some(now - Duration::from_secs(4));
        let messages = vec![user, ChatMessage::assistant()];

        assert_eq!(recent_voice_continuation_index(&messages, now), Some(0));
    }

    #[test]
    fn user_voice_fragment_after_five_seconds_starts_new_message() {
        let now = Instant::now();
        let mut user = ChatMessage::user_voice(vec![1, 2], None);
        user.voice_last_activity_at =
            Some(now - MESSAGE_CONTINUATION_WINDOW - Duration::from_millis(1));
        let messages = vec![user, ChatMessage::assistant()];

        assert_eq!(recent_voice_continuation_index(&messages, now), None);
    }

    #[test]
    fn newer_typed_user_message_blocks_voice_continuation() {
        let now = Instant::now();
        let mut voice = ChatMessage::user_voice(vec![1], None);
        voice.voice_last_activity_at = Some(now - Duration::from_secs(1));
        let messages = vec![
            voice,
            ChatMessage::assistant(),
            ChatMessage::user_text("new typed turn".to_owned(), &[]),
        ];

        assert_eq!(recent_voice_continuation_index(&messages, now), None);
    }

    #[test]
    fn continued_voice_transcript_and_audio_append_to_one_card() {
        let now = Instant::now();
        let mut message = ChatMessage::user_voice(vec![1, 2], None);
        message.text = "I wish we have good".to_owned();
        message.server_item_id = Some("first".to_owned());
        prepare_voice_continuation(&mut message, now);
        update_voice_transcript(&mut message, "second".to_owned(), "luck".to_owned(), now);
        append_voice_audio_fragment(&mut message, &[3, 4], now);
        prepare_voice_continuation(&mut message, now);
        update_voice_transcript(
            &mut message,
            "third".to_owned(),
            "and do you think".to_owned(),
            now,
        );
        append_voice_audio_fragment(&mut message, &[5, 6], now);
        update_voice_transcript(
            &mut message,
            "second".to_owned(),
            "luck indeed".to_owned(),
            now,
        );

        assert_eq!(
            message.text,
            "I wish we have good luck indeed and do you think"
        );
        assert_eq!(message.audio, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(message.voice_transcript_segments.len(), 3);
    }

    #[test]
    fn speech_screenshot_gate_waits_for_one_second_of_loud_audio_and_fires_once() {
        let mut gate = SpeechScreenshotGate::default();

        assert_eq!(SPEECH_SCREENSHOT_SAMPLE_TARGET, 24_000);
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, true));
        gate.begin();
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET - 1, true));
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, false));
        assert!(gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, true));
        gate.mark_sent();
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET * 2, true));

        gate.end();
        gate.begin();
        assert!(gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, true));
    }
}

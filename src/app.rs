use crate::{
    audio::{MessageRecorder, Microphone, Speaker},
    auth,
    codex_account::{self, CodexAccountInfo, CodexUsageInfo, RateLimitWindow},
    image_generation, live_pointer,
    media::{self, Attachment, ScreenInfo},
    notes,
    realtime::{
        CONTEXT_IMAGE_UPLOAD_TIMEOUT, Command, ConnectOptions, Event, RealtimeBackend,
        RealtimeClient, ToolOutput, available_tool_descriptions, default_system_prompt,
        shared_system_prompt,
    },
    tools,
};
use anyhow::Context as _;
use base64::{Engine, engine::general_purpose::STANDARD};
use device_query::{DeviceState, Keycode};
use eframe::egui::{self, Color32, RichText, Stroke};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SETTINGS_KEY: &str = "live_assistant.settings";
/// Capture after roughly half a second of clear 24 kHz speech so the image is
/// already visible and uploading before the user finishes the sentence.
const SPEECH_SCREENSHOT_SAMPLE_TARGET: usize = 24_000 / 2;
const CODEX_USAGE_REFRESH_INTERVAL: Duration = Duration::from_secs(30);
const MESSAGE_CONTINUATION_WINDOW: Duration = Duration::from_secs(5);
const SYSTEM_AUDIO_COMMAND_HOLD_DELAY: Duration = Duration::from_secs(1);
const NOTE_DISK_POLL_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct CommandHoldState {
    pressed_at: Option<Instant>,
    active: bool,
}

impl CommandHoldState {
    fn update(&mut self, command_down: bool, now: Instant) -> bool {
        if !command_down {
            self.pressed_at = None;
            self.active = false;
            return false;
        }

        let pressed_at = self.pressed_at.get_or_insert(now);
        self.active = now.saturating_duration_since(*pressed_at) >= SYSTEM_AUDIO_COMMAND_HOLD_DELAY;
        self.active
    }
}

#[derive(Default)]
struct SystemAudioCommandHold {
    device_state: Option<DeviceState>,
    input_initialization_attempted: bool,
    hold: CommandHoldState,
}

impl SystemAudioCommandHold {
    fn command_is_down(&mut self, ctx: &egui::Context) -> bool {
        if !self.input_initialization_attempted {
            self.input_initialization_attempted = true;
            self.device_state = DeviceState::checked_new();
        }

        if let Some(device_state) = &self.device_state {
            let keys = device_state.query_keymap();
            if keys.iter().any(|key| {
                matches!(
                    key,
                    Keycode::Command | Keycode::RCommand | Keycode::LMeta | Keycode::RMeta
                )
            }) {
                return true;
            }
        }

        // This fallback works while the app is focused if global input access is
        // unavailable or has not yet been granted.
        ctx.input(|input| input.modifiers.command)
    }

    fn poll(&mut self, ctx: &egui::Context, now: Instant) -> bool {
        let command_down = self.command_is_down(ctx);
        self.hold.update(command_down, now)
    }

    fn reset(&mut self) {
        self.hold = CommandHoldState::default();
    }
}

fn app_plays_live_assistant_audio(backend: RealtimeBackend) -> bool {
    backend != RealtimeBackend::CodexText
        && !(backend == RealtimeBackend::CodexGptLive
            && crate::gpt_live_webrtc::uses_platform_audio())
}
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
const FALLBACK_TEXT_MODELS: &[(&str, &str)] = &[
    ("gpt-5.6-sol", "GPT-5.6 Sol"),
    ("gpt-5.6-luna", "GPT-5.6 Luna"),
    ("gpt-5.6", "GPT-5.6"),
];
const LIGHT_BLUE: Color32 = Color32::from_rgb(232, 243, 255);
const LIGHT_BLUE_SELECTED: Color32 = Color32::from_rgb(201, 226, 255);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum AuthMode {
    #[default]
    ApiKey,
    CodexApiKey,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ThinkingLevel {
    Minimal,
    #[default]
    Light,
    Medium,
    High,
    ExtraHigh,
    Ultra,
}

impl ThinkingLevel {
    fn label(self) -> &'static str {
        match self {
            Self::Minimal => "Minimal",
            Self::Light => "Light",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::ExtraHigh => "Extra high",
            Self::Ultra => "Ultra",
        }
    }

    fn wire_value(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Light => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::ExtraHigh => "xhigh",
            Self::Ultra => "ultra",
        }
    }

    const VISIBLE: [Self; 3] = [Self::Light, Self::Medium, Self::High];
    const MORE: [Self; 3] = [Self::Minimal, Self::ExtraHigh, Self::Ultra];
    const ALL: [Self; 6] = [
        Self::Minimal,
        Self::Light,
        Self::Medium,
        Self::High,
        Self::ExtraHigh,
        Self::Ultra,
    ];
}

fn parse_thinking_level(value: &str) -> Option<ThinkingLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "minimal" => Some(ThinkingLevel::Minimal),
        "light" | "low" => Some(ThinkingLevel::Light),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "extra_high" | "extra-high" | "xhigh" => Some(ThinkingLevel::ExtraHigh),
        "ultra" => Some(ThinkingLevel::Ultra),
        _ => None,
    }
}

fn style_light_blue_popup(ui: &mut egui::Ui) {
    let visuals = ui.visuals_mut();
    visuals.window_fill = LIGHT_BLUE;
    visuals.panel_fill = LIGHT_BLUE;
    visuals.extreme_bg_color = LIGHT_BLUE;
    visuals.faint_bg_color = LIGHT_BLUE;
    visuals.selection.bg_fill = LIGHT_BLUE_SELECTED;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    backend: RealtimeBackend,
    model: String,
    default_text_model: String,
    default_image_model: String,
    default_image_resolution: ImageResolution,
    default_thinking_level: ThinkingLevel,
    thinking_level: ThinkingLevel,
    voice: String,
    system_prompt: String,
    append_realtime_tool_prompt: bool,
    /// Kept only to migrate settings saved by older builds.
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
            default_text_model: "gpt-5.6-luna".to_owned(),
            default_image_model: "gpt-image-2".to_owned(),
            default_image_resolution: ImageResolution::Square1024,
            default_thinking_level: ThinkingLevel::Light,
            thinking_level: ThinkingLevel::Light,
            voice: "marin".to_owned(),
            system_prompt: String::new(),
            append_realtime_tool_prompt: true,
            instructions: String::new(),
            auth_mode: AuthMode::ApiKey,
            send_screenshot: true,
            screenshot_width: 1440,
            screenshot_height: 900,
            show_live_pointer: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum ImageResolution {
    #[default]
    Square1024,
    Portrait1024,
    Landscape1536,
    Landscape2560,
    Landscape3840,
}

impl ImageResolution {
    const ALL: [Self; 5] = [
        Self::Square1024,
        Self::Portrait1024,
        Self::Landscape1536,
        Self::Landscape2560,
        Self::Landscape3840,
    ];

    fn wire_value(self) -> &'static str {
        match self {
            Self::Square1024 => "1024x1024",
            Self::Portrait1024 => "1024x1536",
            Self::Landscape1536 => "1536x1024",
            Self::Landscape2560 => "2560x1440",
            Self::Landscape3840 => "3840x2160",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Square1024 => "1024 × 1024",
            Self::Portrait1024 => "1024 × 1536",
            Self::Landscape1536 => "1536 × 1024 · HD Landscape",
            Self::Landscape2560 => "2560 × 1440 · HD Landscape",
            Self::Landscape3840 => "3840 × 2160 · Ultra HD Landscape",
        }
    }
}

const OPENAI_REALTIME_TOOL_GUIDANCE: &str = "- OpenAI Realtime fast tool behavior:
  - When a user request requires a tool, call the required tool as the first response output, before speaking or emitting assistant text. Do not acknowledge, explain, or promise before the tool call.
  - After the tool result arrives, reply to the user briefly and accurately with the real result. This tool-first order is required for faster actions.
  - For screen click requests, call ask_text_model first with include_screenshot=true and instruct it to inspect the fresh screenshot and perform the click with click_screen. Do not estimate coordinates or call click_screen directly in the OpenAI Realtime layer.";
const NOTE_CHANGE_SYSTEM_GUIDANCE: &str = "- When a user message contains a note file change wrapped in a <note_FILENAME >...</note_FILENAME> block, reply exactly: note saved";

fn configured_system_prompt(settings: &Settings, screen: ScreenInfo) -> String {
    let mut prompt = if !settings.system_prompt.trim().is_empty() {
        settings.system_prompt.clone()
    } else if !settings.instructions.trim().is_empty() {
        // Compatibility for settings created before the full prompt became
        // editable. New saves clear this legacy field after migration.
        shared_system_prompt(&settings.instructions, screen)
    } else {
        default_system_prompt(screen)
    };
    if !prompt.contains("reply exactly: note saved") {
        prompt.push_str("\n\n");
        prompt.push_str(NOTE_CHANGE_SYSTEM_GUIDANCE);
    }
    prompt
}

fn connection_system_prompt(settings: &Settings, screen: ScreenInfo) -> String {
    let mut prompt = configured_system_prompt(settings, screen);
    if settings.backend == RealtimeBackend::OpenAiRealtime && settings.append_realtime_tool_prompt {
        prompt.push_str("\n\n");
        prompt.push_str(OPENAI_REALTIME_TOOL_GUIDANCE);
    }
    prompt
}

fn voice_tab_settings(base: &Settings, backend: RealtimeBackend) -> Settings {
    let mut settings = base.clone();
    settings.backend = backend;
    settings.model = "gpt-realtime-2.1".to_owned();
    let voice = match backend {
        RealtimeBackend::OpenAiRealtime => "marin".to_owned(),
        RealtimeBackend::CodexGptLive => "ember".to_owned(),
        RealtimeBackend::CodexText => settings.voice.clone(),
    };
    settings.voice = voice;
    settings
}

fn default_voice_tab_settings(base: &Settings) -> [Settings; 2] {
    [
        voice_tab_settings(base, RealtimeBackend::CodexGptLive),
        voice_tab_settings(base, RealtimeBackend::OpenAiRealtime),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    System(RealtimeBackend),
    User,
    Assistant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageUploadState {
    Uploaded,
    Uploading,
    Failed,
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
    upload_id: Option<u64>,
    upload_state: ImageUploadState,
    upload_deadline: Option<Instant>,
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
                let thumbnail = decode_color_image(thumbnail)
                    // The transport image is also a valid local preview. Falling back
                    // to it keeps a successfully captured screenshot visible even if
                    // its separately encoded thumbnail cannot be decoded.
                    .or_else(|_| decode_color_image(&sent_image))?;
                Ok(Some(Self {
                    name: name.clone(),
                    thumbnail,
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
            upload_id: None,
            upload_state: ImageUploadState::Uploaded,
            upload_deadline: None,
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

    fn begin_upload(&mut self, upload_id: u64, deadline: Instant) {
        self.upload_id = Some(upload_id);
        self.upload_state = ImageUploadState::Uploading;
        self.upload_deadline = Some(deadline);
    }

    fn finish_upload(&mut self) -> bool {
        match self.upload_state {
            ImageUploadState::Uploading => {
                self.upload_state = ImageUploadState::Uploaded;
                self.upload_deadline = None;
                true
            }
            ImageUploadState::Uploaded => true,
            // A response received after the ten-second deadline must not revive
            // an upload the app has already given up on.
            ImageUploadState::Failed => false,
        }
    }

    fn fail_upload(&mut self) -> bool {
        if self.upload_state != ImageUploadState::Uploading {
            return false;
        }
        self.upload_state = ImageUploadState::Failed;
        self.upload_deadline = None;
        true
    }

    fn upload_timed_out(&self, now: Instant) -> bool {
        self.upload_state == ImageUploadState::Uploading
            && self.upload_deadline.is_some_and(|deadline| now >= deadline)
    }

    fn preview_tint(&self) -> Color32 {
        match self.upload_state {
            ImageUploadState::Uploaded => Color32::WHITE,
            // 70% transparent means the image is rendered at 30% opacity.
            ImageUploadState::Uploading | ImageUploadState::Failed => Color32::from_white_alpha(77),
        }
    }

    fn upload_overlay_text(&self) -> Option<&'static str> {
        match self.upload_state {
            ImageUploadState::Uploaded => None,
            ImageUploadState::Uploading => Some("Uploading…"),
            ImageUploadState::Failed => Some("Upload failed"),
        }
    }

    fn upload_is_complete(&self) -> bool {
        self.upload_state == ImageUploadState::Uploaded
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
    response_id: Option<String>,
    started_at: SystemTime,
    finished_at: Option<SystemTime>,
    started_instant: Instant,
    finished_instant: Option<Instant>,
    token_count: Option<u64>,
    token_count_is_estimate: bool,
    actual_token_total: u64,
    last_usage_response_id: Option<String>,
    tokens_per_second: Option<f64>,
}

struct ToolInvocation {
    call_id: String,
    name: String,
    arguments: String,
    output: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VoiceToolRoute {
    AskTextModel,
    CreateImage,
    Local,
}

fn voice_tool_route(name: &str) -> VoiceToolRoute {
    match name {
        "ask_text_model" => VoiceToolRoute::AskTextModel,
        "create_image" => VoiceToolRoute::CreateImage,
        _ => VoiceToolRoute::Local,
    }
}

impl ChatMessage {
    fn timing_start() -> (SystemTime, Instant) {
        (SystemTime::now(), Instant::now())
    }

    fn system(text: String, backend: RealtimeBackend) -> Self {
        let (started_at, started_instant) = Self::timing_start();
        Self {
            role: Role::System(backend),
            text,
            audio: Vec::new(),
            tool_calls: Vec::new(),
            attachment_names: Vec::new(),
            images: Vec::new(),
            included_screen: false,
            voice_turn: false,
            server_item_id: None,
            voice_transcript_segments: Vec::new(),
            voice_last_activity_at: None,
            response_id: None,
            started_at,
            finished_at: None,
            started_instant,
            finished_instant: None,
            token_count: None,
            token_count_is_estimate: false,
            actual_token_total: 0,
            last_usage_response_id: None,
            tokens_per_second: None,
        }
    }

    fn user_text(text: String, attachments: &[Attachment]) -> Self {
        let (started_at, started_instant) = Self::timing_start();
        let audio = attachments
            .iter()
            .find_map(|attachment| match attachment {
                Attachment::Audio { pcm24k, .. } => Some(pcm24k.clone()),
                Attachment::Image { .. } => None,
            })
            .unwrap_or_default();
        let mut message = Self {
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
            response_id: None,
            started_at,
            finished_at: None,
            started_instant,
            finished_instant: None,
            token_count: None,
            token_count_is_estimate: false,
            actual_token_total: 0,
            last_usage_response_id: None,
            tokens_per_second: None,
        };
        message.finish();
        message
    }

    fn user_voice(audio: Vec<i16>, screen: Option<ChatImage>) -> Self {
        let (started_at, started_instant) = Self::timing_start();
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
            response_id: None,
            started_at,
            finished_at: None,
            started_instant,
            finished_instant: None,
            token_count: None,
            token_count_is_estimate: false,
            actual_token_total: 0,
            last_usage_response_id: None,
            tokens_per_second: None,
        }
    }

    fn assistant() -> Self {
        let (started_at, started_instant) = Self::timing_start();
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
            response_id: None,
            started_at,
            finished_at: None,
            started_instant,
            finished_instant: None,
            token_count: None,
            token_count_is_estimate: false,
            actual_token_total: 0,
            last_usage_response_id: None,
            tokens_per_second: None,
        }
    }

    fn refresh_token_estimate(&mut self) {
        if self.last_usage_response_id.is_some() {
            return;
        }
        self.token_count = Some(estimated_message_tokens(self));
        self.token_count_is_estimate = true;
        self.recompute_tokens_per_second();
    }

    fn finish(&mut self) {
        if self.finished_instant.is_none() {
            self.finished_instant = Some(Instant::now());
            self.finished_at = Some(SystemTime::now());
        }
        if self.last_usage_response_id.is_none() {
            self.refresh_token_estimate();
        }
        self.recompute_tokens_per_second();
    }

    fn begin_response(&mut self, response_id: String) {
        self.response_id = Some(response_id);
        self.finished_at = None;
        self.finished_instant = None;
        self.tokens_per_second = None;
    }

    fn reopen_for_voice_continuation(&mut self) {
        self.finished_at = None;
        self.finished_instant = None;
        self.tokens_per_second = None;
    }

    fn set_actual_usage(&mut self, response_id: &str, total_tokens: u64) {
        if self.last_usage_response_id.as_deref() != Some(response_id) {
            self.actual_token_total = self.actual_token_total.saturating_add(total_tokens);
            self.last_usage_response_id = Some(response_id.to_owned());
        }
        self.token_count = Some(self.actual_token_total);
        self.token_count_is_estimate = false;
        self.recompute_tokens_per_second();
    }

    fn recompute_tokens_per_second(&mut self) {
        self.tokens_per_second = self
            .finished_instant
            .and_then(|finished| finished.checked_duration_since(self.started_instant))
            .and_then(|elapsed| {
                let seconds = elapsed.as_secs_f64();
                (seconds > 0.0).then(|| self.token_count.unwrap_or_default() as f64 / seconds)
            });
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
    total = total.saturating_add(
        message
            .attachment_names
            .iter()
            .map(|name| estimated_text_tokens(name))
            .sum::<u64>(),
    );
    for tool in &message.tool_calls {
        total = total
            .saturating_add(estimated_text_tokens(&tool.name))
            .saturating_add(estimated_text_tokens(&tool.arguments));
        if let Some(output) = &tool.output {
            total = total.saturating_add(estimated_text_tokens(output));
        }
    }
    if !message.images.is_empty() {
        // Image tokenization varies by model. Keep the estimate deliberately
        // conservative and visibly marked as approximate in the UI.
        total = total.saturating_add(256 * message.images.len() as u64);
    }
    if !message.audio.is_empty() {
        // Audio is not represented by the text tokenizer, so use a small
        // duration-based placeholder until the backend reports usage.
        total = total.saturating_add((message.audio.len() as u64 / 2_400).max(1));
    }
    total.max(u64::from(
        !message.text.is_empty()
            || !message.audio.is_empty()
            || !message.images.is_empty()
            || !message.attachment_names.is_empty()
            || !message.tool_calls.is_empty(),
    ))
}

fn apply_assistant_usage(
    messages: &mut [ChatMessage],
    response_id: &str,
    total_tokens: u64,
) -> bool {
    messages
        .iter_mut()
        .rev()
        .find(|message| {
            message.role == Role::Assistant && message.response_id.as_deref() == Some(response_id)
        })
        .map(|message| {
            message.set_actual_usage(response_id, total_tokens);
            true
        })
        .unwrap_or(false)
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
    message.reopen_for_voice_continuation();
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
    message.refresh_token_estimate();
}

fn append_voice_audio_fragment(message: &mut ChatMessage, audio: &[i16], now: Instant) {
    if !audio.is_empty() {
        message.audio.extend_from_slice(audio);
    }
    message.voice_last_activity_at = Some(now);
    message.refresh_token_estimate();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnectionState {
    Offline,
    Connecting,
    Live,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self::Offline
    }
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
    result: Result<Attachment, String>,
}

#[derive(Clone)]
struct PendingTurn {
    text: String,
    attachments: Vec<Attachment>,
    thinking_level: String,
}

struct TabSession {
    settings: Settings,
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
    next_context_upload_id: u64,
    confirmed_context_uploads: HashSet<u64>,
    captured_transcript_items: HashSet<String>,
    deferred_voice_response: bool,
    active_voice_message: Option<usize>,
    latest_screen_image: Option<Attachment>,
    image_viewer: Option<(usize, usize)>,
    should_scroll: bool,
    tool_calls_running: usize,
    pending_tool_reply: bool,
    pending_turns: VecDeque<PendingTurn>,
}

impl TabSession {
    fn new(settings: Settings) -> Self {
        Self {
            settings,
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
            next_context_upload_id: 0,
            confirmed_context_uploads: HashSet::new(),
            captured_transcript_items: HashSet::new(),
            deferred_voice_response: false,
            active_voice_message: None,
            latest_screen_image: None,
            image_viewer: None,
            should_scroll: false,
            tool_calls_running: 0,
            pending_tool_reply: false,
            pending_turns: VecDeque::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BottomWorkspace {
    #[default]
    Chat,
    Note,
}

struct AssistantTab {
    session: TabSession,
}

impl AssistantTab {
    fn new(settings: Settings) -> Self {
        Self {
            session: TabSession::new(settings),
        }
    }
}

pub struct LiveAssistantApp {
    realtime: RealtimeClient,
    microphone: Option<Microphone>,
    system_audio_command_hold: SystemAudioCommandHold,
    message_recorder: Option<MessageRecorder>,
    speaker: Option<Speaker>,
    playing_message_audio: Option<usize>,
    settings: Settings,
    api_key: String,
    show_settings: bool,
    state: ConnectionState,
    status: String,
    error: Option<String>,
    composer: String,
    bottom_workspace: BottomWorkspace,
    note_files: Vec<PathBuf>,
    active_note: Option<PathBuf>,
    note_content: String,
    note_saved_content: String,
    note_dirty_since: Option<Instant>,
    note_disk_checked_at: Instant,
    renaming_note: Option<PathBuf>,
    rename_buffer: String,
    rename_needs_focus: bool,
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
    next_context_upload_id: u64,
    confirmed_context_uploads: HashSet<u64>,
    captured_transcript_items: HashSet<String>,
    deferred_voice_response: bool,
    screenshot_result_tx: Sender<SpeechScreenshotResult>,
    screenshot_result_rx: Receiver<SpeechScreenshotResult>,
    active_voice_message: Option<usize>,
    latest_screen_image: Option<Attachment>,
    image_viewer: Option<(usize, usize)>,
    should_scroll: bool,
    tool_calls_running: usize,
    pending_tool_reply: bool,
    pending_turns: VecDeque<PendingTurn>,
    tool_result_tx: Sender<(usize, String, String)>,
    tool_result_rx: Receiver<(usize, String, String)>,
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
    tabs: Vec<AssistantTab>,
    active_tab: usize,
    show_model_picker: bool,
    /// Every tab owns a transport. Inactive transports are drained by
    /// process_background_events instead of being disconnected when the tab
    /// view changes.
    background_clients: HashMap<usize, RealtimeClient>,
    pending_ask_calls: HashMap<String, (usize, usize)>,
    pending_ask_order: VecDeque<String>,
    pending_voice_tool_outputs: Vec<(usize, ToolOutput)>,
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

fn centered_icon_button(
    ui: &mut egui::Ui,
    _id_source: impl std::hash::Hash,
    size: f32,
    paint_icon: impl FnOnce(&egui::Painter, egui::Rect, Stroke),
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    let stroke = if response.hovered() || response.has_focus() {
        ui.visuals().widgets.hovered.fg_stroke
    } else {
        ui.visuals().widgets.inactive.fg_stroke
    };
    paint_icon(ui.painter(), rect, stroke);
    response
}

fn centered_plus_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.5);
        let center = rect.center();
        let arm = (size * 0.20).clamp(4.0, 5.0);
        painter.line_segment(
            [
                egui::pos2(center.x - arm, center.y),
                egui::pos2(center.x + arm, center.y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x, center.y - arm),
                egui::pos2(center.x, center.y + arm),
            ],
            stroke,
        );
    })
}

fn centered_play_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, stroke| {
        let center = rect.center() + egui::vec2(1.0, 0.0);
        painter.add(egui::Shape::convex_polygon(
            vec![
                center + egui::vec2(-4.0, -6.0),
                center + egui::vec2(5.5, 0.0),
                center + egui::vec2(-4.0, 6.0),
            ],
            stroke.color,
            Stroke::NONE,
        ));
    })
}

fn centered_pause_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, stroke| {
        let center = rect.center();
        let bar_width = 3.0;
        let bar_height = 12.0;
        painter.rect_filled(
            egui::Rect::from_center_size(
                center + egui::vec2(-3.0, 0.0),
                egui::vec2(bar_width, bar_height),
            ),
            0.8,
            stroke.color,
        );
        painter.rect_filled(
            egui::Rect::from_center_size(
                center + egui::vec2(3.0, 0.0),
                egui::vec2(bar_width, bar_height),
            ),
            0.8,
            stroke.color,
        );
    })
}

fn centered_save_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.5);
        let center = rect.center();
        let tip_y = center.y + 2.5;
        painter.line_segment(
            [
                egui::pos2(center.x, center.y - 6.0),
                egui::pos2(center.x, tip_y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x - 3.8, tip_y - 3.8),
                egui::pos2(center.x, tip_y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x + 3.8, tip_y - 3.8),
                egui::pos2(center.x, tip_y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x - 5.5, center.y + 6.0),
                egui::pos2(center.x + 5.5, center.y + 6.0),
            ],
            stroke,
        );
    })
}

fn centered_down_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.5);
        let center = rect.center();
        let shaft_top = center.y - 5.0;
        let tip_y = center.y + 5.0;
        painter.line_segment(
            [egui::pos2(center.x, shaft_top), egui::pos2(center.x, tip_y)],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x - 4.0, tip_y - 4.0),
                egui::pos2(center.x, tip_y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(center.x + 4.0, tip_y - 4.0),
                egui::pos2(center.x, tip_y),
            ],
            stroke,
        );
    })
}

fn centered_file_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.4);
        let center = rect.center();
        let left = center.x - 5.0;
        let right = center.x + 5.0;
        let top = center.y - 6.0;
        let bottom = center.y + 6.0;
        let fold = 3.5;
        painter.add(egui::Shape::line(
            vec![
                egui::pos2(left, bottom),
                egui::pos2(left, top),
                egui::pos2(right - fold, top),
                egui::pos2(right, top + fold),
                egui::pos2(right, bottom),
                egui::pos2(left, bottom),
            ],
            stroke,
        ));
        painter.line_segment(
            [
                egui::pos2(right - fold, top),
                egui::pos2(right - fold, top + fold),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(right - fold, top + fold),
                egui::pos2(right, top + fold),
            ],
            stroke,
        );
    })
}

fn centered_attach_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.45);
        let c = rect.center();
        painter.add(egui::Shape::line(
            vec![
                c + egui::vec2(-4.5, 1.5),
                c + egui::vec2(-4.5, -3.0),
                c + egui::vec2(-2.0, -5.5),
                c + egui::vec2(1.5, -5.5),
                c + egui::vec2(4.5, -2.5),
                c + egui::vec2(4.5, 3.0),
                c + egui::vec2(2.0, 5.5),
                c + egui::vec2(-1.0, 5.5),
                c + egui::vec2(-3.0, 3.5),
                c + egui::vec2(-3.0, -1.5),
                c + egui::vec2(-1.5, -3.0),
                c + egui::vec2(1.0, -3.0),
                c + egui::vec2(2.5, -1.5),
                c + egui::vec2(2.5, 2.0),
            ],
            stroke,
        ));
    })
}

fn centered_paste_button(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    size: f32,
) -> egui::Response {
    let panel_fill = ui.visuals().panel_fill;
    centered_icon_button(ui, id_source, size, |painter, rect, mut stroke| {
        stroke.width = stroke.width.max(1.4);
        let body = egui::Rect::from_center_size(
            rect.center() + egui::vec2(0.0, 1.0),
            egui::vec2(10.0, 12.0),
        );
        painter.rect_stroke(body, 1.5, stroke, egui::StrokeKind::Inside);
        let clip = egui::Rect::from_center_size(
            egui::pos2(rect.center().x, body.top()),
            egui::vec2(5.5, 3.0),
        );
        painter.rect_filled(clip, 1.0, panel_fill);
        painter.rect_stroke(clip, 1.0, stroke, egui::StrokeKind::Inside);
    })
}

fn sidebar_text_button(ui: &mut egui::Ui, text: &str, selected: bool) -> egui::Response {
    let height = 24.0;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    let color = if selected {
        Color32::from_rgb(32, 112, 177)
    } else if response.hovered() {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().text_color()
    };
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    ui.painter().text(
        egui::pos2(rect.left() + 4.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        text,
        font_id,
        color,
    );
    if selected {
        ui.painter().line_segment(
            [
                egui::pos2(rect.left(), rect.top() + 3.0),
                egui::pos2(rect.left(), rect.bottom() - 3.0),
            ],
            Stroke::new(2.0, color),
        );
    }
    response
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
        if settings.default_image_model.trim().is_empty() {
            settings.default_image_model = "gpt-image-2".to_owned();
        }
        if let Ok((width, height)) = media::primary_screen_resolution() {
            settings.screenshot_width = width;
            settings.screenshot_height = height;
        }
        let screen_info = Self::screen_info_for_settings(&settings);
        if settings.system_prompt.trim().is_empty()
            || settings
                .system_prompt
                .contains("do not call click_screen directly")
            || settings
                .system_prompt
                .contains("Call ask_text_model as your first output")
        {
            settings.system_prompt = default_system_prompt(screen_info);
        }
        settings.instructions.clear();
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        // Audio devices are opened lazily. Production voice sessions use the
        // app-owned capture/playout path so macOS AEC and Command passthrough are controllable.
        let speaker = None;
        let (tool_result_tx, tool_result_rx) = mpsc::channel();
        let (codex_info_tx, codex_info_rx) = mpsc::channel();
        let (codex_usage_tx, codex_usage_rx) = mpsc::channel();
        let (screenshot_result_tx, screenshot_result_rx) = mpsc::channel();
        let [gpt_live_tab_settings, realtime_tab_settings] = default_voice_tab_settings(&settings);
        settings = gpt_live_tab_settings.clone();
        let note_files = notes::list_notes().unwrap_or_default();
        let mut app = Self {
            realtime: RealtimeClient::spawn(),
            microphone: None,
            system_audio_command_hold: SystemAudioCommandHold::default(),
            message_recorder: None,
            speaker,
            playing_message_audio: None,
            settings,
            api_key,
            show_settings: false,
            state: ConnectionState::Offline,
            status: "Ready".to_owned(),
            error: None,
            composer: String::new(),
            bottom_workspace: BottomWorkspace::Chat,
            note_files,
            active_note: None,
            note_content: String::new(),
            note_saved_content: String::new(),
            note_dirty_since: None,
            note_disk_checked_at: Instant::now(),
            renaming_note: None,
            rename_buffer: String::new(),
            rename_needs_focus: false,
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
            next_context_upload_id: 0,
            confirmed_context_uploads: HashSet::new(),
            captured_transcript_items: HashSet::new(),
            deferred_voice_response: false,
            screenshot_result_tx,
            screenshot_result_rx,
            active_voice_message: None,
            latest_screen_image: None,
            image_viewer: None,
            should_scroll: false,
            tool_calls_running: 0,
            pending_tool_reply: false,
            pending_turns: VecDeque::new(),
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
            tabs: vec![
                AssistantTab::new(gpt_live_tab_settings),
                AssistantTab::new(realtime_tab_settings),
            ],
            active_tab: 0,
            show_model_picker: false,
            background_clients: HashMap::new(),
            pending_ask_calls: HashMap::new(),
            pending_ask_order: VecDeque::new(),
            pending_voice_tool_outputs: Vec::new(),
        };
        app.start();
        app
    }

    fn take_active_session(&mut self) -> TabSession {
        TabSession {
            settings: mem::take(&mut self.settings),
            state: mem::take(&mut self.state),
            status: mem::take(&mut self.status),
            error: self.error.take(),
            composer: mem::take(&mut self.composer),
            pending: mem::take(&mut self.pending),
            messages: mem::take(&mut self.messages),
            active_assistant_message: self.active_assistant_message.take(),
            assistant_group_deadline: self.assistant_group_deadline.take(),
            assistant_text_needs_separator: mem::take(&mut self.assistant_text_needs_separator),
            active_response_id: self.active_response_id.take(),
            last_assistant_item_id: self.last_assistant_item_id.take(),
            speech_screenshot_gate: mem::take(&mut self.speech_screenshot_gate),
            speech_turn_id: mem::replace(&mut self.speech_turn_id, 0),
            screenshot_capture_in_flight: self.screenshot_capture_in_flight.take(),
            screenshot_message_index: self.screenshot_message_index.take(),
            next_context_upload_id: mem::replace(&mut self.next_context_upload_id, 0),
            confirmed_context_uploads: mem::take(&mut self.confirmed_context_uploads),
            captured_transcript_items: mem::take(&mut self.captured_transcript_items),
            deferred_voice_response: mem::take(&mut self.deferred_voice_response),
            active_voice_message: self.active_voice_message.take(),
            latest_screen_image: self.latest_screen_image.take(),
            image_viewer: self.image_viewer.take(),
            should_scroll: mem::take(&mut self.should_scroll),
            tool_calls_running: mem::replace(&mut self.tool_calls_running, 0),
            pending_tool_reply: mem::take(&mut self.pending_tool_reply),
            pending_turns: mem::take(&mut self.pending_turns),
        }
    }

    fn install_active_session(&mut self, session: TabSession) {
        self.settings = session.settings;
        self.state = session.state;
        self.status = session.status;
        self.error = session.error;
        self.composer = session.composer;
        self.pending = session.pending;
        self.messages = session.messages;
        self.active_assistant_message = session.active_assistant_message;
        self.assistant_group_deadline = session.assistant_group_deadline;
        self.assistant_text_needs_separator = session.assistant_text_needs_separator;
        self.active_response_id = session.active_response_id;
        self.last_assistant_item_id = session.last_assistant_item_id;
        self.speech_screenshot_gate = session.speech_screenshot_gate;
        self.speech_turn_id = session.speech_turn_id;
        self.screenshot_capture_in_flight = session.screenshot_capture_in_flight;
        self.screenshot_message_index = session.screenshot_message_index;
        self.next_context_upload_id = session.next_context_upload_id;
        self.confirmed_context_uploads = session.confirmed_context_uploads;
        self.captured_transcript_items = session.captured_transcript_items;
        self.deferred_voice_response = session.deferred_voice_response;
        self.active_voice_message = session.active_voice_message;
        self.latest_screen_image = session.latest_screen_image;
        self.image_viewer = session.image_viewer;
        self.should_scroll = session.should_scroll;
        self.tool_calls_running = session.tool_calls_running;
        self.pending_tool_reply = session.pending_tool_reply;
        self.pending_turns = session.pending_turns;
    }

    fn reset_session_channels(&mut self) {
        let (screenshot_result_tx, screenshot_result_rx) =
            mpsc::channel::<SpeechScreenshotResult>();
        self.screenshot_result_tx = screenshot_result_tx;
        self.screenshot_result_rx = screenshot_result_rx;
        // Tool results are shared by active and background sessions and carry
        // their tab index, so switching tabs never paints a result into the
        // wrong transcript.
    }

    fn switch_tab(&mut self, index: usize) {
        if index >= self.tabs.len() || index == self.active_tab {
            return;
        }
        if self.playing_message_audio.take().is_some()
            && let Some(speaker) = &mut self.speaker
        {
            let _ = speaker.clear();
        }
        let previous_index = self.active_tab;
        // A tab switch is a view operation. Keep the current transport alive
        // and put its event receiver in the per-tab background map; the
        // background pump continues draining it while another tab is shown.
        let previous_realtime = mem::replace(&mut self.realtime, RealtimeClient::spawn());
        self.background_clients
            .insert(previous_index, previous_realtime);
        let previous = self.take_active_session();
        self.tabs[previous_index].session = previous;
        self.active_tab = index;
        let next = mem::replace(
            &mut self.tabs[index].session,
            TabSession::new(Settings::default()),
        );
        let next_realtime = self.background_clients.remove(&index);
        self.install_active_session(next);
        // A text tab used by ask_text_model may already have a live background
        // transport. Adopt that transport when the user opens the tab instead
        // of starting a second Codex thread and losing its in-flight events.
        self.realtime = next_realtime.unwrap_or_else(RealtimeClient::spawn);
        self.reset_session_channels();
        self.show_settings = false;
    }

    fn add_model_tab(&mut self, backend: RealtimeBackend, model: String, voice: &str) {
        let mut settings = self.settings.clone();
        settings.backend = backend;
        settings.model = model;
        settings.voice = voice.to_owned();
        if backend == RealtimeBackend::CodexText {
            settings.thinking_level = settings.default_thinking_level;
        }
        self.tabs.push(AssistantTab::new(settings));
        let index = self.tabs.len() - 1;
        self.switch_tab(index);
        self.show_model_picker = false;
    }

    fn tab_title(&self, index: usize) -> String {
        let settings = if index == self.active_tab {
            &self.settings
        } else {
            &self.tabs[index].session.settings
        };
        match settings.backend {
            RealtimeBackend::OpenAiRealtime => "Realtime".to_owned(),
            RealtimeBackend::CodexGptLive => "GPT-Live".to_owned(),
            RealtimeBackend::CodexText => self
                .text_model_choices()
                .into_iter()
                .find(|(model, _)| model == &settings.model)
                .map(|(_, label)| label)
                .unwrap_or_else(|| settings.model.clone()),
        }
    }

    fn text_model_choices(&self) -> Vec<(String, String)> {
        let mut choices = Vec::new();
        let mut seen = HashSet::new();
        if let Some(info) = &self.codex_info {
            for model in &info.models {
                if !model.hidden && seen.insert(model.name.clone()) {
                    choices.push((model.name.clone(), model.display_name.clone()));
                }
            }
        }
        if !self.settings.default_text_model.trim().is_empty()
            && seen.insert(self.settings.default_text_model.clone())
        {
            let model = self.settings.default_text_model.clone();
            let display_name = FALLBACK_TEXT_MODELS
                .iter()
                .find(|(name, _)| *name == model)
                .map(|(_, label)| (*label).to_owned())
                .unwrap_or_else(|| model.clone());
            choices.push((model, display_name));
        }
        for (name, display_name) in FALLBACK_TEXT_MODELS {
            if seen.insert((*name).to_owned()) {
                choices.push(((*name).to_owned(), (*display_name).to_owned()));
            }
        }
        choices
    }

    fn image_model_choices(&self) -> Vec<(String, String)> {
        let mut choices = Vec::new();
        let mut seen = HashSet::new();
        if let Some(info) = &self.codex_info {
            for model in &info.models {
                let is_image_model = model.name.to_ascii_lowercase().contains("image");
                if is_image_model && !model.hidden && seen.insert(model.name.clone()) {
                    choices.push((model.name.clone(), model.display_name.clone()));
                }
            }
        }
        if !self.settings.default_image_model.trim().is_empty()
            && seen.insert(self.settings.default_image_model.clone())
        {
            let model = self.settings.default_image_model.clone();
            let display_name = image_generation::FALLBACK_MODELS
                .iter()
                .find(|(name, _)| *name == model)
                .map(|(_, label)| (*label).to_owned())
                .unwrap_or_else(|| model.clone());
            choices.push((model, display_name));
        }
        for (name, display_name) in image_generation::FALLBACK_MODELS {
            if seen.insert((*name).to_owned()) {
                choices.push(((*name).to_owned(), (*display_name).to_owned()));
            }
        }
        choices
    }

    fn ensure_text_model_selection(&mut self) {
        if self.settings.backend != RealtimeBackend::CodexText {
            return;
        }
        let choices = self.text_model_choices();
        if !choices
            .iter()
            .any(|(model, _)| model == &self.settings.model)
            && let Some((model, _)) = choices
                .iter()
                .find(|(model, _)| model == &self.settings.default_text_model)
                .or_else(|| choices.first())
        {
            self.settings.model = model.clone();
        }
    }

    fn open_model_picker(&mut self) {
        self.show_model_picker = true;
        if self.codex_info.is_none() && !self.codex_info_loading {
            self.refresh_codex_info();
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
        if !matches!(
            self.settings.backend,
            RealtimeBackend::CodexGptLive | RealtimeBackend::CodexText
        ) || self.codex_usage_loading
        {
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
        self.resolve_credentials_for(self.settings.auth_mode, self.settings.backend)
    }

    fn resolve_credentials_for(
        &self,
        auth_mode: AuthMode,
        backend: RealtimeBackend,
    ) -> anyhow::Result<(String, Option<String>)> {
        match auth_mode {
            AuthMode::ApiKey => {
                let key = self.api_key.trim();
                if key.is_empty() {
                    if backend == RealtimeBackend::CodexText {
                        let creds = auth::codex_credentials().context(
                            "No API key is set and no Codex login was found for this text model",
                        )?;
                        return Ok((creds.bearer_token, creds.chatgpt_account_id));
                    }
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

    fn fail_start(&mut self, message: String, show_settings: bool) {
        let _ = self.realtime.commands.send(Command::Disconnect);
        self.microphone = None;
        self.message_recorder = None;
        self.playing_message_audio = None;
        if let Some(speaker) = &mut self.speaker {
            let _ = speaker.clear();
        }
        self.state = ConnectionState::Offline;
        self.status = "Ready".to_owned();
        self.show_settings |= show_settings;
        self.error = Some(message);
    }

    fn start_text(&mut self) {
        let screen_info = match media::primary_screen_info() {
            Ok(screen) => {
                self.settings.screenshot_width = screen.logical_width;
                self.settings.screenshot_height = screen.logical_height;
                screen
            }
            Err(error) => {
                self.fail_start(
                    format!("Could not read the primary display resolution: {error:#}"),
                    false,
                );
                return;
            }
        };
        let (api_key, chatgpt_account_id) = match self.resolve_credentials() {
            Ok(credentials) => credentials,
            Err(error) => {
                self.fail_start(error.to_string(), true);
                return;
            }
        };
        let system_prompt = configured_system_prompt(&self.settings, screen_info);
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexText,
            api_key,
            chatgpt_account_id,
            model: self.settings.model.clone(),
            voice: self.settings.voice.clone(),
            thinking_level: self.settings.thinking_level.wire_value().to_owned(),
            system_prompt: system_prompt.clone(),
            screen_info,
        };
        self.state = ConnectionState::Connecting;
        self.status = "Connecting…".to_owned();
        if self
            .realtime
            .commands
            .send(Command::Connect(options))
            .is_err()
        {
            self.fail_start(
                "Could not start text connection: app-server transport stopped".to_owned(),
                false,
            );
            return;
        }
        if !self
            .messages
            .iter()
            .any(|message| matches!(message.role, Role::System(RealtimeBackend::CodexText)))
        {
            self.messages.push(ChatMessage::system(
                system_prompt,
                RealtimeBackend::CodexText,
            ));
        }
    }

    fn start(&mut self) {
        self.error = None;
        if self.playing_message_audio.take().is_some()
            && let Some(speaker) = &mut self.speaker
        {
            let _ = speaker.clear();
        }
        if self.settings.backend == RealtimeBackend::CodexText {
            self.start_text();
            return;
        }
        if self.settings.backend == RealtimeBackend::CodexGptLive
            && crate::gpt_live_webrtc::uses_platform_audio()
        {
            // Match Codex's native WebRTC architecture: one platform ADM owns
            // microphone capture, AEC, jitter buffering, and speaker playout.
            // Opening parallel CPAL/VoiceProcessingIO streams here can compete
            // with libWebRTC and cause stalls or broken acoustic timing.
            self.microphone = None;
            if let Some(speaker) = &mut self.speaker {
                let _ = speaker.clear();
            }
            self.speaker = None;
            match MessageRecorder::start() {
                Ok(recorder) => self.message_recorder = Some(recorder),
                Err(error) => {
                    self.fail_start(
                        format!("Could not open message audio recorder: {error:#}"),
                        false,
                    );
                    return;
                }
            }
            self.state = ConnectionState::Connecting;
            self.status = "Connecting native WebRTC audio…".to_owned();
        } else {
            self.message_recorder = None;
            if self.speaker.is_none() {
                match Speaker::new() {
                    Ok(speaker) => self.speaker = Some(speaker),
                    Err(error) => {
                        self.fail_start(format!("Could not open audio output: {error:#}"), false);
                        return;
                    }
                }
            }

            // Open capture as the first connection action. Audio chunks produced while
            // credentials, screen metadata, and the transport are being prepared stay
            // ordered in the realtime supervisor's bounded pre-connect buffer.
            match Microphone::start(self.realtime.commands.clone()) {
                Ok(microphone) => {
                    self.microphone = Some(microphone);
                    self.state = ConnectionState::Connecting;
                    self.status = "Recording while connecting…".to_owned();
                }
                Err(error) => {
                    self.fail_start(format!("{error:#}"), false);
                    return;
                }
            }
        }

        let screen_info = match media::primary_screen_info() {
            Ok(screen) => {
                self.settings.screenshot_width = screen.logical_width;
                self.settings.screenshot_height = screen.logical_height;
                screen
            }
            Err(error) => {
                self.fail_start(
                    format!("Could not read the primary display resolution: {error:#}"),
                    false,
                );
                return;
            }
        };
        match self.resolve_credentials() {
            Ok((api_key, chatgpt_account_id)) => {
                let system_prompt = connection_system_prompt(&self.settings, screen_info);
                let options = ConnectOptions {
                    backend: self.settings.backend,
                    api_key,
                    chatgpt_account_id,
                    model: self.settings.model.clone(),
                    voice: self.settings.voice.clone(),
                    thinking_level: self.settings.thinking_level.wire_value().to_owned(),
                    system_prompt: system_prompt.clone(),
                    screen_info,
                };
                eprintln!(
                    "[live-assistant prompt] backend={:?} bytes={} shared=true pointer_reply=Done exact=true",
                    self.settings.backend,
                    system_prompt.len()
                );
                if self
                    .realtime
                    .commands
                    .send(Command::Connect(options))
                    .is_err()
                {
                    self.fail_start(
                        "Could not start Realtime connection: audio service stopped".to_owned(),
                        false,
                    );
                    return;
                }
                self.messages
                    .push(ChatMessage::system(system_prompt, self.settings.backend));
            }
            Err(error) => {
                self.fail_start(error.to_string(), true);
            }
        }
    }

    fn stop(&mut self) {
        let _ = self.realtime.commands.send(Command::Disconnect);
        self.microphone = None;
        self.message_recorder = None;
        self.playing_message_audio = None;
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
        self.captured_transcript_items.clear();
        self.confirmed_context_uploads.clear();
        self.fail_pending_context_uploads();
        self.cancel_speech_screenshot();
        self.latest_screen_image = None;
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

        while let Ok((tab_index, call_id, output)) = self.tool_result_rx.try_recv() {
            self.apply_note_tool_result(&output);
            if tab_index == self.active_tab {
                apply_tool_result_to_messages(&mut self.messages, &call_id, &output);
            } else if let Some(tab) = self.tabs.get_mut(tab_index) {
                apply_tool_result_to_messages(&mut tab.session.messages, &call_id, &output);
            }
        }

        while let Ok(event) = self.realtime.events.try_recv() {
            match event {
                Event::Connecting => {
                    self.state = ConnectionState::Connecting;
                    self.status = if self.microphone.is_some() {
                        "Recording while connecting…".to_owned()
                    } else if self.settings.backend == RealtimeBackend::CodexGptLive
                        && crate::gpt_live_webrtc::uses_platform_audio()
                    {
                        "Connecting native WebRTC audio…".to_owned()
                    } else {
                        "Connecting…".to_owned()
                    };
                }
                Event::Reconnecting { attempt, reason } => {
                    self.state = ConnectionState::Connecting;
                    self.status = if self.settings.backend == RealtimeBackend::CodexText {
                        format!("Reconnecting… attempt {attempt}")
                    } else if self.microphone.is_some() {
                        format!("Recording during reconnect… attempt {attempt}")
                    } else {
                        format!("Reconnecting GPT-Live… attempt {attempt}")
                    };
                    if self.fail_pending_context_uploads() > 0 {
                        self.error = Some(format!(
                            "Screenshot upload interrupted by reconnect: {reason}"
                        ));
                    } else {
                        self.error = None;
                    }
                    eprintln!(
                        "[live-assistant reconnect-ui] attempt={} reason={}",
                        attempt, reason
                    );
                }
                Event::Connected if self.settings.backend == RealtimeBackend::CodexText => {
                    self.state = ConnectionState::Live;
                    self.status = "Ready".to_owned();
                    self.error = None;
                    self.flush_pending_turn();
                }
                Event::Connected
                    if self.settings.backend == RealtimeBackend::CodexGptLive
                        && crate::gpt_live_webrtc::uses_platform_audio() =>
                {
                    self.state = ConnectionState::Live;
                    self.status = "Listening · Native WebRTC".to_owned();
                    self.error = None;
                    self.flush_pending_turn();
                }
                Event::Connected if self.microphone.is_some() && self.speaker.is_some() => {
                    self.state = ConnectionState::Live;
                    self.status = "Listening".to_owned();
                    self.error = None;
                    self.flush_pending_turn();
                }
                Event::Connected => match Microphone::start(self.realtime.commands.clone()) {
                    Ok(mic) if self.speaker.is_some() => {
                        self.microphone = Some(mic);
                        self.state = ConnectionState::Live;
                        self.status = "AEC listening".to_owned();
                        self.error = None;
                        self.flush_pending_turn();
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
                    self.fail_pending_context_uploads();
                    self.microphone = None;
                    self.message_recorder = None;
                    self.playing_message_audio = None;
                    if let Some(speaker) = &mut self.speaker {
                        let _ = speaker.clear();
                    }
                    self.state = ConnectionState::Offline;
                    self.active_response_id = None;
                    self.last_assistant_item_id = None;
                    self.active_assistant_message = None;
                    self.speech_screenshot_gate.end();
                    self.captured_transcript_items.clear();
                    self.confirmed_context_uploads.clear();
                    self.cancel_speech_screenshot();
                    self.active_voice_message = None;
                    self.latest_screen_image = None;
                    self.tool_calls_running = 0;
                    self.pending_tool_reply = false;
                    self.status = "Offline".to_owned();
                }
                Event::SpeechStarted => {
                    if self.settings.backend == RealtimeBackend::CodexGptLive {
                        // Native libWebRTC owns barge-in and full-duplex transport.
                        // The parallel recorder is capture-only and feeds message replay.
                        self.status = if self.active_response_id.is_some() {
                            "Speaking + hearing you…".to_owned()
                        } else {
                            "Hearing you…".to_owned()
                        };
                        if let Some(recorder) = &self.message_recorder {
                            recorder.begin_turn();
                        }
                        self.ensure_active_voice_message();
                        continue;
                    }
                    // OpenAI Realtime uses interruption/truncation for barge-in.
                    self.status = "Hearing you…".to_owned();
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
                }
                Event::SpeechStopped => {
                    // Server VAD has ended the utterance. Native GPT-Live has no
                    // parallel local PCM capture; finalize the transcript-only turn.
                    self.speech_screenshot_gate.end();
                    if self.settings.backend == RealtimeBackend::CodexGptLive
                        && crate::gpt_live_webrtc::uses_platform_audio()
                    {
                        self.status = "Thinking…".to_owned();
                        let audio = self
                            .message_recorder
                            .as_ref()
                            .map(MessageRecorder::finish_turn)
                            .unwrap_or_default();
                        self.finish_voice_message(audio);
                        continue;
                    }
                    // OpenAI Realtime only replies on locally confirmed speech.
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
                                self.status = "Listening".to_owned();
                            }
                            continue;
                        }
                    }
                    self.status = "Thinking…".to_owned();
                    self.finish_voice_message(audio);
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
                    if self.speech_screenshot_gate.sent && !item_id.trim().is_empty() {
                        self.captured_transcript_items.insert(item_id.clone());
                    }
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
                    let transcript_has_text = !text.trim().is_empty();
                    let transcript_key = if item_id.trim().is_empty() {
                        format!("local-transcript-turn-{}", self.speech_turn_id)
                    } else {
                        item_id.clone()
                    };
                    update_voice_transcript(&mut self.messages[index], item_id, text, now);
                    if transcript_has_text
                        && !self.captured_transcript_items.contains(&transcript_key)
                    {
                        let already_captured = self.speech_screenshot_gate.sent;
                        let capture_started = if already_captured {
                            false
                        } else {
                            self.start_speech_screenshot(ctx, index, "first-transcript")
                        };
                        if already_captured || capture_started {
                            self.captured_transcript_items.insert(transcript_key);
                        }
                    }
                    if continuing_existing_item {
                        eprintln!(
                            "[live-assistant user-group] merged transcript continuation into message={} window_seconds={}",
                            index,
                            MESSAGE_CONTINUATION_WINDOW.as_secs()
                        );
                    }
                    self.status = "Hearing you…".to_owned();
                }
                Event::ContextImageAccepted { upload_id } => {
                    // Accepted means the backend has queued the JPEG. Keep the
                    // overlay visible until its server acknowledgment confirms
                    // that the image reached conversation context.
                    self.status = "Hearing you… · Screen queued".to_owned();
                    ctx.request_repaint();
                    eprintln!("[live-assistant image] backend queued upload_id={upload_id}");
                }
                Event::ContextImageUploaded { upload_id } => {
                    let completed = self.mark_context_image_uploaded(upload_id);
                    if completed {
                        self.status = if self
                            .microphone
                            .as_ref()
                            .map(Microphone::in_speech)
                            .unwrap_or(false)
                        {
                            "Hearing you… · Screen uploaded".to_owned()
                        } else {
                            "Screen uploaded".to_owned()
                        };
                        ctx.request_repaint();
                    }
                    eprintln!(
                        "[live-assistant image] upload confirmed upload_id={upload_id} applied={completed}"
                    );
                }
                Event::ContextImageUploadFailed { upload_id, detail } => {
                    self.confirmed_context_uploads.remove(&upload_id);
                    let failed = self
                        .context_image_mut(upload_id)
                        .is_some_and(ChatImage::fail_upload);
                    if failed {
                        self.error = Some(format!("Screenshot upload failed: {detail}"));
                        self.status = "Screenshot upload failed".to_owned();
                        ctx.request_repaint();
                    }
                    eprintln!(
                        "[live-assistant image] upload failed upload_id={upload_id} applied={failed}: {detail}"
                    );
                }
                Event::AssistantResponseStarted { response_id } => {
                    if self.playing_message_audio.take().is_some()
                        && let Some(speaker) = &mut self.speaker
                    {
                        let _ = speaker.clear();
                    }
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
                    if app_plays_live_assistant_audio(self.settings.backend)
                        && let Some(speaker) = &mut self.speaker
                    {
                        speaker.begin_assistant_response(
                            self.settings.backend == RealtimeBackend::CodexGptLive,
                        );
                    }
                    self.active_response_id = Some(response_id.clone());
                    let assistant_index = ensure_assistant_message_index(
                        &mut self.messages,
                        &mut self.active_assistant_message,
                    );
                    self.messages[assistant_index].begin_response(response_id);
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
                    self.current_assistant().refresh_token_estimate();
                    self.assistant_text_needs_separator = false;
                    self.touch_assistant_group(Instant::now());
                    if self.settings.backend == RealtimeBackend::CodexGptLive
                        && crate::gpt_live_webrtc::uses_platform_audio()
                    {
                        self.status = "Speaking… · Native WebRTC".to_owned();
                    }
                }
                Event::AssistantAudio {
                    response_id,
                    samples,
                } => {
                    if !self.response_is_active(&response_id) {
                        continue;
                    }
                    if app_plays_live_assistant_audio(self.settings.backend)
                        && let Some(speaker) = &mut self.speaker
                    {
                        speaker.append_assistant(samples.clone());
                    }
                    self.current_assistant().audio.extend_from_slice(&samples);
                    self.touch_assistant_group(Instant::now());
                    self.status = "Speaking…".to_owned();
                }
                Event::AssistantSegmentDone { response_id } => {
                    if self.response_is_active(&response_id) {
                        // Keep the complete GPT-Live reply in one message WAV. The
                        // transcript boundary only extends the continuation window.
                        self.touch_assistant_group(Instant::now());
                    }
                }
                Event::AssistantDone { response_id } => {
                    if self.response_is_active(&response_id) {
                        let reply_is_complete = self.tool_calls_running == 0;
                        // The backend's final response event is the message end.
                        // Speaker playback may continue draining after this timestamp.
                        if reply_is_complete
                            && let Some(index) = self.active_assistant_message
                            && let Some(message) = self.messages.get_mut(index)
                        {
                            message.finish();
                        }
                        if app_plays_live_assistant_audio(self.settings.backend)
                            && let Some(speaker) = &mut self.speaker
                        {
                            speaker.finish_assistant_response();
                        }
                        self.active_response_id = None;
                        self.touch_assistant_group(Instant::now());
                        if self.tool_calls_running > 0 {
                            self.status = format!(
                                "Running {} tool{}…",
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
                            self.status = if self.settings.backend == RealtimeBackend::CodexText {
                                "Ready".to_owned()
                            } else {
                                "Listening".to_owned()
                            };
                            self.flush_pending_turn();
                        }
                    }
                }
                Event::AssistantUsage {
                    response_id,
                    total_tokens,
                } => {
                    apply_assistant_usage(&mut self.messages, &response_id, total_tokens);
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
                    self.status =
                        format!("Running {count} tool{}…", if count == 1 { "" } else { "s" });
                    let delegated_calls = calls
                        .iter()
                        .filter(|call| voice_tool_route(&call.name) == VoiceToolRoute::AskTextModel)
                        .cloned()
                        .collect::<Vec<_>>();
                    for call in delegated_calls {
                        self.start_ask_text_model(call, self.active_tab);
                    }
                    let image_calls = calls
                        .iter()
                        .filter(|call| voice_tool_route(&call.name) == VoiceToolRoute::CreateImage)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !image_calls.is_empty() {
                        self.start_image_generation(
                            self.active_tab,
                            self.realtime.commands.clone(),
                            image_calls,
                            self.settings.clone(),
                        );
                    }
                    let local_calls = calls
                        .into_iter()
                        .filter(|call| voice_tool_route(&call.name) == VoiceToolRoute::Local)
                        .map(|call| self.prepare_note_tool_call(call))
                        .collect::<Vec<_>>();
                    if local_calls.is_empty() {
                        continue;
                    }
                    let tab_index = self.active_tab;
                    let screenshot_width = self.settings.screenshot_width;
                    let screenshot_height = self.settings.screenshot_height;
                    for call in local_calls {
                        let commands = self.realtime.commands.clone();
                        let tool_result_tx = self.tool_result_tx.clone();
                        thread::spawn(move || {
                            let screen_context = tools::ScreenContext {
                                screenshot_width,
                                screenshot_height,
                            };
                            let call_id = call.call_id;
                            let name = call.name;
                            let arguments = call.arguments;
                            let queue_ms = call.requested_at.elapsed().as_millis();
                            let execute_started = Instant::now();
                            let output =
                                tools::execute_with_context(&name, &arguments, screen_context);
                            eprintln!(
                                "[live-assistant latency] call_id={} name={} stage=local.execute_complete queue_ms={} execute_ms={} total_ms={}",
                                call_id,
                                name,
                                queue_ms,
                                execute_started.elapsed().as_millis(),
                                call.requested_at.elapsed().as_millis(),
                            );
                            let tool_output = ToolOutput { call_id, output };
                            let _ = tool_result_tx.send((
                                tab_index,
                                tool_output.call_id.clone(),
                                tool_output.output.clone(),
                            ));
                            // Submit each result as soon as it is available instead
                            // of waiting for unrelated parallel calls to finish.
                            let _ = commands.send(Command::ToolOutputs(vec![tool_output]));
                        });
                    }
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

        // Voice turns can delegate to a text tab that continues while this
        // active transport waits for the result. Drain those sessions after
        // the active event loop so a newly-created background client is also
        // eligible on the same repaint.
        self.process_background_events(ctx);

        let now = Instant::now();
        self.finalize_expired_assistant_group(now);
        let expired_uploads = self.expire_context_image_uploads(now);
        if expired_uploads > 0 {
            self.error = Some(if expired_uploads == 1 {
                "Screenshot upload gave up after 10 seconds".to_owned()
            } else {
                format!("{expired_uploads} screenshot uploads gave up after 10 seconds")
            });
            self.status = "Screenshot upload timed out".to_owned();
            ctx.request_repaint();
        }

        if self.playing_message_audio.is_some() {
            let still_playing = self
                .speaker
                .as_ref()
                .map(Speaker::is_playing)
                .unwrap_or(false);
            if still_playing {
                ctx.request_repaint_after(Duration::from_millis(50));
            } else {
                self.playing_message_audio = None;
                ctx.request_repaint();
            }
        }

        if self.state == ConnectionState::Live
            && self.settings.backend != RealtimeBackend::CodexText
            && self.active_response_id.is_none()
            && !self
                .speaker
                .as_ref()
                .map(Speaker::assistant_is_playing)
                .unwrap_or(false)
            && self.status.contains("Speaking")
        {
            self.status = "Listening".to_owned();
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
        if let Some(index) = self.playing_message_audio {
            self.playing_message_audio = Some(remap_message_index(index, placement));
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
        if self.playing_message_audio == Some(index)
            && let Some(speaker) = &mut self.speaker
        {
            let _ = speaker.clear();
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
        remap(&mut self.playing_message_audio);
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

    fn allocate_context_upload_id(&mut self) -> u64 {
        self.next_context_upload_id = self.next_context_upload_id.wrapping_add(1).max(1);
        self.next_context_upload_id
    }

    fn context_image_mut(&mut self, upload_id: u64) -> Option<&mut ChatImage> {
        self.messages
            .iter_mut()
            .flat_map(|message| message.images.iter_mut())
            .find(|image| image.upload_id == Some(upload_id))
    }

    /// Apply an acknowledgment even if it arrives before the preview has been
    /// installed. The latter case is retained and consumed during insertion so
    /// an uploaded image can never be left behind with an "Uploading…" overlay.
    fn mark_context_image_uploaded(&mut self, upload_id: u64) -> bool {
        if let Some(image) = self.context_image_mut(upload_id) {
            return image.finish_upload();
        }
        self.confirmed_context_uploads.insert(upload_id);
        false
    }

    fn expire_context_image_uploads(&mut self, now: Instant) -> usize {
        let mut expired_ids = Vec::new();
        for image in self
            .messages
            .iter_mut()
            .flat_map(|message| message.images.iter_mut())
        {
            if image.upload_timed_out(now)
                && image.fail_upload()
                && let Some(upload_id) = image.upload_id
            {
                expired_ids.push(upload_id);
                eprintln!(
                    "[live-assistant image] upload timeout upload_id={upload_id} seconds={}",
                    CONTEXT_IMAGE_UPLOAD_TIMEOUT.as_secs()
                );
            }
        }
        for upload_id in &expired_ids {
            self.confirmed_context_uploads.remove(upload_id);
        }
        expired_ids.len()
    }

    fn fail_pending_context_uploads(&mut self) -> usize {
        let mut failed = 0usize;
        for image in self
            .messages
            .iter_mut()
            .flat_map(|message| message.images.iter_mut())
        {
            if image.upload_state == ImageUploadState::Uploading {
                image.fail_upload();
                failed = failed.saturating_add(1);
            }
        }
        failed
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
                Ok(image) => {
                    if self.state == ConnectionState::Offline {
                        continue;
                    }
                    // Keep the exact captured attachment available as the newest
                    // visual context for the realtime model. Direct click_screen
                    // calls use coordinates from this capture.
                    self.latest_screen_image = Some(image.clone());
                    let message_index = screenshot_message_index.unwrap_or_else(|| {
                        self.append_user_message(ChatMessage::user_voice(Vec::new(), None))
                    });
                    let upload_id = self.allocate_context_upload_id();
                    let upload_deadline = Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT;
                    let image_index = self.messages[message_index].images.len();

                    let prepared = match PreparedChatImage::from_attachment(&image) {
                        Ok(Some(prepared_image)) => {
                            let mut chat_image = prepared_image.into_chat_image();
                            chat_image.begin_upload(upload_id, upload_deadline);
                            if self.confirmed_context_uploads.remove(&upload_id) {
                                let _ = chat_image.finish_upload();
                            }
                            if let Err(error) = chat_image.ensure_thumbnail_texture(
                                ctx,
                                format!("chat-thumbnail-{message_index}-{image_index}"),
                            ) {
                                self.error = Some(format!(
                                    "Screenshot was captured but could not be displayed: {error:#}"
                                ));
                            }
                            Some(chat_image)
                        }
                        Ok(None) => {
                            self.error = Some(
                                "Screenshot capture did not produce an image attachment".to_owned(),
                            );
                            None
                        }
                        Err(error) => {
                            self.error = Some(format!(
                                "Screenshot was captured but could not be prepared: {error:#}"
                            ));
                            None
                        }
                    };

                    if let Some(chat_image) = prepared {
                        // Install the local preview first. Any backend acknowledgment now
                        // has a matching image, and the next repaint shows the dimmed image
                        // plus its uploading overlay immediately.
                        let message = &mut self.messages[message_index];
                        message.images.push(chat_image);
                        message.included_screen = true;
                        message.refresh_token_estimate();
                        eprintln!(
                            "[live-assistant image] preview ready upload_id={} turn={} message={} image={} total_images={}",
                            upload_id,
                            capture.turn_id,
                            message_index,
                            image_index,
                            message.images.len()
                        );
                        ctx.request_repaint();

                        let send_result = self.realtime.commands.send(Command::SendContextImage {
                            upload_id,
                            image,
                            deadline: upload_deadline,
                        });
                        if send_result.is_err() {
                            if let Some(chat_image) = self.context_image_mut(upload_id) {
                                chat_image.fail_upload();
                            }
                            self.error = Some(
                                "Screenshot captured but upload could not start: Realtime connection closed"
                                    .to_owned(),
                            );
                            self.status = "Screenshot upload failed".to_owned();
                        } else if self.state == ConnectionState::Connecting {
                            self.status = "Connecting · Uploading screen…".to_owned();
                        } else {
                            self.status = "Hearing you… · Uploading screen…".to_owned();
                        }
                        if capture.turn_id == self.speech_turn_id
                            && self
                                .microphone
                                .as_ref()
                                .map(Microphone::in_speech)
                                .unwrap_or(false)
                        {
                            self.active_voice_message = Some(message_index);
                        }
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
                    self.status = "Thinking… · Screen uploading".to_owned();
                }
            }
        }
    }

    fn start_speech_screenshot(
        &mut self,
        ctx: &egui::Context,
        message_index: usize,
        trigger: &'static str,
    ) -> bool {
        if self.state == ConnectionState::Offline
            || !self.settings.send_screenshot
            || self.screenshot_capture_in_flight.is_some()
            || self.speech_screenshot_gate.sent
        {
            return false;
        }
        if !self.speech_screenshot_gate.speech_active {
            self.speech_turn_id = self.speech_turn_id.wrapping_add(1);
            self.speech_screenshot_gate.begin();
        }

        self.speech_screenshot_gate.mark_sent();
        let turn_id = self.speech_turn_id;
        self.screenshot_capture_in_flight = Some(turn_id);
        self.screenshot_message_index = Some(message_index);
        self.status = if self.state == ConnectionState::Connecting {
            "Connecting · Capturing screen".to_owned()
        } else {
            "Hearing you… · Capturing screen".to_owned()
        };
        eprintln!(
            "[live-assistant image] capture triggered trigger={} turn={} message={}",
            trigger, turn_id, message_index
        );

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
            .map_err(|error| format!("{error:#}"));
            let _ = result_tx.send(SpeechScreenshotResult { turn_id, result });
            repaint.request_repaint();
        });
        true
    }

    fn maybe_send_speech_screenshot(&mut self, ctx: &egui::Context) {
        if self.state == ConnectionState::Offline
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

        // Transcript deltas are the primary trigger. This audio path is only a fallback
        // for speech that has not produced its first transcription token yet.
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

        let message_index = self.ensure_active_voice_message();
        self.start_speech_screenshot(ctx, message_index, "audio-fallback");
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
            message.finish();
            return;
        }
        let index = self.append_user_message(ChatMessage::user_voice(Vec::new(), None));
        append_voice_audio_fragment(&mut self.messages[index], &audio, now);
        self.messages[index].finish();
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

    fn send_turn_now(&mut self, turn: PendingTurn) -> bool {
        let result = self.realtime.commands.send(Command::SendTurn {
            text: turn.text,
            attachments: turn.attachments,
            thinking_level: turn.thinking_level,
        });
        if let Err(error) = result {
            self.error = Some(format!("Could not send message: {error}"));
            return false;
        }
        self.status = "Thinking…".to_owned();
        true
    }

    fn flush_pending_turn(&mut self) {
        if self.state != ConnectionState::Live {
            return;
        }
        if let Some(turn) = self.pending_turns.pop_front()
            && !self.send_turn_now(turn.clone())
        {
            // The command channel can close during a reconnect. Keep the turn
            // visible and retry on the next successful connection.
            self.pending_turns.push_front(turn);
            self.status = "Waiting to send…".to_owned();
        }
    }

    fn ensure_text_tab_for_request(&mut self, model: &str, thinking_level: ThinkingLevel) -> usize {
        if let Some(index) = (0..self.tabs.len()).find(|&index| {
            let settings = if index == self.active_tab {
                &self.settings
            } else {
                &self.tabs[index].session.settings
            };
            settings.backend == RealtimeBackend::CodexText && settings.model == model
        }) {
            if index == self.active_tab {
                self.settings.thinking_level = thinking_level;
            } else {
                self.tabs[index].session.settings.thinking_level = thinking_level;
            }
            return index;
        }

        let mut settings = self.settings.clone();
        settings.backend = RealtimeBackend::CodexText;
        settings.model = model.to_owned();
        settings.voice = "marin".to_owned();
        settings.thinking_level = thinking_level;
        self.tabs.push(AssistantTab::new(settings));
        self.tabs.len() - 1
    }

    fn screen_info_for_settings(settings: &Settings) -> ScreenInfo {
        ScreenInfo {
            origin_x: 0,
            origin_y: 0,
            logical_width: settings.screenshot_width,
            logical_height: settings.screenshot_height,
            backing_width: settings.screenshot_width,
            backing_height: settings.screenshot_height,
            scale_factor: 1.0,
        }
    }

    fn background_text_screen(&mut self, tab_index: usize) -> ScreenInfo {
        match media::primary_screen_info() {
            Ok(screen) => {
                if let Some(tab) = self.tabs.get_mut(tab_index) {
                    tab.session.settings.screenshot_width = screen.logical_width;
                    tab.session.settings.screenshot_height = screen.logical_height;
                }
                screen
            }
            Err(_) => self
                .tabs
                .get(tab_index)
                .map(|tab| Self::screen_info_for_settings(&tab.session.settings))
                .unwrap_or_else(|| Self::screen_info_for_settings(&self.settings)),
        }
    }

    fn start_background_text_request(
        &mut self,
        tab_index: usize,
        prompt: String,
        attachments: Vec<Attachment>,
        thinking_level: ThinkingLevel,
    ) {
        if tab_index >= self.tabs.len() || prompt.trim().is_empty() {
            return;
        }
        let screen_info = self.background_text_screen(tab_index);
        let settings_snapshot = self.tabs[tab_index].session.settings.clone();
        let system_prompt = configured_system_prompt(&settings_snapshot, screen_info);
        {
            let session = &mut self.tabs[tab_index].session;
            session.settings.thinking_level = thinking_level;
            let placement = append_user_message_before_active_assistant(
                &mut session.messages,
                &mut session.active_assistant_message,
                ChatMessage::user_text(prompt.clone(), &attachments),
            );
            session.active_voice_message = None;
            session.active_response_id = None;
            session.last_assistant_item_id = None;
            session.pending_tool_reply = false;
            session.pending_turns.push_back(PendingTurn {
                text: prompt,
                attachments,
                thinking_level: thinking_level.wire_value().to_owned(),
            });
            if placement.moved_assistant.is_some() {
                session.assistant_group_deadline = None;
            }
            if !session
                .messages
                .iter()
                .any(|message| matches!(message.role, Role::System(RealtimeBackend::CodexText)))
            {
                session.messages.push(ChatMessage::system(
                    system_prompt.clone(),
                    RealtimeBackend::CodexText,
                ));
            }
        }

        if tab_index == self.active_tab && self.settings.backend == RealtimeBackend::CodexText {
            if self.state == ConnectionState::Offline {
                self.start_text();
            } else if self.state == ConnectionState::Live {
                self.flush_pending_turn();
            }
            return;
        }

        if self.background_clients.contains_key(&tab_index) {
            if self.tabs[tab_index].session.state == ConnectionState::Live {
                self.flush_background_turn(tab_index);
            } else {
                self.tabs[tab_index].session.status =
                    "Connecting text model… will send when ready".to_owned();
            }
            return;
        }

        let (api_key, chatgpt_account_id) = match self
            .resolve_credentials_for(settings_snapshot.auth_mode, RealtimeBackend::CodexText)
        {
            Ok(credentials) => credentials,
            Err(error) => {
                let session = &mut self.tabs[tab_index].session;
                session.state = ConnectionState::Offline;
                session.status = "Text request failed".to_owned();
                session.error = Some(error.to_string());
                self.fail_pending_asks_for_tab(tab_index, error.to_string());
                return;
            }
        };
        let options = ConnectOptions {
            backend: RealtimeBackend::CodexText,
            api_key,
            chatgpt_account_id,
            model: settings_snapshot.model,
            voice: settings_snapshot.voice,
            thinking_level: thinking_level.wire_value().to_owned(),
            system_prompt,
            screen_info,
        };
        let client = RealtimeClient::spawn();
        if client.commands.send(Command::Connect(options)).is_err() {
            let session = &mut self.tabs[tab_index].session;
            session.state = ConnectionState::Offline;
            session.status = "Text request failed".to_owned();
            session.error = Some("Could not start the background text session".to_owned());
            self.fail_pending_asks_for_tab(
                tab_index,
                "Could not start the background text session",
            );
            return;
        }
        self.background_clients.insert(tab_index, client);
        let session = &mut self.tabs[tab_index].session;
        session.state = ConnectionState::Connecting;
        session.status = "Connecting text model… will send when ready".to_owned();
        session.error = None;
    }

    fn flush_background_turn(&mut self, tab_index: usize) {
        if self
            .tabs
            .get(tab_index)
            .is_none_or(|tab| tab.session.state != ConnectionState::Live)
        {
            return;
        }
        let Some(commands) = self
            .background_clients
            .get(&tab_index)
            .map(|client| client.commands.clone())
        else {
            return;
        };
        let Some(turn) = self.tabs[tab_index].session.pending_turns.pop_front() else {
            return;
        };
        let command = Command::SendTurn {
            text: turn.text.clone(),
            attachments: turn.attachments.clone(),
            thinking_level: turn.thinking_level.clone(),
        };
        if commands.send(command).is_err() {
            self.tabs[tab_index].session.pending_turns.push_front(turn);
            self.tabs[tab_index].session.state = ConnectionState::Offline;
            self.tabs[tab_index].session.status = "Text session stopped".to_owned();
            self.fail_pending_asks_for_tab(tab_index, "The background text session stopped");
        } else {
            self.tabs[tab_index].session.status = "Thinking…".to_owned();
        }
    }

    fn capture_screen_for_delegation(&mut self) -> anyhow::Result<Attachment> {
        let screen = media::primary_screen_info()?;
        self.settings.screenshot_width = screen.logical_width;
        self.settings.screenshot_height = screen.logical_height;
        let image = media::capture_screenshot(
            screen.logical_width,
            screen.logical_height,
            self.settings.show_live_pointer,
            self.pointer_overlay.snapshot(),
        )?;
        self.latest_screen_image = Some(image.clone());
        Ok(image)
    }

    fn start_ask_text_model(&mut self, call: crate::realtime::ToolCall, origin_tab: usize) {
        let arguments = match serde_json::from_str::<serde_json::Value>(&call.arguments) {
            Ok(arguments) => arguments,
            Err(error) => {
                self.queue_voice_tool_output(
                    origin_tab,
                    ToolOutput {
                        call_id: call.call_id,
                        output: serde_json::json!({
                            "ok": false,
                            "error": format!("ask_text_model arguments were invalid: {error}"),
                        })
                        .to_string(),
                    },
                );
                return;
            }
        };
        let prompt = arguments
            .get("prompt")
            .or_else(|| arguments.get("question"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .map(str::to_owned);
        let Some(prompt) = prompt else {
            self.queue_voice_tool_output(
                origin_tab,
                ToolOutput {
                    call_id: call.call_id,
                    output: serde_json::json!({
                        "ok": false,
                        "error": "ask_text_model requires a non-empty prompt",
                    })
                    .to_string(),
                },
            );
            return;
        };
        let screen_click = prompt_requires_screen_click(&prompt);
        let include_screenshot = arguments
            .get("include_screenshot")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            || screen_click;
        let mut attachments = Vec::new();
        if let Some(data_url) = arguments
            .get("image")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|data_url| !data_url.is_empty())
        {
            match media::image_from_data_url("ask_text_model image", data_url) {
                Ok(image) => attachments.push(image),
                Err(error) => {
                    self.queue_voice_tool_output(
                        origin_tab,
                        ToolOutput {
                            call_id: call.call_id,
                            output: serde_json::json!({
                                "ok": false,
                                "error": format!("ask_text_model image was invalid: {error:#}"),
                            })
                            .to_string(),
                        },
                    );
                    return;
                }
            }
        }
        if include_screenshot {
            let screenshot = if screen_click {
                // A click must always be grounded in a fresh capture, even if
                // automatic voice screenshots are disabled or an older image
                // is still visible in the voice transcript.
                self.capture_screen_for_delegation()
            } else if let Some(image) = &self.latest_screen_image {
                Ok(image.clone())
            } else {
                self.capture_screen_for_delegation()
            };
            match screenshot {
                Ok(image) => attachments.push(image),
                Err(error) => {
                    self.queue_voice_tool_output(
                        origin_tab,
                        ToolOutput {
                            call_id: call.call_id,
                            output: serde_json::json!({
                                "ok": false,
                                "error": format!("Could not capture the screen for ask_text_model: {error:#}"),
                            })
                            .to_string(),
                        },
                    );
                    return;
                }
            }
        }
        let prompt = if screen_click {
            format!(
                "Inspect the attached newest screen capture and complete this click task with click_screen. Return the real tool result: {prompt}"
            )
        } else {
            prompt
        };
        let model = arguments
            .get("model")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| self.settings.default_text_model.clone());
        let thinking_level = arguments
            .get("thinking_level")
            .and_then(serde_json::Value::as_str)
            .and_then(parse_thinking_level)
            .unwrap_or(self.settings.default_thinking_level);
        let tab_index = self.ensure_text_tab_for_request(&model, thinking_level);
        let call_id = call.call_id;
        self.pending_ask_calls
            .insert(call_id.clone(), (tab_index, origin_tab));
        self.pending_ask_order.push_back(call_id);
        self.start_background_text_request(tab_index, prompt, attachments, thinking_level);
    }

    fn queue_voice_tool_output(&mut self, origin_tab: usize, output: ToolOutput) {
        self.pending_voice_tool_outputs.push((origin_tab, output));
    }

    fn fail_pending_asks_for_tab(&mut self, tab_index: usize, detail: impl Into<String>) {
        let detail = detail.into();
        let call_ids = self
            .pending_ask_calls
            .iter()
            .filter_map(|(call_id, (target, _))| (*target == tab_index).then_some(call_id.clone()))
            .collect::<Vec<_>>();
        for call_id in call_ids {
            let Some((_, origin_tab)) = self.pending_ask_calls.remove(&call_id) else {
                continue;
            };
            self.pending_ask_order.retain(|queued| queued != &call_id);
            self.queue_voice_tool_output(
                origin_tab,
                ToolOutput {
                    call_id,
                    output: serde_json::json!({
                        "ok": false,
                        "error": detail,
                    })
                    .to_string(),
                },
            );
        }
    }

    fn complete_pending_asks_for_tab(&mut self, tab_index: usize) {
        let Some(tab) = self.tabs.get(tab_index) else {
            return;
        };
        let response = tab
            .session
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::Assistant)
            .map(|message| message.text.trim().to_owned())
            .unwrap_or_default();
        let model = tab.session.settings.model.clone();
        let thinking_level = tab.session.settings.thinking_level.wire_value();
        let Some(order_index) = self.pending_ask_order.iter().position(|call_id| {
            self.pending_ask_calls
                .get(call_id)
                .is_some_and(|(target, _)| *target == tab_index)
        }) else {
            return;
        };
        let Some(call_id) = self.pending_ask_order.remove(order_index) else {
            return;
        };
        let Some((_, origin_tab)) = self.pending_ask_calls.remove(&call_id) else {
            return;
        };
        self.queue_voice_tool_output(
            origin_tab,
            ToolOutput {
                call_id,
                output: serde_json::json!({
                    "ok": true,
                    "model": model,
                    "thinking_level": thinking_level,
                    "response": response,
                })
                .to_string(),
            },
        );
    }

    fn process_background_events(&mut self, ctx: &egui::Context) {
        let indices = self.background_clients.keys().copied().collect::<Vec<_>>();
        for tab_index in indices {
            let mut events = Vec::new();
            if let Some(client) = self.background_clients.get(&tab_index) {
                while let Ok(event) = client.events.try_recv() {
                    events.push(event);
                }
            }
            for event in events {
                self.handle_background_text_event(tab_index, event, ctx);
            }
        }
        self.flush_pending_voice_tool_outputs();
    }

    fn handle_background_text_event(
        &mut self,
        tab_index: usize,
        event: Event,
        ctx: &egui::Context,
    ) {
        if tab_index >= self.tabs.len() {
            return;
        }
        match event {
            Event::Connecting => {
                let session = &mut self.tabs[tab_index].session;
                session.state = ConnectionState::Connecting;
                session.status = "Connecting text model…".to_owned();
            }
            Event::Reconnecting { attempt, reason } => {
                let session = &mut self.tabs[tab_index].session;
                session.state = ConnectionState::Connecting;
                session.status = format!("Reconnecting text model… attempt {attempt}");
                session.error = Some(reason);
            }
            Event::Connected => {
                let session = &mut self.tabs[tab_index].session;
                session.state = ConnectionState::Live;
                session.status = "Ready".to_owned();
                session.error = None;
                self.flush_background_turn(tab_index);
                ctx.request_repaint();
            }
            Event::Disconnected => {
                self.background_clients.remove(&tab_index);
                let session = &mut self.tabs[tab_index].session;
                session.state = ConnectionState::Offline;
                session.status = "Offline".to_owned();
                self.fail_pending_asks_for_tab(
                    tab_index,
                    "The background text session disconnected",
                );
                ctx.request_repaint();
            }
            Event::AssistantResponseStarted { response_id } => {
                let session = &mut self.tabs[tab_index].session;
                session.active_response_id = Some(response_id.clone());
                session.last_assistant_item_id = None;
                session.active_assistant_message = None;
                session.assistant_group_deadline = None;
                session.assistant_text_needs_separator = false;
                session.pending_tool_reply = false;
                let index = ensure_assistant_message_index(
                    &mut session.messages,
                    &mut session.active_assistant_message,
                );
                session.messages[index].begin_response(response_id);
                session.status = "Thinking…".to_owned();
            }
            Event::AssistantItem {
                response_id,
                item_id,
            } => {
                let session = &mut self.tabs[tab_index].session;
                if session.active_response_id.as_deref() == Some(&response_id) {
                    session.last_assistant_item_id = Some(item_id.clone());
                    let index = ensure_assistant_message_index(
                        &mut session.messages,
                        &mut session.active_assistant_message,
                    );
                    session.messages[index].server_item_id = Some(item_id);
                }
            }
            Event::AssistantTranscriptDelta { response_id, delta } => {
                let session = &mut self.tabs[tab_index].session;
                if session.active_response_id.as_deref() != Some(&response_id) {
                    return;
                }
                let index = ensure_assistant_message_index(
                    &mut session.messages,
                    &mut session.active_assistant_message,
                );
                session.messages[index].text.push_str(&delta);
                session.messages[index].refresh_token_estimate();
                session.status = "Thinking…".to_owned();
                ctx.request_repaint();
            }
            Event::AssistantDone { response_id } => {
                let complete = {
                    let session = &mut self.tabs[tab_index].session;
                    if session.active_response_id.as_deref() != Some(&response_id) {
                        false
                    } else {
                        let reply_is_complete = session.tool_calls_running == 0;
                        // Background/text completion also uses the final model event,
                        // independent of any later UI or audio playback work.
                        if reply_is_complete
                            && let Some(index) = session.active_assistant_message
                            && let Some(message) = session.messages.get_mut(index)
                        {
                            message.finish();
                        }
                        session.active_response_id = None;
                        session.last_assistant_item_id = None;
                        session.status = if session.tool_calls_running > 0 {
                            format!(
                                "Running {} tool{}…",
                                session.tool_calls_running,
                                if session.tool_calls_running == 1 {
                                    ""
                                } else {
                                    "s"
                                }
                            )
                        } else {
                            "Ready".to_owned()
                        };
                        session.tool_calls_running == 0
                    }
                };
                if complete {
                    self.complete_pending_asks_for_tab(tab_index);
                    self.flush_background_turn(tab_index);
                }
                ctx.request_repaint();
            }
            Event::AssistantUsage {
                response_id,
                total_tokens,
            } => {
                let session = &mut self.tabs[tab_index].session;
                if apply_assistant_usage(&mut session.messages, &response_id, total_tokens) {
                    ctx.request_repaint();
                }
            }
            Event::ToolCalls(calls) => {
                let count = calls.len();
                {
                    let session = &mut self.tabs[tab_index].session;
                    session.tool_calls_running = session.tool_calls_running.saturating_add(count);
                    session.pending_tool_reply = true;
                    let index = ensure_assistant_message_index(
                        &mut session.messages,
                        &mut session.active_assistant_message,
                    );
                    session.messages[index]
                        .tool_calls
                        .extend(calls.iter().map(|call| ToolInvocation {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: pretty_tool_arguments(&call.arguments),
                            output: None,
                        }));
                    session.status =
                        format!("Running {count} tool{}…", if count == 1 { "" } else { "s" });
                }
                self.start_background_local_tools(tab_index, calls);
                ctx.request_repaint();
            }
            Event::ToolOutputsSubmitted { count } => {
                let session = &mut self.tabs[tab_index].session;
                session.tool_calls_running = session.tool_calls_running.saturating_sub(count);
                session.status = if session.tool_calls_running == 0 {
                    "Thinking…".to_owned()
                } else {
                    format!("Running {} tools…", session.tool_calls_running)
                };
                if session.tool_calls_running == 0 && session.active_response_id.is_none() {
                    self.complete_pending_asks_for_tab(tab_index);
                    self.flush_background_turn(tab_index);
                }
                ctx.request_repaint();
            }
            Event::Error(message) => {
                let transport_exists = self.background_clients.contains_key(&tab_index);
                let session = &mut self.tabs[tab_index].session;
                session.error = Some(message.clone());
                session.status = "Text request failed".to_owned();
                // A failed turn does not necessarily close the app-server
                // thread. Keep a live background transport reusable; a real
                // disconnect removes it below and marks the tab offline.
                if !transport_exists {
                    session.state = ConnectionState::Offline;
                }
                self.fail_pending_asks_for_tab(tab_index, message);
                ctx.request_repaint();
            }
            Event::ContextImageAccepted { .. }
            | Event::ContextImageUploaded { .. }
            | Event::ContextImageUploadFailed { .. }
            | Event::AssistantAudio { .. }
            | Event::AssistantSegmentDone { .. }
            | Event::SpeechStarted
            | Event::SpeechStopped
            | Event::InputCommitted { .. }
            | Event::InputTranscript { .. } => {}
        }
    }

    fn start_background_local_tools(
        &mut self,
        tab_index: usize,
        calls: Vec<crate::realtime::ToolCall>,
    ) {
        let Some(commands) = self
            .background_clients
            .get(&tab_index)
            .map(|client| client.commands.clone())
        else {
            self.fail_pending_asks_for_tab(
                tab_index,
                "The background text transport is unavailable",
            );
            return;
        };
        let image_calls = calls
            .iter()
            .filter(|call| call.name == "create_image")
            .cloned()
            .collect::<Vec<_>>();
        if !image_calls.is_empty() {
            let settings = self
                .tabs
                .get(tab_index)
                .map(|tab| tab.session.settings.clone())
                .unwrap_or_else(|| self.settings.clone());
            self.start_image_generation(tab_index, commands.clone(), image_calls, settings);
        }
        let calls = calls
            .into_iter()
            .filter(|call| call.name != "create_image")
            .map(|call| self.prepare_note_tool_call(call))
            .collect::<Vec<_>>();
        if calls.is_empty() {
            return;
        }
        let (screenshot_width, screenshot_height) = self
            .tabs
            .get(tab_index)
            .map(|tab| {
                (
                    tab.session.settings.screenshot_width,
                    tab.session.settings.screenshot_height,
                )
            })
            .unwrap_or((1440, 900));
        for call in calls {
            let commands = commands.clone();
            let tool_result_tx = self.tool_result_tx.clone();
            thread::spawn(move || {
                let screen_context = tools::ScreenContext {
                    screenshot_width,
                    screenshot_height,
                };
                let call_id = call.call_id;
                let name = call.name;
                let arguments = call.arguments;
                let queue_ms = call.requested_at.elapsed().as_millis();
                let execute_started = Instant::now();
                let output = tools::execute_with_context(&name, &arguments, screen_context);
                eprintln!(
                    "[live-assistant latency] call_id={} name={} stage=background.execute_complete queue_ms={} execute_ms={} total_ms={}",
                    call_id,
                    name,
                    queue_ms,
                    execute_started.elapsed().as_millis(),
                    call.requested_at.elapsed().as_millis(),
                );
                let tool_output = ToolOutput { call_id, output };
                let _ = tool_result_tx.send((
                    tab_index,
                    tool_output.call_id.clone(),
                    tool_output.output.clone(),
                ));
                let _ = commands.send(Command::ToolOutputs(vec![tool_output]));
            });
        }
    }

    fn resolve_image_credentials(
        &self,
        auth_mode: AuthMode,
    ) -> anyhow::Result<auth::CodexCredentials> {
        if auth_mode == AuthMode::CodexApiKey || self.api_key.trim().is_empty() {
            return auth::codex_credentials();
        }
        Ok(auth::CodexCredentials {
            bearer_token: self.api_key.trim().to_owned(),
            chatgpt_account_id: None,
        })
    }

    fn start_image_generation(
        &mut self,
        tab_index: usize,
        commands: tokio::sync::mpsc::UnboundedSender<Command>,
        calls: Vec<crate::realtime::ToolCall>,
        settings: Settings,
    ) {
        if calls.is_empty() {
            return;
        }
        let default_model = settings.default_image_model.clone();
        let default_resolution = settings.default_image_resolution.wire_value().to_owned();
        let credentials = self
            .resolve_image_credentials(settings.auth_mode)
            .map_err(|error| format!("{error:#}"));
        let tool_result_tx = self.tool_result_tx.clone();
        thread::spawn(move || {
            for call in calls {
                let output = match &credentials {
                    Ok(credentials) => image_generation::request_from_tool_arguments(
                        &call.arguments,
                        &default_model,
                        &default_resolution,
                        &call.call_id,
                    )
                    .and_then(|request| {
                        image_generation::generate(request, credentials).map(|result| {
                            serde_json::json!({
                                "ok": true,
                                "model": result.model,
                                "resolution": result.resolution,
                                "image_url": result.data_url,
                            })
                            .to_string()
                        })
                    })
                    .unwrap_or_else(|error| {
                        serde_json::json!({
                            "ok": false,
                            "error": format!("{error:#}"),
                        })
                        .to_string()
                    }),
                    Err(error) => serde_json::json!({
                        "ok": false,
                        "error": error,
                    })
                    .to_string(),
                };
                let tool_output = ToolOutput {
                    call_id: call.call_id,
                    output,
                };
                let _ = tool_result_tx.send((
                    tab_index,
                    tool_output.call_id.clone(),
                    tool_output.output.clone(),
                ));
                let _ = commands.send(Command::ToolOutputs(vec![tool_output]));
            }
        });
    }

    fn flush_pending_voice_tool_outputs(&mut self) {
        if self.pending_voice_tool_outputs.is_empty()
            || self.state != ConnectionState::Live
            || self.settings.backend == RealtimeBackend::CodexText
        {
            return;
        }
        let active_tab = self.active_tab;
        let (ready, waiting): (Vec<_>, Vec<_>) = self
            .pending_voice_tool_outputs
            .drain(..)
            .partition(|(origin, _)| *origin == active_tab);
        self.pending_voice_tool_outputs = waiting;
        if ready.is_empty() {
            return;
        }
        let outputs = ready
            .into_iter()
            .map(|(_, output)| output)
            .collect::<Vec<_>>();
        let ui_outputs = outputs
            .iter()
            .map(|output| (active_tab, output.call_id.clone(), output.output.clone()))
            .collect::<Vec<_>>();
        if self
            .realtime
            .commands
            .send(Command::ToolOutputs(outputs))
            .is_err()
        {
            self.error =
                Some("Could not return ask_text_model result to the voice model".to_owned());
            return;
        }
        for result in ui_outputs {
            let _ = self.tool_result_tx.send(result);
        }
    }

    fn send_composer(&mut self) {
        if self.composer.trim().is_empty() && self.pending.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.composer);
        let attachments = std::mem::take(&mut self.pending);
        self.append_user_message(ChatMessage::user_text(text.clone(), &attachments));
        self.pending_turns.push_back(PendingTurn {
            text,
            attachments,
            thinking_level: self.settings.thinking_level.wire_value().to_owned(),
        });
        if self.state == ConnectionState::Offline {
            // Text sessions are deliberately lazy: the first Send both opens
            // the app-server session and waits for Connected before sending.
            // The same queue makes typed turns on voice tabs auto-connect too.
            self.start();
        } else if self.state == ConnectionState::Live {
            self.flush_pending_turn();
        } else {
            self.status = "Connecting… will send when ready".to_owned();
        }
    }

    fn draw_thinking_level_controls(&mut self, ui: &mut egui::Ui) {
        let wide = ui.available_width() >= 520.0;
        if wide {
            for level in ThinkingLevel::VISIBLE {
                let selected = self.settings.thinking_level == level;
                if ui
                    .add(egui::Button::new(level.label()).fill(if selected {
                        LIGHT_BLUE_SELECTED
                    } else {
                        Color32::WHITE
                    }))
                    .on_hover_text(format!(
                        "Use {} reasoning for the next request",
                        level.label()
                    ))
                    .clicked()
                {
                    self.settings.thinking_level = level;
                }
            }
            let more_selected = ThinkingLevel::MORE.contains(&self.settings.thinking_level);
            egui::Frame::new()
                .fill(if more_selected {
                    LIGHT_BLUE_SELECTED
                } else {
                    Color32::WHITE
                })
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(2, 0))
                .show(ui, |ui| {
                    egui::ComboBox::from_id_salt("thinking_more")
                        .selected_text(if more_selected {
                            self.settings.thinking_level.label()
                        } else {
                            "More…"
                        })
                        .show_ui(ui, |ui| {
                            style_light_blue_popup(ui);
                            for level in ThinkingLevel::MORE {
                                ui.selectable_value(
                                    &mut self.settings.thinking_level,
                                    level,
                                    level.label(),
                                );
                            }
                        });
                });
        } else {
            egui::ComboBox::from_id_salt("thinking_level_narrow")
                .selected_text(format!("Think: {}", self.settings.thinking_level.label()))
                .show_ui(ui, |ui| {
                    style_light_blue_popup(ui);
                    for level in ThinkingLevel::ALL {
                        ui.selectable_value(
                            &mut self.settings.thinking_level,
                            level,
                            level.label(),
                        );
                    }
                });
        }
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
        let mut requested_switch = None;
        ui.horizontal(|ui| {
            for index in 0..self.tabs.len() {
                let title = self.tab_title(index);
                if ui
                    .selectable_label(index == self.active_tab, title)
                    .clicked()
                {
                    requested_switch = Some(index);
                }
            }
            if centered_plus_button(ui, "new-model-tab", 22.0)
                .on_hover_text("Open a new model tab")
                .clicked()
            {
                self.open_model_picker();
            }
        });
        if let Some(index) = requested_switch {
            self.switch_tab(index);
        }
        ui.separator();
        ui.horizontal(|ui| {
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
                if self.state == ConnectionState::Offline
                    && self.settings.backend == RealtimeBackend::CodexText
                {
                    ui.label(RichText::new("Text starts when you send").small().weak())
                        .on_hover_text(
                            "Text tabs connect automatically when the Send button is pressed.",
                        );
                } else {
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

    fn draw_model_picker(&mut self, ctx: &egui::Context) {
        if !self.show_model_picker {
            return;
        }
        let text_models = self.text_model_choices();
        let mut open = true;
        let mut selection: Option<(RealtimeBackend, String, String)> = None;
        egui::Window::new("New model tab")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(380.0)
            .frame(
                egui::Frame::new()
                    .fill(LIGHT_BLUE)
                    .stroke(Stroke::new(1.0, Color32::from_rgb(183, 211, 241)))
                    .corner_radius(10.0)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.label("Choose a model for a new conversation tab.");
                ui.add_space(8.0);
                ui.label(RichText::new("Voice").strong());
                if ui
                    .add(egui::Button::new("Realtime · gpt-realtime-2.1").fill(LIGHT_BLUE))
                    .clicked()
                {
                    selection = Some((
                        RealtimeBackend::OpenAiRealtime,
                        "gpt-realtime-2.1".to_owned(),
                        "marin".to_owned(),
                    ));
                }
                if ui
                    .add(egui::Button::new("Live · Codex GPT-Live").fill(LIGHT_BLUE))
                    .clicked()
                {
                    selection = Some((
                        RealtimeBackend::CodexGptLive,
                        "gpt-realtime-2.1".to_owned(),
                        "ember".to_owned(),
                    ));
                }
                ui.add_space(8.0);
                ui.label(RichText::new("Text").strong());
                for (model, display_name) in &text_models {
                    if ui
                        .add(egui::Button::new(display_name).fill(LIGHT_BLUE))
                        .on_hover_text(model)
                        .clicked()
                    {
                        selection = Some((
                            RealtimeBackend::CodexText,
                            model.clone(),
                            "marin".to_owned(),
                        ));
                    }
                }
                if self.codex_info_loading {
                    ui.add_space(6.0);
                    ui.label("Loading additional Codex models…");
                } else if let Some(error) = &self.codex_info_error {
                    ui.add_space(6.0);
                    ui.label(RichText::new(error).small().weak());
                }
                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "Text tabs use the Codex app-server and include the same computer tools as voice tabs.",
                    )
                    .small()
                    .weak(),
                );
            });
        self.show_model_picker = open;
        if let Some((backend, model, voice)) = selection {
            self.add_model_tab(backend, model, &voice);
        }
    }

    fn draw_empty(&self, ui: &mut egui::Ui) {
        let text_model = self.settings.backend == RealtimeBackend::CodexText;
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label(
                RichText::new(if text_model {
                    "Chat with this model"
                } else {
                    "Talk, type, or share what you see"
                })
                .size(28.0),
            );
            ui.add_space(10.0);
            ui.label(
                RichText::new(if text_model {
                    "Send a message, attach an image, or ask the model to use the computer tools."
                } else {
                    "Start voice, then speak naturally. After half a second of clear speech, \
                     the current screen is sent while you are still talking."
                })
                .weak(),
            );
            ui.add_space(24.0);
            ui.horizontal_wrapped(|ui| {
                if !text_model {
                    ui.label("🎙 Semantic turn detection");
                    ui.separator();
                } else {
                    ui.label("Text model");
                    ui.separator();
                    ui.label("Computer tools");
                    ui.separator();
                }
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

            let role = self.messages[index].role;
            let is_user = role == Role::User;
            let is_assistant = role == Role::Assistant;
            let role_label = match role {
                Role::System(RealtimeBackend::OpenAiRealtime) => "SYSTEM · OPENAI REALTIME",
                Role::System(RealtimeBackend::CodexGptLive) => "SYSTEM · GPT-LIVE",
                Role::System(RealtimeBackend::CodexText) => "SYSTEM · CODEX TEXT",
                Role::User => "YOU",
                Role::Assistant => "ASSISTANT",
            };
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
                            ui.label(RichText::new(role_label).small().color(if is_user {
                                Color32::from_rgb(37, 92, 158)
                            } else {
                                Color32::from_rgb(92, 102, 116)
                            }));
                            if is_user || is_assistant {
                                let metadata = format_message_metadata(message, is_assistant);
                                ui.label(RichText::new(metadata).small().weak())
                                    .on_hover_text(
                                        "The first value is start-end time and elapsed cost. Backend-reported token totals are exact when available; a ~ prefix means a local estimate.",
                                    );
                            }
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
                                    let complete = image.upload_is_complete();
                                    let response = ui.add(
                                        egui::Image::from_texture(texture)
                                            .max_size(egui::vec2(width.min(210.0), 140.0))
                                            .corner_radius(8)
                                            .tint(image.preview_tint())
                                            .sense(if complete {
                                                egui::Sense::click()
                                            } else {
                                                egui::Sense::hover()
                                            }),
                                    );
                                    paint_image_metadata(
                                        ui,
                                        response.rect,
                                        image.width,
                                        image.height,
                                        image.byte_size,
                                    );
                                    if let Some(label) = image.upload_overlay_text() {
                                        paint_centered_image_status(ui, response.rect, label);
                                    }
                                    if complete && response.clicked() {
                                        self.image_viewer = Some((index, image_index));
                                    }
                                    if complete {
                                        response.on_hover_text(
                                            "Click to view the exact image sent to AI",
                                        );
                                    }
                                }
                                let caption = if message.included_screen {
                                    "▣ Current screen"
                                } else {
                                    &image.name
                                };
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(RichText::new(caption).small().weak());
                                    if ui.small_button("Save image").clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_file_name(image.name.clone())
                                            .add_filter("JPEG image", &["jpg", "jpeg"])
                                            .save_file()
                                        && let Err(error) =
                                            media::save_image(&path, &image.sent_image)
                                    {
                                        self.error = Some(format!("{error:#}"));
                                    }
                                });
                            }
                            if is_assistant {
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
                            } else if is_assistant {
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
                                    let play_tooltip = if is_user {
                                        "Play voice input"
                                    } else {
                                        "Play full reply"
                                    };
                                    let is_playing = self.playing_message_audio == Some(index)
                                        && self
                                            .speaker
                                            .as_ref()
                                            .map(Speaker::is_playing)
                                            .unwrap_or(false);
                                    let play_response = if is_playing {
                                        centered_pause_button(
                                            ui,
                                            ("pause-message-audio", index),
                                            24.0,
                                        )
                                        .on_hover_text("Stop playback")
                                    } else {
                                        centered_play_button(
                                            ui,
                                            ("play-message-audio", index),
                                            24.0,
                                        )
                                        .on_hover_text(play_tooltip)
                                    };
                                    if play_response.clicked() {
                                        if is_playing {
                                            if let Some(speaker) = &mut self.speaker
                                                && let Err(error) = speaker.clear()
                                            {
                                                self.error = Some(error.to_string());
                                            }
                                            self.playing_message_audio = None;
                                        } else {
                                            let audio = message.audio.clone();
                                            if self.speaker.is_none() {
                                                match Speaker::new() {
                                                    Ok(speaker) => self.speaker = Some(speaker),
                                                    Err(error) => {
                                                        self.error = Some(format!(
                                                            "Could not open audio output: {error:#}"
                                                        ));
                                                    }
                                                }
                                            }
                                            if let Some(speaker) = &mut self.speaker {
                                                match speaker.play_clip(&audio) {
                                                    Ok(()) => {
                                                        self.playing_message_audio = Some(index);
                                                    }
                                                    Err(error) => {
                                                        self.playing_message_audio = None;
                                                        self.error = Some(error.to_string());
                                                    }
                                                }
                                            }
                                        }
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
                                    if centered_save_button(
                                        ui,
                                        ("save-message-audio", index),
                                        24.0,
                                    )
                                    .on_hover_text(format!("Save WAV · {end_time}"))
                                    .clicked()
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

    fn refresh_note_files(&mut self) {
        match notes::list_notes() {
            Ok(files) => self.note_files = files,
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn active_note_name(&self) -> Option<String> {
        self.active_note.as_deref().map(notes::display_name)
    }

    fn queue_note_changed(&mut self, path: &Path, content: &str) {
        let text = note_change_message(path, content);
        self.append_user_message(ChatMessage::user_text(text.clone(), &[]));
        self.pending_turns.push_back(PendingTurn {
            text,
            attachments: Vec::new(),
            thinking_level: self.settings.thinking_level.wire_value().to_owned(),
        });
        if self.state == ConnectionState::Offline {
            self.start();
        } else if self.state == ConnectionState::Live {
            self.flush_pending_turn();
        } else {
            self.status = "Connecting… will send note when ready".to_owned();
        }
    }

    fn flush_dirty_note(&mut self, notify_ai: bool) -> bool {
        if self.note_content == self.note_saved_content {
            self.note_dirty_since = None;
            return true;
        }
        let Some(path) = self.active_note.clone() else {
            return false;
        };
        if let Err(error) = notes::save_note(&path, &self.note_content) {
            self.error = Some(format!("{error:#}"));
            return false;
        }
        self.note_saved_content = self.note_content.clone();
        self.note_dirty_since = None;
        if notify_ai {
            let content = self.note_content.clone();
            self.queue_note_changed(&path, &content);
        }
        true
    }

    fn maybe_autosave_note(&mut self) {
        let ready = self
            .note_dirty_since
            .is_some_and(|changed_at| changed_at.elapsed() >= Duration::from_secs(3));
        if ready {
            self.flush_dirty_note(true);
        }
    }

    fn sync_notes_from_disk(&mut self) {
        if self.note_disk_checked_at.elapsed() < NOTE_DISK_POLL_INTERVAL {
            return;
        }
        self.note_disk_checked_at = Instant::now();

        match notes::list_notes() {
            Ok(files) => self.note_files = files,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        }

        let Some(path) = self.active_note.clone() else {
            return;
        };
        let Ok(content) = notes::read_note(&path) else {
            return;
        };
        if content == self.note_saved_content {
            return;
        }

        self.note_content = content.clone();
        self.note_saved_content = content.clone();
        self.note_dirty_since = None;
        self.queue_note_changed(&path, &content);
    }

    fn select_note(&mut self, path: PathBuf) {
        if self.active_note.as_ref() == Some(&path)
            && self.bottom_workspace == BottomWorkspace::Note
        {
            return;
        }
        if !self.flush_dirty_note(true) {
            return;
        }
        match notes::read_note(&path) {
            Ok(content) => {
                self.active_note = Some(path.clone());
                self.note_content = content.clone();
                self.note_saved_content = content.clone();
                self.note_dirty_since = None;
                self.renaming_note = None;
                self.rename_buffer.clear();
                self.bottom_workspace = BottomWorkspace::Note;
                self.queue_note_changed(&path, &content);
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn create_note_from_ui(&mut self) {
        if !self.flush_dirty_note(true) {
            return;
        }
        match notes::create_unique_note("Untitled.md", "") {
            Ok(path) => {
                self.refresh_note_files();
                self.select_note(path.clone());
                self.rename_buffer = notes::display_name(&path);
                self.renaming_note = Some(path);
                self.rename_needs_focus = true;
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn import_note_from_ui(&mut self) {
        let Some(source) = rfd::FileDialog::new()
            .add_filter("Markdown or text", &["md", "markdown", "txt", "text"])
            .pick_file()
        else {
            return;
        };
        if !self.flush_dirty_note(true) {
            return;
        }
        match notes::import_text_file(&source) {
            Ok(path) => {
                self.refresh_note_files();
                self.select_note(path);
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn start_note_rename(&mut self, path: PathBuf) {
        self.rename_buffer = notes::display_name(&path);
        self.renaming_note = Some(path);
        self.rename_needs_focus = true;
    }

    fn commit_note_rename(&mut self) {
        let Some(path) = self.renaming_note.take() else {
            return;
        };
        let new_name = self.rename_buffer.trim().to_owned();
        self.rename_buffer.clear();
        if new_name.is_empty() {
            return;
        }
        if !self.flush_dirty_note(true) {
            self.renaming_note = Some(path);
            self.rename_buffer = new_name;
            return;
        }
        match notes::rename_note(&path, &new_name) {
            Ok(new_path) => {
                let renamed_active_note = self.active_note.as_ref() == Some(&path);
                if renamed_active_note {
                    self.active_note = Some(new_path.clone());
                }
                self.refresh_note_files();
                if renamed_active_note {
                    let content = self.note_content.clone();
                    self.queue_note_changed(&new_path, &content);
                }
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                self.renaming_note = Some(path);
                self.rename_buffer = new_name;
            }
        }
    }

    fn draw_note_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if centered_plus_button(ui, "new-note", 24.0)
                .on_hover_text("New note")
                .clicked()
            {
                self.create_note_from_ui();
            }
            if centered_file_button(ui, "open-note", 24.0)
                .on_hover_text("Open text file")
                .clicked()
            {
                self.import_note_from_ui();
            }
        });
        ui.add_space(3.0);
        let chat_selected = self.bottom_workspace == BottomWorkspace::Chat;
        if sidebar_text_button(ui, "Chat", chat_selected).clicked() {
            if self.flush_dirty_note(true) {
                self.bottom_workspace = BottomWorkspace::Chat;
                self.renaming_note = None;
            }
        }
        ui.separator();

        let files = self.note_files.clone();
        egui::ScrollArea::vertical()
            .id_salt("note-file-list")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for path in files {
                    let selected = self.active_note.as_ref() == Some(&path)
                        && self.bottom_workspace == BottomWorkspace::Note;
                    if self.renaming_note.as_ref() == Some(&path) {
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.rename_buffer)
                                .desired_width(ui.available_width()),
                        );
                        if self.rename_needs_focus {
                            response.request_focus();
                            self.rename_needs_focus = false;
                        }
                        let commit = response.lost_focus()
                            || ui.input(|input| input.key_pressed(egui::Key::Enter));
                        let cancel = ui.input(|input| input.key_pressed(egui::Key::Escape));
                        if cancel {
                            self.renaming_note = None;
                            self.rename_buffer.clear();
                            self.rename_needs_focus = false;
                        } else if commit {
                            self.commit_note_rename();
                        }
                        continue;
                    }
                    let name = notes::display_name(&path);
                    let response = sidebar_text_button(ui, &name, selected);
                    if response.clicked() {
                        if selected {
                            self.start_note_rename(path);
                        } else {
                            self.select_note(path);
                        }
                    }
                }
            });
    }

    fn draw_note_editor(&mut self, ui: &mut egui::Ui) {
        if self.active_note.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Create or open a Markdown/text note from the file list.");
            });
            return;
        }
        let editor_size = ui.available_size().max(egui::vec2(1.0, 1.0));
        let editor = egui::ScrollArea::both()
            .id_salt(("note-editor-scroll", self.active_note.as_ref()))
            .auto_shrink([false, false])
            .max_width(editor_size.x)
            .max_height(editor_size.y)
            .min_scrolled_width(editor_size.x)
            .min_scrolled_height(editor_size.y)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.note_content)
                        .code_editor()
                        .desired_rows(1)
                        .desired_width(f32::INFINITY)
                        .min_size(editor_size)
                        .lock_focus(true)
                        .hint_text("Write Markdown…"),
                )
            });
        if editor.inner.changed() {
            self.note_dirty_since = Some(Instant::now());
        }
    }

    fn draw_bottom_workspace(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::SidePanel::left("bottom-note-files")
            .resizable(true)
            .default_width(100.0)
            .width_range(75.0..=320.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(241, 245, 250))
                    .inner_margin(egui::Margin::symmetric(6, 6)),
            )
            .show_inside(ui, |ui| self.draw_note_sidebar(ui));

        ui.vertical(|ui| {
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
            match self.bottom_workspace {
                BottomWorkspace::Chat => self.draw_composer(ui, ctx),
                BottomWorkspace::Note => self.draw_note_editor(ui),
            }
        });
    }

    fn prepare_note_tool_call(
        &self,
        mut call: crate::realtime::ToolCall,
    ) -> crate::realtime::ToolCall {
        if !notes::uses_current_note(&call.name) {
            return call;
        }
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&call.arguments) else {
            return call;
        };
        let Some(object) = value.as_object_mut() else {
            return call;
        };
        let needs_name = object
            .get("note_name")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|name| name.trim().is_empty());
        if needs_name && let Some(name) = self.active_note_name() {
            object.insert("note_name".to_owned(), serde_json::Value::String(name));
            call.arguments = value.to_string();
        }
        call
    }

    fn apply_note_tool_result(&mut self, output: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(output) else {
            return;
        };
        if value.get("ok").and_then(serde_json::Value::as_bool) != Some(true)
            || value
                .get("note_changed")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
        {
            return;
        }
        let Some(name) = value.get("note_name").and_then(serde_json::Value::as_str) else {
            return;
        };
        let Ok(path) = notes::path_for_name(name) else {
            return;
        };
        let content = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .or_else(|| notes::read_note(&path).ok())
            .unwrap_or_default();
        self.refresh_note_files();
        self.active_note = Some(path);
        self.note_content = content.clone();
        self.note_saved_content = content;
        self.note_dirty_since = None;
        self.renaming_note = None;
        self.rename_buffer.clear();
        self.rename_needs_focus = false;
        self.bottom_workspace = BottomWorkspace::Note;
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
                let editor_height = (ui.available_height() - 44.0).max(120.0);
                let response = ui.add_sized(
                    [ui.available_width(), editor_height],
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
                    if centered_attach_button(ui, "attach-file", 24.0)
                        .on_hover_text("Attach image or audio")
                        .clicked()
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
                    if centered_paste_button(ui, "paste-image", 24.0)
                        .on_hover_text("Paste image")
                        .clicked()
                    {
                        match media::image_from_clipboard() {
                            Ok(image) => self.pending.push(image),
                            Err(error) => self.error = Some(error.to_string()),
                        }
                    }
                    if self.settings.backend == RealtimeBackend::CodexText {
                        ui.separator();
                        self.draw_thinking_level_controls(ui);
                    }
                    let level = self
                        .microphone
                        .as_ref()
                        .map(Microphone::level)
                        .unwrap_or(0.0);
                    if let Some(microphone) = &self.microphone {
                        ui.add(
                            egui::ProgressBar::new((level * 5.0).clamp(0.0, 1.0))
                                .desired_width(70.0),
                        );
                        if microphone.system_audio_passthrough() {
                            ui.label(
                                RichText::new("System audio")
                                    .color(Color32::from_rgb(185, 90, 20))
                                    .strong(),
                            )
                            .on_hover_text(
                                "Command has been held for 1 second. Release it to restore echo cancellation.",
                            );
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                true,
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
        self.ensure_text_model_selection();
        let settings_screen = Self::screen_info_for_settings(&self.settings);
        let default_prompt = default_system_prompt(settings_screen);
        let available_tools = available_tool_descriptions(settings_screen);
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
                        ui.heading("Session and model");
                        ui.add_space(8.0);
                        egui::Grid::new("settings_grid")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .show(ui, |ui| {
                                ui.label("Session type");
                                let previous_backend = self.settings.backend;
                                egui::ComboBox::from_id_salt("voice_engine")
                                    .selected_text(match self.settings.backend {
                                        RealtimeBackend::OpenAiRealtime => "Realtime · OpenAI",
                                        RealtimeBackend::CodexGptLive => "Live · Codex GPT-Live",
                                        RealtimeBackend::CodexText => "Text · Codex model",
                                    })
                                    .show_ui(ui, |ui| {
                                        style_light_blue_popup(ui);
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
                                        ui.selectable_value(
                                            &mut self.settings.backend,
                                            RealtimeBackend::CodexText,
                                            "Text model via Codex app-server",
                                        );
                                    });
                                if previous_backend != self.settings.backend {
                                    self.settings.voice = match self.settings.backend {
                                        RealtimeBackend::OpenAiRealtime => "marin",
                                        RealtimeBackend::CodexGptLive => "ember",
                                        RealtimeBackend::CodexText => "marin",
                                    }
                                    .to_owned();
                                    if self.settings.backend == RealtimeBackend::CodexText {
                                        self.ensure_text_model_selection();
                                        self.settings.thinking_level =
                                            self.settings.default_thinking_level;
                                    }
                                }
                                ui.end_row();

                                ui.label(if self.settings.backend == RealtimeBackend::CodexText {
                                    "Text model"
                                } else {
                                    "Realtime model"
                                });
                                if self.settings.backend == RealtimeBackend::CodexGptLive {
                                    ui.label("gpt-live-1-boulder-alpha · managed by Codex");
                                } else if self.settings.backend == RealtimeBackend::CodexText {
                                    let text_models = self.text_model_choices();
                                    egui::ComboBox::from_id_salt("text_model")
                                        .selected_text(&self.settings.model)
                                        .show_ui(ui, |ui| {
                                            style_light_blue_popup(ui);
                                            for (model, display_name) in text_models {
                                                ui.selectable_value(
                                                    &mut self.settings.model,
                                                    model,
                                                    display_name,
                                                );
                                            }
                                        });
                                } else {
                                    egui::ComboBox::from_id_salt("model")
                                        .selected_text(&self.settings.model)
                                        .show_ui(ui, |ui| {
                                            style_light_blue_popup(ui);
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

                                ui.label("Default text model");
                                let default_text_models = self.text_model_choices();
                                egui::ComboBox::from_id_salt("default_text_model")
                                    .selected_text(&self.settings.default_text_model)
                                    .show_ui(ui, |ui| {
                                        style_light_blue_popup(ui);
                                        for (model, display_name) in default_text_models {
                                            ui.selectable_value(
                                                &mut self.settings.default_text_model,
                                                model,
                                                display_name,
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Default image model");
                                let image_models = self.image_model_choices();
                                egui::ComboBox::from_id_salt("default_image_model")
                                    .selected_text(&self.settings.default_image_model)
                                    .show_ui(ui, |ui| {
                                        style_light_blue_popup(ui);
                                        for (model, display_name) in image_models {
                                            ui.selectable_value(
                                                &mut self.settings.default_image_model,
                                                model,
                                                display_name,
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Default image resolution");
                                egui::ComboBox::from_id_salt("default_image_resolution")
                                    .selected_text(self.settings.default_image_resolution.label())
                                    .show_ui(ui, |ui| {
                                        style_light_blue_popup(ui);
                                        for resolution in ImageResolution::ALL {
                                            ui.selectable_value(
                                                &mut self.settings.default_image_resolution,
                                                resolution,
                                                resolution.label(),
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Default thinking level");
                                egui::ComboBox::from_id_salt("default_thinking_level")
                                    .selected_text(self.settings.default_thinking_level.label())
                                    .show_ui(ui, |ui| {
                                        style_light_blue_popup(ui);
                                        for level in ThinkingLevel::ALL {
                                            ui.selectable_value(
                                                &mut self.settings.default_thinking_level,
                                                level,
                                                level.label(),
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label(if self.settings.backend == RealtimeBackend::CodexText {
                                    "Output"
                                } else {
                                    "Voice"
                                });
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
                                    RealtimeBackend::CodexText => Vec::new(),
                                };
                                if self.settings.backend == RealtimeBackend::CodexText {
                                    ui.label("Text response");
                                } else {
                                    egui::ComboBox::from_id_salt("voice")
                                        .selected_text(&self.settings.voice)
                                        .show_ui(ui, |ui| {
                                            style_light_blue_popup(ui);
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
                                }
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
                                ui.label(RichText::new("Available session types").strong());
                                ui.label(
                                    "gpt-realtime-2.1 · current selectable Realtime API model",
                                );
                                ui.label(
                                    "GPT-Live · experimental V3 WebRTC through `codex app-server`; \
                                     supports Codex-managed ChatGPT login",
                                );
                                ui.label(
                                    RichText::new(
                                        "GPT-Live adds screenshots as Codex multimodal thread context. \
                                         Its screen clicks and other local tools remain on the direct \
                                         low-latency realtime path.",
                                    )
                                    .small()
                                    .weak(),
                                );
                                ui.label(
                                    "Text models · Codex app-server threads with the same computer tools and image attachments",
                                );
                            });

                        ui.add_space(8.0);
                        egui::Frame::new()
                            .fill(LIGHT_BLUE)
                            .stroke(Stroke::new(1.0, Color32::from_rgb(183, 211, 241)))
                            .corner_radius(7.0)
                            .inner_margin(egui::Margin::same(9))
                            .show(ui, |ui| {
                                ui.label(RichText::new("Available tools").strong());
                                ui.label(
                                    RichText::new(
                                        "OpenAI Realtime delegates screen clicks to ask_text_model with a fresh screenshot for accuracy. GPT-Live calls click_screen directly. Text tabs omit ask_text_model to prevent recursive text sessions.",
                                    )
                                    .small()
                                    .weak(),
                                );
                                for (name, description) in &available_tools {
                                    ui.add_space(4.0);
                                    ui.label(RichText::new(format!("⚙ {name}")).strong());
                                    ui.label(RichText::new(description).small());
                                }
                            });

                        if self.settings.auth_mode == AuthMode::CodexApiKey {
                            ui.add_space(10.0);
                            self.draw_codex_account(ui);
                        }

                        ui.add_space(8.0);
                        if self.settings.backend == RealtimeBackend::CodexText {
                            ui.label(
                                RichText::new(
                                    "Text models can inspect the screen through computer tools or an attached image.",
                                )
                                .weak(),
                            );
                        } else {
                            ui.checkbox(
                                &mut self.settings.send_screenshot,
                                "Send the primary display after 0.5 seconds of clear speech",
                            );
                        }
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
                        let append_prompt_changed = ui
                            .checkbox(
                                &mut self.settings.append_realtime_tool_prompt,
                                "Append fast tool-call instructions for OpenAI Realtime",
                            )
                            .on_hover_text(
                                "When enabled, OpenAI Realtime is told to call tools before speaking, then reply after the real result. GPT-Live is unaffected.",
                            )
                            .changed();
                        if append_prompt_changed {
                            let enabled = self.settings.append_realtime_tool_prompt;
                            for tab in &mut self.tabs {
                                tab.session.settings.append_realtime_tool_prompt = enabled;
                            }
                        }
                        ui.label(
                            RichText::new(
                                "This optional appendix applies only when connecting an OpenAI Realtime tab. Disable it to send only the editable system prompt below.",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label("System prompt");
                            if ui.button("Reset to default").clicked() {
                                self.settings.system_prompt = default_prompt.clone();
                            }
                        });
                        ui.label(
                            RichText::new(
                                "This complete prompt is sent to the selected model. Edit it \
                                 directly to change the assistant's behavior; Reset restores the \
                                 built-in prompt for the current screen.",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_sized(
                            [ui.available_width(), 220.0],
                            egui::TextEdit::multiline(&mut self.settings.system_prompt)
                                .font(egui::TextStyle::Monospace)
                                .hint_text("System prompt…"),
                        );
                        ui.label(
                            RichText::new(if self.settings.system_prompt == default_prompt {
                                "Using the built-in default."
                            } else {
                                "Using a customized system prompt."
                            })
                            .small()
                            .weak(),
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

impl LiveAssistantApp {
    fn update_system_audio_command_hold(&mut self, ctx: &egui::Context) {
        let enabled = self.system_audio_command_hold.poll(ctx, Instant::now());
        let Some(microphone) = &self.microphone else {
            self.system_audio_command_hold.reset();
            return;
        };

        if let Err(error) = microphone.set_system_audio_passthrough(enabled) {
            self.error = Some(format!(
                "Could not switch system-audio microphone passthrough: {error:#}"
            ));
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
        self.flush_dirty_note(false);
        eframe::set_value(storage, SETTINGS_KEY, &self.settings);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_system_audio_command_hold(ctx);
        if self.state == ConnectionState::Live && self.settings.show_live_pointer {
            self.pointer_overlay.poll();
        }
        self.process_events(ctx);
        self.sync_notes_from_disk();
        self.maybe_autosave_note();
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

        if self.show_model_picker {
            self.draw_model_picker(ctx);
        }

        egui::TopBottomPanel::bottom("composer-workspace-v2")
            .resizable(true)
            .default_height((ctx.screen_rect().height() * 0.48).max(240.0))
            .min_height(200.0)
            .max_height((ctx.screen_rect().height() * 0.82).max(320.0))
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(247, 249, 252))
                    .inner_margin(egui::Margin::symmetric(6, 6)),
            )
            .show(ctx, |ui| self.draw_bottom_workspace(ui, ctx));

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
                    let scroll_height = ui.available_height().max(0.0);
                    let scroll_output = egui::ScrollArea::vertical()
                        .id_salt(("messages", self.active_tab))
                        .auto_shrink([false, false])
                        .max_height(scroll_height)
                        .stick_to_bottom(false)
                        .show(ui, |ui| {
                            self.draw_messages(ui);
                            if self.should_scroll {
                                ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                                self.should_scroll = false;
                            }
                        });

                    let max_offset =
                        (scroll_output.content_size.y - scroll_output.inner_rect.height()).max(0.0);
                    let at_latest =
                        max_offset <= 1.0 || scroll_output.state.offset.y >= max_offset - 2.0;
                    if !at_latest {
                        let button_size = 24.0;
                        let button_position = egui::pos2(
                            scroll_output.inner_rect.center().x - button_size * 0.5,
                            scroll_output.inner_rect.bottom() - button_size - 8.0,
                        );
                        egui::Area::new(egui::Id::new(("scroll-to-latest", self.active_tab)))
                            .order(egui::Order::Foreground)
                            .fixed_pos(button_position)
                            .show(ctx, |ui| {
                                if centered_down_button(
                                    ui,
                                    ("scroll-to-latest-button", self.active_tab),
                                    button_size,
                                )
                                .on_hover_text("Scroll to latest message")
                                .clicked()
                                {
                                    self.should_scroll = true;
                                    ctx.request_repaint();
                                }
                            });
                    }
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

fn prompt_requires_screen_click(prompt: &str) -> bool {
    prompt.to_ascii_lowercase().contains("click")
}

fn format_duration(seconds: f32) -> String {
    let total = seconds.round() as u32;
    format!("{}:{:02}", total / 60, total % 60)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClockParts {
    day_key: i64,
    hour: u32,
    minute: u32,
    second: u32,
}

fn clock_parts_utc(time: SystemTime) -> ClockParts {
    let total_seconds = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let seconds = total_seconds % (24 * 60 * 60);
    ClockParts {
        day_key: (total_seconds / (24 * 60 * 60)) as i64,
        hour: (seconds / 3_600) as u32,
        minute: ((seconds % 3_600) / 60) as u32,
        second: (seconds % 60) as u32,
    }
}

#[cfg(unix)]
fn clock_parts(time: SystemTime) -> ClockParts {
    let timestamp = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as libc::time_t;
    let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
    let result = unsafe { libc::localtime_r(&timestamp, local.as_mut_ptr()) };
    if result.is_null() {
        return clock_parts_utc(time);
    }
    let local = unsafe { local.assume_init() };
    ClockParts {
        day_key: i64::from(local.tm_year) * 400 + i64::from(local.tm_yday),
        hour: local.tm_hour.max(0) as u32,
        minute: local.tm_min.max(0) as u32,
        second: local.tm_sec.max(0) as u32,
    }
}

#[cfg(not(unix))]
fn clock_parts(time: SystemTime) -> ClockParts {
    clock_parts_utc(time)
}

fn format_clock_parts(parts: ClockParts) -> String {
    format!("{:02}:{:02}:{:02}", parts.hour, parts.minute, parts.second)
}

fn same_clock_minute(start: ClockParts, end: ClockParts) -> bool {
    start.day_key == end.day_key && start.hour == end.hour && start.minute == end.minute
}

fn format_elapsed_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    if milliseconds < 1_000 {
        return format!("{milliseconds}ms");
    }
    if milliseconds < 60_000 {
        if milliseconds % 1_000 == 0 {
            return format!("{}s", milliseconds / 1_000);
        }
        return format!("{:.1}s", milliseconds as f64 / 1_000.0);
    }

    let minutes = milliseconds / 60_000;
    let remaining_ms = milliseconds % 60_000;
    if remaining_ms % 1_000 == 0 {
        format!("{minutes}m{:02}s", remaining_ms / 1_000)
    } else {
        format!("{minutes}m{:04.1}s", remaining_ms as f64 / 1_000.0)
    }
}

fn format_time_range(start: ClockParts, end: ClockParts, elapsed: Duration) -> String {
    let end_text = if same_clock_minute(start, end) {
        format!("{:02}", end.second)
    } else {
        format_clock_parts(end)
    };
    format!(
        "{}-{end_text},{}",
        format_clock_parts(start),
        format_elapsed_duration(elapsed)
    )
}

fn format_token_count(value: u64) -> String {
    if value <= i64::MAX as u64 {
        format_count(value as i64)
    } else {
        value.to_string()
    }
}

fn note_change_message(path: &Path, content: &str) -> String {
    let note_name = notes::display_name(path);
    format!(
        "Note file changed.\nName: {note_name}\nPath: {}\n\n<note_{note_name} >\n{content}\n</note_{note_name}>",
        path.display()
    )
}

fn format_message_metadata(message: &ChatMessage, is_assistant: bool) -> String {
    let start = clock_parts(message.started_at);
    let timing = match (message.finished_at, message.finished_instant) {
        (Some(finished_at), Some(finished_instant)) => {
            let end = clock_parts(finished_at);
            let elapsed = finished_instant
                .checked_duration_since(message.started_instant)
                .unwrap_or_default();
            format_time_range(start, end, elapsed)
        }
        _ => format!("{}-…", format_clock_parts(start)),
    };

    let mut parts = vec![timing];
    if let Some(tokens) = message.token_count {
        let prefix = if message.token_count_is_estimate {
            "~"
        } else {
            ""
        };
        parts.push(format!("Tokens {prefix}{}", format_token_count(tokens)));
    }
    if is_assistant && let Some(rate) = message.tokens_per_second {
        parts.push(format!("{rate:.1} token/s"));
    }
    parts.join(" · ")
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

fn paint_centered_image_status(ui: &egui::Ui, image_rect: egui::Rect, text: &str) {
    let font = egui::FontId::proportional(16.0);
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, Color32::WHITE);
    let padding = egui::vec2(10.0, 6.0);
    let label_rect =
        egui::Rect::from_center_size(image_rect.center(), galley.size() + padding * 2.0);
    ui.painter()
        .rect_filled(label_rect, 7.0, Color32::from_black_alpha(180));
    ui.painter()
        .galley(label_rect.min + padding, galley, Color32::WHITE);
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

fn tool_output_preview(output: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(output) else {
        return output.to_owned();
    };
    if let Some(object) = value.as_object_mut()
        && object.remove("image_url").is_some()
    {
        object.insert(
            "image_url".to_owned(),
            serde_json::Value::String("[generated image attached]".to_owned()),
        );
    }
    pretty_tool_arguments(&value.to_string())
}

fn generated_chat_image(output: &str) -> Option<ChatImage> {
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    let data_url = value.get("image_url").and_then(serde_json::Value::as_str)?;
    let attachment = media::image_from_data_url("Generated image", data_url).ok()?;
    ChatImage::from_attachment(&attachment)
}

fn apply_tool_result_to_messages(
    messages: &mut [ChatMessage],
    call_id: &str,
    output: &str,
) -> bool {
    let generated_image = generated_chat_image(output);
    let preview = tool_output_preview(output);
    for message in messages {
        let Some(tool) = message
            .tool_calls
            .iter_mut()
            .find(|tool| tool.call_id == call_id)
        else {
            continue;
        };
        tool.output = Some(preview);
        if let Some(image) = generated_image {
            message.images.push(image);
        }
        message.refresh_token_estimate();
        return true;
    }
    false
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

    #[test]
    fn command_hold_enables_system_audio_only_after_one_second() {
        let started = Instant::now();
        let mut hold = CommandHoldState::default();

        assert!(!hold.update(true, started));
        assert!(!hold.update(
            true,
            started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY - Duration::from_millis(1)
        ));
        assert!(hold.update(true, started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY));
        assert!(hold.update(
            true,
            started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY + Duration::from_secs(2)
        ));
    }

    #[test]
    fn command_hold_release_immediately_restores_cancellation_and_resets_timer() {
        let started = Instant::now();
        let mut hold = CommandHoldState::default();

        assert!(!hold.update(true, started));
        assert!(hold.update(true, started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY));
        assert!(!hold.update(false, started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY));
        assert!(!hold.update(
            true,
            started + SYSTEM_AUDIO_COMMAND_HOLD_DELAY + Duration::from_millis(1)
        ));
    }

    #[test]
    fn note_change_message_includes_name_path_and_content() {
        assert_eq!(
            note_change_message(Path::new("/tmp/ideas.md"), "hello"),
            "Note file changed.\nName: ideas.md\nPath: /tmp/ideas.md\n\n<note_ideas.md >\nhello\n</note_ideas.md>"
        );
    }

    #[test]
    fn default_voice_tabs_put_gpt_live_first() {
        let base = Settings::default();
        let [live, realtime] = default_voice_tab_settings(&base);

        assert_eq!(live.backend, RealtimeBackend::CodexGptLive);
        assert_eq!(live.model, "gpt-realtime-2.1");
        assert_eq!(live.voice, "ember");
        assert_eq!(realtime.backend, RealtimeBackend::OpenAiRealtime);
        assert_eq!(realtime.model, "gpt-realtime-2.1");
        assert_eq!(realtime.voice, "marin");
    }

    #[test]
    fn openai_realtime_connection_prompt_delegates_clicks_only_for_openai() {
        let screen = ScreenInfo {
            origin_x: 0,
            origin_y: 0,
            logical_width: 1408,
            logical_height: 881,
            backing_width: 2816,
            backing_height: 1762,
            scale_factor: 2.0,
        };
        let mut realtime = Settings::default();
        realtime.system_prompt = "base prompt".to_owned();
        realtime.backend = RealtimeBackend::OpenAiRealtime;
        let realtime_prompt = connection_system_prompt(&realtime, screen);
        assert!(realtime_prompt.contains("call the required tool as the first response output"));
        assert!(realtime_prompt.contains("After the tool result arrives"));
        assert!(realtime_prompt.contains("ask_text_model"));
        assert!(realtime_prompt.contains("include_screenshot=true"));
        assert!(
            realtime_prompt.contains("Do not estimate coordinates or call click_screen directly")
        );

        realtime.append_realtime_tool_prompt = false;
        let realtime_prompt = connection_system_prompt(&realtime, screen);
        assert!(realtime_prompt.starts_with("base prompt"));
        assert!(realtime_prompt.contains("reply exactly: note saved"));

        let mut live = realtime.clone();
        live.backend = RealtimeBackend::CodexGptLive;
        let live_prompt = connection_system_prompt(&live, screen);
        assert!(live_prompt.starts_with("base prompt"));
        assert!(live_prompt.contains("reply exactly: note saved"));
    }

    #[test]
    fn compact_message_time_range_shortens_same_minute_end_time() {
        let start = ClockParts {
            day_key: 1,
            hour: 12,
            minute: 1,
            second: 1,
        };
        let same_minute_end = ClockParts {
            day_key: 1,
            hour: 12,
            minute: 1,
            second: 6,
        };
        let next_minute_end = ClockParts {
            day_key: 1,
            hour: 12,
            minute: 2,
            second: 3,
        };

        assert_eq!(
            format_time_range(start, same_minute_end, Duration::from_secs(5)),
            "12:01:01-06,5s"
        );
        assert_eq!(
            format_time_range(start, next_minute_end, Duration::from_secs(62)),
            "12:01:01-12:02:03,1m02s"
        );
    }

    #[test]
    fn unexpected_direct_voice_clicks_still_use_the_local_tool_path() {
        assert_eq!(voice_tool_route("click_screen"), VoiceToolRoute::Local);
        assert_eq!(
            voice_tool_route("ask_text_model"),
            VoiceToolRoute::AskTextModel
        );
        assert_eq!(
            voice_tool_route("create_image"),
            VoiceToolRoute::CreateImage
        );
    }

    #[test]
    fn text_defaults_and_thinking_levels_use_the_expected_wire_values() {
        let settings = Settings::default();
        assert_eq!(settings.default_text_model, "gpt-5.6-luna");
        assert_eq!(settings.default_image_model, "gpt-image-2");
        assert_eq!(settings.default_image_resolution.wire_value(), "1024x1024");
        assert_eq!(settings.default_thinking_level, ThinkingLevel::Light);
        assert_eq!(settings.thinking_level, ThinkingLevel::Light);

        for level in ThinkingLevel::ALL {
            assert_eq!(parse_thinking_level(level.wire_value()), Some(level));
        }
        assert_eq!(parse_thinking_level("light"), Some(ThinkingLevel::Light));
        assert_eq!(
            parse_thinking_level("extra-high"),
            Some(ThinkingLevel::ExtraHigh)
        );
        assert_eq!(parse_thinking_level("not-a-level"), None);
    }

    #[test]
    fn message_metadata_uses_backend_usage_and_deduplicates_retries() {
        let mut message = ChatMessage::assistant();
        message.started_instant = Instant::now() - Duration::from_secs(2);
        message.started_at = SystemTime::now() - Duration::from_secs(2);
        message.begin_response("response-1".to_owned());
        message.text = "A short answer".to_owned();
        message.set_actual_usage("response-1", 42);
        message.set_actual_usage("response-1", 42);
        message.finish();

        assert_eq!(message.token_count, Some(42));
        assert!(!message.token_count_is_estimate);
        assert_eq!(message.actual_token_total, 42);
        assert!(message.finished_at.is_some());
        assert!(message.tokens_per_second.is_some());

        let metadata = format_message_metadata(&message, true);
        assert!(metadata.contains('-'));
        assert!(metadata.contains(','));
        assert!(!metadata.contains("Start "));
        assert!(!metadata.contains("Finish "));
        assert!(metadata.contains("Tokens 42"));
        assert!(metadata.contains("token/s"));
    }

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
    fn assistant_group_window_is_five_seconds() {
        assert_eq!(MESSAGE_CONTINUATION_WINDOW, Duration::from_secs(5));
    }

    #[test]
    fn assistant_text_and_audio_can_append_across_a_user_card() {
        let now = Instant::now();
        let mut assistant = ChatMessage::assistant();
        assistant.text = "first AI part".to_owned();
        assistant.audio.extend_from_slice(&[1, 2]);
        let messages = [ChatMessage::user_voice(vec![9], None), assistant];
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
    fn speech_screenshot_audio_fallback_waits_for_half_second_and_fires_once() {
        let mut gate = SpeechScreenshotGate::default();

        assert_eq!(SPEECH_SCREENSHOT_SAMPLE_TARGET, 12_000);
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, true));
        gate.begin();
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET - 1, true));
        assert!(gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET, true));
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET + 1, false));
        assert!(gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET + 1, true));
        gate.mark_sent();
        assert!(!gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET * 2, true));

        gate.end();
        gate.begin();
        assert!(gate.should_capture(SPEECH_SCREENSHOT_SAMPLE_TARGET + 1, true));
    }

    #[test]
    fn context_image_preview_is_dim_until_upload_confirmation() {
        let mut image = ChatImage {
            name: "Current screen".to_owned(),
            thumbnail: None,
            sent_image: vec![1],
            width: 100,
            height: 100,
            byte_size: 1,
            thumbnail_texture: None,
            full_texture: None,
            thumbnail_load_attempted: false,
            full_load_attempted: false,
            upload_id: None,
            upload_state: ImageUploadState::Uploaded,
            upload_deadline: None,
        };
        image.begin_upload(7, Instant::now() + CONTEXT_IMAGE_UPLOAD_TIMEOUT);
        assert_eq!(image.upload_overlay_text(), Some("Uploading…"));
        assert_eq!(image.preview_tint().a(), 77);
        assert!(!image.upload_is_complete());

        assert!(image.finish_upload());
        assert_eq!(image.upload_overlay_text(), None);
        assert_eq!(image.preview_tint(), Color32::WHITE);
        assert!(image.upload_is_complete());
    }

    #[test]
    fn context_image_preview_falls_back_to_uploaded_image_bytes() {
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgba8(2, 1)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let sent_image = encoded.into_inner();
        let attachment = Attachment::Image {
            name: "Current screen".to_owned(),
            data_url: format!("data:image/png;base64,{}", STANDARD.encode(&sent_image)),
            thumbnail: b"not a decodable thumbnail".to_vec(),
            width: 2,
            height: 1,
            byte_size: sent_image.len(),
        };

        let prepared = PreparedChatImage::from_attachment(&attachment)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.thumbnail.size, [2, 1]);
        assert_eq!(prepared.sent_image, sent_image);
    }

    #[test]
    fn timed_out_context_image_ignores_late_success() {
        let mut image = ChatImage {
            name: "Current screen".to_owned(),
            thumbnail: None,
            sent_image: vec![1],
            width: 100,
            height: 100,
            byte_size: 1,
            thumbnail_texture: None,
            full_texture: None,
            thumbnail_load_attempted: false,
            full_load_attempted: false,
            upload_id: None,
            upload_state: ImageUploadState::Uploaded,
            upload_deadline: None,
        };
        let now = Instant::now();
        image.begin_upload(8, now + CONTEXT_IMAGE_UPLOAD_TIMEOUT);

        assert!(
            !image.upload_timed_out(now + CONTEXT_IMAGE_UPLOAD_TIMEOUT - Duration::from_millis(1))
        );
        assert!(image.upload_timed_out(now + CONTEXT_IMAGE_UPLOAD_TIMEOUT));
        assert!(image.fail_upload());
        assert!(!image.finish_upload());
        assert_eq!(image.upload_overlay_text(), Some("Upload failed"));
    }
}

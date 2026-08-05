#[path = "gpt_live_webrtc_fallback.rs"]
mod fallback;

#[cfg(target_os = "macos")]
#[path = "gpt_live_webrtc_native.rs"]
mod native;

// Production uses the app-owned microphone and speaker path on every platform.
// This gives macOS VoiceProcessingIO a controllable AEC path and avoids the
// platform ADM's inability to cancel unrelated system playback.
pub use fallback::GptLivePeer;

// Keep the deterministic raw-RTP peer for transport probes on every platform.
pub(crate) use fallback::GptLivePeer as GptLiveProbePeer;

#[cfg(target_os = "macos")]
pub(crate) use native::GptLivePeer as GptLiveNativePeer;

/// Production audio is owned by the app so echo cancellation and the temporary
/// Command-key passthrough can be controlled without changing output volume.
pub const fn uses_platform_audio() -> bool {
    false
}

#[path = "gpt_live_webrtc_fallback.rs"]
mod fallback;

#[cfg(target_os = "macos")]
#[path = "gpt_live_webrtc_native.rs"]
mod native;

#[cfg(not(target_os = "macos"))]
pub use fallback::GptLivePeer;
#[cfg(target_os = "macos")]
pub use native::GptLivePeer;

// Keep the deterministic raw-RTP peer for the transport probe on every platform.
pub(crate) use fallback::GptLivePeer as GptLiveProbePeer;

/// True when GPT-Live audio capture and playout are owned by the native
/// platform audio device module rather than the app's CPAL streams.
pub const fn uses_platform_audio() -> bool {
    cfg!(target_os = "macos")
}

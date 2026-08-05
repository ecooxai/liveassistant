#[path = "gpt_live_webrtc_fallback.rs"]
mod fallback;

#[cfg(target_os = "macos")]
#[path = "gpt_live_webrtc_native.rs"]
mod native;

#[cfg(not(target_os = "macos"))]
pub use fallback::GptLivePeer;
#[cfg(target_os = "macos")]
pub use native::GptLivePeer;

// Keep the deterministic raw-RTP peer for transport probes on every platform.
pub(crate) use fallback::GptLivePeer as GptLiveProbePeer;

#[cfg(target_os = "macos")]
pub(crate) use native::GptLivePeer as GptLiveNativePeer;

/// True when GPT-Live capture, acoustic echo cancellation, jitter buffering,
/// packet-loss concealment, clock correction, and speaker playout are owned by
/// the native platform WebRTC audio device instead of the app's PCM streams.
pub const fn uses_platform_audio() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(test)]
mod tests {
    use super::uses_platform_audio;

    #[test]
    fn macos_uses_native_platform_audio() {
        assert_eq!(uses_platform_audio(), cfg!(target_os = "macos"));
    }
}

#[path = "gpt_live_webrtc_fallback.rs"]
mod fallback;

#[cfg(target_os = "macos")]
#[path = "gpt_live_webrtc_native.rs"]
mod native;

// macOS combines app-owned VoiceProcessingIO devices with libWebRTC's
// external-audio encoder and NetEQ decoder. Other platforms retain raw RTP.
#[cfg(not(target_os = "macos"))]
pub use fallback::GptLivePeer;
#[cfg(target_os = "macos")]
pub use native::GptLivePeer;

// Keep the deterministic raw-RTP peer for transport probes on every platform.
pub(crate) use fallback::GptLivePeer as GptLiveProbePeer;

#[cfg(target_os = "macos")]
pub(crate) use native::GptLivePeer as GptLiveNativePeer;

/// Audio devices stay app-owned so VoiceProcessingIO can cancel unrelated
/// system playback and receive the exact assistant-output reference.
pub const fn uses_platform_audio() -> bool {
    false
}

/// macOS remote audio comes from libWebRTC NetEQ and includes continuous
/// decoded frames, so apply the existing remote speech gate.
pub const fn uses_native_remote_audio() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(test)]
mod tests {
    use super::{uses_native_remote_audio, uses_platform_audio};

    #[test]
    fn production_keeps_audio_devices_app_owned() {
        assert!(!uses_platform_audio());
    }

    #[test]
    fn macos_uses_native_remote_audio_processing() {
        assert_eq!(uses_native_remote_audio(), cfg!(target_os = "macos"));
    }
}

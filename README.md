# Live Assistant

A native Rust desktop assistant for macOS and Linux using the OpenAI Realtime
WebSocket API. It supports:

- `gpt-realtime-2.1` and `gpt-realtime-2`
- full-duplex microphone/audio: the mic stays live while the assistant speaks
- instant barge-in that stops and truncates assistant playback when the user talks
- OS voice-processing capture with acoustic echo cancellation for speaker/system audio
- one primary-display screenshot per detected voice turn
- logical-resolution screenshots on Retina/HiDPI displays
- transparent Live pointer overlay with one-second left/right click colors and
  bottom-right coordinates while a mouse button is held
- visible and replayable user audio for every voice turn
- typed text, Command/Ctrl+V image paste, uploaded images/audio, and drag-and-drop
- sent image and screen-context previews directly in user message bubbles
- click-to-enlarge viewing of the exact encoded image payload sent to the model
- compact per-message start/end timing and elapsed cost, plus token totals and completed assistant token rate
- GPT-Live and Realtime tabs open by default with GPT-Live first and auto-connecting, plus a manual down-arrow control instead of automatic transcript scrolling
- OpenAI Realtime has an optional tool-first prompt appendix, delegates screenshot-backed clicks to a text model for accuracy, and GPT-Live clicks directly
- both voice backends pause streaming playback below one second of queued audio and retry every two seconds
- Realtime function tools for screenshot-relative clicks, Bash commands, and text insertion
- light interface theme
- a persistent conversation until the voice session is stopped

## Run

Install the platform dependencies, then:

```sh
cargo run --release
```

On macOS, the first use should prompt for Microphone, Screen Recording, and
Accessibility permissions. Screen clicks and text insertion require
**System Settings → Privacy & Security → Accessibility**. If screen capture
remains unavailable, enable Screen Recording for the built application or
terminal there as well.

To verify the complete screenshot-coordinate → native-click path, run:

```sh
cargo run -- --test-click
```

This opens a small target window, clicks its center through the production
`click_screen` implementation, and fails if the target does not receive the
click or macOS reports a different pointer position.

To verify JPEG screenshot upload and latest-image ordering on both OpenAI
Realtime and GPT-Live, run:

```sh
cargo run -- --test-image-upload
```

The probe keeps one connection open per backend, uploads a real cat JPEG and
requires the first voice turn to identify the cat, then uploads a real dog JPEG
and requires the following voice turn to identify the dog rather than the stale
cat. It fails if either upload exceeds the deadline or either backend sees the
wrong image.

On Debian/Ubuntu Linux, the typical build dependencies are:

```sh
sudo apt install build-essential pkg-config libasound2-dev libx11-dev \
  libxcb-shape0-dev libxcb-xfixes0-dev libgtk-3-dev
```

Wayland screenshot support depends on the desktop portal/compositor. X11 is
supported directly by the screenshot library and is currently required for the
global pointer/click overlay on Linux.

## Authentication

Open **Settings** and choose one of:

1. **OpenAI Platform API key** — paste a key (or launch with `OPENAI_API_KEY`).
   The app keeps it in memory only and does not write it to disk.
2. **Reuse Codex login** — reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`).
   Supports both:
   - Platform API-key Codex logins (`sk-…`)
   - ChatGPT/Codex OAuth (`access_token` + optional `account_id`)

For OpenAI Realtime, Platform API keys connect directly to
`wss://api.openai.com/v1/realtime`. For GPT-Live V3, Codex app-server manages
ChatGPT OAuth and call creation while Live Assistant supplies the native WebRTC
audio peer. If the stored OAuth access token is near expiry and a `refresh_token`
is present, Live Assistant refreshes it via `https://auth.openai.com/oauth/token`
and updates `auth.json`.

When Codex authentication is selected, Settings also uses the installed
`codex app-server` to show the complete model catalog visible to that account,
rolling usage-limit percentages and reset times, credit state, and reported
token usage. It also loads the Codex v1/v2 voice-persona catalog. Set
`CODEX_BIN` if the `codex` executable is not on the app's `PATH`. Codex coding
models and GPT-Live information are displayed separately from the selectable
Realtime voice model because the Realtime WebSocket requires a Realtime model
ID. Settings is scrollable, and the complete system prompt is editable directly
with a Reset to default button. Settings can also disable the OpenAI Realtime
fast tool-call appendix without changing the editable base prompt. Message timing uses a compact form such as
`12:01:01-06,5s`; assistant end time is recorded when the backend finishes the
reply, not when audio playback drains. Token totals use backend usage when a
transport reports it and a visibly marked local estimate otherwise.

## Privacy and behavior

Microphone capture starts as soon as **Start voice** is clicked. Audio recorded
while the transport connects is kept in order (up to the latest 60 seconds),
then flushed when the session is ready before live audio continues. A screen
capture is taken after roughly half a second of clear speech or the first live
transcript token, not continuously. Disable per-turn screenshots in Settings.
Press **Stop** to close the transport and microphone.

Computer tools run with the current user's permissions. The session prompt
treats screenshot/application text as untrusted content. Bash commands time out
after 30 seconds, and their stdout and stderr are capped before being returned
to the model.

Assistant speech uses the normal native output device at full fidelity and
volume. The always-open microphone uses OS voice-processing capture, which
monitors/links system output as its acoustic echo-cancellation reference. On
Linux, system-wide AEC requires PulseAudio `module-echo-cancel`; macOS uses
VoiceProcessingIO and Windows uses WASAPI AEC. On macOS 14 and newer, the app
requests activity-aware, minimum-level media ducking so other audio is not
reduced throughout the entire connection; macOS can still apply a small
reduction while VoiceProcessingIO detects speech.

Uploaded audio is decoded locally, converted to mono 24 kHz PCM, and then sent
as an `input_audio` conversation item. User voice turns can be replayed or
saved as WAV from the chat.

## Package

For a distributable macOS `.app`, install `cargo-bundle` and run:

```sh
cargo install cargo-bundle
cargo bundle --release
```

Linux can run the release binary from `target/release/live-assistant` or package
it with your distribution's preferred format.

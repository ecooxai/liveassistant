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
ID. Settings is scrollable, and its custom system prompt is appended after the
app's built-in instructions.

## Privacy and behavior

Audio streams while the voice session is live. A screen capture is taken only
after the server reports the start of speech, not continuously. Disable
per-turn screenshots in Settings. Press **Stop** to close the WebSocket and
microphone.

Computer tools run with the current user's permissions. The session prompt
treats screenshot/application text as untrusted content. Bash commands time out
after 30 seconds, and their stdout and stderr are capped before being returned
to the model.

Assistant speech uses the normal native output device at full fidelity and
volume. The always-open microphone uses OS voice-processing capture, which
monitors/links system output as its acoustic echo-cancellation reference. On
Linux, system-wide AEC requires PulseAudio `module-echo-cancel`; macOS uses
VoiceProcessingIO and Windows uses WASAPI AEC.

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

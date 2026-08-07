# Live Assistant

A local Rust assistant backend with a lightweight browser interface for OpenAI Realtime, GPT Live, and Codex text sessions.

The browser is an interaction and rendering surface. Rust owns credentials, independent model transports, conversation state, screenshots, uploads, timing and usage data, tools, and image generation.

## Restored web features

- Two compact recent-session tabs live in the topbar; the adjacent overflow menu switches, closes, or creates any opened tab.
- GPT Live (`gpt-live-*`), OpenAI Realtime API (`gpt-realtime-*`), and Codex text tabs with separate model catalogs.
- Dynamic model discovery from the connected Codex account and, when an API key is supplied, the OpenAI Models API.
- Radio-card model and transport selection in Settings and in the Add model tab dialog.
- Dynamic voice persona lists for GPT Live and OpenAI Realtime.
- Automatic active-tab connection when the page opens, with the browser microphone enabled by default after a voice transport connects.
- Typed messages and browser microphone input with assistant audio playback; a stable single-window audio lease prevents duplicate microphones/speakers, microphone PCM uses a dedicated `/ws/audio` stream at 24 kHz, and GPT Live uses WebRTC's own bounded realtime audio queue.
- Image and audio upload, manual screen capture, and removable pending attachment previews.
- Optional automatic current-screen context for typed and voice turns.
- Screenshot upload lifecycle states: preparing, uploading, uploaded, and failed.
- Incremental keyed message rendering: streaming snapshots update only the affected card and preserve audio controls, selection, scroll context, and microphone state.
- Message start and end clocks, elapsed time, token totals, estimated-token marking, and completed assistant tokens per second.
- Replayable and downloadable WAV cards for uploaded audio and recorded user/assistant PCM, plus Realtime audio-alias de-duplication and a continuous 24 kHz playback AudioWorklet with a small jitter buffer.
- A compact resizable bottom workspace with one shared editor surface for Chat and Markdown/text files, shared attachment/voice controls, autosave, and Ctrl/Cmd+Enter current-line sending from notes.
- Codex account rate limits, reset times, credit state, and token-usage snapshots in Settings.
- Generated-image previews and local computer-tool progress/results.
- In-memory API keys; credentials are never returned in browser state snapshots or persisted by the web UI.

## Architecture

- **Axum server:** binds only to `127.0.0.1`, serves the embedded interface, and accepts explicit action requests.
- **Rust state worker:** owns every tab and polls each tab's independent Realtime client.
- **WebSocket:** sends Rust state snapshots and assistant PCM to the browser; a keyed DOM reconciler applies only changed tabs/cards, while microphone PCM travels back to the active Rust tab.
- **Model catalog:** merges account/API discoveries with built-in official fallbacks while preserving the source of each model option.
- **Attachments:** browser images are decoded and normalized in Rust before entering a turn. Screenshots are captured by Rust.

## Run

```sh
cargo run --release
```

The app starts at `http://127.0.0.1:4317` and opens that address in the default browser.

Choose another port:

```sh
cargo run --release -- --port 8080
```

or:

```sh
LIVE_ASSISTANT_PORT=8080 cargo run --release
```

Prevent automatic browser launch:

```sh
cargo run --release -- --no-open
```

For automatic rebuilds while editing Rust or browser assets:

```sh
./dev.sh
```

## Authentication and model discovery

Use either:

1. **Reuse Codex login** — the default for GPT Live, Realtime, and text tabs. It reads existing Codex credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json` and discovers the account's models and voice personas.
2. **OpenAI API key** — enter a key in Settings or set `OPENAI_API_KEY`. The **Find available** action asks the Rust backend to refresh the model catalog from the OpenAI Models API.

A key entered in the browser is sent only to the localhost Rust process and retained in memory.

## Permissions

On macOS, allow the built application or terminal to use:

- Microphone
- Screen Recording
- Accessibility

Accessibility is required for screen clicks and text insertion. Screen Recording is required for manual or automatic screen context. Browser microphone access also requires permission for the browser.

## Validation

```sh
node --check web/app.js
cargo check --locked
cargo test --no-fail-fast
```

Backend transport probes remain available:

```sh
cargo run -- --test-click
cargo run -- --test-gpt-live-native
cargo run -- --test-gpt-live
cargo run -- --test-gpt-live-tool-latency
cargo run -- --test-image-upload
```

## Linux build dependencies

On Debian/Ubuntu:

```sh
sudo apt install build-essential pkg-config libasound2-dev \
  libx11-dev libxcb-shape0-dev libxcb-xfixes0-dev libgtk-3-dev
```

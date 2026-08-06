# Live Assistant

A local Rust assistant backend with a lightweight browser interface for OpenAI Realtime, GPT Live, and Codex text sessions.

The browser is an interaction and rendering surface. Rust owns credentials, independent model transports, conversation state, screenshots, uploads, timing and usage data, tools, and image generation.

## Restored web features

- Independent top tabs, each with its own Rust transport, settings, messages, uploads, and connection state.
- GPT Live, OpenAI Realtime API, and Codex text tabs.
- Dynamic model discovery from the connected Codex account and, when an API key is supplied, the OpenAI Models API.
- Radio-card model and transport selection in Settings and in the Add model tab dialog.
- Dynamic voice persona lists for GPT Live and OpenAI Realtime.
- Typed messages and browser microphone input with assistant audio playback.
- Image upload, manual screen capture, and removable pending attachment previews.
- Optional automatic current-screen context for typed and voice turns.
- Screenshot upload lifecycle states: preparing, uploading, uploaded, and failed.
- Message start and end clocks, elapsed time, token totals, estimated-token marking, and completed assistant tokens per second.
- Generated-image previews and local computer-tool progress/results.
- In-memory API keys; credentials are never returned in browser state snapshots or persisted by the web UI.

## Architecture

- **Axum server:** binds only to `127.0.0.1`, serves the embedded interface, and accepts explicit action requests.
- **Rust state worker:** owns every tab and polls each tab's independent Realtime client.
- **WebSocket:** sends full state snapshots and assistant PCM to the active browser session; microphone PCM travels back to the active Rust tab.
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

1. **Reuse Codex login** — reads existing Codex credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json` and discovers the account's text models and voice personas.
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

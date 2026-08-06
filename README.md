# Live Assistant

A local Rust assistant with a lightweight browser interface for OpenAI Realtime, GPT Live, and Codex text models.

The browser is intentionally thin: it collects user input and renders backend state. Credentials, model connections, screenshots, tool execution, image generation, and conversation state stay in the Rust process.

## Architecture

- **Rust backend:** owns the in-memory app state, authentication, Realtime/Codex transports, model requests, tools, screenshots, and image generation.
- **Web UI:** a small dependency-free HTML/CSS/JavaScript client with an aurora-style light interface.
- **State sync:** WebSocket snapshots update the UI; no conversation or credential state is persisted by the browser.
- **Audio:** browser microphone PCM is sent to Rust over the local WebSocket; assistant PCM events are streamed back for playback when the transport exposes them.
- **Network scope:** the server binds only to `127.0.0.1`.

## Run

```sh
cargo run --release
```

The app starts on `http://127.0.0.1:4317` and opens that address in the default browser.

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

## Authentication

Use either:

1. **Codex login** — reads the existing Codex credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json`.
2. **OpenAI API key** — enter it in Settings or set `OPENAI_API_KEY` before launch.

An API key entered in the browser is sent only to the localhost Rust server and retained in memory. It is never returned in state snapshots.

## Permissions

On macOS, allow the built application or the terminal running it to use:

- Microphone
- Screen Recording
- Accessibility

Accessibility is required for screen clicks and text insertion. Screen Recording is required when screen context is enabled.

## Backend probes

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

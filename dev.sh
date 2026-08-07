#!/usr/bin/env bash

set -Eeuo pipefail

# Run this script from anywhere; it always watches the project containing it.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEBOUNCE_SECONDS=5
POLL_INTERVAL_SECONDS=0.5
WATCH_PATHS=("$ROOT_DIR/src" "$ROOT_DIR/web" "$ROOT_DIR/vendor")
WATCH_FILES=("$ROOT_DIR/Cargo.toml" "$ROOT_DIR/Cargo.lock" "$ROOT_DIR/build.rs")

if ! command -v cksum >/dev/null 2>&1; then
    printf 'dev.sh requires the POSIX cksum command.\n' >&2
    exit 1
fi

app_pid=""
last_snapshot=""

snapshot() {
    local file

    {
        find "${WATCH_PATHS[@]}" -type f -print 2>/dev/null
        for file in "${WATCH_FILES[@]}"; do
            if [[ -f "$file" ]]; then
                printf '%s\n' "$file"
            fi
        done
    } | LC_ALL=C sort -u | while IFS= read -r file; do
        # Include both the path and contents so additions, removals, and edits
        # all produce a different snapshot without requiring fswatch/inotify.
        printf '%s\t' "${file#"$ROOT_DIR/"}"
        cksum "$file"
    done
}

signal_process_tree() {
    local pid="$1"
    local signal="$2"
    local child
    local children

    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        signal_process_tree "$child" "$signal"
    done
    kill -"$signal" "$pid" 2>/dev/null || true
}

wait_for_process_exit() {
    local pid="$1"
    local attempt

    for attempt in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

stop_app() {
    local pid="$app_pid"

    if [[ -z "$pid" ]]; then
        return
    fi

    if kill -0 "$pid" 2>/dev/null; then
        printf '\nStopping app (pid %s)…\n' "$pid"
        signal_process_tree "$pid" INT
        if ! wait_for_process_exit "$pid"; then
            signal_process_tree "$pid" TERM
        fi
        if ! wait_for_process_exit "$pid"; then
            signal_process_tree "$pid" KILL
        fi
    fi

    wait "$pid" 2>/dev/null || true
    app_pid=""
}

start_app() {
    printf '\nStarting compiled app…\n'
    "$ROOT_DIR/target/debug/live-assistant" &
    app_pid=$!
}

build_until_current() {
    local current_snapshot

    while true; do
        printf '\nCompiling changed sources while the current app keeps running…\n'
        if ! cargo build; then
            if [[ -n "$app_pid" ]]; then
                printf '\nCompilation failed; keeping the current app running and waiting for the next source change.\n' >&2
            else
                printf '\nCompilation failed; no app is running. Waiting for the next source change.\n' >&2
            fi
            return 1
        fi

        current_snapshot="$(snapshot)"
        if [[ "$current_snapshot" == "$last_snapshot" ]]; then
            return 0
        fi

        last_snapshot="$current_snapshot"
        printf 'Another change arrived during compilation; waiting %ss more before compiling again…\n' \
            "$DEBOUNCE_SECONDS"
        wait_until_quiet
    done
}

wait_until_quiet() {
    local current_snapshot

    printf 'Waiting %ss for source changes to settle…\n' "$DEBOUNCE_SECONDS"
    while true; do
        sleep "$DEBOUNCE_SECONDS"
        current_snapshot="$(snapshot)"
        if [[ "$current_snapshot" == "$last_snapshot" ]]; then
            return
        fi

        last_snapshot="$current_snapshot"
        printf 'Another change arrived during the debounce window; waiting %ss more…\n' \
            "$DEBOUNCE_SECONDS"
    done
}

cleanup() {
    trap - EXIT INT TERM
    stop_app
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

last_snapshot="$(snapshot)"
if build_until_current; then
    start_app
else
    printf 'Initial compilation failed; dev.sh will keep watching for changes.\n' >&2
fi

while true; do
    if [[ -n "$app_pid" ]] && ! kill -0 "$app_pid" 2>/dev/null; then
        wait "$app_pid" 2>/dev/null || true
        printf '\nApp exited; waiting for the next source change.\n'
        app_pid=""
    fi

    current_snapshot="$(snapshot)"
    if [[ "$current_snapshot" != "$last_snapshot" ]]; then
        last_snapshot="$current_snapshot"
        wait_until_quiet
        if build_until_current; then
            stop_app
            start_app
        fi
    fi
    sleep "$POLL_INTERVAL_SECONDS"
done

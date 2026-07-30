use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_COMMAND_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct ScreenContext {
    pub screenshot_width: u32,
    pub screenshot_height: u32,
}

impl Default for ScreenContext {
    fn default() -> Self {
        Self {
            screenshot_width: 1440,
            screenshot_height: 900,
        }
    }
}

pub fn execute_with_context(name: &str, arguments: &str, screen_context: ScreenContext) -> String {
    let result = execute_inner(name, arguments, screen_context)
        .unwrap_or_else(|error| json!({"ok": false, "error": format!("{error:#}")}));
    let result = apply_assistant_reply_policy(name, result);
    serde_json::to_string(&result).unwrap_or_else(|error| {
        format!(r#"{{"ok":false,"error":"Could not encode tool result: {error}"}}"#)
    })
}

fn is_pointer_tool(name: &str) -> bool {
    matches!(name, "move_pointer" | "click_screen")
}

fn apply_assistant_reply_policy(name: &str, mut result: Value) -> Value {
    if !is_pointer_tool(name) {
        return result;
    }
    let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if let Some(object) = result.as_object_mut() {
        if ok {
            object.insert(
                "assistant_reply".to_owned(),
                Value::String("Done".to_owned()),
            );
            object.insert("assistant_reply_exact".to_owned(), Value::Bool(true));
            object.insert("speak_before_tool".to_owned(), Value::Bool(false));
        } else {
            object.insert(
                "assistant_reply_policy".to_owned(),
                Value::String(
                    "Briefly report the real pointer-tool failure; do not say Done".to_owned(),
                ),
            );
        }
    }
    result
}

fn execute_inner(name: &str, arguments: &str, screen_context: ScreenContext) -> Result<Value> {
    match name {
        "move_pointer" => move_pointer(arguments, screen_context),
        "click_screen" => click_screen(arguments, screen_context),
        "run_bash" => run_bash(arguments),
        "insert_text" => insert_text(arguments),
        _ => bail!("Unknown tool: {name}"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClickArgs {
    x: i32,
    y: i32,
}

fn move_pointer(arguments: &str, screen_context: ScreenContext) -> Result<Value> {
    let args: ClickArgs =
        serde_json::from_str(arguments).context("Invalid move_pointer arguments")?;
    let screens = screenshots::Screen::all().context("Could not enumerate displays")?;
    let screen = screens
        .into_iter()
        .find(|screen| screen.display_info.is_primary)
        .context("No primary display was found")?;
    let info = screen.display_info;
    let (global_x, global_y) = map_click_to_global(
        args.x,
        args.y,
        screen_context,
        info.x,
        info.y,
        info.width,
        info.height,
    )?;
    let (actual_x, actual_y) = post_native_move(global_x, global_y).context(
        "Pointer movement failed. Enable Accessibility permission for Live Assistant (or the terminal running cargo) in System Settings → Privacy & Security → Accessibility",
    )?;
    anyhow::ensure!(
        (actual_x - global_x as f64).abs() <= 2.0 && (actual_y - global_y as f64).abs() <= 2.0,
        "macOS moved the pointer to ({actual_x:.1}, {actual_y:.1}) instead of the requested ({global_x}, {global_y})"
    );
    Ok(json!({
        "ok": true,
        "x": args.x,
        "y": args.y,
        "global_x": global_x,
        "global_y": global_y,
        "actual_pointer_x": actual_x,
        "actual_pointer_y": actual_y
    }))
}

fn click_screen(arguments: &str, screen_context: ScreenContext) -> Result<Value> {
    let args: ClickArgs =
        serde_json::from_str(arguments).context("Invalid click_screen arguments")?;
    let screens = screenshots::Screen::all().context("Could not enumerate displays")?;
    let screen = screens
        .into_iter()
        .find(|screen| screen.display_info.is_primary)
        .context("No primary display was found")?;
    let info = screen.display_info;
    let (global_x, global_y) = map_click_to_global(
        args.x,
        args.y,
        screen_context,
        info.x,
        info.y,
        info.width,
        info.height,
    )?;
    let screen_x = global_x - info.x;
    let screen_y = global_y - info.y;
    let (actual_x, actual_y) = post_native_click(global_x, global_y).context(
        "Screen click failed. Enable Accessibility permission for Live Assistant (or the terminal \
         running cargo) in System Settings → Privacy & Security → Accessibility",
    )?;
    anyhow::ensure!(
        (actual_x - global_x as f64).abs() <= 2.0 && (actual_y - global_y as f64).abs() <= 2.0,
        "macOS moved the pointer to ({actual_x:.1}, {actual_y:.1}) instead of the requested \
         ({global_x}, {global_y})"
    );

    Ok(json!({
        "ok": true,
        "x": args.x,
        "y": args.y,
        "screen_x": screen_x,
        "screen_y": screen_y,
        "screenshot_width": screen_context.screenshot_width,
        "screenshot_height": screen_context.screenshot_height,
        "global_x": global_x,
        "global_y": global_y,
        "actual_pointer_x": actual_x,
        "actual_pointer_y": actual_y
    }))
}

fn map_click_to_global(
    x: i32,
    y: i32,
    screen_context: ScreenContext,
    display_x: i32,
    display_y: i32,
    display_width: u32,
    display_height: u32,
) -> Result<(i32, i32)> {
    anyhow::ensure!(
        screen_context.screenshot_width > 0 && screen_context.screenshot_height > 0,
        "Screenshot dimensions are invalid"
    );
    anyhow::ensure!(
        x >= 0 && x < screen_context.screenshot_width as i32,
        "x must be between 0 and {}",
        screen_context.screenshot_width.saturating_sub(1)
    );
    anyhow::ensure!(
        y >= 0 && y < screen_context.screenshot_height as i32,
        "y must be between 0 and {}",
        screen_context.screenshot_height.saturating_sub(1)
    );

    anyhow::ensure!(
        display_width == screen_context.screenshot_width
            && display_height == screen_context.screenshot_height,
        "The screen changed from {} × {} to {} × {} after the screenshot was sent; take a new \
         screenshot before clicking",
        screen_context.screenshot_width,
        screen_context.screenshot_height,
        display_width,
        display_height
    );

    // Current-screen images are encoded at the display's logical dimensions,
    // which is also the coordinate space CoreGraphics uses for mouse events.
    // Keeping this 1:1 avoids accidental double application of Retina scaling.
    Ok((display_x + x, display_y + y))
}

#[cfg(target_os = "macos")]
fn post_native_move(x: i32, y: i32) -> Result<(f64, f64)> {
    use core_graphics::{
        event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton},
        event_source::{CGEventSource, CGEventSourceStateID},
        geometry::CGPoint,
    };

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }

    if !unsafe { AXIsProcessTrusted() } {
        bail!("macOS has not granted Accessibility access to this process");
    }
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|()| anyhow::anyhow!("Could not create a macOS HID event source"))?;
    CGEvent::new_mouse_event(
        source,
        CGEventType::MouseMoved,
        CGPoint::new(x as f64, y as f64),
        CGMouseButton::Left,
    )
    .map_err(|()| anyhow::anyhow!("Could not create a macOS pointer event"))?
    .post(CGEventTapLocation::HID);
    thread::sleep(Duration::from_millis(20));
    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .map_err(|()| anyhow::anyhow!("Could not verify the macOS pointer position"))?;
    let location = CGEvent::new(source)
        .map_err(|()| anyhow::anyhow!("Could not read the macOS pointer position"))?
        .location();
    Ok((location.x, location.y))
}

#[cfg(not(target_os = "macos"))]
fn post_native_move(_x: i32, _y: i32) -> Result<(f64, f64)> {
    bail!("Pointer movement currently requires macOS")
}

#[cfg(target_os = "macos")]
fn post_native_click(x: i32, y: i32) -> Result<(f64, f64)> {
    use core_graphics::{
        event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton},
        event_source::{CGEventSource, CGEventSourceStateID},
        geometry::CGPoint,
    };

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }

    if !unsafe { AXIsProcessTrusted() } {
        bail!("macOS has not granted Accessibility access to this process");
    }

    let point = CGPoint::new(x as f64, y as f64);
    let make_event = |event_type| -> Result<CGEvent> {
        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|()| anyhow::anyhow!("Could not create a macOS HID event source"))?;
        CGEvent::new_mouse_event(source, event_type, point, CGMouseButton::Left)
            .map_err(|()| anyhow::anyhow!("Could not create a macOS mouse event"))
    };

    // Posting native HID events from this process is more reliable than asking
    // System Events/osascript to click on behalf of an unbundled Cargo binary.
    make_event(CGEventType::MouseMoved)?.post(CGEventTapLocation::HID);
    thread::sleep(Duration::from_millis(15));
    make_event(CGEventType::LeftMouseDown)?.post(CGEventTapLocation::HID);
    thread::sleep(Duration::from_millis(35));
    make_event(CGEventType::LeftMouseUp)?.post(CGEventTapLocation::HID);
    thread::sleep(Duration::from_millis(10));
    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .map_err(|()| anyhow::anyhow!("Could not verify the macOS pointer position"))?;
    let location = CGEvent::new(source)
        .map_err(|()| anyhow::anyhow!("Could not read the macOS pointer position"))?
        .location();
    Ok((location.x, location.y))
}

#[cfg(not(target_os = "macos"))]
fn post_native_click(_x: i32, _y: i32) -> Result<(f64, f64)> {
    bail!("Screen clicking currently requires macOS")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InsertTextArgs {
    text: String,
}

fn insert_text(arguments: &str) -> Result<Value> {
    let args: InsertTextArgs =
        serde_json::from_str(arguments).context("Invalid insert_text arguments")?;
    anyhow::ensure!(!args.text.is_empty(), "text cannot be empty");
    anyhow::ensure!(
        args.text.len() <= MAX_COMMAND_BYTES,
        "text is too large (maximum {MAX_COMMAND_BYTES} UTF-8 bytes)"
    );
    run_osascript(
        r#"on run argv
set textToInsert to item 1 of argv
tell application "System Events" to keystroke textToInsert
end run"#,
        std::slice::from_ref(&args.text),
    )
    .context(
        "Text insertion failed. Enable Accessibility permission for Live Assistant in System Settings",
    )?;

    Ok(json!({"ok": true, "inserted_characters": args.text.chars().count()}))
}

fn run_osascript(script: &str, arguments: &[String]) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (script, arguments);
        bail!("Screen interaction tools currently require macOS");
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("/usr/bin/osascript")
            .arg("-e")
            .arg(script)
            .arg("--")
            .args(arguments)
            .output()
            .context("Could not launch macOS System Events")?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            bail!(
                "{}",
                if detail.is_empty() {
                    "macOS System Events rejected the action".to_owned()
                } else {
                    detail
                }
            );
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BashArgs {
    command: String,
}

fn run_bash(arguments: &str) -> Result<Value> {
    let args: BashArgs = serde_json::from_str(arguments).context("Invalid run_bash arguments")?;
    anyhow::ensure!(!args.command.trim().is_empty(), "command cannot be empty");
    anyhow::ensure!(
        args.command.len() <= MAX_COMMAND_BYTES,
        "command is too large (maximum {MAX_COMMAND_BYTES} UTF-8 bytes)"
    );

    let mut command = Command::new("/bin/bash");
    command
        .arg("-lc")
        .arg(&args.command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let mut child = command.spawn().context("Could not start /bin/bash")?;
    let child_id = child.id();
    let stdout = child.stdout.take().context("Could not capture stdout")?;
    let stderr = child.stderr.take().context("Could not capture stderr")?;
    let stdout_reader = thread::spawn(move || read_capped(stdout));
    let stderr_reader = thread::spawn(move || read_capped(stderr));

    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child
            .try_wait()
            .context("Could not wait for Bash command")?
        {
            break (status, false);
        }
        if started.elapsed() >= COMMAND_TIMEOUT {
            terminate_process_group(child_id);
            let status = child
                .wait()
                .context("Could not collect timed-out Bash command")?;
            break (status, true);
        }
        thread::sleep(Duration::from_millis(20));
    };

    let (stdout, stdout_truncated) = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("stdout reader thread failed"))??;
    let (stderr, stderr_truncated) = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("stderr reader thread failed"))??;
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_default();

    Ok(json!({
        "ok": status.success() && !timed_out,
        "exit_code": status.code(),
        "timed_out": timed_out,
        "timeout_seconds": COMMAND_TIMEOUT.as_secs(),
        "cwd": cwd,
        "stdout": String::from_utf8_lossy(&stdout),
        "stderr": String::from_utf8_lossy(&stderr),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated
    }))
}

fn read_capped(mut reader: impl Read) -> Result<(Vec<u8>, bool)> {
    let mut saved = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader
            .read(&mut buffer)
            .context("Could not read command output")?;
        if count == 0 {
            break;
        }
        let remaining = MAX_OUTPUT_BYTES.saturating_sub(saved.len());
        let keep = remaining.min(count);
        saved
            .write_all(&buffer[..keep])
            .context("Could not buffer command output")?;
        truncated |= keep < count;
    }
    Ok((saved, truncated))
}

fn terminate_process_group(child_id: u32) {
    #[cfg(unix)]
    {
        // The Bash child starts a fresh process group, so this also terminates
        // any descendants that still hold the stdout/stderr pipes open.
        unsafe {
            libc::kill(-(child_id as i32), libc::SIGKILL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ScreenContext, apply_assistant_reply_policy, execute_with_context, map_click_to_global,
    };
    use serde_json::{Value, json};

    #[test]
    fn bash_tool_returns_output_and_exit_code() {
        let raw = execute_with_context(
            "run_bash",
            r#"{"command":"printf tool-ok"}"#,
            ScreenContext::default(),
        );
        let result: Value = serde_json::from_str(&raw).expect("valid tool JSON");
        assert_eq!(result["ok"], true);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "tool-ok");
    }

    #[test]
    fn unknown_tool_returns_structured_error() {
        let raw = execute_with_context("not_a_tool", "{}", ScreenContext::default());
        let result: Value = serde_json::from_str(&raw).expect("valid tool JSON");
        assert_eq!(result["ok"], false);
        assert!(result["error"].as_str().unwrap().contains("Unknown tool"));
    }

    #[test]
    fn successful_pointer_result_requires_exact_done_reply() {
        let result = apply_assistant_reply_policy("move_pointer", json!({"ok": true}));
        assert_eq!(result["assistant_reply"], "Done");
        assert_eq!(result["assistant_reply_exact"], true);
        assert_eq!(result["speak_before_tool"], false);
    }

    #[test]
    fn failed_pointer_result_forbids_done_reply() {
        let result = apply_assistant_reply_policy(
            "click_screen",
            json!({"ok": false, "error": "permission denied"}),
        );
        assert!(
            result["assistant_reply_policy"]
                .as_str()
                .unwrap()
                .contains("do not say Done")
        );
        assert!(result.get("assistant_reply").is_none());
    }

    #[test]
    fn non_pointer_result_is_not_forced_to_say_done() {
        let result = apply_assistant_reply_policy("run_bash", json!({"ok": true}));
        assert!(result.get("assistant_reply").is_none());
    }

    #[test]
    fn screenshot_coordinates_map_one_to_one_to_macos_points() {
        let context = ScreenContext {
            screenshot_width: 1408,
            screenshot_height: 881,
        };
        assert_eq!(
            map_click_to_global(0, 0, context, 0, 0, 1408, 881).unwrap(),
            (0, 0)
        );
        assert_eq!(
            map_click_to_global(704, 440, context, 0, 0, 1408, 881).unwrap(),
            (704, 440)
        );
        assert_eq!(
            map_click_to_global(1407, 880, context, 0, 0, 1408, 881).unwrap(),
            (1407, 880)
        );
    }

    #[test]
    fn coordinate_mapping_rejects_stale_screen_geometry() {
        let context = ScreenContext {
            screenshot_width: 1440,
            screenshot_height: 900,
        };
        let error = map_click_to_global(700, 440, context, 0, 0, 1408, 881).unwrap_err();
        assert!(error.to_string().contains("screen changed"));
    }
}

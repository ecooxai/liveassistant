mod auth;
mod click_test;
mod codex_account;
mod gpt_live_webrtc;
mod image_generation;
mod live_pointer;
mod media;
mod notes;
mod realtime;
mod resample;
mod tools;
mod web_app;

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

struct SingleInstanceGuard {
    path: PathBuf,
    pid: u32,
}

impl SingleInstanceGuard {
    fn acquire() -> Result<Self, String> {
        let path = std::env::temp_dir().join("live-assistant.pid");
        let pid = std::process::id();
        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{pid}").map_err(|error| error.to_string())?;
                    return Ok(Self { path, pid });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let existing = fs::read_to_string(&path)
                        .ok()
                        .and_then(|text| text.trim().parse::<i32>().ok());
                    if let Some(existing) = existing
                        && existing > 0
                        && unsafe { libc::kill(existing, 0) } == 0
                    {
                        return Err(format!(
                            "Live Assistant is already running as process {existing}."
                        ));
                    }
                    let _ = fs::remove_file(&path);
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Err("Could not acquire the Live Assistant process lock".to_owned())
    }
}

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        let owned = fs::read_to_string(&self.path)
            .ok()
            .is_some_and(|text| text.trim() == self.pid.to_string());
        if owned {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn argument_value(name: &str) -> Option<String> {
    let mut arguments = std::env::args();
    while let Some(argument) = arguments.next() {
        if argument == name {
            return arguments.next();
        }
        if let Some(value) = argument.strip_prefix(&format!("{name}=")) {
            return Some(value.to_owned());
        }
    }
    None
}

fn has_argument(name: &str) -> bool {
    std::env::args().any(|argument| argument == name)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if has_argument("--test-click") {
        click_test::run().map_err(|error| anyhow::anyhow!("{error}"))?;
        return Ok(());
    }
    if has_argument("--test-gpt-live-tool-latency") {
        realtime::probe_codex_gpt_live_tool_latency()?;
        println!("GPT-Live tool latency probe succeeded under the realtime target.");
        return Ok(());
    }
    if has_argument("--test-gpt-live-native") {
        realtime::probe_codex_gpt_live_native()?;
        println!("GPT-Live native platform-ADM probe succeeded.");
        return Ok(());
    }
    if has_argument("--test-gpt-live") {
        realtime::probe_codex_gpt_live()?;
        println!("GPT-Live V3 WebRTC probe succeeded.");
        return Ok(());
    }
    if has_argument("--test-image-upload") {
        realtime::probe_context_image_uploads()?;
        println!(
            "JPEG upload and latest-image ordering probe succeeded for OpenAI Realtime and GPT-Live."
        );
        return Ok(());
    }

    let _instance_guard = match SingleInstanceGuard::acquire() {
        Ok(guard) => guard,
        Err(message) => {
            eprintln!("{message}");
            return Ok(());
        }
    };

    let port = argument_value("--port")
        .or_else(|| std::env::var("LIVE_ASSISTANT_PORT").ok())
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(4317);
    let open_browser = !has_argument("--no-open");
    web_app::run(port, open_browser).await
}

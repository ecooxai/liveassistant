mod app;
mod audio;
mod auth;
mod click_test;
mod codex_account;
mod gpt_live_webrtc;
mod image_generation;
mod live_pointer;
mod media;
mod realtime;
mod resample;
mod tools;

use app::LiveAssistantApp;
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

fn main() -> eframe::Result<()> {
    if std::env::args().any(|argument| argument == "--test-click") {
        return click_test::run();
    }
    if std::env::args().any(|argument| argument == "--test-gpt-live") {
        match realtime::probe_codex_gpt_live() {
            Ok(()) => {
                println!("GPT-Live V3 WebRTC probe succeeded.");
                return Ok(());
            }
            Err(error) => {
                eprintln!("GPT-Live V3 WebRTC probe failed: {error:#}");
                std::process::exit(1);
            }
        }
    }
    if std::env::args().any(|argument| argument == "--test-image-upload") {
        match realtime::probe_context_image_uploads() {
            Ok(()) => {
                println!(
                    "JPEG upload and latest-image ordering probe succeeded for OpenAI Realtime and GPT-Live."
                );
                return Ok(());
            }
            Err(error) => {
                eprintln!("JPEG upload probe failed: {error:#}");
                std::process::exit(1);
            }
        }
    }

    let _instance_guard = match SingleInstanceGuard::acquire() {
        Ok(guard) => guard,
        Err(message) => {
            eprintln!("{message}");
            return Ok(());
        }
    };

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("")
            .with_inner_size([1080.0, 760.0])
            .with_min_inner_size([760.0, 560.0])
            // glow shares one GL config across every viewport, and its alpha
            // channel comes from the main viewport. Without this the pointer
            // overlay window can never be transparent.
            .with_transparent(true),
        ..Default::default()
    };

    eframe::run_native(
        "",
        options,
        Box::new(|cc| Ok(Box::new(LiveAssistantApp::new(cc)))),
    )
}

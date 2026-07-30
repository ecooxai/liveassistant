use crate::{
    media,
    tools::{self, ScreenContext},
};
use eframe::egui;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Default)]
struct TestState {
    clicked: AtomicBool,
    scheduled: AtomicBool,
    expected: Mutex<Option<(i32, i32)>>,
    tool_output: Mutex<Option<String>>,
}

struct ClickTestApp {
    state: Arc<TestState>,
    started: Instant,
}

impl eframe::App for ClickTestApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::light());
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Live Assistant click test");
                ui.label("The app will click the blue target through the real click_screen tool.");
                ui.add_space(34.0);

                let response = ui.add_sized(
                    [28.0, 28.0],
                    egui::Button::new("")
                        .fill(egui::Color32::from_rgb(91, 190, 245))
                        .corner_radius(14.0),
                );
                if response.clicked() {
                    self.state.clicked.store(true, Ordering::SeqCst);
                }

                ui.add_space(24.0);
                ui.label(if self.state.clicked.load(Ordering::SeqCst) {
                    "PASS — target received the native click"
                } else {
                    "Waiting for the native click…"
                });

                let viewport = ctx.input(|input| input.viewport().clone());
                if !self.state.scheduled.load(Ordering::SeqCst)
                    && viewport.focused == Some(true)
                    && let Some(inner_rect) = viewport.inner_rect
                {
                    let local_screen = ctx.screen_rect();
                    let local_target = response.rect.center();
                    let global_target = inner_rect.min + (local_target - local_screen.min);
                    let target_x = global_target.x.round() as i32;
                    let target_y = global_target.y.round() as i32;
                    self.state.scheduled.store(true, Ordering::SeqCst);
                    *self.state.expected.lock().expect("expected target lock") =
                        Some((target_x, target_y));

                    let state = Arc::clone(&self.state);
                    thread::spawn(move || {
                        thread::sleep(Duration::from_millis(350));
                        let output = match media::primary_screen_resolution() {
                            Ok((width, height)) => tools::execute_with_context(
                                "click_screen",
                                &format!(r#"{{"x":{target_x},"y":{target_y}}}"#),
                                ScreenContext {
                                    screenshot_width: width,
                                    screenshot_height: height,
                                },
                            ),
                            Err(error) => {
                                format!(r#"{{"ok":false,"error":"{error:#}"}}"#)
                            }
                        };
                        *state.tool_output.lock().expect("tool output lock") = Some(output);
                    });
                }
            });
        });

        if self.state.clicked.load(Ordering::SeqCst)
            || (self.started.elapsed() > Duration::from_secs(5)
                && self
                    .state
                    .tool_output
                    .lock()
                    .expect("tool output lock")
                    .is_some())
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

pub fn run() -> eframe::Result<()> {
    let state = Arc::new(TestState::default());
    let app_state = Arc::clone(&state);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Live Assistant — Click Test")
            .with_position([360.0, 240.0])
            .with_inner_size([430.0, 250.0])
            .with_resizable(false)
            .with_always_on_top(),
        ..Default::default()
    };

    eframe::run_native(
        "Live Assistant Click Test",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(ClickTestApp {
                state: app_state,
                started: Instant::now(),
            }))
        }),
    )?;

    let expected = state.expected.lock().expect("expected target lock");
    let output = state.tool_output.lock().expect("tool output lock");
    eprintln!("Expected target: {expected:?}");
    eprintln!(
        "click_screen result: {}",
        output.as_deref().unwrap_or("tool did not run")
    );
    assert!(
        state.clicked.load(Ordering::SeqCst),
        "click_screen did not activate the 28 × 28 target at {expected:?}"
    );
    Ok(())
}

use device_query::DeviceState;
use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};
use image::RgbaImage;
use std::time::{Duration, Instant};

pub const POINTER_WINDOW_TITLE: &str = "Live pointer";
pub const COORDINATE_WINDOW_TITLE: &str = "Pointer coordinates";
pub const WINDOW_SIZE: f32 = 27.0;
pub const WINDOW_OFFSET: f32 = 3.0;
pub const COORDINATE_WINDOW_WIDTH: f32 = 190.0;
pub const COORDINATE_WINDOW_HEIGHT: f32 = 42.0;

const CLICK_FLASH_DURATION: Duration = Duration::from_secs(1);
/// The on-screen overlay is 1.5x its previous half-size marker. Screenshots
/// sent to the model keep the existing full-size marker.
const OVERLAY_SCALE: f32 = 0.75;
const POINTER_POINTS: [(f32, f32); 7] = [
    (0.0, 0.0),
    (0.9, 21.5),
    (6.4, 16.4),
    (11.3, 26.5),
    (15.1, 24.7),
    (10.1, 14.7),
    (17.5, 14.2),
];

#[cfg(target_os = "linux")]
const RIGHT_BUTTON_INDEX: usize = 3;
#[cfg(not(target_os = "linux"))]
const RIGHT_BUTTON_INDEX: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Appearance {
    #[default]
    Normal,
    LeftClick,
    RightClick,
}

#[derive(Clone, Copy, Debug)]
pub struct PointerSnapshot {
    pub position: Pos2,
    pub left_down: bool,
    pub right_down: bool,
    pub appearance: Appearance,
}

impl PointerSnapshot {
    pub fn any_button_down(self) -> bool {
        self.left_down || self.right_down
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoordinateOverlay {
    pub window_origin: Pos2,
    pub x: i32,
    pub y: i32,
}

pub struct OverlayState {
    device_state: Option<DeviceState>,
    input_initialization_attempted: bool,
    previous_left_down: bool,
    previous_right_down: bool,
    flash: Option<(Appearance, Instant)>,
    snapshot: Option<PointerSnapshot>,
}

impl OverlayState {
    pub fn new() -> Self {
        Self {
            device_state: None,
            input_initialization_attempted: false,
            previous_left_down: false,
            previous_right_down: false,
            flash: None,
            snapshot: None,
        }
    }

    pub fn poll(&mut self) {
        if !self.input_initialization_attempted {
            self.input_initialization_attempted = true;
            self.device_state = DeviceState::checked_new();
        }

        let (position, left_down, right_down) = match &self.device_state {
            Some(device_state) => {
                let mouse = device_state.query_pointer();
                (
                    Pos2::new(mouse.coords.0 as f32, mouse.coords.1 as f32),
                    button_is_down(&mouse.button_pressed, 1),
                    button_is_down(&mouse.button_pressed, RIGHT_BUTTON_INDEX),
                )
            }
            None => {
                let Some(position) = fallback_position() else {
                    self.snapshot = None;
                    return;
                };
                (position, false, false)
            }
        };

        let now = Instant::now();
        if right_down && !self.previous_right_down {
            self.flash = Some((Appearance::RightClick, now + CLICK_FLASH_DURATION));
        } else if left_down && !self.previous_left_down {
            self.flash = Some((Appearance::LeftClick, now + CLICK_FLASH_DURATION));
        }
        if self.flash.is_some_and(|(_, flash_ends)| now >= flash_ends) {
            self.flash = None;
        }

        self.previous_left_down = left_down;
        self.previous_right_down = right_down;
        self.snapshot = Some(PointerSnapshot {
            position,
            left_down,
            right_down,
            appearance: self
                .flash
                .map(|(appearance, _)| appearance)
                .unwrap_or_default(),
        });
    }

    pub fn snapshot(&self) -> Option<PointerSnapshot> {
        self.snapshot
    }
}

#[cfg(target_os = "macos")]
fn fallback_position() -> Option<Pos2> {
    global_position()
}

#[cfg(not(target_os = "macos"))]
fn fallback_position() -> Option<Pos2> {
    None
}

fn button_is_down(buttons: &[bool], index: usize) -> bool {
    buttons.get(index).copied().unwrap_or(false)
}

#[cfg(target_os = "macos")]
pub fn global_position() -> Option<Pos2> {
    use core_graphics::{
        event::CGEvent,
        event_source::{CGEventSource, CGEventSourceStateID},
    };

    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok()?;
    let location = CGEvent::new(source).ok()?.location();
    Some(Pos2::new(location.x as f32, location.y as f32))
}

#[cfg(not(target_os = "macos"))]
pub fn global_position() -> Option<Pos2> {
    DeviceState::checked_new().map(|device_state| {
        let mouse = device_state.query_pointer();
        Pos2::new(mouse.coords.0 as f32, mouse.coords.1 as f32)
    })
}

pub fn coordinate_overlay(position: Pos2) -> Option<CoordinateOverlay> {
    let screens = screenshots::Screen::all().ok()?;
    let display = screens
        .iter()
        .map(|screen| screen.display_info)
        .find(|display| {
            position.x >= display.x as f32
                && position.y >= display.y as f32
                && position.x < (display.x as f32 + display.width as f32)
                && position.y < (display.y as f32 + display.height as f32)
        })
        .or_else(|| {
            screens
                .iter()
                .map(|screen| screen.display_info)
                .find(|display| display.is_primary)
        })?;

    let margin = 14.0;
    Some(CoordinateOverlay {
        window_origin: Pos2::new(
            display.x as f32 + display.width as f32 - COORDINATE_WINDOW_WIDTH - margin,
            display.y as f32 + display.height as f32 - COORDINATE_WINDOW_HEIGHT - margin,
        ),
        x: (position.x - display.x as f32).round() as i32,
        y: (position.y - display.y as f32).round() as i32,
    })
}

pub fn paint_egui(painter: &egui::Painter, hotspot: Pos2, appearance: Appearance) {
    let points = translated_points(hotspot);
    let (fill, outer, inner) = egui_palette(appearance);
    let shadow_points = points
        .iter()
        .map(|point| *point + Vec2::new(1.1, 1.3) * OVERLAY_SCALE)
        .collect::<Vec<_>>();

    paint_filled_pointer(painter, shadow_points, Color32::from_black_alpha(80));
    painter.add(egui::Shape::closed_line(
        points.clone(),
        Stroke::new(2.4 * OVERLAY_SCALE, outer),
    ));
    paint_filled_pointer(painter, points.clone(), fill);
    painter.add(egui::Shape::closed_line(
        points,
        Stroke::new(0.85 * OVERLAY_SCALE, inner),
    ));
}

fn paint_filled_pointer(painter: &egui::Painter, points: Vec<Pos2>, color: Color32) {
    debug_assert_eq!(points.len(), POINTER_POINTS.len());
    let mut mesh = egui::Mesh::default();
    for point in points {
        mesh.colored_vertex(point, color);
    }
    // Triangulation for the concave arrow polygon.
    for [a, b, c] in [[0, 1, 2], [0, 2, 6], [2, 5, 6], [2, 3, 4], [2, 4, 5]] {
        mesh.add_triangle(a, b, c);
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub fn paint_coordinates(painter: &egui::Painter, x: i32, y: i32) {
    let text = format!("x: {x}   y: {y}");
    let anchor = Pos2::new(
        COORDINATE_WINDOW_WIDTH - 5.0,
        COORDINATE_WINDOW_HEIGHT - 5.0,
    );
    let font = egui::FontId::monospace(17.0);
    painter.text(
        anchor + Vec2::new(1.5, 1.5),
        egui::Align2::RIGHT_BOTTOM,
        &text,
        font.clone(),
        Color32::from_black_alpha(230),
    );
    painter.text(
        anchor,
        egui::Align2::RIGHT_BOTTOM,
        text,
        font,
        Color32::WHITE,
    );
}

pub fn paint_image(image: &mut RgbaImage, hotspot_x: f32, hotspot_y: f32, appearance: Appearance) {
    let points = POINTER_POINTS.map(|(x, y)| (hotspot_x + x, hotspot_y + y));
    let shadow = points.map(|(x, y)| (x + 1.1, y + 1.3));
    let (fill, outer, inner) = image_palette(appearance);

    fill_polygon(image, &shadow, [0, 0, 0], 80);
    draw_polygon_outline(image, &points, outer, 1);
    fill_polygon(image, &points, fill, 255);
    draw_polygon_outline(image, &points, inner, 0);
}

/// AppKit gives borderless windows a rectangular native shadow by default.
/// Removing it is necessary for a pointer-only overlay even when the surface
/// itself is already alpha-transparent.
#[cfg(target_os = "macos")]
pub fn harden_native_transparency(window_title: &str) {
    use objc2_app_kit::{NSApplication, NSColor};
    use objc2_foundation::MainThreadMarker;

    let Some(main_thread) = MainThreadMarker::new() else {
        return;
    };
    let application = NSApplication::sharedApplication(main_thread);
    let windows = application.windows();
    let clear = unsafe { NSColor::clearColor() };

    for index in 0..windows.len() {
        let Some(window) = windows.get(index) else {
            continue;
        };
        if window.title().to_string() == window_title {
            window.setHasShadow(false);
            window.setOpaque(false);
            window.setBackgroundColor(Some(&clear));
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn harden_native_transparency(_window_title: &str) {}

fn translated_points(hotspot: Pos2) -> Vec<Pos2> {
    POINTER_POINTS
        .iter()
        .map(|(x, y)| hotspot + Vec2::new(*x, *y) * OVERLAY_SCALE)
        .collect()
}

fn egui_palette(appearance: Appearance) -> (Color32, Color32, Color32) {
    match appearance {
        Appearance::Normal => (
            Color32::from_rgb(99, 199, 247),
            Color32::WHITE,
            Color32::from_rgb(29, 111, 153),
        ),
        Appearance::LeftClick => (
            Color32::from_rgb(10, 12, 15),
            Color32::WHITE,
            Color32::BLACK,
        ),
        Appearance::RightClick => (
            Color32::WHITE,
            Color32::from_rgb(20, 22, 25),
            Color32::from_rgb(20, 22, 25),
        ),
    }
}

fn image_palette(appearance: Appearance) -> ([u8; 3], [u8; 3], [u8; 3]) {
    match appearance {
        Appearance::Normal => ([99, 199, 247], [255, 255, 255], [29, 111, 153]),
        Appearance::LeftClick => ([10, 12, 15], [255, 255, 255], [0, 0, 0]),
        Appearance::RightClick => ([255, 255, 255], [20, 22, 25], [20, 22, 25]),
    }
}

fn fill_polygon(image: &mut RgbaImage, points: &[(f32, f32)], rgb: [u8; 3], alpha: u8) {
    let min_x = points
        .iter()
        .map(|point| point.0.floor() as i32)
        .min()
        .unwrap_or_default();
    let max_x = points
        .iter()
        .map(|point| point.0.ceil() as i32)
        .max()
        .unwrap_or_default();
    let min_y = points
        .iter()
        .map(|point| point.1.floor() as i32)
        .min()
        .unwrap_or_default();
    let max_y = points
        .iter()
        .map(|point| point.1.ceil() as i32)
        .max()
        .unwrap_or_default();

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, points) {
                blend_if_inside(image, x, y, rgb, alpha);
            }
        }
    }
}

fn draw_polygon_outline(image: &mut RgbaImage, points: &[(f32, f32)], rgb: [u8; 3], radius: i32) {
    for index in 0..points.len() {
        let start = points[index];
        let end = points[(index + 1) % points.len()];
        draw_line(
            image,
            start.0.round() as i32,
            start.1.round() as i32,
            end.0.round() as i32,
            end.1.round() as i32,
            rgb,
            radius,
        );
    }
}

fn point_in_polygon(x: f32, y: f32, points: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut previous = points.len() - 1;
    for current in 0..points.len() {
        let (current_x, current_y) = points[current];
        let (previous_x, previous_y) = points[previous];
        if (current_y > y) != (previous_y > y)
            && x < (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn draw_line(
    image: &mut RgbaImage,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    rgb: [u8; 3],
    radius: i32,
) {
    let dx = (x1 - x0).abs();
    let step_x = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let step_y = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;

    loop {
        for offset_y in -radius..=radius {
            for offset_x in -radius..=radius {
                if offset_x * offset_x + offset_y * offset_y <= radius * radius {
                    blend_if_inside(image, x0 + offset_x, y0 + offset_y, rgb, 255);
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = error * 2;
        if doubled >= dy {
            error += dy;
            x0 += step_x;
        }
        if doubled <= dx {
            error += dx;
            y0 += step_y;
        }
    }
}

fn blend_if_inside(image: &mut RgbaImage, x: i32, y: i32, rgb: [u8; 3], alpha: u8) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        blend_pixel(image, x as u32, y as u32, rgb, alpha);
    }
}

fn blend_pixel(image: &mut RgbaImage, x: u32, y: u32, rgb: [u8; 3], alpha: u8) {
    let pixel = image.get_pixel_mut(x, y);
    let amount = alpha as u16;
    let inverse = 255 - amount;
    for channel in 0..3 {
        pixel[channel] =
            ((rgb[channel] as u16 * amount + pixel[channel] as u16 * inverse) / 255) as u8;
    }
    pixel[3] = 255;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_overlay_is_one_and_a_half_times_previous_size() {
        assert_eq!(OVERLAY_SCALE, 0.75);
        assert_eq!(WINDOW_SIZE, 27.0);
        assert_eq!(WINDOW_OFFSET, 3.0);
        let points = translated_points(Pos2::ZERO);
        assert!((points[1].y - 21.5 * 0.75).abs() < f32::EPSILON);
    }
}

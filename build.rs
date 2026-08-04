fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // libWebRTC uses Objective-C categories from native archives. Keep
        // category symbols alive in the final application link, matching Codex.
        println!("cargo:rustc-link-arg=-ObjC");
        println!("cargo:rustc-link-arg=-lc++");
    }
}

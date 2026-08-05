use anyhow::{Context, Result, bail};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn notes_dir() -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .context("Could not find the home directory")?
        .join("liveassistant");
    fs::create_dir_all(&dir)
        .with_context(|| format!("Could not create notes directory {}", dir.display()))?;
    Ok(dir)
}

fn is_supported_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "md" | "markdown" | "txt" | "text"
            )
        })
        .unwrap_or(false)
}

pub fn list_notes() -> Result<Vec<PathBuf>> {
    let dir = notes_dir()?;
    let mut notes = fs::read_dir(&dir)
        .with_context(|| format!("Could not read notes directory {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_supported_text_file(path))
        .collect::<Vec<_>>();
    notes.sort_by_key(|path| display_name(path).to_ascii_lowercase());
    Ok(notes)
}

pub fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Untitled.md")
        .to_owned()
}

fn normalized_name(name: &str, add_markdown_extension: bool) -> Result<String> {
    let name = name.trim();
    anyhow::ensure!(!name.is_empty(), "Note name cannot be empty");
    anyhow::ensure!(name.len() <= 180, "Note name is too long");
    let path = Path::new(name);
    anyhow::ensure!(
        path.components().count() == 1 && path.file_name().is_some(),
        "Note name must not contain a directory path"
    );
    anyhow::ensure!(name != "." && name != "..", "Invalid note name");
    let mut result = name.to_owned();
    if add_markdown_extension && path.extension().is_none() {
        result.push_str(".md");
    }
    Ok(result)
}

pub fn read_note(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("Could not read note {}", path.display()))
}

pub fn save_note(path: &Path, content: &str) -> Result<()> {
    let dir = notes_dir()?;
    let parent = path.parent().context("Note path has no parent directory")?;
    anyhow::ensure!(parent == dir, "Notes must be saved in {}", dir.display());
    fs::write(path, content).with_context(|| format!("Could not save note {}", path.display()))
}

pub fn create_unique_note(base_name: &str, content: &str) -> Result<PathBuf> {
    let normalized = normalized_name(base_name, true)?;
    let requested = Path::new(&normalized);
    let stem = requested
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("Untitled");
    let extension = requested
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("md");
    let dir = notes_dir()?;
    for index in 1..10_000 {
        let name = if index == 1 {
            format!("{stem}.{extension}")
        } else {
            format!("{stem} {index}.{extension}")
        };
        let path = dir.join(name);
        if !path.exists() {
            fs::write(&path, content)
                .with_context(|| format!("Could not create note {}", path.display()))?;
            return Ok(path);
        }
    }
    bail!("Could not find an available note name")
}

pub fn import_text_file(source: &Path) -> Result<PathBuf> {
    anyhow::ensure!(source.is_file(), "Selected path is not a file");
    anyhow::ensure!(
        is_supported_text_file(source),
        "Select a Markdown or text file"
    );
    let content = fs::read_to_string(source)
        .with_context(|| format!("Could not open text file {}", source.display()))?;
    let name = source
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Imported.md");
    create_unique_note(name, &content)
}

pub fn rename_note(path: &Path, new_name: &str) -> Result<PathBuf> {
    let dir = notes_dir()?;
    anyhow::ensure!(
        path.parent() == Some(dir.as_path()),
        "Only workspace notes can be renamed"
    );
    let new_name = normalized_name(new_name, true)?;
    let destination = dir.join(new_name);
    if destination == path {
        return Ok(destination);
    }
    anyhow::ensure!(
        !destination.exists(),
        "A note named {} already exists",
        display_name(&destination)
    );
    fs::rename(path, &destination).with_context(|| {
        format!(
            "Could not rename note {} to {}",
            path.display(),
            destination.display()
        )
    })?;
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::normalized_name;

    #[test]
    fn markdown_extension_is_added_for_new_notes() {
        assert_eq!(normalized_name("Ideas", true).unwrap(), "Ideas.md");
        assert_eq!(normalized_name("Ideas.txt", true).unwrap(), "Ideas.txt");
    }

    #[test]
    fn directory_traversal_is_rejected() {
        assert!(normalized_name("../secret.md", true).is_err());
        assert!(normalized_name("folder/note.md", true).is_err());
    }
}

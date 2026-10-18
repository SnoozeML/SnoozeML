//! Hugging Face Hub model repository downloader for SnoozeML.
//!
//! Downloads models directly from the Hugging Face Hub using the official `hf-hub` client
//! into the configured models directory (`./models` or environment variable `SNOOZE_MODELS_DIR`).

use crate::config::ModelDirectoryConfig;
use hf_hub::{split_id, HFClient};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Configuration options for downloading a Hugging Face repository.
#[derive(Debug, Clone)]
pub struct HfDownloadOptions {
    /// Repository identifier on Hugging Face (e.g. `"openai-community/gpt2"`, `"sentence-transformers/all-MiniLM-L6-v2"`).
    pub repo_id: String,

    /// Root models directory path where the model repository will be saved.
    pub models_dir: PathBuf,

    /// Optional custom destination subfolder name within the models directory.
    pub folder: Option<String>,

    /// Optional git revision, branch, or tag (defaults to Hub default, usually `"main"`).
    pub revision: Option<String>,

    /// Optional file glob patterns to include in the snapshot download (e.g. `vec!["*.onnx".into()]`).
    pub include: Option<Vec<String>>,

    /// Optional file glob patterns to exclude from the snapshot download.
    pub exclude: Option<Vec<String>>,

    /// Optional Hugging Face authorization token (defaults to `HF_TOKEN` environment variable).
    pub token: Option<String>,

    /// Whether to skip automatically initializing or updating `models.toml` after download.
    pub no_init: bool,
}

/// Computes the target folder path where repository files will be downloaded.
///
/// If `custom_folder` is provided, it is joined to `models_dir`. Otherwise, the repository
/// name extracted from `repo_id` (e.g. `"all-MiniLM-L6-v2"` from `"sentence-transformers/all-MiniLM-L6-v2"`)
/// is used.
#[must_use]
pub fn determine_target_dir(
    models_dir: &Path,
    repo_id: &str,
    custom_folder: Option<&str>,
) -> PathBuf {
    if let Some(folder) = custom_folder {
        let trimmed = folder.trim();
        if !trimmed.is_empty() {
            return models_dir.join(trimmed);
        }
    }

    let (_owner, name) = split_id(repo_id);
    let folder_name = if name.is_empty() {
        repo_id.replace('/', "--")
    } else {
        name.to_string()
    };
    models_dir.join(folder_name)
}

/// Executes the download of a Hugging Face model repository.
///
/// # Errors
/// Returns an error if the Hugging Face client cannot be constructed, network requests fail,
/// or writing to the target directory fails.
pub async fn run_download(
    options: HfDownloadOptions,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let target_dir = determine_target_dir(
        &options.models_dir,
        &options.repo_id,
        options.folder.as_deref(),
    );

    println!("========================================================================");
    println!("  SnoozeML Hugging Face Model Downloader");
    println!("========================================================================\n");
    println!("  Repository:       {}", options.repo_id);
    println!("  Destination Dir:  {}", target_dir.display());
    println!("  Models Root:      {}", options.models_dir.display());
    if let Some(rev) = &options.revision {
        println!("  Revision:         {rev}");
    }
    if let Some(inc) = &options.include {
        println!("  Include Filters:  {}", inc.join(", "));
    }
    if let Some(exc) = &options.exclude {
        println!("  Exclude Filters:  {}", exc.join(", "));
    }
    println!("\n  Connecting to Hugging Face Hub and downloading snapshot...");

    tokio::fs::create_dir_all(&target_dir).await.map_err(|e| {
        format!(
            "Failed to create target destination directory '{}': {e}",
            target_dir.display()
        )
    })?;

    let client = if let Some(token) = &options.token {
        HFClient::builder()
            .token(token)
            .build()
            .map_err(|e| format!("Failed to initialize Hugging Face client with token: {e}"))?
    } else {
        HFClient::new().map_err(|e| format!("Failed to initialize Hugging Face client: {e}"))?
    };

    let (owner, name) = split_id(&options.repo_id);
    let repo = client.model(owner, name);

    let start_time = Instant::now();
    let downloaded_path = repo
        .snapshot_download()
        .local_dir(&target_dir)
        .maybe_revision(options.revision)
        .maybe_allow_patterns(options.include)
        .maybe_ignore_patterns(options.exclude)
        .send()
        .await
        .map_err(|e| {
            format!(
                "Failed to download Hugging Face repository '{}': {e}",
                options.repo_id
            )
        })?;
    let duration = start_time.elapsed();

    println!(
        "\n  Successfully downloaded snapshot in {:.2}s!",
        duration.as_secs_f64()
    );

    // Discover any model files downloaded in target directory
    if let Ok(discovered) = ModelDirectoryConfig::discover_models_in_dir(&options.models_dir).await
    {
        let matching: Vec<_> = discovered
            .into_iter()
            .filter(|m| m.absolute_path.starts_with(&target_dir))
            .collect();

        if !matching.is_empty() {
            println!("\n  Discovered Model Weights:");
            for m in &matching {
                println!("    * ID: {:<20} Path: {}", m.id, m.relative_path);
            }
        }
    }

    if !options.no_init {
        match ModelDirectoryConfig::init_or_update_dir(&options.models_dir, None).await {
            Ok((config_path, added_count)) => {
                if added_count > 0 {
                    println!(
                        "\n  Automatically configured {} new model(s) in '{}'.",
                        added_count,
                        config_path.display()
                    );
                } else {
                    println!(
                        "\n  Configuration file '{}' is up to date.",
                        config_path.display()
                    );
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "Failed to initialize/update models configuration file");
            }
        }
    }

    println!("========================================================================");
    Ok(downloaded_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_target_dir_default_repo_name() {
        let base = PathBuf::from("./models");
        let target = determine_target_dir(&base, "sentence-transformers/all-MiniLM-L6-v2", None);
        assert_eq!(target, PathBuf::from("./models/all-MiniLM-L6-v2"));
    }

    #[test]
    fn test_determine_target_dir_root_model() {
        let base = PathBuf::from("./models");
        let target = determine_target_dir(&base, "gpt2", None);
        assert_eq!(target, PathBuf::from("./models/gpt2"));
    }

    #[test]
    fn test_determine_target_dir_custom_folder() {
        let base = PathBuf::from("./models");
        let target = determine_target_dir(
            &base,
            "sentence-transformers/all-MiniLM-L6-v2",
            Some("my-custom-model"),
        );
        assert_eq!(target, PathBuf::from("./models/my-custom-model"));
    }
}

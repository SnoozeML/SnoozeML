//! Integration and behavioral unit tests for the Terminal User Interface (TUI) dashboard.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use snooze_ml::cli::tui::{render_dashboard, render_ui, AppState, MetricsSummary, Theme};
use snooze_ml::types::{ModelMetadata, ModelState};
use std::time::Duration;

#[test]
fn test_theme_cycling() {
    assert_eq!(Theme::default(), Theme::Dracula);

    let mut theme = Theme::Ocean;
    assert_eq!(theme, Theme::Ocean);

    theme = theme.next();
    assert_eq!(theme, Theme::Dracula);

    theme = theme.next();
    assert_eq!(theme, Theme::Catppuccin);

    theme = theme.next();
    assert_eq!(theme, Theme::Nord);

    theme = theme.next();
    assert_eq!(theme, Theme::Cyberpunk);

    theme = theme.next();
    assert_eq!(theme, Theme::Monochrome);

    theme = theme.next();
    assert_eq!(theme, Theme::Ocean);

    theme = theme.previous();
    assert_eq!(theme, Theme::Monochrome);
}

#[test]
fn test_metrics_summary_parsing() {
    let raw = "\
# HELP snoozeml_requests_total Total requests
# TYPE snoozeml_requests_total counter
snoozeml_requests_total{status=\"all\"} 42
snoozeml_requests_total{status=\"success\"} 40
snoozeml_requests_total{status=\"error\"} 2
snoozeml_inferences_total 15
";
    let summary = MetricsSummary::parse(raw);
    assert_eq!(summary.total_requests, 42);
    assert_eq!(summary.success_requests, 40);
    assert_eq!(summary.failed_requests, 2);
    assert_eq!(summary.total_inferences, 15);
    assert_eq!(summary.raw_lines.len(), 4);
}

#[test]
fn test_app_state_navigation() {
    let mut state = AppState::new("http://localhost:8080".to_string(), Duration::from_secs(1));
    assert!(state.selected_model().is_none());

    let models = vec![
        ModelMetadata {
            id: "model-a".to_string(),
            state: ModelState::Ready,
            onnx_path: "path/a.onnx".to_string(),
            idle_timeout_secs: 60,
            max_batch_size: 8,
            max_batch_delay_ms: 10,
            last_accessed_secs_ago: 2,
            description: None,
            pinned: true,
        },
        ModelMetadata {
            id: "model-b".to_string(),
            state: ModelState::Idle,
            onnx_path: "path/b.onnx".to_string(),
            idle_timeout_secs: 120,
            max_batch_size: 16,
            max_batch_delay_ms: 20,
            last_accessed_secs_ago: 100,
            description: None,
            pinned: false,
        },
    ];

    state.update_data(models, "");
    assert_eq!(
        state.selected_model().map(|m| m.id.as_str()),
        Some("model-a")
    );

    state.select_next();
    assert_eq!(
        state.selected_model().map(|m| m.id.as_str()),
        Some("model-b")
    );

    state.select_next(); // Wrap around
    assert_eq!(
        state.selected_model().map(|m| m.id.as_str()),
        Some("model-a")
    );

    state.select_previous(); // Wrap around back
    assert_eq!(
        state.selected_model().map(|m| m.id.as_str()),
        Some("model-b")
    );
}

#[test]
fn test_ratatui_render_in_test_backend() {
    let mut state = AppState::new("http://127.0.0.1:8080".to_string(), Duration::from_secs(1));
    let models = vec![ModelMetadata {
        id: "bert-base".to_string(),
        state: ModelState::Ready,
        onnx_path: "/data/bert.onnx".to_string(),
        idle_timeout_secs: 300,
        max_batch_size: 16,
        max_batch_delay_ms: 10,
        last_accessed_secs_ago: 5,
        description: None,
        pinned: true,
    }];
    state.update_data(models, "snoozeml_requests_total{status=\"all\"} 99\n");

    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| render_ui(f, &mut state)).unwrap();

    let buffer = terminal.backend().buffer();
    let content: String = buffer.content().iter().map(|cell| cell.symbol()).collect();

    assert!(content.contains("SnoozeML"));
    assert!(content.contains("bert-base"));
    assert!(content.contains("Ready"));
    assert!(content.contains("99"));
}

#[test]
fn test_render_dashboard_formatting() {
    let models = vec![ModelMetadata {
        id: "bert-base".to_string(),
        state: ModelState::Ready,
        onnx_path: "/tmp/bert.onnx".to_string(),
        idle_timeout_secs: 300,
        max_batch_size: 16,
        max_batch_delay_ms: 10,
        last_accessed_secs_ago: 5,
        description: None,
        pinned: true,
    }];

    let metrics = "snoozeml_requests_total{status=\"all\"} 100\n";
    let view = render_dashboard("http://127.0.0.1:8080", &models, metrics);
    assert!(view.contains("bert-base"));
    assert!(view.contains("Ready"));
    assert!(view.contains("yes"));
    assert!(view.contains("100"));
}

#[test]
fn test_render_dashboard_empty_models() {
    let view = render_dashboard("http://localhost:8080", &[], "");
    assert!(view.contains("No models registered yet"));
    assert!(view.contains("Host: http://localhost:8080"));
}

#[test]
fn test_render_dashboard_filters_metric_comments() {
    let raw_metrics = "# HELP req Total\n# TYPE req counter\nsnoozeml_requests 42\n\n";
    let view = render_dashboard("http://127.0.0.1:8080", &[], raw_metrics);
    assert!(!view.contains("# HELP"));
    assert!(!view.contains("# TYPE"));
    assert!(view.contains("snoozeml_requests 42"));
}

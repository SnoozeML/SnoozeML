//! Terminal User Interface (TUI) dashboard for live monitoring of model states, batching, and metrics.
//! Powered by `ratatui` and `crossterm`.

use crate::types::{ModelMetadata, ModelState};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
    Frame, Terminal,
};
use std::io::{self, stdout, Stdout};
use std::time::{Duration, Instant};

/// Predefined color themes available for the SnoozeML TUI dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// Default Popular Dracula dark palette.
    #[default]
    Dracula,
    ///  Blue/cyan ocean theme.    
    Ocean,
    /// Catppuccin Mocha pastel palette.
    Catppuccin,
    /// Arctic, north-bluish Nord palette.
    Nord,
    /// High-contrast neon synthwave cyberpunk palette.
    Cyberpunk,
    /// High-contrast monochrome palette.
    Monochrome,
}

impl Theme {
    /// Returns human-readable name of the theme.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ocean => "Ocean",
            Self::Dracula => "Dracula",
            Self::Catppuccin => "Catppuccin",
            Self::Nord => "Nord",
            Self::Cyberpunk => "Cyberpunk",
            Self::Monochrome => "Monochrome",
        }
    }

    /// Cycles forward to the next predefined theme.
    #[must_use]
    pub fn next(&self) -> Self {
        match self {
            Self::Ocean => Self::Dracula,
            Self::Dracula => Self::Catppuccin,
            Self::Catppuccin => Self::Nord,
            Self::Nord => Self::Cyberpunk,
            Self::Cyberpunk => Self::Monochrome,
            Self::Monochrome => Self::Ocean,
        }
    }

    /// Cycles backward to the previous predefined theme.
    #[must_use]
    pub fn previous(&self) -> Self {
        match self {
            Self::Ocean => Self::Monochrome,
            Self::Dracula => Self::Ocean,
            Self::Catppuccin => Self::Dracula,
            Self::Nord => Self::Catppuccin,
            Self::Cyberpunk => Self::Nord,
            Self::Monochrome => Self::Cyberpunk,
        }
    }

    /// Primary accent color (borders, titles, highlights).
    #[must_use]
    pub fn primary(&self) -> Color {
        match self {
            Self::Ocean => Color::Cyan,
            Self::Dracula => Color::Rgb(189, 147, 249), // Purple #bd93f9
            Self::Catppuccin => Color::Rgb(203, 166, 247), // Mauve #cba6f7
            Self::Nord => Color::Rgb(136, 192, 208),    // Frost blue #88c0d0
            Self::Cyberpunk => Color::Rgb(255, 0, 128), // Neon pink #ff0080
            Self::Monochrome => Color::White,
        }
    }

    /// Secondary accent color (subheadings, table headers, endpoints).
    #[must_use]
    pub fn secondary(&self) -> Color {
        match self {
            Self::Ocean => Color::Yellow,
            Self::Dracula => Color::Rgb(255, 121, 198), // Pink #ff79c6
            Self::Catppuccin => Color::Rgb(137, 180, 250), // Blue #89b4fa
            Self::Nord => Color::Rgb(129, 161, 193),    // Frost #81a1c1
            Self::Cyberpunk => Color::Rgb(0, 255, 255), // Neon cyan #00ffff
            Self::Monochrome => Color::Gray,
        }
    }

    /// Border stroke color.
    #[must_use]
    pub fn border(&self) -> Color {
        match self {
            Self::Ocean => Color::Cyan,
            Self::Dracula => Color::Rgb(189, 147, 249),
            Self::Catppuccin => Color::Rgb(180, 190, 254), // Lavender #b4befe
            Self::Nord => Color::Rgb(136, 192, 208),
            Self::Cyberpunk => Color::Rgb(255, 0, 128),
            Self::Monochrome => Color::White,
        }
    }

    /// Row selection background highlight color.
    #[must_use]
    pub fn highlight_bg(&self) -> Color {
        match self {
            Self::Ocean => Color::Rgb(30, 60, 100),
            Self::Dracula => Color::Rgb(68, 71, 90), // Selection #44475a
            Self::Catppuccin => Color::Rgb(69, 71, 90), // Surface1 #45475a
            Self::Nord => Color::Rgb(67, 76, 94),    // Polar night #434c5e
            Self::Cyberpunk => Color::Rgb(80, 0, 80), // Deep purple
            Self::Monochrome => Color::White,
        }
    }

    /// Row selection foreground text color.
    #[must_use]
    pub fn highlight_fg(&self) -> Color {
        match self {
            Self::Ocean | Self::Dracula | Self::Catppuccin | Self::Nord => Color::White,
            Self::Cyberpunk => Color::Rgb(255, 255, 0), // Neon yellow
            Self::Monochrome => Color::Black,
        }
    }

    /// Standard foreground text color.
    #[must_use]
    pub fn text(&self) -> Color {
        match self {
            Self::Ocean => Color::White,
            Self::Dracula => Color::Rgb(248, 248, 242),
            Self::Catppuccin => Color::Rgb(205, 214, 244), // Text #cdd6f4
            Self::Nord => Color::Rgb(236, 239, 244),       // Snow storm
            Self::Cyberpunk => Color::Rgb(240, 240, 255),
            Self::Monochrome => Color::White,
        }
    }

    /// Dimmed / muted text color (labels, timestamps).
    #[must_use]
    pub fn muted(&self) -> Color {
        match self {
            Self::Ocean => Color::DarkGray,
            Self::Dracula => Color::Rgb(98, 114, 164), // Comment #6272a4
            Self::Catppuccin => Color::Rgb(147, 153, 178), // Overlay0 #9399b2
            Self::Nord => Color::Rgb(76, 86, 106),     // Polar night 3 #4c566a
            Self::Cyberpunk => Color::Rgb(120, 80, 140),
            Self::Monochrome => Color::DarkGray,
        }
    }

    /// Ready model status badge color.
    #[must_use]
    pub fn status_ready(&self) -> Color {
        match self {
            Self::Ocean => Color::Green,
            Self::Dracula => Color::Rgb(80, 250, 123), // Green #50fa7b
            Self::Catppuccin => Color::Rgb(166, 227, 161), // Green #a6e3a1
            Self::Nord => Color::Rgb(163, 190, 140),   // Aurora green #a3be8c
            Self::Cyberpunk => Color::Rgb(0, 255, 128), // Neon green #00ff80
            Self::Monochrome => Color::White,
        }
    }

    /// Busy model status badge color.
    #[must_use]
    pub fn status_busy(&self) -> Color {
        match self {
            Self::Ocean => Color::Cyan,
            Self::Dracula => Color::Rgb(139, 233, 253), // Cyan #8be9fd
            Self::Catppuccin => Color::Rgb(148, 226, 213), // Teal #94e2d5
            Self::Nord => Color::Rgb(143, 188, 187),    // Frost teal #8fbcbb
            Self::Cyberpunk => Color::Rgb(0, 255, 255), // Neon cyan #00ffff
            Self::Monochrome => Color::Gray,
        }
    }

    /// Loading model status badge color.
    #[must_use]
    pub fn status_loading(&self) -> Color {
        match self {
            Self::Ocean => Color::Yellow,
            Self::Dracula => Color::Rgb(241, 250, 140), // Yellow #f1fa8c
            Self::Catppuccin => Color::Rgb(249, 226, 175), // Yellow #f9e2af
            Self::Nord => Color::Rgb(235, 203, 139),    // Aurora yellow #ebcb8b
            Self::Cyberpunk => Color::Rgb(255, 170, 0), // Amber #ffaa00
            Self::Monochrome => Color::DarkGray,
        }
    }

    /// Idle model status badge color.
    #[must_use]
    pub fn status_idle(&self) -> Color {
        match self {
            Self::Ocean => Color::DarkGray,
            Self::Dracula => Color::Rgb(98, 114, 164),
            Self::Catppuccin => Color::Rgb(108, 112, 134),
            Self::Nord => Color::Rgb(76, 86, 106),
            Self::Cyberpunk => Color::Rgb(90, 60, 110),
            Self::Monochrome => Color::DarkGray,
        }
    }

    /// Error badge color.
    #[must_use]
    pub fn status_error(&self) -> Color {
        match self {
            Self::Ocean => Color::Red,
            Self::Dracula => Color::Rgb(255, 85, 85), // Red #ff5555
            Self::Catppuccin => Color::Rgb(243, 139, 168), // Red #f38ba8
            Self::Nord => Color::Rgb(191, 97, 106),   // Aurora red #bf616a
            Self::Cyberpunk => Color::Rgb(255, 50, 50),
            Self::Monochrome => Color::White,
        }
    }

    /// Connection badge colors (Background, Foreground).
    #[must_use]
    pub fn badge_connected(&self) -> (Color, Color) {
        match self {
            Self::Ocean => (Color::Green, Color::Black),
            Self::Dracula => (Color::Rgb(80, 250, 123), Color::Black),
            Self::Catppuccin => (Color::Rgb(166, 227, 161), Color::Black),
            Self::Nord => (Color::Rgb(163, 190, 140), Color::Black),
            Self::Cyberpunk => (Color::Rgb(0, 255, 128), Color::Black),
            Self::Monochrome => (Color::White, Color::Black),
        }
    }

    /// Disconnection badge colors (Background, Foreground).
    #[must_use]
    pub fn badge_disconnected(&self) -> (Color, Color) {
        match self {
            Self::Ocean => (Color::Red, Color::White),
            Self::Dracula => (Color::Rgb(255, 85, 85), Color::White),
            Self::Catppuccin => (Color::Rgb(243, 139, 168), Color::Black),
            Self::Nord => (Color::Rgb(191, 97, 106), Color::White),
            Self::Cyberpunk => (Color::Rgb(255, 50, 50), Color::White),
            Self::Monochrome => (Color::DarkGray, Color::White),
        }
    }
}

/// Parsed metrics summary for high-level dashboard display.
#[derive(Debug, Default, Clone)]
pub struct MetricsSummary {
    /// Total incoming requests recorded by the server.
    pub total_requests: u64,
    /// Total successful requests.
    pub success_requests: u64,
    /// Total failed requests.
    pub failed_requests: u64,
    /// Total inference executions completed.
    pub total_inferences: u64,
    /// Total raw key-value lines for metric inspector.
    pub raw_lines: Vec<String>,
}

impl MetricsSummary {
    /// Parses raw Prometheus exposition text into a [`MetricsSummary`].
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let mut summary = Self::default();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            summary.raw_lines.push(trimmed.to_string());

            let mut parts = trimmed.split_whitespace();
            let metric_name = parts.next().unwrap_or_default();
            let metric_value = parts.next().and_then(|v| v.parse::<f64>().ok());

            if let Some(val) = metric_value {
                let count = if val >= 0.0 { val as u64 } else { 0 };
                if metric_name.starts_with("snoozeml_requests_total") {
                    if metric_name.contains("status=\"all\"") || !metric_name.contains("status=") {
                        summary.total_requests = count;
                    } else if metric_name.contains("status=\"success\"") {
                        summary.success_requests = count;
                    } else if metric_name.contains("status=\"error\"") {
                        summary.failed_requests = count;
                    }
                } else if metric_name.starts_with("snoozeml_inferences_total") {
                    summary.total_inferences = count;
                }
            }
        }

        if summary.total_requests == 0
            && (summary.success_requests > 0 || summary.failed_requests > 0)
        {
            summary.total_requests = summary
                .success_requests
                .checked_add(summary.failed_requests)
                .unwrap_or(summary.success_requests);
        }

        summary
    }
}

/// State machine backing the live TUI dashboard.
#[derive(Debug)]
pub struct AppState {
    /// Remote server endpoint being monitored.
    pub endpoint: String,
    /// Polling interval for querying server status.
    pub refresh_interval: Duration,
    /// Active color theme.
    pub theme: Theme,
    /// List of registered models discovered on the remote daemon.
    pub models: Vec<ModelMetadata>,
    /// Summary of Prometheus metrics collected from the server.
    pub metrics: MetricsSummary,
    /// Currently selected index in the models table.
    pub table_state: TableState,
    /// Status message displayed in the notification bar.
    pub status_message: String,
    /// Last error encountered when connecting to the server.
    pub last_error: Option<String>,
    /// Server connection state.
    pub is_connected: bool,
    /// Timestamp of last successful refresh.
    pub last_refresh: Option<Instant>,
    /// Flag signaling TUI exit requested by user.
    pub should_quit: bool,
}

impl AppState {
    /// Creates a new [`AppState`] with initial defaults.
    #[must_use]
    pub fn new(endpoint: String, refresh_interval: Duration) -> Self {
        let mut table_state = TableState::default();
        table_state.select(Some(0));

        Self {
            endpoint,
            refresh_interval,
            theme: Theme::default(),
            models: Vec::new(),
            metrics: MetricsSummary::default(),
            table_state,
            status_message: "Initializing monitor...".to_string(),
            last_error: None,
            is_connected: false,
            last_refresh: None,
            should_quit: false,
        }
    }

    /// Cycles to the next color theme and updates the status message.
    pub fn next_theme(&mut self) {
        self.theme = self.theme.next();
        self.status_message = format!("Color scheme switched to: {}", self.theme.name());
    }

    /// Cycles to the previous color theme and updates the status message.
    pub fn previous_theme(&mut self) {
        self.theme = self.theme.previous();
        self.status_message = format!("Color scheme switched to: {}", self.theme.name());
    }

    /// Moves the table selection cursor down by one row.
    pub fn select_next(&mut self) {
        if self.models.is_empty() {
            self.table_state.select(None);
            return;
        }

        let i = match self.table_state.selected() {
            Some(i) => {
                if i.saturating_add(1) >= self.models.len() {
                    0
                } else {
                    i.saturating_add(1)
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Moves the table selection cursor up by one row.
    pub fn select_previous(&mut self) {
        if self.models.is_empty() {
            self.table_state.select(None);
            return;
        }

        let i = match self.table_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.models.len().saturating_sub(1)
                } else {
                    i.saturating_sub(1)
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Returns a reference to the currently selected model metadata, if any.
    #[must_use]
    pub fn selected_model(&self) -> Option<&ModelMetadata> {
        let index = self.table_state.selected()?;
        self.models.get(index)
    }

    /// Updates the state with freshly polled models and metrics.
    pub fn update_data(&mut self, models: Vec<ModelMetadata>, metrics_raw: &str) {
        self.models = models;
        self.metrics = MetricsSummary::parse(metrics_raw);
        self.is_connected = true;
        self.last_error = None;
        self.last_refresh = Some(Instant::now());

        if self.models.is_empty() {
            self.table_state.select(None);
        } else {
            let cur = self.table_state.selected().unwrap_or(0);
            if cur >= self.models.len() {
                self.table_state
                    .select(Some(self.models.len().saturating_sub(1)));
            } else {
                self.table_state.select(Some(cur));
            }
        }
    }

    /// Records a polling error and updates telemetry indicators.
    pub fn set_error(&mut self, err: String) {
        self.is_connected = false;
        self.last_error = Some(err.clone());
        self.status_message = format!("Error: {err}");
    }
}

/// Renders the full SnoozeML dashboard onto the provided Ratatui frame.
pub fn render_ui(frame: &mut Frame, state: &mut AppState) {
    let size = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Body (Models Table + Details / Metrics)
            Constraint::Length(3), // Footer / Controls
        ])
        .split(size);

    let header_area = *chunks.first().unwrap_or(&Rect::default());
    let body_area = *chunks.get(1).unwrap_or(&Rect::default());
    let footer_area = *chunks.get(2).unwrap_or(&Rect::default());

    render_header(frame, header_area, state);
    render_body(frame, body_area, state);
    render_footer(frame, footer_area, state);
}

/// Renders the title banner, server host info, theme tag, and status indicators.
fn render_header(frame: &mut Frame, area: Rect, state: &AppState) {
    let (badge_bg, badge_fg) = if state.is_connected {
        state.theme.badge_connected()
    } else {
        state.theme.badge_disconnected()
    };

    let status_span = Span::styled(
        if state.is_connected {
            " ● CONNECTED "
        } else {
            " ● DISCONNECTED "
        },
        Style::default()
            .fg(badge_fg)
            .bg(badge_bg)
            .add_modifier(Modifier::BOLD),
    );

    let refresh_str = format!("Interval: {}s", state.refresh_interval.as_secs());
    let last_refresh_str = match state.last_refresh {
        Some(t) => format!("Updated: {}s ago", t.elapsed().as_secs()),
        None => "Updated: never".to_string(),
    };

    let title_line = Line::from(vec![
        Span::styled(
            " SnoozeML ",
            Style::default()
                .fg(state.theme.primary())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "Inference Daemon Dashboard  ",
            Style::default().fg(state.theme.text()),
        ),
        status_span,
        Span::styled("  Endpoint: ", Style::default().fg(state.theme.text())),
        Span::styled(
            &state.endpoint,
            Style::default().fg(state.theme.secondary()),
        ),
        Span::raw("  |  "),
        Span::styled(
            format!("Theme: {}", state.theme.name()),
            Style::default()
                .fg(state.theme.primary())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(refresh_str, Style::default().fg(state.theme.muted())),
        Span::raw("  |  "),
        Span::styled(last_refresh_str, Style::default().fg(state.theme.muted())),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.border()));

    let paragraph = Paragraph::new(title_line).block(block);
    frame.render_widget(paragraph, area);
}

/// Renders the main dashboard body split into model registry table and telemetry inspector.
fn render_body(frame: &mut Frame, area: Rect, state: &mut AppState) {
    let horizontal_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Model table
            Constraint::Percentage(40), // Telemetry & Selected Model info
        ])
        .split(area);

    let left_area = *horizontal_chunks.first().unwrap_or(&Rect::default());
    let right_area = *horizontal_chunks.get(1).unwrap_or(&Rect::default());

    render_models_table(frame, left_area, state);
    render_telemetry_panel(frame, right_area, state);
}

/// Renders the table of registered models with state highlighting and row selection.
fn render_models_table(frame: &mut Frame, area: Rect, state: &mut AppState) {
    let header_cells = [
        "MODEL ID",
        "STATE",
        "PINNED",
        "MAX BATCH",
        "DELAY",
        "LAST ACTIVE",
    ]
    .iter()
    .map(|h| {
        Cell::from(*h).style(
            Style::default()
                .fg(state.theme.secondary())
                .add_modifier(Modifier::BOLD),
        )
    });
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows: Vec<Row> = state
        .models
        .iter()
        .map(|m| {
            let state_style = match m.state {
                ModelState::Ready => Style::default()
                    .fg(state.theme.status_ready())
                    .add_modifier(Modifier::BOLD),
                ModelState::Busy => Style::default()
                    .fg(state.theme.status_busy())
                    .add_modifier(Modifier::BOLD),
                ModelState::Loading => Style::default().fg(state.theme.status_loading()),
                ModelState::Idle => Style::default().fg(state.theme.status_idle()),
            };

            let pin_style = if m.pinned {
                Style::default()
                    .fg(state.theme.primary())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(state.theme.muted())
            };

            let cells = vec![
                Cell::from(m.id.clone()).style(Style::default().fg(state.theme.text())),
                Cell::from(format!("{:?}", m.state)).style(state_style),
                Cell::from(if m.pinned { "yes" } else { "no" }).style(pin_style),
                Cell::from(m.max_batch_size.to_string())
                    .style(Style::default().fg(state.theme.text())),
                Cell::from(format!("{}ms", m.max_batch_delay_ms))
                    .style(Style::default().fg(state.theme.text())),
                Cell::from(format!("{}s", m.last_accessed_secs_ago))
                    .style(Style::default().fg(state.theme.text())),
            ];
            Row::new(cells).height(1)
        })
        .collect();

    let title = format!(" Registered Models ({}) ", state.models.len());
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(28),
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Percentage(15),
            Constraint::Percentage(14),
            Constraint::Percentage(16),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(state.theme.border())),
    )
    .row_highlight_style(
        Style::default()
            .bg(state.theme.highlight_bg())
            .fg(state.theme.highlight_fg())
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("▶ ");

    frame.render_stateful_widget(table, area, &mut state.table_state);
}

/// Renders telemetry panels: selected model inspect card and Prometheus metrics snapshot.
fn render_telemetry_panel(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Selected model details
            Constraint::Min(6),     // Metrics snapshot
        ])
        .split(area);

    let details_area = *chunks.first().unwrap_or(&Rect::default());
    let metrics_area = *chunks.get(1).unwrap_or(&Rect::default());

    // Selected model details
    let details_lines = match state.selected_model() {
        Some(m) => {
            let state_str = format!("{:?}", m.state);
            let desc_str = m.description.as_deref().unwrap_or("None");
            let pin_str = if m.pinned {
                "Yes (exempt from eviction)"
            } else {
                "No"
            };
            vec![
                Line::from(vec![
                    Span::styled("Model ID:       ", Style::default().fg(state.theme.muted())),
                    Span::styled(
                        &m.id,
                        Style::default()
                            .fg(state.theme.primary())
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Description:    ", Style::default().fg(state.theme.muted())),
                    Span::styled(desc_str, Style::default().fg(state.theme.text())),
                ]),
                Line::from(vec![
                    Span::styled("State:          ", Style::default().fg(state.theme.muted())),
                    Span::styled(state_str, Style::default().fg(state.theme.secondary())),
                ]),
                Line::from(vec![
                    Span::styled("Pinned:         ", Style::default().fg(state.theme.muted())),
                    Span::styled(
                        pin_str,
                        if m.pinned {
                            Style::default()
                                .fg(state.theme.primary())
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(state.theme.text())
                        },
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Weights Path:   ", Style::default().fg(state.theme.muted())),
                    Span::styled(&m.onnx_path, Style::default().fg(state.theme.text())),
                ]),
                Line::from(vec![
                    Span::styled("Idle Timeout:   ", Style::default().fg(state.theme.muted())),
                    Span::styled(
                        format!("{}s", m.idle_timeout_secs),
                        Style::default().fg(state.theme.text()),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Micro-Batching: ", Style::default().fg(state.theme.muted())),
                    Span::styled(
                        format!(
                            "Max {} items / {}ms latency window",
                            m.max_batch_size, m.max_batch_delay_ms
                        ),
                        Style::default().fg(state.theme.text()),
                    ),
                ]),
            ]
        }
        None => vec![Line::from(Span::styled(
            "No model selected. Use ↑/↓ to navigate.",
            Style::default().fg(state.theme.muted()),
        ))],
    };

    let details_block = Block::default()
        .borders(Borders::ALL)
        .title(" Selected Model Inspector ")
        .border_style(Style::default().fg(state.theme.border()));
    let details_paragraph = Paragraph::new(details_lines).block(details_block);
    frame.render_widget(details_paragraph, details_area);

    // Metrics snapshot
    let mut metric_lines = vec![
        Line::from(vec![
            Span::styled(
                "Total Requests:       ",
                Style::default().fg(state.theme.muted()),
            ),
            Span::styled(
                state.metrics.total_requests.to_string(),
                Style::default()
                    .fg(state.theme.status_ready())
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "Successful Requests:  ",
                Style::default().fg(state.theme.muted()),
            ),
            Span::styled(
                state.metrics.success_requests.to_string(),
                Style::default().fg(state.theme.status_ready()),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "Failed Requests:      ",
                Style::default().fg(state.theme.muted()),
            ),
            Span::styled(
                state.metrics.failed_requests.to_string(),
                Style::default().fg(state.theme.status_error()),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "Total Batched Runs:   ",
                Style::default().fg(state.theme.muted()),
            ),
            Span::styled(
                state.metrics.total_inferences.to_string(),
                Style::default().fg(state.theme.status_busy()),
            ),
        ]),
        Line::raw(""),
        Line::styled(
            "Telemetry Sample Lines:",
            Style::default()
                .fg(state.theme.muted())
                .add_modifier(Modifier::UNDERLINED),
        ),
    ];

    for line in state.metrics.raw_lines.iter().take(6) {
        metric_lines.push(Line::styled(
            format!("  {line}"),
            Style::default().fg(state.theme.text()),
        ));
    }

    let metrics_block = Block::default()
        .borders(Borders::ALL)
        .title(" Prometheus Metrics Snapshot ")
        .border_style(Style::default().fg(state.theme.border()));
    let metrics_paragraph = Paragraph::new(metric_lines)
        .block(metrics_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(metrics_paragraph, metrics_area);
}

/// Renders the footer with key binding legends and actionable feedback message.
fn render_footer(frame: &mut Frame, area: Rect, state: &AppState) {
    let keybindings = Line::from(vec![
        Span::styled(
            " [q/Esc] ",
            Style::default()
                .bg(state.theme.muted())
                .fg(state.theme.highlight_fg())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Quit  "),
        Span::styled(
            " [r] ",
            Style::default()
                .bg(state.theme.muted())
                .fg(state.theme.highlight_fg())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Refresh  "),
        Span::styled(
            " [t] ",
            Style::default()
                .bg(state.theme.primary())
                .fg(state.theme.highlight_fg())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Theme  "),
        Span::styled(
            " [↑/↓/j/k] ",
            Style::default()
                .bg(state.theme.muted())
                .fg(state.theme.highlight_fg())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Select  "),
        Span::styled(
            " [l] ",
            Style::default()
                .bg(state.theme.status_busy())
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Load  "),
        Span::styled(
            " [u] ",
            Style::default()
                .bg(state.theme.secondary())
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Unload  "),
        Span::styled(
            " [p] ",
            Style::default()
                .bg(state.theme.primary())
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Pin/Unpin  "),
        Span::raw(" | "),
        Span::styled(
            &state.status_message,
            Style::default().fg(state.theme.secondary()),
        ),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.muted()));
    let paragraph = Paragraph::new(keybindings).block(block);
    frame.render_widget(paragraph, area);
}

/// RAII Terminal guard that ensures alternate screen and raw mode are cleaned up cleanly.
pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    /// Initializes crossterm into raw mode and alternate screen, wrapping the terminal.
    ///
    /// # Errors
    /// Returns [`io::Error`] if raw mode or alternate screen setup fails.
    pub fn init() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Borrows the internal mutable [`Terminal`].
    pub fn terminal_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

/// Runs the interactive Ratatui event loop for monitoring SnoozeML.
///
/// # Errors
/// Returns error if terminal setup fails or unrecoverable error occurs.
pub async fn run_tui(endpoint: &str, interval_secs: u64) -> Result<(), Box<dyn std::error::Error>> {
    let refresh_interval = Duration::from_secs(interval_secs.max(1));
    let mut state = AppState::new(endpoint.to_string(), refresh_interval);
    let mut guard = TerminalGuard::init()?;

    let ep = endpoint.to_string();
    // Initial fetch
    if let Ok((models, metrics)) = fetch_remote_data(&ep).await {
        state.update_data(models, &metrics);
        state.status_message = "Connected to daemon.".to_string();
    } else {
        state.set_error("Initial connection failed. Retrying...".to_string());
    }

    let mut last_poll = Instant::now();

    while !state.should_quit {
        // Render UI
        guard.terminal_mut().draw(|f| render_ui(f, &mut state))?;

        // Poll for terminal events with a 50ms budget so the loop remains responsive
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            state.should_quit = true;
                        }
                        KeyCode::Char('c')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            state.should_quit = true;
                        }
                        KeyCode::Char('t') => {
                            state.next_theme();
                        }
                        KeyCode::Char('T') => {
                            state.previous_theme();
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            state.select_next();
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            state.select_previous();
                        }
                        KeyCode::Char('r') => {
                            state.status_message = "Refreshing...".to_string();
                            let ep_clone = state.endpoint.clone();
                            match fetch_remote_data(&ep_clone).await {
                                Ok((models, metrics)) => {
                                    state.update_data(models, &metrics);
                                    state.status_message = "Refreshed.".to_string();
                                }
                                Err(e) => {
                                    state.set_error(e);
                                }
                            }
                            last_poll = Instant::now();
                        }
                        KeyCode::Char('l') => {
                            if let Some(model) = state.selected_model().cloned() {
                                state.status_message = format!("Loading '{}'...", model.id);
                                let ep_clone = state.endpoint.clone();
                                let mid = model.id.clone();
                                let res = tokio::task::spawn_blocking(move || {
                                    let url = format!(
                                        "{}/v1/models/{mid}/load",
                                        ep_clone.trim_end_matches('/')
                                    );
                                    ureq::post(&url).send_empty()
                                })
                                .await;

                                match res {
                                    Ok(Ok(_)) => {
                                        state.status_message =
                                            format!("Loaded model '{}' into memory.", model.id);
                                        // Refresh immediately
                                        if let Ok((models, metrics)) =
                                            fetch_remote_data(&state.endpoint).await
                                        {
                                            state.update_data(models, &metrics);
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        state.status_message =
                                            format!("Failed to load model '{}': {e}", model.id);
                                    }
                                    Err(join_err) => {
                                        state.status_message =
                                            format!("Internal task failure: {join_err}");
                                    }
                                }
                            }
                        }
                        KeyCode::Char('u') => {
                            if let Some(model) = state.selected_model().cloned() {
                                state.status_message = format!("Unloading '{}'...", model.id);
                                let ep_clone = state.endpoint.clone();
                                let mid = model.id.clone();
                                let res = tokio::task::spawn_blocking(move || {
                                    let url = format!(
                                        "{}/v1/models/{mid}/unload",
                                        ep_clone.trim_end_matches('/')
                                    );
                                    ureq::post(&url).send_empty()
                                })
                                .await;

                                match res {
                                    Ok(Ok(_)) => {
                                        state.status_message = format!(
                                            "Unloaded model '{}' (scale to zero).",
                                            model.id
                                        );
                                        // Refresh immediately
                                        if let Ok((models, metrics)) =
                                            fetch_remote_data(&state.endpoint).await
                                        {
                                            state.update_data(models, &metrics);
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        state.status_message =
                                            format!("Failed to unload model '{}': {e}", model.id);
                                    }
                                    Err(join_err) => {
                                        state.status_message =
                                            format!("Internal task failure: {join_err}");
                                    }
                                }
                            }
                        }
                        KeyCode::Char('p') => {
                            if let Some(model) = state.selected_model().cloned() {
                                let action = if model.pinned { "unpin" } else { "pin" };
                                state.status_message = format!(
                                    "{}ning '{}'...",
                                    if model.pinned { "Unpin" } else { "Pin" },
                                    model.id
                                );
                                let ep_clone = state.endpoint.clone();
                                let mid = model.id.clone();
                                let res = tokio::task::spawn_blocking(move || {
                                    let url = format!(
                                        "{}/v1/models/{mid}/{action}",
                                        ep_clone.trim_end_matches('/')
                                    );
                                    ureq::post(&url).send_empty()
                                })
                                .await;

                                match res {
                                    Ok(Ok(_)) => {
                                        state.status_message = format!(
                                            "{}ned model '{}'.",
                                            if model.pinned { "Unpin" } else { "Pin" },
                                            model.id
                                        );
                                        if let Ok((models, metrics)) =
                                            fetch_remote_data(&state.endpoint).await
                                        {
                                            state.update_data(models, &metrics);
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        state.status_message =
                                            format!("Failed to {action} model '{}': {e}", model.id);
                                    }
                                    Err(join_err) => {
                                        state.status_message =
                                            format!("Internal task failure: {join_err}");
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Periodic background poll
        if last_poll.elapsed() >= state.refresh_interval {
            let ep_clone = state.endpoint.clone();
            match fetch_remote_data(&ep_clone).await {
                Ok((models, metrics)) => {
                    state.update_data(models, &metrics);
                }
                Err(e) => {
                    state.set_error(e);
                }
            }
            last_poll = Instant::now();
        }
    }

    Ok(())
}

/// Helper to fetch remote models and metrics concurrently using `ureq` in a worker thread.
async fn fetch_remote_data(endpoint: &str) -> Result<(Vec<ModelMetadata>, String), String> {
    let ep = endpoint.to_string();
    tokio::task::spawn_blocking(move || {
        let models = crate::cli::fetch_models(&ep)?;
        let metrics = crate::cli::fetch_metrics(&ep).unwrap_or_default();
        Ok((models, metrics))
    })
    .await
    .map_err(|e| format!("Task join failed: {e}"))?
}

/// Formats and renders a static fallback dashboard string.
/// Preserved for headless assertions and legacy CLI testing.
#[must_use]
pub fn render_dashboard(endpoint: &str, models: &[ModelMetadata], raw_metrics: &str) -> String {
    let mut out = String::new();
    out.push_str("\x1B[2J\x1B[1;1H");
    out.push_str("========================================================================\n");
    out.push_str(&format!(
        "  SnoozeML ONNX Inference Daemon Monitor | Host: {endpoint}\n"
    ));
    out.push_str("========================================================================\n\n");

    out.push_str("Registered Models:\n");
    out.push_str(&format!(
        "  {:<20} {:<10} {:<8} {:<15} {:<12} {:<16}\n",
        "MODEL ID", "STATE", "PINNED", "MAX BATCH", "DELAY (ms)", "LAST ACTIVE (s)"
    ));
    out.push_str(
        "  ----------------------------------------------------------------------------\n",
    );

    if models.is_empty() {
        out.push_str("  No models registered yet. Deploy models using the CLI or API.\n");
    } else {
        for m in models {
            let state_str = format!("{:?}", m.state);
            let pin_str = if m.pinned { "yes" } else { "no" };
            out.push_str(&format!(
                "  {:<20} {:<10} {:<8} {:<15} {:<12} {:<16}\n",
                m.id,
                state_str,
                pin_str,
                m.max_batch_size,
                m.max_batch_delay_ms,
                m.last_accessed_secs_ago
            ));
        }
    }

    out.push_str("\nServer Metrics Snapshot:\n");
    out.push_str("  --------------------------------------------------------------------\n");
    for line in raw_metrics.lines() {
        if !line.starts_with('#') && !line.trim().is_empty() {
            out.push_str(&format!("  {line}\n"));
        }
    }
    out.push_str("\n========================================================================\n");
    out.push_str("Press Ctrl+C to exit monitor.\n");

    out
}

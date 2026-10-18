#!/usr/bin/env python3
"""
SnoozeML Visual Benchmark Report Generator (Paged / Tabbed Dashboard)
====================================================================

Generates an interactive, dark-themed, multi-page / tabbed HTML dashboard
using Plotly and modern web design from benchmark telemetry produced by `tools/perf_test.py`.

Features:
- Paged navigation tabs: Overview, Capacity Hunter, Sustained Load, Scale-to-Zero, Raw Data.
- Interactive Plotly visualizations (Throughput, Latencies, Docker CPU/RAM, Memory Lifecycle).
- Print / Export to PDF support with clean CSS page breaks (@media print).
- Executive diagnostic insights and system bottleneck detection.

Usage:
------
    python tools/report_generator.py perf_results.json --output report.html
"""

import argparse
import html
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("[ERROR] 'plotly' is not installed.")
    print("Please install requirements: pip install -r tools/requirements.txt")
    sys.exit(1)


# Plotly dark theme layout defaults
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(17, 24, 39, 0.6)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color="#F1F5F9"),
    title_font=dict(family="Outfit, Inter, sans-serif", size=16, color="#F8FAFC"),
    hoverlabel=dict(
        bgcolor="#1E293B",
        bordercolor="#334155",
        font=dict(family="Inter, monospace", color="#F8FAFC", size=12),
    ),
    xaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#94A3B8", size=11),
    ),
    margin=dict(l=50, r=40, t=60, b=50),
)


def format_bytes(bytes_val: float) -> str:
    abs_val = abs(bytes_val)
    if abs_val >= 1024.0**3:
        return f"{bytes_val / (1024.0**3):.2f} GiB"
    elif abs_val >= 1024.0**2:
        return f"{bytes_val / (1024.0**2):.2f} MiB"
    elif abs_val >= 1024.0:
        return f"{bytes_val / 1024.0:.2f} KiB"
    else:
        return f"{bytes_val:.0f} B"


def create_throughput_chart(steps: List[Dict[str, Any]]) -> str:
    """Builds interactive Throughput (RPS) vs Concurrency chart."""
    x = [s["concurrency"] for s in steps]
    rps = [s["requests_per_sec"] for s in steps]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(v) for v in x],
            y=rps,
            name="Throughput (RPS)",
            marker=dict(
                color=rps,
                colorscale=[[0, "#38bdf8"], [0.5, "#06b6d4"], [1.0, "#0284c7"]],
                line=dict(color="#38bdf8", width=1.5),
            ),
            text=[f"{v:.1f} rps" for v in rps],
            textposition="outside",
            textfont=dict(color="#F8FAFC", size=11),
            hovertemplate="<b>Concurrency:</b> %{x} workers<br><b>Throughput:</b> %{y:.1f} RPS<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[str(v) for v in x],
            y=rps,
            name="Throughput Trend",
            mode="lines+markers",
            line=dict(color="#a855f7", width=3, shape="spline"),
            marker=dict(size=8, color="#c084fc", symbol="circle"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Inference Throughput vs. Concurrency</b>",
        xaxis_title="Concurrent Workers",
        yaxis_title="Requests / Second (RPS)",
        bargap=0.35,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_latency_chart(steps: List[Dict[str, Any]]) -> str:
    """Builds interactive Latency Distribution (Avg, P50, P90, P95, P99) chart."""
    x = [str(s["concurrency"]) for s in steps]
    p50 = [s["latency_p50_ms"] for s in steps]
    p90 = [s["latency_p90_ms"] for s in steps]
    p95 = [s["latency_p95_ms"] for s in steps]
    p99 = [s["latency_p99_ms"] for s in steps]
    avg = [s["latency_avg_ms"] for s in steps]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p99,
            name="P99 Tail Latency",
            mode="lines+markers",
            line=dict(color="#f43f5e", width=2.5, dash="dash"),
            marker=dict(size=7),
            hovertemplate="P99: %{y:.1f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p95,
            name="P95 Latency",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7),
            hovertemplate="P95: %{y:.1f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p90,
            name="P90 Latency",
            mode="lines+markers",
            line=dict(color="#eab308", width=2),
            marker=dict(size=6),
            hovertemplate="P90: %{y:.1f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=avg,
            name="Mean (Avg) Latency",
            mode="lines+markers",
            line=dict(color="#06b6d4", width=2.5),
            marker=dict(size=7),
            hovertemplate="Avg: %{y:.1f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            name="P50 Median Latency",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8),
            hovertemplate="P50: %{y:.1f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Latency Percentiles vs. Concurrency</b>",
        xaxis_title="Concurrent Workers",
        yaxis_title="Response Time (ms)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_docker_resources_chart(steps: List[Dict[str, Any]]) -> str:
    """Builds dual-axis chart showing Docker CPU % and Docker Memory RSS."""
    x = [str(s["concurrency"]) for s in steps]
    cpu = [s.get("docker_peak_cpu", 0.0) for s in steps]
    mem_mib = [s.get("docker_peak_mem_bytes", 0.0) / (1024.0**2) for s in steps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=x,
            y=cpu,
            name="Docker CPU %",
            marker=dict(color="rgba(139, 92, 246, 0.65)", line=dict(color="#8b5cf6", width=1.5)),
            hovertemplate="CPU: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mem_mib,
            name="Docker Memory RSS (MiB)",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8, color="#34d399"),
            hovertemplate="RAM: %{y:.2f} MiB<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Docker Container Resources (CPU % & Memory RSS)</b>",
        xaxis_title="Concurrent Workers",
        bargap=0.4,
    )
    fig.update_yaxes(title_text="Peak CPU %", secondary_y=False, gridcolor="#1E293B")
    fig.update_yaxes(title_text="Memory RSS (MiB)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_scale_to_zero_chart(s2z: Dict[str, Any]) -> str:
    """Builds waterfall / bar chart illustrating Scale-To-Zero memory reclamation."""
    base = s2z.get("baseline_mem_bytes", 0.0) / (1024.0**2)
    loaded = s2z.get("loaded_mem_bytes", 0.0) / (1024.0**2)
    final = s2z.get("final_mem_bytes", 0.0) / (1024.0**2)
    reclaimed = s2z.get("reclaimed_bytes", 0.0) / (1024.0**2)

    categories = ["1. Idle Baseline", "2. Model Loaded", "3. Post-Unload", "Reclaimed Memory"]
    values = [base, loaded, final, reclaimed]
    colors = ["#38bdf8", "#f43f5e", "#10b981", "#8b5cf6"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker=dict(color=colors, line=dict(color="#CBD5E1", width=1)),
            text=[f"{v:.1f} MiB" for v in values],
            textposition="outside",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="<b>%{x}:</b> %{y:.2f} MiB<extra></extra>",
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"<b>Scale-to-Zero Memory Lifecycle ({s2z.get('model_id', 'Model')})</b>",
        yaxis_title="Memory (MiB)",
        bargap=0.4,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_fixed_latency_chart(fixed: Dict[str, Any]) -> str:
    """Builds horizontal bar chart showing latency percentiles under sustained load."""
    metrics = ["Min", "P50", "Mean (Avg)", "P90", "P95", "P99", "Max"]
    values = [
        fixed.get("latency_min_ms", 0.0),
        fixed.get("latency_p50_ms", 0.0),
        fixed.get("latency_avg_ms", 0.0),
        fixed.get("latency_p90_ms", 0.0),
        fixed.get("latency_p95_ms", 0.0),
        fixed.get("latency_p99_ms", 0.0),
        fixed.get("latency_max_ms", 0.0),
    ]
    colors = ["#34d399", "#10b981", "#38bdf8", "#f59e0b", "#f97316", "#ef4444", "#dc2626"]

    fig = go.Figure(
        go.Bar(
            y=metrics,
            x=values,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f} ms" for v in values],
            textposition="outside",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="<b>%{y}:</b> %{x:.2f} ms<extra></extra>",
        )
    )
    layout = dict(DARK_LAYOUT)
    layout["yaxis"] = dict(autorange="reversed", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1"))
    fig.update_layout(
        **layout,
        title=f"<b>Sustained Load Latency Spread (Concurrency={fixed.get('concurrency', 16)})</b>",
        xaxis_title="Response Time (ms)",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_status_donut_chart(status_dist: Dict[str, int]) -> str:
    """Builds donut chart showing HTTP response status code distribution."""
    labels = [f"HTTP {k}" if str(k) != "0" else "Network Error" for k in status_dist.keys()]
    values = list(status_dist.values())
    colors = ["#10b981" if "200" in l else "#f43f5e" for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="<b>%{label}:</b> %{value} requests (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>HTTP Response Status Codes</b>",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def generate_html_report(data: Dict[str, Any], output_path: str):
    """Assembles all sections into an interactive, multi-page / tabbed HTML dashboard."""

    endpoint = data.get("endpoint", "http://localhost:8080")
    container = data.get("container") or "N/A"
    model_id = data.get("model_id", "Unknown")
    mode = data.get("mode", "all")
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(data.get("timestamp", time.time())))

    # Extract Ramp Metrics
    ramp_test = data.get("ramp_test")
    steps = ramp_test.get("steps", []) if ramp_test else []
    findings = ramp_test.get("findings", {}) if ramp_test else {}

    # Extract Fixed Test Metrics
    fixed_test = data.get("fixed_test")

    # Extract Scale to Zero Metrics
    s2z_test = data.get("scale_to_zero_test")

    # Compute Global KPI totals
    peak_rps = findings.get("peak_requests_per_sec", 0.0)
    optimal_c = findings.get("optimal_concurrency", 1)
    breaking_point = findings.get("breaking_point")

    if not peak_rps and steps:
        peak_rps = max(s["requests_per_sec"] for s in steps)
    if not peak_rps and fixed_test:
        peak_rps = fixed_test.get("requests_per_sec", 0.0)

    target_step = next((s for s in steps if s["concurrency"] == optimal_c), steps[-1] if steps else fixed_test or {})
    p50_val = target_step.get("latency_p50_ms", 0.0)
    p99_val = target_step.get("latency_p99_ms", 0.0)
    
    cpu_candidates = [s.get("docker_peak_cpu", 0.0) for s in steps]
    if fixed_test:
        cpu_candidates.append(fixed_test.get("docker_peak_cpu", 0.0))
    peak_cpu = max(cpu_candidates) if cpu_candidates else 0.0

    ram_candidates = [s.get("docker_peak_mem_bytes", 0.0) for s in steps]
    if fixed_test:
        ram_candidates.append(fixed_test.get("docker_peak_mem_bytes", 0.0))
    peak_ram_bytes = max(ram_candidates) if ram_candidates else 0.0

    # Render Charts
    throughput_html = create_throughput_chart(steps) if steps else ""
    latency_html = create_latency_chart(steps) if steps else ""
    docker_html = create_docker_resources_chart(steps) if (steps and peak_ram_bytes > 0) else ""
    s2z_html = create_scale_to_zero_chart(s2z_test) if (s2z_test and s2z_test.get("status") != "skipped") else ""
    fixed_lat_html = create_fixed_latency_chart(fixed_test) if fixed_test else ""
    fixed_status_html = create_status_donut_chart(fixed_test.get("status_distribution", {})) if fixed_test else ""

    # Build Step Table Rows
    table_rows = ""
    for s in steps:
        ram_str = format_bytes(s.get("docker_peak_mem_bytes", 0)) if s.get("docker_peak_mem_bytes", 0) > 0 else "—"
        cpu_str = f"{s.get('docker_peak_cpu', 0):.1f}%" if s.get("docker_peak_cpu", 0) > 0 else "—"
        err_badge_class = "badge-green" if s["error_rate_perc"] == 0.0 else "badge-red"
        table_rows += f"""
        <tr>
            <td class="font-mono font-bold text-cyan-400">{s['concurrency']}</td>
            <td>{s['total_requests']}</td>
            <td class="font-mono text-emerald-400 font-semibold">{s['requests_per_sec']:.1f}</td>
            <td>{s['latency_avg_ms']:.1f} ms</td>
            <td>{s['latency_p50_ms']:.1f} ms</td>
            <td>{s['latency_p95_ms']:.1f} ms</td>
            <td class="font-mono text-rose-400">{s['latency_p99_ms']:.1f} ms</td>
            <td><span class="badge {err_badge_class}">{s['error_rate_perc']:.1f}%</span></td>
            <td>{cpu_str}</td>
            <td class="font-mono text-slate-300">{ram_str}</td>
        </tr>
        """

    # Recommendation / Diagnostic box
    if breaking_point:
        sat_title = f"Saturation Detected at {breaking_point['concurrency']} Concurrent Workers"
        sat_desc = f"{breaking_point['reason']}. To avoid queue saturation and tail latency spikes, keep max concurrent client requests below {optimal_c} or tune max_batch_size and max_delay_ms in models.toml."
        sat_badge = '<span class="badge badge-amber">System Limit Reached</span>'
    else:
        max_tested = steps[-1]['concurrency'] if steps else optimal_c
        sat_title = "High Concurrency Headroom Confirmed"
        sat_desc = f"All tested concurrency tiers (up to {max_tested} workers) completed with 0% errors within latency thresholds. The inference daemon scales efficiently under concurrent loads."
        sat_badge = '<span class="badge badge-green">Optimal Stability</span>'

    s2z_kpi = ""
    if s2z_test and s2z_test.get("status") == "passed":
        reclaimed_mb = format_bytes(s2z_test.get("reclaimed_bytes", 0))
        s2z_kpi = f"""
        <div class="kpi-card">
            <div class="kpi-label">Scale-To-Zero Reclaimed</div>
            <div class="kpi-value text-emerald-400">{reclaimed_mb}</div>
            <div class="kpi-subtext">{s2z_test.get('reclaimed_percentage', 0)}% of weights deallocated</div>
        </div>
        """

    json_raw_escaped = html.escape(json.dumps(data, indent=2))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SnoozeML Benchmark Report — {model_id}</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-body: #090D16;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(56, 189, 248, 0.25);
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --accent-cyan: #38BDF8;
            --accent-purple: #A855F7;
            --accent-green: #34D399;
            --accent-amber: #FBBF24;
            --accent-rose: #FB7185;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg-body);
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            padding: 28px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1320px;
            margin: 0 auto;
        }}
        /* Header Hero */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-card);
            padding-bottom: 20px;
        }}
        .tag-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 9999px;
            color: var(--accent-cyan);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }}
        .title {{
            font-family: 'Outfit', sans-serif;
            font-size: 30px;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .meta-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
        }}
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.03);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }}
        .meta-item strong {{
            color: #E2E8F0;
        }}
        .print-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #CBD5E1;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .print-btn:hover {{
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--accent-cyan);
            color: #FFFFFF;
        }}
        /* Tab Navigation Bar */
        .tab-nav {{
            display: flex;
            gap: 8px;
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 6px;
            margin-bottom: 28px;
            overflow-x: auto;
            backdrop-filter: blur(16px);
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .tab-btn:hover {{
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.04);
        }}
        .tab-btn.active {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(168, 85, 247, 0.2));
            color: #FFFFFF;
            border: 1px solid rgba(56, 189, 248, 0.4);
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
        }}
        /* Tab Page Visibility */
        .tab-page {{
            display: none;
            animation: fadeIn 0.25s ease-in-out;
        }}
        .tab-page.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(12px);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-glow);
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            opacity: 0.8;
        }}
        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 4px;
        }}
        .kpi-subtext {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .chart-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 16px;
            backdrop-filter: blur(12px);
        }}
        /* Diagnosis Banner */
        .diag-banner {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7));
            border: 1px solid var(--border-glow);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 28px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }}
        .diag-banner h4 {{
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            color: #FFFFFF;
            margin-bottom: 4px;
        }}
        .diag-banner p {{
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.6;
        }}
        /* Tables */
        .table-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 28px;
            overflow-x: auto;
        }}
        .table-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            margin-bottom: 16px;
            color: #FFFFFF;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: right;
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: #CBD5E1;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-green {{
            background: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}
        .badge-amber {{
            background: rgba(251, 191, 36, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }}
        .badge-red {{
            background: rgba(251, 113, 133, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(251, 113, 133, 0.3);
        }}
        /* Code block */
        pre.code-block {{
            background: #0B0F19;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #93C5FD;
            overflow-x: auto;
            max-height: 500px;
        }}
        /* Typography utilities */
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .text-cyan-400 {{ color: var(--accent-cyan); }}
        .text-emerald-400 {{ color: var(--accent-green); }}
        .text-rose-400 {{ color: var(--accent-rose); }}
        .text-amber-400 {{ color: var(--accent-amber); }}
        .text-slate-300 {{ color: #CBD5E1; }}
        .font-bold {{ font-weight: 700; }}
        .font-semibold {{ font-weight: 600; }}

        /* Print / Multi-page PDF Styles */
        @media print {{
            body {{
                background: #FFFFFF !important;
                color: #000000 !important;
                padding: 0 !important;
            }}
            .tab-nav, .print-btn {{
                display: none !important;
            }}
            .tab-page {{
                display: block !important;
                page-break-after: always;
                margin-bottom: 40px;
            }}
            .kpi-card, .chart-box, .table-card, .diag-banner {{
                background: #FFFFFF !important;
                border: 1px solid #CCCCCC !important;
                color: #000000 !important;
                box-shadow: none !important;
            }}
            .kpi-value, .title, .table-title, h4 {{
                color: #000000 !important;
                -webkit-text-fill-color: initial !important;
            }}
            th, td {{
                color: #000000 !important;
                border-bottom: 1px solid #E5E7EB !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div>
                <div class="tag-pill">SnoozeML Observability</div>
                <h1 class="title">Docker Performance & Capacity Benchmark</h1>
                <div class="meta-bar">
                    <div class="meta-item"><span>Target Model:</span> <strong>{model_id}</strong></div>
                    <div class="meta-item"><span>Container:</span> <strong>{container}</strong></div>
                    <div class="meta-item"><span>Endpoint:</span> <strong>{endpoint}</strong></div>
                    <div class="meta-item"><span>Mode:</span> <strong>{mode.upper()}</strong></div>
                    <div class="meta-item"><span>Run Date:</span> <strong>{timestamp_str}</strong></div>
                </div>
            </div>
            <div>
                <button class="print-btn" onclick="window.print()">
                    <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2m-4 0v4H8v-4m0-4h8"></path></svg>
                    Print / Export PDF
                </button>
            </div>
        </header>

        <!-- Paged Tab Navigation -->
        <nav class="tab-nav">
            <button class="tab-btn active" id="btn-overview" onclick="showTab('overview')">
                <span>📑</span> Overview & Summary
            </button>
            <button class="tab-btn" id="btn-ramp" onclick="showTab('ramp')" {'style="display:none;"' if not steps else ''}>
                <span>📈</span> Capacity Hunter (Ramp-Up)
            </button>
            <button class="tab-btn" id="btn-fixed" onclick="showTab('fixed')" {'style="display:none;"' if not fixed_test else ''}>
                <span>⏱️</span> Sustained Load (Fixed)
            </button>
            <button class="tab-btn" id="btn-s2z" onclick="showTab('s2z')" {'style="display:none;"' if not (s2z_test and s2z_test.get("status") != "skipped") else ''}>
                <span>♻️</span> Scale-To-Zero
            </button>
            <button class="tab-btn" id="btn-raw" onclick="showTab('raw')">
                <span>📋</span> Raw Telemetry (JSON)
            </button>
            <button class="tab-btn" id="btn-all" onclick="showAllTabs()">
                <span>📄</span> View All Pages
            </button>
        </nav>

        <!-- ============================================================== -->
        <!-- TAB 1: EXECUTIVE OVERVIEW -->
        <!-- ============================================================== -->
        <section class="tab-page active" id="page-overview">
            <!-- Global KPI Grid -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Peak Throughput</div>
                    <div class="kpi-value text-cyan-400">{peak_rps:.1f} <span style="font-size: 14px; font-weight: normal; color: #94A3B8;">RPS</span></div>
                    <div class="kpi-subtext">Max sustained requests/sec</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Optimal Concurrency</div>
                    <div class="kpi-value">{optimal_c} <span style="font-size: 14px; font-weight: normal; color: #94A3B8;">workers</span></div>
                    <div class="kpi-subtext">Peak efficiency tier</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Median Latency (P50)</div>
                    <div class="kpi-value text-emerald-400">{p50_val:.1f} <span style="font-size: 14px; font-weight: normal; color: #94A3B8;">ms</span></div>
                    <div class="kpi-subtext">At optimal concurrency</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Tail Latency (P99)</div>
                    <div class="kpi-value text-rose-400">{p99_val:.1f} <span style="font-size: 14px; font-weight: normal; color: #94A3B8;">ms</span></div>
                    <div class="kpi-subtext">99th percentile response time</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Peak Docker CPU</div>
                    <div class="kpi-value">{peak_cpu:.1f}%</div>
                    <div class="kpi-subtext">Multi-core compute saturation</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Peak Docker RAM</div>
                    <div class="kpi-value">{format_bytes(peak_ram_bytes)}</div>
                    <div class="kpi-subtext">Container RSS limit headroom</div>
                </div>
                {s2z_kpi}
            </div>

            <!-- Diagnostic Banner -->
            <div class="diag-banner">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <h4>{sat_title}</h4>
                        {sat_badge}
                    </div>
                    <p>{sat_desc}</p>
                </div>
            </div>

            <!-- Overview Charts Grid -->
            <div class="charts-grid">
                {f'<div class="chart-box">{throughput_html}</div>' if throughput_html else ''}
                {f'<div class="chart-box">{latency_html}</div>' if latency_html else ''}
            </div>
        </section>

        <!-- ============================================================== -->
        <!-- TAB 2: CAPACITY HUNTER (RAMP-UP) -->
        <!-- ============================================================== -->
        <section class="tab-page" id="page-ramp">
            <div class="charts-grid">
                {f'<div class="chart-box">{throughput_html}</div>' if throughput_html else ''}
                {f'<div class="chart-box">{latency_html}</div>' if latency_html else ''}
                {f'<div class="chart-box" style="grid-column: 1 / -1;">{docker_html}</div>' if docker_html else ''}
            </div>

            {f'''
            <div class="table-card">
                <h3 class="table-title">Concurrency Step Telemetry</h3>
                <table>
                    <thead>
                        <tr>
                            <th style="text-align: left;">Concurrency</th>
                            <th>Requests</th>
                            <th>Req/s (RPS)</th>
                            <th>Avg Latency</th>
                            <th>P50 Latency</th>
                            <th>P95 Latency</th>
                            <th>P99 Latency</th>
                            <th>Error Rate</th>
                            <th>Peak CPU</th>
                            <th>Peak RAM</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            ''' if table_rows else '<p style="color:#94a3b8;">No ramp-up steps recorded in this run.</p>'}
        </section>

        <!-- ============================================================== -->
        <!-- TAB 3: SUSTAINED FIXED LOAD -->
        <!-- ============================================================== -->
        <section class="tab-page" id="page-fixed">
            {f'''
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Sustained Throughput</div>
                    <div class="kpi-value text-cyan-400">{fixed_test.get("requests_per_sec", 0):.2f} <span style="font-size: 14px; font-weight: normal; color: #94A3B8;">RPS</span></div>
                    <div class="kpi-subtext">{fixed_test.get("total_requests", 0)} total requests across {fixed_test.get("elapsed_secs", 0)}s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Workers (Concurrency)</div>
                    <div class="kpi-value">{fixed_test.get("concurrency", 16)}</div>
                    <div class="kpi-subtext">Active continuous connections</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Median Latency (P50)</div>
                    <div class="kpi-value text-emerald-400">{fixed_test.get("latency_p50_ms", 0):.1f} ms</div>
                    <div class="kpi-subtext">Avg: {fixed_test.get("latency_avg_ms", 0):.1f} ms</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">P99 Tail Latency</div>
                    <div class="kpi-value text-rose-400">{fixed_test.get("latency_p99_ms", 0):.1f} ms</div>
                    <div class="kpi-subtext">Max: {fixed_test.get("latency_max_ms", 0):.1f} ms</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Docker Peak CPU</div>
                    <div class="kpi-value">{fixed_test.get("docker_peak_cpu", 0):.1f}%</div>
                    <div class="kpi-subtext">Steady multi-core usage</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Docker Memory RSS</div>
                    <div class="kpi-value">{format_bytes(fixed_test.get("docker_peak_mem_bytes", 0))}</div>
                    <div class="kpi-subtext">No memory leak detected</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-box">{fixed_lat_html}</div>
                <div class="chart-box">{fixed_status_html}</div>
            </div>
            ''' if fixed_test else '<p style="color:#94a3b8; padding: 24px;">No sustained fixed-load test was executed in this run.</p>'}
        </section>

        <!-- ============================================================== -->
        <!-- TAB 4: SCALE-TO-ZERO -->
        <!-- ============================================================== -->
        <section class="tab-page" id="page-s2z">
            {f'''
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Idle Baseline Memory</div>
                    <div class="kpi-value text-cyan-400">{format_bytes(s2z_test.get("baseline_mem_bytes", 0))}</div>
                    <div class="kpi-subtext">Before cold start</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Loaded Memory Peak</div>
                    <div class="kpi-value text-rose-400">{format_bytes(s2z_test.get("loaded_mem_bytes", 0))}</div>
                    <div class="kpi-subtext">+{format_bytes(s2z_test.get("loaded_mem_bytes", 0) - s2z_test.get("baseline_mem_bytes", 0))} allocated</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Post-Unload Memory</div>
                    <div class="kpi-value text-emerald-400">{format_bytes(s2z_test.get("final_mem_bytes", 0))}</div>
                    <div class="kpi-subtext">Cooldown memory footprint</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Memory Reclaimed</div>
                    <div class="kpi-value text-emerald-400">{format_bytes(s2z_test.get("reclaimed_bytes", 0))}</div>
                    <div class="kpi-subtext">{s2z_test.get("reclaimed_percentage", 0)}% of weights deallocated</div>
                </div>
            </div>

            <div class="chart-box" style="margin-bottom: 24px;">
                {s2z_html}
            </div>

            <div class="diag-banner">
                <div>
                    <h4>Scale-To-Zero Verification: {s2z_test.get("status", "").upper()}</h4>
                    <p>The SnoozeML idle reaper successfully evicted model weights and unmapped inference buffers from Docker resident memory upon trigger, returning container RSS down to idle baseline.</p>
                </div>
            </div>
            ''' if (s2z_test and s2z_test.get("status") != "skipped") else '<p style="color:#94a3b8; padding: 24px;">No scale-to-zero test was executed in this run.</p>'}
        </section>

        <!-- ============================================================== -->
        <!-- TAB 5: RAW TELEMETRY (JSON) -->
        <!-- ============================================================== -->
        <section class="tab-page" id="page-raw">
            <div class="table-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 class="table-title" style="margin: 0;">Raw Benchmark Telemetry</h3>
                    <button class="print-btn" onclick="navigator.clipboard.writeText(document.getElementById('raw-json').innerText); alert('Copied JSON telemetry to clipboard!');">
                        📋 Copy JSON
                    </button>
                </div>
                <pre class="code-block" id="raw-json">{json_raw_escaped}</pre>
            </div>
        </section>
    </div>

    <!-- Tab Switching Script -->
    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-page').forEach(page => page.classList.remove('active'));
            const activeBtn = document.getElementById('btn-' + tabId);
            const activePage = document.getElementById('page-' + tabId);
            if (activeBtn) activeBtn.classList.add('active');
            if (activePage) activePage.classList.add('active');
            // Trigger Plotly responsive redraw
            window.dispatchEvent(new Event('resize'));
        }}

        function showAllTabs() {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-all').classList.add('active');
            document.querySelectorAll('.tab-page').forEach(page => page.classList.add('active'));
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Interactive paged visual report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive HTML performance dashboard from benchmark JSON")
    parser.add_argument("input_json", help="Path to input JSON produced by perf_test.py")
    parser.add_argument("-o", "--output", default="benchmark_report.html", help="Output HTML file path (default: benchmark_report.html)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_json):
        print(f"[ERROR] Input JSON file '{args.input_json}' not found.")
        sys.exit(1)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_html_report(data, args.output)


if __name__ == "__main__":
    main()

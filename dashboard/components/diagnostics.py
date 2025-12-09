"""
Diagnostics Component — State file health and system diagnostics.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_diagnostics_block(
    state_health: List[Dict[str, Any]],
    executor_status: Dict[str, Any] = None,
) -> None:
    """Render diagnostics panel with state file health grid."""
    # Build state file health grid
    if not state_health:
        st.html('''
        <div class="quant-card empty-card">
            <div class="empty-title">No state files configured</div>
        </div>
        ''')
        return
    
    # State file rows
    rows_html = []
    healthy_count = 0
    stale_count = 0
    missing_count = 0
    
    for sf in state_health:
        name = sf.get("name") or sf.get("file") or "?"
        status = sf.get("status") or "unknown"
        age_s = sf.get("age_s") or sf.get("age_seconds")
        size_b = sf.get("size_bytes") or sf.get("size")
        
        # Status badge
        if status == "ok" or status == "healthy":
            badge_class = "normal"
            badge_text = "OK"
            healthy_count += 1
        elif status == "stale":
            badge_class = "warning"
            badge_text = "STALE"
            stale_count += 1
        else:
            badge_class = "critical"
            badge_text = "MISSING"
            missing_count += 1
        
        # Format age
        age_str = "n/a"
        if age_s is not None:
            try:
                age = float(age_s)
                if age < 60:
                    age_str = f"{age:.0f}s"
                elif age < 3600:
                    age_str = f"{age/60:.1f}m"
                else:
                    age_str = f"{age/3600:.1f}h"
            except Exception:
                pass
        
        # Format size
        size_str = "n/a"
        if size_b is not None:
            try:
                size = int(size_b)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
            except Exception:
                pass
        
        rows_html.append(f'''
        <tr>
            <td class="text-mono">{name}</td>
            <td><span class="status-badge {badge_class}">{badge_text}</span></td>
            <td class="text-mono text-right">{age_str}</td>
            <td class="text-mono text-right">{size_str}</td>
        </tr>
        ''')
    
    # Summary status
    if missing_count > 0:
        summary_badge = '<span class="status-badge critical">DEGRADED</span>'
    elif stale_count > 0:
        summary_badge = '<span class="status-badge warning">PARTIAL</span>'
    else:
        summary_badge = '<span class="status-badge normal">HEALTHY</span>'
    
    # Executor status section
    executor_html = ""
    if executor_status:
        exec_running = executor_status.get("running", False)
        exec_uptime = executor_status.get("uptime_s") or executor_status.get("uptime")
        exec_last_cycle = executor_status.get("last_cycle_s") or executor_status.get("last_cycle")
        
        exec_badge_class = "normal" if exec_running else "critical"
        exec_badge_text = "RUNNING" if exec_running else "STOPPED"
        
        uptime_str = "n/a"
        if exec_uptime is not None:
            try:
                u = float(exec_uptime)
                if u < 3600:
                    uptime_str = f"{u/60:.0f}m"
                elif u < 86400:
                    uptime_str = f"{u/3600:.1f}h"
                else:
                    uptime_str = f"{u/86400:.1f}d"
            except Exception:
                pass
        
        cycle_str = "n/a"
        if exec_last_cycle is not None:
            try:
                c = float(exec_last_cycle)
                cycle_str = f"{c:.0f}s ago"
            except Exception:
                pass
        
        executor_html = f'''
        <div class="diagnostics-executor">
            <div class="executor-row">
                <span>Executor</span>
                <span class="status-badge {exec_badge_class}">{exec_badge_text}</span>
            </div>
            <div class="executor-row text-muted text-xs">
                <span>Uptime: {uptime_str}</span>
                <span>Last cycle: {cycle_str}</span>
            </div>
        </div>
        '''
    
    html = f'''
    <div class="quant-card">
        <div class="diagnostics-header">
            <span class="diagnostics-title">State Files</span>
            {summary_badge}
        </div>
        <table class="quant-table diagnostics-table">
            <thead>
                <tr>
                    <th>File</th>
                    <th>Status</th>
                    <th class="text-right">Age</th>
                    <th class="text-right">Size</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        {executor_html}
    </div>
    '''
    
    st.html(html)

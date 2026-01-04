import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.spatial import cKDTree
import pickle
from datetime import datetime, timedelta
import gc
import time

st.set_page_config(page_title="Supply & Demand Dashboard", layout="wide", page_icon="📊")

# ============================================================================
# CONSTANTS
# ============================================================================

# Area names - EXACT as they appear in raw data
AREA_NAMES = [
    "Alexandria",
    "Downtown 1", 
    "Maadi",
    "Masr El Gedida",
    "Zahraa El Maadi"
]

# Improved color palette - more distinct and professional
AREA_COLORS = {
    "Alexandria": "#2E86DE",      # Blue
    "Downtown 1": "#8854D0",      # Purple
    "Maadi": "#20BF55",           # Green
    "Masr El Gedida": "#EE5A6F",  # Red/Pink
    "Zahraa El Maadi": "#F79F1F",  # Orange
    "Total": "#00D2D3",           # Cyan (for total view)
}

# Performance thresholds
FULFILLMENT_TARGET = 85  # Target fulfillment %
FULFILLMENT_WARNING = 70  # Warning threshold

# Persistent storage
DATA_DIR = Path("persistent_data")
DATA_DIR.mkdir(exist_ok=True)
DIST_POINTS_FILE = DATA_DIR / "distribution_points.pkl"
ANALYZED_DATA_FILE = DATA_DIR / "analyzed_data.pkl"

# ============================================================================
# HELPER FUNCTIONS - FILE I/O
# ============================================================================

def save_distribution_points(points):
    """Save distribution points to disk"""
    with open(DIST_POINTS_FILE, 'wb') as f:
        pickle.dump(points, f)

def load_distribution_points():
    """Load distribution points from disk"""
    if DIST_POINTS_FILE.exists():
        with open(DIST_POINTS_FILE, 'rb') as f:
            return pickle.load(f)
    return []

def save_analyzed_data(sessions_agg, heat, rides):
    """Save analyzed data to disk"""
    data = {
        'sessions_agg': sessions_agg,
        'heat': heat,
        'rides': rides,
        'timestamp': datetime.now()
    }
    with open(ANALYZED_DATA_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_analyzed_data():
    """Load analyzed data from disk - returns dict or None"""
    if ANALYZED_DATA_FILE.exists():
        with open(ANALYZED_DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            # Ensure it's a dictionary
            if isinstance(data, dict):
                return data
            else:
                # Old format - convert to dict
                return {
                    'sessions_agg': data[0] if len(data) > 0 else None,
                    'heat': data[1] if len(data) > 1 else None,
                    'timestamp': datetime.now()
                }
    return None

# ============================================================================
# HELPER FUNCTIONS - ASSIGNMENT
# ============================================================================

def assign_sessions_to_areas(sessions, dist_points):
    """
    Assign sessions to areas based on distribution points.
    Optimized for large datasets (1M+ rows).
    """
    # Convert distribution points to DataFrame
    all_points = pd.DataFrame(dist_points)
    
    # Find latitude/longitude columns (case-insensitive)
    cols_lower = {c.lower(): c for c in sessions.columns}
    lat_col = cols_lower.get('latitude', cols_lower.get('lat'))
    lon_col = cols_lower.get('longitude', cols_lower.get('lon', cols_lower.get('lng')))
    
    if not lat_col or not lon_col:
        raise ValueError(f"Sessions file must have 'Latitude' and 'Longitude' columns. Found: {sessions.columns.tolist()}")
    
    # Remove sessions with invalid coordinates
    valid_mask = (
        sessions[lat_col].notna() & 
        sessions[lon_col].notna() &
        np.isfinite(sessions[lat_col]) & 
        np.isfinite(sessions[lon_col]) &
        (sessions[lat_col] != 0) &
        (sessions[lon_col] != 0)
    )
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        st.warning(f"⚠️ Found {invalid_count:,} sessions with invalid coordinates - marking as 'Out of Fence'")
    
    sessions.loc[~valid_mask, 'Area'] = 'Out of Fence'
    sessions.loc[~valid_mask, 'Neighborhood'] = 'Out of Fence'
    
    valid_sessions = sessions[valid_mask].copy()
    
    if len(valid_sessions) == 0:
        st.error("No valid sessions found with proper coordinates!")
        return sessions
    
    # Extract coordinates
    session_coords = valid_sessions[[lat_col, lon_col]].values
    point_coords = all_points[["lat", "lon"]].values
    
    # Convert to radians
    session_rad = np.radians(session_coords)
    point_rad = np.radians(point_coords)
    
    # Convert to Cartesian coordinates
    session_cartesian = np.column_stack([
        np.cos(session_rad[:, 0]) * np.cos(session_rad[:, 1]),
        np.cos(session_rad[:, 0]) * np.sin(session_rad[:, 1]),
        np.sin(session_rad[:, 0])
    ])
    
    point_cartesian = np.column_stack([
        np.cos(point_rad[:, 0]) * np.cos(point_rad[:, 1]),
        np.cos(point_rad[:, 0]) * np.sin(point_rad[:, 1]),
        np.sin(point_rad[:, 0])
    ])
    
    # Build KD-tree and find nearest points
    tree = cKDTree(point_cartesian)
    _, nearest_indices = tree.query(session_cartesian, k=1)
    
    # Calculate distances
    nearest_point_coords = point_rad[nearest_indices]
    dlat = nearest_point_coords[:, 0] - session_rad[:, 0]
    dlon = nearest_point_coords[:, 1] - session_rad[:, 1]
    a = np.sin(dlat/2)**2 + np.cos(session_rad[:, 0]) * np.cos(nearest_point_coords[:, 0]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distances_meters = c * 6371000
    
    # Assign areas
    within_threshold = distances_meters <= 200
    valid_sessions.loc[:, 'Area'] = np.where(
        within_threshold, 
        all_points['area'].iloc[nearest_indices].values, 
        'Out of Fence'
    )
    valid_sessions.loc[:, 'Neighborhood'] = np.where(
        within_threshold,
        all_points['neighborhood'].iloc[nearest_indices].values,
        'Out of Fence'
    )
    
    sessions.loc[valid_mask, 'Area'] = valid_sessions['Area'].values
    sessions.loc[valid_mask, 'Neighborhood'] = valid_sessions['Neighborhood'].values
    
    return sessions

# ============================================================================
# HELPER FUNCTIONS - AGGREGATION
# ============================================================================

def aggregate_sessions(sessions):
    """Aggregate sessions by Area, Neighborhood, and Date"""
    if 'timestamp' not in sessions.columns:
        raise ValueError("Sessions must have 'timestamp' column before aggregation")
    
    sessions = sessions[sessions['timestamp'].notna()].copy()
    
    if len(sessions) == 0:
        return pd.DataFrame(columns=['Area', 'Neighborhood', 'date', 'sessions_count'])
    
    sessions['date'] = sessions['timestamp'].dt.floor('D')
    
    # Aggregate by Area, Neighborhood, AND date to preserve neighborhood-level data
    agg = sessions.groupby(['Area', 'Neighborhood', 'date']).size().reset_index(name='sessions_count')
    
    return agg

# ============================================================================
# HELPER FUNCTIONS - VISUAL IMPROVEMENTS
# ============================================================================

def create_sparkline(data_series, color="#2E86DE", height=80):
    """Create a mini sparkline chart for metrics"""
    df = pd.DataFrame({'x': range(len(data_series)), 'y': data_series.values})
    
    chart = alt.Chart(df).mark_line(
        color=color,
        strokeWidth=3,
        point=alt.OverlayMarkDef(filled=True, size=50, color=color)
    ).encode(
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None, scale=alt.Scale(zero=False)),
        tooltip=alt.Tooltip('y:Q', format=',.0f')
    ).properties(
        width='container',
        height=height
    )
    
    return chart

def get_trend_indicator(current, previous):
    """Get trend arrow and color based on comparison"""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "", ""
    
    change_pct = ((current - previous) / previous) * 100
    
    if abs(change_pct) < 1:
        return "➡️", "gray"
    elif change_pct > 0:
        return "📈", "green"
    else:
        return "📉", "red"

def format_metric_card(label, value, delta=None, delta_color="off", prefix="", suffix="", sparkline_data=None, color="#2E86DE"):
    """Create an enhanced metric card"""
    if delta is not None:
        st.metric(
            label=label,
            value=f"{prefix}{value:,.0f}{suffix}",
            delta=delta,
            delta_color=delta_color
        )
    else:
        st.metric(label=label, value=f"{prefix}{value:,.0f}{suffix}")

def show_data_table(df, title="Data Table"):
    """Display a clean, formatted data table"""
    if st.session_state.get("show_data_tables", False):
        with st.expander(f"📋 {title}", expanded=False):
            # Format the dataframe for display
            df_display = df.reset_index()
            
            # Format numbers with commas
            for col in df_display.columns:
                if df_display[col].dtype in ['float64', 'int64'] and col != df_display.columns[0]:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=min(400, len(df_display) * 35 + 38)  # Dynamic height
            )

# ============================================================================
# HELPER FUNCTIONS - CHARTING
# ============================================================================

def bucket_data(df, date_col, range_choice):
    """Apply time bucketing based on range choice"""
    if range_choice == "Daily":
        return df[date_col].dt.floor("D")
    elif range_choice == "Weekly":
        return df[date_col].dt.to_period("W").dt.to_timestamp()
    else:  # Monthly
        return df[date_col].dt.to_period("M").dt.to_timestamp()

def line_chart(df, title="", y_label="", show_values=False):
    """Create detailed line chart with advanced insights"""
    df = df.reset_index()
    date_col = df.columns[0]
    
    # Store original data before melting for comparison calculations
    df_original = df.copy()
    
    df = df.melt(date_col, var_name="Series", value_name="Value")
    
    # Get colors
    series_in_data = df["Series"].unique()
    color_domain = [s for s in AREA_COLORS.keys() if s in series_in_data]
    color_range = [AREA_COLORS[s] for s in color_domain]
    
    selection = alt.selection_multi(fields=["Series"], bind="legend")
    
    # Initialize layers list
    layers = []
    
    # ========================================================================
    # WEEKEND HIGHLIGHTING (Thursday, Friday & Saturday in Egypt)
    # ========================================================================
    if st.session_state.get("highlight_weekends", False):
        # Create weekend rectangles
        all_dates = pd.Series(pd.to_datetime(df[date_col].unique()))
        weekend_dates = all_dates[all_dates.dt.dayofweek.isin([3, 4, 5])]  # Thursday=3, Friday=4, Saturday=5
        
        if len(weekend_dates) > 0:
            weekend_df = pd.DataFrame({
                'start': weekend_dates.values,
                'end': weekend_dates.values + pd.Timedelta(days=1),
                'label': 'Weekend (Thu-Fri-Sat)'
            })
            
            weekend_layer = (
                alt.Chart(weekend_df)
                .mark_rect(opacity=0.15, color='#FFA726')
                .encode(
                    x='start:T',
                    x2='end:T',
                    tooltip=alt.Tooltip('label:N', title='Day Type')
                )
            )
            layers.append(weekend_layer)
    
    # ========================================================================
    # BASE LINE CHART
    # ========================================================================
    line = (
        alt.Chart(df)
        .mark_line(
            point=alt.OverlayMarkDef(size=100, filled=True),
            strokeWidth=4
        )
        .encode(
            x=alt.X(
                f"{date_col}:T", 
                title="",
                axis=alt.Axis(
                    format="%b %d",
                    labelAngle=-45,
                    labelFontSize=12,
                    labelOverlap="greedy",
                    grid=True,
                    gridOpacity=0.2
                )
            ),
            y=alt.Y(
                "Value:Q", 
                title=y_label,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    labelFontSize=13,
                    titleFontSize=14,
                    grid=True,
                    gridOpacity=0.2
                )
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(
                    orient="bottom",
                    labelFontSize=14,
                    titleFontSize=15,
                    symbolSize=200,
                    symbolStrokeWidth=4,
                    title="Click to highlight →"
                ),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
            strokeWidth=alt.condition(selection, alt.value(4), alt.value(2)),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Series:N", title="Area"),
                alt.Tooltip("Value:Q", title=y_label, format=",.0f")
            ],
        )
        .add_selection(selection)
    )
    layers.append(line)
    
    # ========================================================================
    # PERIOD COMPARISON
    # ========================================================================
    compare_period = st.session_state.get("compare_period", "None")
    if compare_period != "None":
        # Calculate offset
        if compare_period == "Last Week":
            offset_days = 7
        else:  # Last Month
            offset_days = 30
        
        # Create comparison data
        df_compare = df_original.copy()
        df_compare[date_col] = df_compare[date_col] + pd.Timedelta(days=offset_days)
        
        # Only keep dates that overlap with current range
        min_date = df_original[date_col].min()
        max_date = df_original[date_col].max()
        df_compare = df_compare[
            (df_compare[date_col] >= min_date) & 
            (df_compare[date_col] <= max_date)
        ]
        
        if len(df_compare) > 0:
            df_compare_melted = df_compare.melt(date_col, var_name="Series", value_name="Value")
            
            comparison_line = (
                alt.Chart(df_compare_melted)
                .mark_line(
                    strokeDash=[5, 5],
                    strokeWidth=2,
                    opacity=0.5
                )
                .encode(
                    x=f"{date_col}:T",
                    y="Value:Q",
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(domain=color_domain, range=color_range),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip(f"{date_col}:T", title=f"Date ({compare_period})", format="%b %d, %Y"),
                        alt.Tooltip("Series:N", title="Area"),
                        alt.Tooltip("Value:Q", title=y_label, format=",.0f")
                    ]
                )
            )
            layers.append(comparison_line)
    
    # ========================================================================
    # ANOMALIES DETECTION
    # ========================================================================
    if st.session_state.get("show_anomalies", False):
        # Calculate anomalies (> 2 standard deviations)
        anomalies_list = []
        
        for series_name in df["Series"].unique():
            series_data = df[df["Series"] == series_name].copy()
            mean_val = series_data["Value"].mean()
            std_val = series_data["Value"].std()
            
            if std_val > 0:
                series_data["z_score"] = (series_data["Value"] - mean_val) / std_val
                anomalies = series_data[abs(series_data["z_score"]) > 2].copy()
                
                if len(anomalies) > 0:
                    anomalies["anomaly_label"] = "⚠️ Anomaly"
                    anomalies_list.append(anomalies)
        
        if anomalies_list:
            df_anomalies = pd.concat(anomalies_list, ignore_index=True)
            
            anomaly_points = (
                alt.Chart(df_anomalies)
                .mark_point(
                    size=200,
                    filled=False,
                    strokeWidth=3,
                    color='#FF6B6B'
                )
                .encode(
                    x=f"{date_col}:T",
                    y="Value:Q",
                    tooltip=[
                        alt.Tooltip(f"{date_col}:T", title="Date", format="%b %d, %Y"),
                        alt.Tooltip("Series:N", title="Area"),
                        alt.Tooltip("Value:Q", title=y_label, format=",.0f"),
                        alt.Tooltip("anomaly_label:N", title="Alert")
                    ]
                )
            )
            layers.append(anomaly_points)
    
    # ========================================================================
    # PEAKS & VALLEYS
    # ========================================================================
    if st.session_state.get("highlight_peaks_valleys", False):
        peaks_list = []
        valleys_list = []
        
        for series_name in df["Series"].unique():
            series_data = df[df["Series"] == series_name].copy()
            
            if len(series_data) > 0:
                # Find peak (max)
                peak_idx = series_data["Value"].idxmax()
                peak_row = series_data.loc[peak_idx].copy()
                peak_row["marker"] = "🔺 Peak"
                peaks_list.append(peak_row)
                
                # Find valley (min)
                valley_idx = series_data["Value"].idxmin()
                valley_row = series_data.loc[valley_idx].copy()
                valley_row["marker"] = "🔻 Valley"
                valleys_list.append(valley_row)
        
        if peaks_list:
            df_peaks = pd.DataFrame(peaks_list)
            
            peak_markers = (
                alt.Chart(df_peaks)
                .mark_point(
                    size=250,
                    filled=True,
                    shape='triangle-up',
                    color='#20BF55'
                )
                .encode(
                    x=f"{date_col}:T",
                    y="Value:Q",
                    tooltip=[
                        alt.Tooltip(f"{date_col}:T", title="Date", format="%b %d, %Y"),
                        alt.Tooltip("Series:N", title="Area"),
                        alt.Tooltip("Value:Q", title=y_label, format=",.0f"),
                        alt.Tooltip("marker:N", title="Type")
                    ]
                )
            )
            layers.append(peak_markers)
        
        if valleys_list:
            df_valleys = pd.DataFrame(valleys_list)
            
            valley_markers = (
                alt.Chart(df_valleys)
                .mark_point(
                    size=250,
                    filled=True,
                    shape='triangle-down',
                    color='#EE5A6F'
                )
                .encode(
                    x=f"{date_col}:T",
                    y="Value:Q",
                    tooltip=[
                        alt.Tooltip(f"{date_col}:T", title="Date", format="%b %d, %Y"),
                        alt.Tooltip("Series:N", title="Area"),
                        alt.Tooltip("Value:Q", title=y_label, format=",.0f"),
                        alt.Tooltip("marker:N", title="Type")
                    ]
                )
            )
            layers.append(valley_markers)
    
    # ========================================================================
    # VALUE LABELS (LARGER TEXT)
    # ========================================================================
    if show_values:
        text = (
            alt.Chart(df)
            .mark_text(
                align='center',
                baseline='bottom',
                dy=-12,
                fontSize=13,  # Increased from 11 to 13
                fontWeight=700,
                opacity=0.95
            )
            .encode(
                x=f"{date_col}:T",
                y="Value:Q",
                text=alt.Text("Value:Q", format=",.0f"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=None
                ),
                opacity=alt.condition(selection, alt.value(0.95), alt.value(0.25))
            )
        )
        layers.append(text)
    
    # ========================================================================
    # COMBINE ALL LAYERS
    # ========================================================================
    chart = alt.layer(*layers).properties(title=title, height=500)
    
    st.altair_chart(chart, use_container_width=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "distribution_points" not in st.session_state:
    st.session_state.distribution_points = load_distribution_points()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    
if "range_choice" not in st.session_state:
    st.session_state.range_choice = "Daily"

if "show_chart_values" not in st.session_state:
    st.session_state.show_chart_values = True  # Show values by default

if "show_data_tables" not in st.session_state:
    st.session_state.show_data_tables = False

if "compare_period" not in st.session_state:
    st.session_state.compare_period = "None"

if "highlight_peaks_valleys" not in st.session_state:
    st.session_state.highlight_peaks_valleys = False

if "highlight_weekends" not in st.session_state:
    st.session_state.highlight_weekends = False

if "show_anomalies" not in st.session_state:
    st.session_state.show_anomalies = False

# Try to load previously analyzed data
if not st.session_state.data_loaded:
    loaded_data = load_analyzed_data()
    if loaded_data:
        st.session_state.sessions_agg = loaded_data['sessions_agg']
        st.session_state.heat = loaded_data['heat']
        st.session_state.rides = loaded_data['rides']
        st.session_state.data_updated = loaded_data['timestamp']
        st.session_state.data_loaded = True

# ============================================================================
# SIDEBAR - DISTRIBUTION POINTS
# ============================================================================

st.sidebar.title("📍 1. Distribution Points")

# Add cache clearing button at the top
if st.sidebar.button("🗑️ Clear All Cached Data", type="secondary"):
    import shutil
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(exist_ok=True)
    st.session_state.clear()
    st.success("✅ All cached data cleared! Please reload your data.")
    st.rerun()

with st.sidebar.expander("➕ Add Points from File", expanded=False):
    area_for_upload = st.selectbox("Select Area", AREA_NAMES, key="area_upload_select")
    
    points_file = st.file_uploader(
        f"Upload points for {area_for_upload}",
        type=["csv", "xlsx"],
        key=f"points_upload_{area_for_upload}",
        help="File should have columns: area, lat, lng, neighborhood (point name will be used if neighborhood column doesn't exist)"
    )
    
    if points_file and st.button(f"Load Points for {area_for_upload}"):
        try:
            if points_file.name.endswith('.csv'):
                df = pd.read_csv(points_file)
            else:
                df = pd.read_excel(points_file)
            
            cols_lower = {c.lower(): c for c in df.columns}
            
            lat_col = cols_lower.get('lat', cols_lower.get('latitude'))
            lng_col = cols_lower.get('lng', cols_lower.get('longitude', cols_lower.get('lon')))
            # Prioritize 'neighborhood' column, fall back to 'point name' if not found
            neighborhood_col = cols_lower.get('neighborhood', cols_lower.get('point name', cols_lower.get('name')))
            
            if not (lat_col and lng_col and neighborhood_col):
                st.error(f"File must have columns: 'lat', 'lng', and 'neighborhood' (or 'point name')")
                st.info(f"Found columns: {df.columns.tolist()}")
            else:
                new_points = []
                for _, row in df.iterrows():
                    new_points.append({
                        "area": area_for_upload,
                        "neighborhood": str(row[neighborhood_col]),
                        "lat": float(row[lat_col]),
                        "lon": float(row[lng_col])
                    })
                
                st.session_state.distribution_points.extend(new_points)
                save_distribution_points(st.session_state.distribution_points)
                
                st.success(f"✅ Added {len(new_points)} points for {area_for_upload}!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if st.session_state.distribution_points:
    st.sidebar.success(f"✅ {len(st.session_state.distribution_points)} points saved")
    
    points_df = pd.DataFrame(st.session_state.distribution_points)
    
    if 'area' in points_df.columns:
        area_counts = points_df.groupby('area').size()
        
        with st.sidebar.expander("📋 View Points by Area", expanded=False):
            for area in AREA_NAMES:
                count = area_counts.get(area, 0)
                st.write(f"**{area}:** {count} points")
            
            st.markdown("---")
            st.dataframe(points_df, use_container_width=True, hide_index=True)
    else:
        with st.sidebar.expander("📋 View Points", expanded=False):
            st.dataframe(points_df, use_container_width=True, hide_index=True)
    
    if st.sidebar.button("🗑️ Clear All Points"):
        st.session_state.distribution_points = []
        if DIST_POINTS_FILE.exists():
            DIST_POINTS_FILE.unlink()
        st.rerun()

# ============================================================================
# SIDEBAR - DATA FILES
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.title("📂 2. Data Files")

if st.session_state.data_loaded:
    st.sidebar.success("✅ Data loaded")
    if hasattr(st.session_state, 'data_updated'):
        st.sidebar.caption(f"Updated: {st.session_state.data_updated.strftime('%Y-%m-%d %H:%M')}")
    
    # Export Data Option
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💾 Save Your Work**")
    st.sidebar.caption("⏱️ Quick restore after app sleep!")
    
    if ANALYZED_DATA_FILE.exists():
        # Get file size
        file_size_mb = ANALYZED_DATA_FILE.stat().st_size / (1024 * 1024)
        
        with open(ANALYZED_DATA_FILE, 'rb') as f:
            st.sidebar.download_button(
                label=f"📥 Export Data ({file_size_mb:.1f} MB)",
                data=f.read(),
                file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                help="Download processed data. Upload this file to quickly restore your dashboard after app sleep!"
            )
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Re-analyze", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.data_updated = None
            if ANALYZED_DATA_FILE.exists():
                ANALYZED_DATA_FILE.unlink()
            st.rerun()
    
    # ADD MORE DATA SECTION (when data is already loaded)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**➕ Add More Data**")
    st.sidebar.caption("📅 Append new dates to existing data")
    
    with st.sidebar.expander("📊 Upload Additional Files", expanded=False):
        st.info("💡 Upload files with NEW dates to add to your existing dataset")
        
        append_sfiles = st.file_uploader(
            "Sessions (new dates)", 
            type=["csv", "xlsx"], 
            accept_multiple_files=True, 
            key="append_sessions_upload"
        )
        append_hfiles = st.file_uploader(
            "Heat Data (new dates)", 
            type=["csv", "xlsx"], 
            accept_multiple_files=True, 
            key="append_heat_upload"
        )
        
        if st.button("➕ Append & Merge", type="primary", use_container_width=True):
            if not (append_sfiles and append_hfiles):
                st.error("Upload files for both Sessions and Heat Data")
            elif not st.session_state.distribution_points:
                st.error("Add distribution points first")
            else:
                # Set append mode flag
                append_mode = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load existing data
                status_text.text("📂 Loading existing data...")
                existing_sessions = st.session_state.sessions_agg
                existing_heat = st.session_state.heat
                
                st.info(f"✅ Found existing data: {len(existing_sessions):,} session records")
                
                try:
                    # Process new Sessions files
                    status_text.text("📊 Processing new sessions files...")
                    sessions_agg_list = []
                    
                    for i, sfile in enumerate(append_sfiles):
                        progress = (i / (len(append_sfiles) + len(append_hfiles)))
                        progress_bar.progress(progress)
                        status_text.text(f"📊 Processing session file {i+1}/{len(append_sfiles)}: {sfile.name}")
                        
                        if sfile.name.endswith('.csv'):
                            sessions = pd.read_csv(sfile)
                        else:
                            sessions = pd.read_excel(sfile)
                        
                        if 'Session_Date' in sessions.columns:
                            sessions['timestamp'] = pd.to_datetime(sessions['Session_Date'], errors='coerce')
                        elif 'Created At (Local)' in sessions.columns:
                            sessions['timestamp'] = pd.to_datetime(sessions['Created At (Local)'], errors='coerce')
                        else:
                            for col in sessions.columns:
                                if 'date' in col.lower():
                                    sessions['timestamp'] = pd.to_datetime(sessions[col], errors='coerce')
                                    break
                        
                        sessions = sessions[sessions['timestamp'].notna()].copy()
                        sessions = assign_sessions_to_areas(sessions, st.session_state.distribution_points)
                        agg = aggregate_sessions(sessions)
                        sessions_agg_list.append(agg)
                        
                        del sessions
                        gc.collect()
                    
                    new_sessions_agg = pd.concat(sessions_agg_list, ignore_index=True)
                    new_sessions_agg = new_sessions_agg.groupby(['Area', 'Neighborhood', 'date']).sum().reset_index()
                    
                    # Merge with existing
                    status_text.text("🔗 Merging with existing session data...")
                    combined_sessions = pd.concat([existing_sessions, new_sessions_agg], ignore_index=True)
                    sessions_agg = combined_sessions.groupby(['Area', 'Neighborhood', 'date']).sum().reset_index()
                    status_text.text(f"✅ Merged: {sessions_agg['sessions_count'].sum():,} total sessions")
                    
                    del sessions_agg_list, new_sessions_agg, combined_sessions
                    gc.collect()
                    
                    # Process new Heat files
                    status_text.text("🔥 Processing new heat files...")
                    heat_list = []
                    
                    for i, hfile in enumerate(append_hfiles):
                        progress = ((len(append_sfiles) + i) / (len(append_sfiles) + len(append_hfiles)))
                        progress_bar.progress(progress)
                        status_text.text(f"🔥 Loading heat file {i+1}/{len(append_hfiles)}: {hfile.name}")
                        
                        if hfile.name.endswith('.csv'):
                            heat = pd.read_csv(hfile)
                        else:
                            heat = pd.read_excel(hfile)
                        
                        heat_list.append(heat)
                    
                    new_heat = pd.concat(heat_list, ignore_index=True)
                    
                    if 'Start Date - Local' in new_heat.columns:
                        new_heat['timestamp'] = pd.to_datetime(new_heat['Start Date - Local'], errors='coerce')
                    else:
                        new_heat['timestamp'] = pd.to_datetime(new_heat['Start Date'], errors='coerce')
                    
                    new_heat = new_heat[new_heat['timestamp'].notna()].copy()
                    new_heat['Area'] = new_heat['Area'].astype('category')
                    
                    # Merge with existing
                    status_text.text("🔗 Merging with existing heat data...")
                    combined_heat = pd.concat([existing_heat, new_heat], ignore_index=True)
                    heat = combined_heat.drop_duplicates(subset=['timestamp', 'Area', 'Neighborhood Name'], keep='last')
                    status_text.text(f"✅ Merged: {len(heat):,} total heat records")
                    
                    del heat_list, new_heat, combined_heat
                    gc.collect()
                    
                    # Save merged data
                    progress_bar.progress(0.95)
                    status_text.text("💾 Saving merged data...")
                    
                    rides = pd.DataFrame()
                    save_analyzed_data(sessions_agg, heat, rides)
                    
                    st.session_state.sessions_agg = sessions_agg
                    st.session_state.heat = heat
                    st.session_state.rides = rides
                    st.session_state.data_updated = datetime.now()
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Merge complete!")
                    st.success("🎉 Data successfully appended! Don't forget to export the combined data.")
                    
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Append failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

if not st.session_state.data_loaded:
    # Check if there's an existing analyzed data file
    if ANALYZED_DATA_FILE.exists():
        st.sidebar.info("💡 Previous data found!")
        if st.sidebar.button("⚡ Load Previous Data", use_container_width=True, type="primary"):
            try:
                loaded_data = load_analyzed_data()
                if loaded_data and 'sessions_agg' in loaded_data and 'heat' in loaded_data:
                    st.session_state.sessions_agg = loaded_data['sessions_agg']
                    st.session_state.heat = loaded_data['heat']
                    st.session_state.data_loaded = True
                    if 'timestamp' in loaded_data:
                        st.session_state.data_updated = loaded_data['timestamp']
                    else:
                        st.session_state.data_updated = datetime.now()
                    st.sidebar.success("✅ Previous data loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Invalid data format")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load: {str(e)}")
        
        st.sidebar.markdown("---")
    
    # Import Previously Exported Data
    st.sidebar.markdown("**📤 Quick Restore**")
    st.sidebar.caption("Upload previously exported data")
    
    import_file = st.sidebar.file_uploader(
        "Import Saved Data",
        type=["pkl"],
        key="import_data",
        help="Upload a .pkl file you previously exported to quickly restore your dashboard"
    )
    
    if import_file is not None:
        if st.sidebar.button("⚡ Import & Load", use_container_width=True, type="primary"):
            try:
                # Save the uploaded file
                with open(ANALYZED_DATA_FILE, 'wb') as f:
                    f.write(import_file.read())
                
                # Load it
                loaded_data = load_analyzed_data()
                if loaded_data and 'sessions_agg' in loaded_data and 'heat' in loaded_data:
                    st.session_state.sessions_agg = loaded_data['sessions_agg']
                    st.session_state.heat = loaded_data['heat']
                    st.session_state.data_loaded = True
                    if 'timestamp' in loaded_data:
                        st.session_state.data_updated = loaded_data['timestamp']
                    else:
                        st.session_state.data_updated = datetime.now()
                    st.sidebar.success("✅ Data imported successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Invalid data file format")
            except Exception as e:
                st.sidebar.error(f"❌ Import failed: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Upload New Files:**")
    st.sidebar.caption("💡 You can upload multiple files - they will be combined automatically")
    
    # Append mode option
    append_mode = st.sidebar.checkbox(
        "➕ Append to existing data",
        value=False,
        help="Add new dates to your existing data instead of replacing it. Great for adding Jan 11-20 to existing Jan 1-10 data!"
    )
    
    if append_mode:
        st.sidebar.info("📌 Append Mode: New data will be added to existing data")
    
    sfiles = st.sidebar.file_uploader("Sessions", type=["csv", "xlsx"], accept_multiple_files=True, key="sessions_upload")
    hfiles = st.sidebar.file_uploader("Heat Data", type=["csv", "xlsx"], accept_multiple_files=True, key="heat_upload")
    
    if st.sidebar.button("🚀 Analyze", type="primary", use_container_width=True):
        if not (sfiles and hfiles):
            st.sidebar.error("Upload at least one file for each type")
        elif not st.session_state.distribution_points:
            st.sidebar.error("Add distribution points first")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load existing data if in append mode
            existing_sessions = None
            existing_heat = None
            
            if append_mode:
                status_text.text("📂 Loading existing data...")
                loaded_data = load_analyzed_data()
                if loaded_data and 'sessions_agg' in loaded_data and 'heat' in loaded_data:
                    existing_sessions = loaded_data['sessions_agg']
                    existing_heat = loaded_data['heat']
                    st.sidebar.info(f"✅ Found existing data: {len(existing_sessions):,} session records")
                else:
                    st.sidebar.warning("⚠️ No existing data found. Will process as new data.")
                    append_mode = False  # Turn off append mode if no existing data
            
            try:
                # Process Sessions
                status_text.text("📊 Processing sessions files...")
                sessions_agg_list = []
                
                for i, sfile in enumerate(sfiles):
                    progress = (i / (len(sfiles) + len(hfiles)))
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Processing sessions file {i+1}/{len(sfiles)}: {sfile.name}")
                    
                    if sfile.name.endswith('.csv'):
                        sessions = pd.read_csv(sfile)
                    else:
                        sessions = pd.read_excel(sfile)
                    
                    status_text.text(f"📊 Loaded {len(sessions):,} sessions from {sfile.name}")
                    
                    if 'Session_Date' in sessions.columns:
                        sessions['timestamp'] = pd.to_datetime(sessions['Session_Date'], errors='coerce')
                    elif 'Created At (Local)' in sessions.columns:
                        sessions['timestamp'] = pd.to_datetime(sessions['Created At (Local)'], errors='coerce')
                    else:
                        for col in sessions.columns:
                            if 'date' in col.lower():
                                sessions['timestamp'] = pd.to_datetime(sessions[col], errors='coerce')
                                break
                    
                    if 'timestamp' not in sessions.columns:
                        st.error(f"❌ No date column found in {sfile.name}!")
                        continue
                    
                    invalid_timestamps = sessions['timestamp'].isna().sum()
                    if invalid_timestamps > 0:
                        st.warning(f"⚠️ Removing {invalid_timestamps:,} sessions with invalid timestamps")
                        sessions = sessions[sessions['timestamp'].notna()].copy()
                    
                    if len(sessions) == 0:
                        st.error(f"❌ No valid sessions in {sfile.name}!")
                        continue
                    
                    status_text.text(f"🗺️ Assigning {len(sessions):,} sessions to areas...")
                    sessions = assign_sessions_to_areas(sessions, st.session_state.distribution_points)
                    
                    status_text.text(f"📉 Aggregating sessions from {sfile.name}...")
                    agg = aggregate_sessions(sessions)
                    sessions_agg_list.append(agg)
                    
                    del sessions
                    gc.collect()
                    
                    status_text.text(f"✅ Processed {sfile.name}")
                
                status_text.text("🔗 Combining aggregated sessions...")
                
                if len(sessions_agg_list) == 0:
                    st.error("❌ No valid sessions processed!")
                    st.stop()
                
                sessions_agg = pd.concat(sessions_agg_list, ignore_index=True)
                # Group by Area, Neighborhood, AND date to preserve neighborhood-level data
                sessions_agg = sessions_agg.groupby(['Area', 'Neighborhood', 'date']).sum().reset_index()
                
                # Merge with existing data if in append mode
                if append_mode and existing_sessions is not None:
                    status_text.text("🔗 Merging with existing session data...")
                    # Combine new and existing data
                    combined_sessions = pd.concat([existing_sessions, sessions_agg], ignore_index=True)
                    # Remove duplicates (keep latest), group by Area, Neighborhood, date
                    sessions_agg = combined_sessions.groupby(['Area', 'Neighborhood', 'date']).sum().reset_index()
                    status_text.text(f"✅ Merged: {sessions_agg['sessions_count'].sum():,} total sessions")
                
                status_text.text(f"✅ Total: {sessions_agg['sessions_count'].sum():,} sessions")
                
                del sessions_agg_list
                gc.collect()
                
                # Process Heat Data
                status_text.text("🔥 Processing heatdata files...")
                heat_list = []
                
                for i, hfile in enumerate(hfiles):
                    progress = ((len(sfiles) + i) / (len(sfiles) + len(hfiles)))
                    progress_bar.progress(progress)
                    status_text.text(f"🔥 Loading heatdata file {i+1}/{len(hfiles)}: {hfile.name}")
                    
                    if hfile.name.endswith('.csv'):
                        heat = pd.read_csv(hfile)
                    else:
                        heat = pd.read_excel(hfile)
                    
                    heat_list.append(heat)
                    status_text.text(f"✅ Loaded {len(heat):,} rows")
                
                status_text.text("🔗 Combining heatdata files...")
                heat = pd.concat(heat_list, ignore_index=True)
                
                if 'Start Date - Local' in heat.columns:
                    heat['timestamp'] = pd.to_datetime(heat['Start Date - Local'], errors='coerce')
                else:
                    heat['timestamp'] = pd.to_datetime(heat['Start Date'], errors='coerce')
                
                heat = heat[heat['timestamp'].notna()].copy()
                heat['Area'] = heat['Area'].astype('category')
                
                # Merge with existing data if in append mode
                if append_mode and existing_heat is not None:
                    status_text.text("🔗 Merging with existing heat data...")
                    # Combine new and existing heat data
                    combined_heat = pd.concat([existing_heat, heat], ignore_index=True)
                    # Remove exact duplicates based on timestamp and other key columns
                    heat = combined_heat.drop_duplicates(subset=['timestamp', 'Area', 'Neighborhood Name'], keep='last')
                    status_text.text(f"✅ Merged: {len(heat):,} total heat records")
                
                status_text.text(f"✅ Total heatdata: {len(heat):,} rows")
                
                del heat_list
                gc.collect()
                
                # Save
                progress_bar.progress(0.95)
                status_text.text("💾 Saving analyzed data...")
                
                rides = pd.DataFrame()
                
                save_analyzed_data(sessions_agg, heat, rides)
                
                st.session_state.sessions_agg = sessions_agg
                st.session_state.heat = heat
                st.session_state.rides = rides
                st.session_state.data_updated = datetime.now()
                st.session_state.data_loaded = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                st.success(f"✅ Processed {sessions_agg['sessions_count'].sum():,} sessions and {len(heat):,} heatdata rows!")
                st.balloons()
                
                import time
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

if not st.session_state.data_loaded:
    st.title("📊 Supply & Demand Dashboard")
    st.info("👈 Upload data files and add distribution points in the sidebar to get started")
    st.stop()

# Get data
sessions_agg = st.session_state.sessions_agg
heat = st.session_state.heat
rides = st.session_state.rides

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Supply & Demand Dashboard")
st.markdown("---")

# ============================================================================
# FILTERS - More compact and organized
# ============================================================================

with st.expander("🔍 **Filters & Settings**", expanded=True):
    # Row 1: Main filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_areas = st.multiselect(
            "**Select Areas**",
            AREA_NAMES,
            default=AREA_NAMES
        )
    
    with col2:
        date_min = sessions_agg['date'].min()
        date_max = sessions_agg['date'].max()
        
        if pd.isna(date_min) or pd.isna(date_max):
            st.error("⚠️ No valid dates found")
            st.stop()
        
        # Convert to date objects
        min_date = date_min.date()
        max_date = date_max.date()
        
        # Calculate default start date (20 days back, but not before min_date)
        default_start = max_date - timedelta(days=20)
        if default_start < min_date:
            default_start = min_date
        
        # Use date_input with proper default values
        date_range = st.date_input(
            "**Date Range**",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        
        # Handle both tuple and single date returns
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        elif isinstance(date_range, tuple) and len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date = end_date = date_range
    
    with col3:
        st.session_state.range_choice = st.selectbox(
            "**Time Grouping**",
            ["Daily", "Weekly", "Monthly"]
        )
    
    with col4:
        st.session_state.compare_period = st.selectbox(
            "**Period Comparison**",
            ["None", "Last Week", "Last Month"],
            index=0
        )
    
    st.markdown("")  # Small spacing
    
    # Row 2: Chart settings
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.session_state.show_chart_values = st.checkbox("🔢 Show Values on Charts", value=True)
    
    with col6:
        st.session_state.highlight_peaks_valleys = st.checkbox("🔺 Highlight Peaks & Valleys", value=False)
    
    with col7:
        st.session_state.show_data_tables = st.checkbox("📋 Show Data Tables", value=False)
    
    st.markdown("")  # Small spacing
    
    # Row 3: Highlight options
    col8, col9, col10 = st.columns(3)
    
    with col8:
        st.session_state.show_anomalies = st.checkbox("⚠️ Mark Anomalies", value=False)
    
    with col9:
        st.session_state.highlight_weekends = st.checkbox("📅 Highlight Weekends (Thu-Fri-Sat)", value=False)

st.markdown("")  # Spacing

# ============================================================================
# APPLY FILTERS
# ============================================================================

sessions_filtered = sessions_agg[
    (sessions_agg['Area'].isin(selected_areas)) &
    (sessions_agg['date'] >= pd.to_datetime(start_date)) &
    (sessions_agg['date'] <= pd.to_datetime(end_date))
].copy()

heat_filtered = heat[
    (heat['Area'].isin(selected_areas)) &
    (heat['timestamp'].dt.date >= start_date) &
    (heat['timestamp'].dt.date <= end_date)
].copy()

if heat_filtered['Area'].dtype.name == 'category':
    heat_filtered['Area'] = heat_filtered['Area'].cat.remove_unused_categories()

if len(rides) > 0 and 'timestamp' in rides.columns:
    rides_filtered = rides[
        (rides['timestamp'].dt.date >= start_date) &
        (rides['timestamp'].dt.date <= end_date)
    ].copy()
else:
    rides_filtered = rides.copy()

# ============================================================================
# ENHANCED SCORECARDS with Previous Period Comparison
# ============================================================================

st.markdown("### 📊 Key Metrics")

# Get current period length and calculate previous period
period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1  # +1 to include both dates
prev_start = pd.to_datetime(start_date) - timedelta(days=period_days)
prev_end = pd.to_datetime(start_date) - timedelta(days=1)

# Get previous period data
prev_sessions_agg = sessions_agg[
    (sessions_agg['Area'].isin(selected_areas)) &
    (sessions_agg['date'] >= prev_start) &
    (sessions_agg['date'] <= prev_end)
]

prev_heat = heat[
    (heat['Area'].isin(selected_areas)) &
    (heat['timestamp'].dt.date >= prev_start.date()) &
    (heat['timestamp'].dt.date <= prev_end.date())
]

# Current period totals
total_sessions = int(sessions_filtered['sessions_count'].sum())
total_rides = int(heat_filtered['Rides'].sum())
fulfillment = (total_rides / total_sessions * 100) if total_sessions > 0 else 0

# Previous period totals
prev_total_sessions = int(prev_sessions_agg['sessions_count'].sum())
prev_total_rides = int(prev_heat['Rides'].sum())
prev_fulfillment = (prev_total_rides / prev_total_sessions * 100) if prev_total_sessions > 0 else 0

# Calculate % changes
sessions_change_pct = ((total_sessions - prev_total_sessions) / prev_total_sessions * 100) if prev_total_sessions > 0 else 0
rides_change_pct = ((total_rides - prev_total_rides) / prev_total_rides * 100) if prev_total_rides > 0 else 0
fulfillment_change_pct = fulfillment - prev_fulfillment  # Absolute change for fulfillment

# Average vehicles
active_per_snapshot = heat_filtered.groupby('timestamp')['Active Vehicles'].sum()
avg_active_vehicles = active_per_snapshot.mean() * 1.075

prev_active_per_snapshot = prev_heat.groupby('timestamp')['Active Vehicles'].sum()
prev_avg_active_vehicles = prev_active_per_snapshot.mean() * 1.075 if len(prev_active_per_snapshot) > 0 else 0
vehicles_change_pct = ((avg_active_vehicles - prev_avg_active_vehicles) / prev_avg_active_vehicles * 100) if prev_avg_active_vehicles > 0 else 0

# Display metrics in 4 columns
m1, m2, m3, m4 = st.columns(4)

with m1:
    format_metric_card(
        "📍 Total Sessions",
        total_sessions,
        delta=f"{sessions_change_pct:+.1f}%",
        delta_color="normal",
        sparkline_data=None,
        color=AREA_COLORS["Alexandria"]
    )

with m2:
    format_metric_card(
        "🚗 Total Rides",
        total_rides,
        delta=f"{rides_change_pct:+.1f}%",
        delta_color="normal",
        sparkline_data=None,
        color=AREA_COLORS["Downtown 1"]
    )

with m3:
    delta_color = "normal" if fulfillment >= FULFILLMENT_TARGET else "inverse"
    format_metric_card(
        "📈 Fulfillment",
        fulfillment,
        suffix="%",
        delta=f"{fulfillment_change_pct:+.1f}%",
        delta_color=delta_color,
        sparkline_data=None,
        color="#20BF55" if fulfillment >= FULFILLMENT_TARGET else "#EE5A6F"
    )

with m4:
    format_metric_card(
        "🚕 Avg Active Vehicles",
        avg_active_vehicles,
        delta=f"{vehicles_change_pct:+.1f}%",
        delta_color="normal",
        sparkline_data=None,
        color=AREA_COLORS["Zahraa El Maadi"]
    )

# Show comparison period info
st.caption(f"📊 Comparing to previous {period_days} day(s): {prev_start.strftime('%b %d')} - {prev_end.strftime('%b %d, %Y')}")

st.markdown("")  # Single line spacing

# ============================================================================
# CHARTS - Cleaner, easier to scan
# ============================================================================

# Create tabs
tab1, tab2 = st.tabs(["📊 Overview", "🏘️ Neighborhood View"])

with tab1:
    # Sessions Chart
    col_title, col_toggle = st.columns([5, 1])
    with col_title:
        st.markdown("### Sessions Over Time")
    with col_toggle:
        sessions_total_view = st.checkbox("Total", value=False, key="sessions_total_toggle")
    
    sessions_filtered['bucket'] = bucket_data(sessions_filtered, 'date', st.session_state.range_choice)
    sessions_chart = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    
    if len(sessions_chart) > 0:
        if sessions_total_view:
            # Show total (sum of all areas)
            sessions_total = sessions_chart.sum(axis=1).to_frame(name='Total')
            line_chart(sessions_total, "", "Sessions", show_values=st.session_state.show_chart_values)
            show_data_table(sessions_total, "Sessions Data (Total)")
        else:
            # Show individual areas
            line_chart(sessions_chart, "", "Sessions", show_values=st.session_state.show_chart_values)
            show_data_table(sessions_chart, "Sessions Data")
    else:
        st.info("No session data for selected filters")
    
    st.markdown("")  # Spacing
    
    # Rides Chart
    col_title, col_toggle = st.columns([5, 1])
    with col_title:
        st.markdown("### Rides Over Time")
    with col_toggle:
        rides_total_view = st.checkbox("Total", value=False, key="rides_total_toggle")
    
    heat_filtered['bucket'] = bucket_data(heat_filtered, 'timestamp', st.session_state.range_choice)
    rides_chart = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    if len(rides_chart) > 0:
        if rides_total_view:
            # Show total (sum of all areas)
            rides_total = rides_chart.sum(axis=1).to_frame(name='Total')
            line_chart(rides_total, "", "Rides", show_values=st.session_state.show_chart_values)
            show_data_table(rides_total, "Rides Data (Total)")
        else:
            # Show individual areas
            line_chart(rides_chart, "", "Rides", show_values=st.session_state.show_chart_values)
            show_data_table(rides_chart, "Rides Data")
    else:
        st.info("No rides data for selected filters")
    
    st.markdown("")  # Spacing
    
    # Fulfillment Chart
    col_title, col_toggle = st.columns([5, 1])
    with col_title:
        st.markdown("### Fulfillment % Over Time")
    with col_toggle:
        fulfillment_total_view = st.checkbox("Total", value=False, key="fulfillment_total_toggle")
    
    sessions_for_fulfill = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    rides_for_fulfill = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    sessions_aligned, rides_aligned = sessions_for_fulfill.align(rides_for_fulfill, fill_value=0)
    fulfillment_chart = (rides_aligned / sessions_aligned.replace(0, np.nan)) * 100
    fulfillment_chart = fulfillment_chart.fillna(0)
    
    if len(fulfillment_chart) > 0:
        if fulfillment_total_view:
            # Calculate overall fulfillment % (total rides / total sessions)
            total_sessions = sessions_aligned.sum(axis=1)
            total_rides = rides_aligned.sum(axis=1)
            fulfillment_total = (total_rides / total_sessions.replace(0, np.nan) * 100).fillna(0).to_frame(name='Total')
            line_chart(fulfillment_total, "", "Percent", show_values=st.session_state.show_chart_values)
            show_data_table(fulfillment_total, "Fulfillment % Data (Total)")
        else:
            # Show individual areas
            line_chart(fulfillment_chart, "", "Percent", show_values=st.session_state.show_chart_values)
            show_data_table(fulfillment_chart, "Fulfillment % Data")
    else:
        st.info("No data for fulfillment calculation")
    
    st.markdown("")  # Spacing
    
    # Comprehensive Sessions Breakdown Chart with toggle
    st.markdown("### 📊 Sessions Performance: Demand, Fulfillment & Missed Opportunity")
    
    view_mode = st.checkbox("🔥 Heatmap View", value=False, key="performance_heatmap_toggle", help="Toggle between bar charts and heatmap visualization")
    
    # Calculate data for the breakdown
    sessions_breakdown = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    rides_breakdown = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    # Align data
    sessions_aligned_bd, rides_aligned_bd = sessions_breakdown.align(rides_breakdown, fill_value=0)
    missed_bd = (sessions_aligned_bd - rides_aligned_bd).clip(lower=0)
    fulfillment_bd = (rides_aligned_bd / sessions_aligned_bd.replace(0, np.nan) * 100).fillna(0)
    
    if len(sessions_aligned_bd) > 0:
        
        if view_mode:
            # HEATMAP VIEW - Three heatmaps stacked vertically (standardized)
            
            # Prepare combined data for all areas
            sessions_data = sessions_breakdown.T  # Transpose so areas are rows, dates are columns
            rides_data = rides_breakdown.T
            missed_data = missed_bd.T
            fulfillment_data = fulfillment_bd.T
            
            # Calculate vehicle metrics per area per date
            vehicle_metrics = {}
            for area in selected_areas:
                area_heat = heat_filtered[heat_filtered['Area'] == area].copy()
                if len(area_heat) > 0:
                    area_heat['bucket'] = bucket_data(area_heat, 'timestamp', st.session_state.range_choice)
                    
                    # Group by bucket and calculate averages
                    active_by_date = area_heat.groupby('bucket')['Active Vehicles'].mean() * 1.075
                    urgent_by_date = area_heat.groupby('bucket')['Urgent Vehicles'].mean() * 1.075
                    effective_by_date = area_heat.groupby('bucket')['Effective Active Vehicles'].mean() * 1.075
                    
                    vehicle_metrics[area] = {
                        'active': active_by_date,
                        'urgent': urgent_by_date,
                        'effective': effective_by_date
                    }
            
            # Standard height for all heatmaps
            standard_height = max(200, len(selected_areas) * 60)
            
            # ===== HEATMAP 1: SESSIONS (TOTAL DEMAND) =====
            st.markdown("#### 📊 Sessions (Total Demand)")
            st.caption("Total session requests per area and date (colors normalized per area)")
            
            # Normalize sessions per area
            sessions_normalized = sessions_data.copy()
            for idx in range(len(sessions_normalized)):
                row = sessions_normalized.iloc[idx]
                row_min = row.min()
                row_max = row.max()
                if row_max > row_min:
                    sessions_normalized.iloc[idx] = (row - row_min) / (row_max - row_min)
                else:
                    sessions_normalized.iloc[idx] = 0
            
            # Build hover text and display text for sessions
            hover_sessions = []
            display_sessions = []
            for area_idx, area in enumerate(sessions_data.index):
                row_hover = []
                row_display = []
                for date_idx, date in enumerate(sessions_data.columns):
                    sessions = sessions_data.iloc[area_idx, date_idx]
                    rides = rides_data.iloc[area_idx, date_idx] if area in rides_data.index else 0
                    missed = missed_data.iloc[area_idx, date_idx] if area in missed_data.index else 0
                    fulfillment = fulfillment_data.iloc[area_idx, date_idx] if area in fulfillment_data.index else 0
                    
                    active = 0
                    urgent = 0
                    effective = 0
                    if area in vehicle_metrics and date in vehicle_metrics[area]['active'].index:
                        active = vehicle_metrics[area]['active'][date]
                        urgent = vehicle_metrics[area]['urgent'][date]
                        effective = vehicle_metrics[area]['effective'][date]
                    
                    hover_text = (
                        f"<b>{area}</b><br>"
                        f"{date.strftime('%b %d, %Y')}<br>"
                        f"<br>"
                        f"<b>Demand & Performance:</b><br>"
                        f"Sessions: {sessions:,.0f}<br>"
                        f"Rides: {rides:,.0f}<br>"
                        f"Missed: {missed:,.0f}<br>"
                        f"Fulfillment: {fulfillment:.1f}%<br>"
                        f"<br>"
                        f"<b>Fleet Metrics:</b><br>"
                        f"Active Scooters: {active:.0f}<br>"
                        f"Urgent Scooters: {urgent:.0f}<br>"
                        f"Effective Active: {effective:.0f}"
                    )
                    row_hover.append(hover_text)
                    
                    # Format display text
                    if sessions >= 1000:
                        display_text = f"{sessions/1000:.1f}k"
                    else:
                        display_text = f"{sessions:.0f}"
                    row_display.append(display_text)
                    
                hover_sessions.append(row_hover)
                display_sessions.append(row_display)
            
            fig_sessions = go.Figure(data=go.Heatmap(
                z=sessions_normalized.values,
                x=sessions_data.columns.strftime('%b %d'),
                y=sessions_data.index,
                text=display_sessions,  # Formatted display text
                texttemplate='%{text}',
                textfont=dict(size=12, family='Arial Black'),
                hovertext=hover_sessions,
                hovertemplate='%{hovertext}<extra></extra>',
                colorscale=[
                    [0, '#E8F5E9'],      # Very light green
                    [0.25, '#81C784'],   # Light green
                    [0.5, '#4CAF50'],    # Green
                    [0.75, '#388E3C'],   # Dark green
                    [1, '#1B5E20']       # Very dark green
                ],
                colorbar=dict(
                    title="Relative<br>Demand",
                    thickness=20,
                    len=0.7,
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Med', 'High']
                ),
                showscale=True,
                zmin=0,
                zmax=1
            ))
            
            fig_sessions.update_layout(
                height=standard_height,
                xaxis_title="Date",
                yaxis_title="Area",
                xaxis=dict(tickangle=-45, side='bottom'),
                margin=dict(l=0, r=0, t=10, b=50),
                autosize=True
            )
            
            st.plotly_chart(fig_sessions, use_container_width=True)
            
            # ===== HEATMAP 2: FULFILLMENT % =====
            st.markdown("#### ✅ Fulfillment %")
            st.caption("Fulfillment percentage per area and date (colors normalized per area)")
            
            # Normalize fulfillment per area
            fulfillment_normalized = fulfillment_data.copy()
            for idx in range(len(fulfillment_normalized)):
                row = fulfillment_normalized.iloc[idx]
                row_min = row.min()
                row_max = row.max()
                if row_max > row_min:
                    fulfillment_normalized.iloc[idx] = (row - row_min) / (row_max - row_min)
                else:
                    fulfillment_normalized.iloc[idx] = 0
            
            # Build hover text and display text for fulfillment
            hover_fulfillment = []
            display_fulfillment = []
            for area_idx, area in enumerate(fulfillment_data.index):
                row_hover = []
                row_display = []
                for date_idx, date in enumerate(fulfillment_data.columns):
                    sessions = sessions_data.iloc[area_idx, date_idx]
                    rides = rides_data.iloc[area_idx, date_idx] if area in rides_data.index else 0
                    missed = missed_data.iloc[area_idx, date_idx] if area in missed_data.index else 0
                    fulfillment = fulfillment_data.iloc[area_idx, date_idx]
                    
                    active = 0
                    urgent = 0
                    effective = 0
                    if area in vehicle_metrics and date in vehicle_metrics[area]['active'].index:
                        active = vehicle_metrics[area]['active'][date]
                        urgent = vehicle_metrics[area]['urgent'][date]
                        effective = vehicle_metrics[area]['effective'][date]
                    
                    hover_text = (
                        f"<b>{area}</b><br>"
                        f"{date.strftime('%b %d, %Y')}<br>"
                        f"<br>"
                        f"<b>Demand & Performance:</b><br>"
                        f"Sessions: {sessions:,.0f}<br>"
                        f"Rides: {rides:,.0f}<br>"
                        f"Missed: {missed:,.0f}<br>"
                        f"Fulfillment: {fulfillment:.1f}%<br>"
                        f"<br>"
                        f"<b>Fleet Metrics:</b><br>"
                        f"Active Scooters: {active:.0f}<br>"
                        f"Urgent Scooters: {urgent:.0f}<br>"
                        f"Effective Active: {effective:.0f}"
                    )
                    row_hover.append(hover_text)
                    row_display.append(f"{fulfillment:.0f}%")
                    
                hover_fulfillment.append(row_hover)
                display_fulfillment.append(row_display)
            
            fig_fulfillment = go.Figure(data=go.Heatmap(
                z=fulfillment_normalized.values,
                x=fulfillment_data.columns.strftime('%b %d'),
                y=fulfillment_data.index,
                text=display_fulfillment,  # Formatted display text
                texttemplate='%{text}',
                textfont=dict(size=12, family='Arial Black'),
                hovertext=hover_fulfillment,
                hovertemplate='%{hovertext}<extra></extra>',
                colorscale=[
                    [0, '#E3F2FD'],      # Very light blue (low fulfillment)
                    [0.25, '#90CAF9'],   # Light blue
                    [0.5, '#42A5F5'],    # Medium blue
                    [0.75, '#1E88E5'],   # Dark blue
                    [1, '#0D47A1']       # Very dark blue (high fulfillment)
                ],
                colorbar=dict(
                    title="Relative<br>Performance",
                    thickness=20,
                    len=0.7,
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Med', 'High']
                ),
                showscale=True,
                zmin=0,
                zmax=1
            ))
            
            fig_fulfillment.update_layout(
                height=standard_height,
                xaxis_title="Date",
                yaxis_title="Area",
                xaxis=dict(tickangle=-45, side='bottom'),
                margin=dict(l=0, r=0, t=10, b=50),
                autosize=True
            )
            
            st.plotly_chart(fig_fulfillment, use_container_width=True)
            
            # ===== HEATMAP 3: MISSED OPPORTUNITY =====
            st.markdown("#### ❌ Missed Opportunity")
            st.caption("Unfulfilled sessions per area and date (colors normalized per area)")
            
            # Normalize missed per area
            missed_normalized = missed_data.copy()
            for idx in range(len(missed_normalized)):
                row = missed_normalized.iloc[idx]
                row_min = row.min()
                row_max = row.max()
                if row_max > row_min:
                    missed_normalized.iloc[idx] = (row - row_min) / (row_max - row_min)
                else:
                    missed_normalized.iloc[idx] = 0
            
            # Build hover text and display text for missed
            hover_missed = []
            display_missed = []
            for area_idx, area in enumerate(missed_data.index):
                row_hover = []
                row_display = []
                for date_idx, date in enumerate(missed_data.columns):
                    sessions = sessions_data.iloc[area_idx, date_idx]
                    rides = rides_data.iloc[area_idx, date_idx] if area in rides_data.index else 0
                    missed = missed_data.iloc[area_idx, date_idx]
                    fulfillment = fulfillment_data.iloc[area_idx, date_idx] if area in fulfillment_data.index else 0
                    
                    active = 0
                    urgent = 0
                    effective = 0
                    if area in vehicle_metrics and date in vehicle_metrics[area]['active'].index:
                        active = vehicle_metrics[area]['active'][date]
                        urgent = vehicle_metrics[area]['urgent'][date]
                        effective = vehicle_metrics[area]['effective'][date]
                    
                    hover_text = (
                        f"<b>{area}</b><br>"
                        f"{date.strftime('%b %d, %Y')}<br>"
                        f"<br>"
                        f"<b>Demand & Performance:</b><br>"
                        f"Sessions: {sessions:,.0f}<br>"
                        f"Rides: {rides:,.0f}<br>"
                        f"Missed: {missed:,.0f}<br>"
                        f"Fulfillment: {fulfillment:.1f}%<br>"
                        f"<br>"
                        f"<b>Fleet Metrics:</b><br>"
                        f"Active Scooters: {active:.0f}<br>"
                        f"Urgent Scooters: {urgent:.0f}<br>"
                        f"Effective Active: {effective:.0f}"
                    )
                    row_hover.append(hover_text)
                    
                    # Format display text
                    if missed >= 1000:
                        display_text = f"{missed/1000:.1f}k"
                    else:
                        display_text = f"{missed:.0f}"
                    row_display.append(display_text)
                    
                hover_missed.append(row_hover)
                display_missed.append(row_display)
            
            fig_missed = go.Figure(data=go.Heatmap(
                z=missed_normalized.values,
                x=missed_data.columns.strftime('%b %d'),
                y=missed_data.index,
                text=display_missed,  # Formatted display text
                texttemplate='%{text}',
                textfont=dict(size=12, family='Arial Black'),
                hovertext=hover_missed,
                hovertemplate='%{hovertext}<extra></extra>',
                colorscale=[
                    [0, '#FFF3E0'],      # Very light orange (low missed)
                    [0.25, '#FFB74D'],   # Light orange
                    [0.5, '#FF9800'],    # Orange
                    [0.75, '#F57C00'],   # Dark orange
                    [1, '#E65100']       # Very dark orange/red (high missed)
                ],
                colorbar=dict(
                    title="Relative<br>Missed",
                    thickness=20,
                    len=0.7,
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Med', 'High']
                ),
                showscale=True,
                zmin=0,
                zmax=1
            ))
            
            fig_missed.update_layout(
                height=standard_height,
                xaxis_title="Date",
                yaxis_title="Area",
                xaxis=dict(tickangle=-45, side='bottom'),
                margin=dict(l=0, r=0, t=10, b=50),
                autosize=True
            )
            
            st.plotly_chart(fig_missed, use_container_width=True)
            
            # Summary metrics for combined view
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            total_sessions = sessions_data.values.sum()
            total_rides = rides_data.values.sum()
            total_missed = missed_data.values.sum()
            avg_fulfillment = (total_rides / total_sessions * 100) if total_sessions > 0 else 0
            
            with col1:
                st.metric("📊 Total Sessions", f"{total_sessions:,.0f}",
                         help="Total session requests across all areas and dates")
            with col2:
                st.metric("✅ Total Rides", f"{total_rides:,.0f}",
                         delta=f"{avg_fulfillment:.1f}% fulfillment",
                         help="Total rides fulfilled")
            with col3:
                st.metric("❌ Total Missed", f"{total_missed:,.0f}",
                         delta=f"-{(total_missed/total_sessions*100):.1f}%",
                         delta_color="inverse",
                         help="Total unfulfilled sessions")
            with col4:
                worst_area = missed_data.sum(axis=1).idxmax()
                worst_area_missed = missed_data.sum(axis=1).max()
                st.metric("⚠️ Highest Impact", worst_area,
                         help=f"Area with most missed: {worst_area_missed:,.0f}")
            
        else:
            # ORIGINAL BAR CHART VIEW
            # Show comprehensive chart for each area
            for area in selected_areas:
                if area not in sessions_aligned_bd.columns:
                    continue
                    
                st.markdown(f"#### 📍 {area}")
                
                # Prepare data
                breakdown_df = pd.DataFrame({
                    'Date': sessions_aligned_bd.index,
                    'Total Sessions': sessions_aligned_bd[area],
                    'Fulfilled (Rides)': rides_aligned_bd[area] if area in rides_aligned_bd.columns else 0,
                    'Missed Opportunity': missed_bd[area] if area in missed_bd.columns else 0,
                    'Fulfillment %': fulfillment_bd[area] if area in fulfillment_bd.columns else 0
                })
                
                # Calculate vehicle metrics for this area
                area_heat = heat_filtered[heat_filtered['Area'] == area].copy()
                area_heat['bucket'] = bucket_data(area_heat, 'timestamp', st.session_state.range_choice)
                
                active_by_date = area_heat.groupby('bucket')['Active Vehicles'].mean() * 1.075
                urgent_by_date = area_heat.groupby('bucket')['Urgent Vehicles'].mean() * 1.075
                effective_by_date = area_heat.groupby('bucket')['Effective Active Vehicles'].mean() * 1.075
                
                # Add vehicle metrics to breakdown_df
                breakdown_df['Active Scooters'] = breakdown_df['Date'].map(active_by_date).fillna(0)
                breakdown_df['Urgent Scooters'] = breakdown_df['Date'].map(urgent_by_date).fillna(0)
                breakdown_df['Effective Active'] = breakdown_df['Date'].map(effective_by_date).fillna(0)
                
                # Create dual-axis chart with plotly
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add stacked bars with enhanced tooltip
                fig.add_trace(
                    go.Bar(
                        x=breakdown_df['Date'],
                        y=breakdown_df['Fulfilled (Rides)'],
                        name='Fulfilled (Rides)',
                        marker_color='#20BF55',
                        customdata=np.column_stack((
                            breakdown_df['Total Sessions'],
                            breakdown_df['Fulfilled (Rides)'],
                            breakdown_df['Missed Opportunity'],
                            breakdown_df['Fulfillment %'],
                            breakdown_df['Active Scooters'],
                            breakdown_df['Urgent Scooters'],
                            breakdown_df['Effective Active']
                        )),
                        hovertemplate='<b>Fulfilled</b><br>' +
                                      '%{x|%b %d, %Y}<br><br>' +
                                      '<b>Demand & Performance:</b><br>' +
                                      'Sessions: %{customdata[0]:,.0f}<br>' +
                                      'Rides: %{customdata[1]:,.0f}<br>' +
                                      'Missed: %{customdata[2]:,.0f}<br>' +
                                      'Fulfillment: %{customdata[3]:.1f}%<br><br>' +
                                      '<b>Fleet Metrics:</b><br>' +
                                      'Active Scooters: %{customdata[4]:.0f}<br>' +
                                      'Urgent Scooters: %{customdata[5]:.0f}<br>' +
                                      'Effective Active: %{customdata[6]:.0f}' +
                                      '<extra></extra>'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Bar(
                        x=breakdown_df['Date'],
                        y=breakdown_df['Missed Opportunity'],
                        name='Missed Opportunity',
                        marker_color='#FF6B6B',
                        customdata=np.column_stack((
                            breakdown_df['Total Sessions'],
                            breakdown_df['Fulfilled (Rides)'],
                            breakdown_df['Missed Opportunity'],
                            breakdown_df['Fulfillment %'],
                            breakdown_df['Active Scooters'],
                            breakdown_df['Urgent Scooters'],
                            breakdown_df['Effective Active']
                        )),
                        hovertemplate='<b>Missed</b><br>' +
                                      '%{x|%b %d, %Y}<br><br>' +
                                      '<b>Demand & Performance:</b><br>' +
                                      'Sessions: %{customdata[0]:,.0f}<br>' +
                                      'Rides: %{customdata[1]:,.0f}<br>' +
                                      'Missed: %{customdata[2]:,.0f}<br>' +
                                      'Fulfillment: %{customdata[3]:.1f}%<br><br>' +
                                      '<b>Fleet Metrics:</b><br>' +
                                      'Active Scooters: %{customdata[4]:.0f}<br>' +
                                      'Urgent Scooters: %{customdata[5]:.0f}<br>' +
                                      'Effective Active: %{customdata[6]:.0f}' +
                                      '<extra></extra>'
                    ),
                    secondary_y=False
                )
                
                # Add fulfillment % line with enhanced tooltip
                fig.add_trace(
                    go.Scatter(
                        x=breakdown_df['Date'],
                        y=breakdown_df['Fulfillment %'],
                        name='Fulfillment %',
                        mode='lines+markers',
                        line=dict(color='#FFC107', width=3),
                        marker=dict(size=8, color='#FFC107'),
                        yaxis='y2',
                        customdata=np.column_stack((
                            breakdown_df['Total Sessions'],
                            breakdown_df['Fulfilled (Rides)'],
                            breakdown_df['Missed Opportunity'],
                            breakdown_df['Fulfillment %'],
                            breakdown_df['Active Scooters'],
                            breakdown_df['Urgent Scooters'],
                            breakdown_df['Effective Active']
                        )),
                        hovertemplate='<b>Fulfillment</b><br>' +
                                      '%{x|%b %d, %Y}<br><br>' +
                                      '<b>Demand & Performance:</b><br>' +
                                      'Sessions: %{customdata[0]:,.0f}<br>' +
                                      'Rides: %{customdata[1]:,.0f}<br>' +
                                      'Missed: %{customdata[2]:,.0f}<br>' +
                                      'Fulfillment: %{customdata[3]:.1f}%<br><br>' +
                                      '<b>Fleet Metrics:</b><br>' +
                                      'Active Scooters: %{customdata[4]:.0f}<br>' +
                                      'Urgent Scooters: %{customdata[5]:.0f}<br>' +
                                      'Effective Active: %{customdata[6]:.0f}' +
                                      '<extra></extra>'
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_layout(
                    barmode='stack',
                    hovermode='x unified',
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                # Update axes
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="Sessions Count", secondary_y=False)
                fig.update_yaxes(title_text="Fulfillment %", range=[0, 100], secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insight cards below chart
                col1, col2, col3, col4 = st.columns(4)
                
                total_sessions = breakdown_df['Total Sessions'].sum()
                total_fulfilled = breakdown_df['Fulfilled (Rides)'].sum()
                total_missed = breakdown_df['Missed Opportunity'].sum()
                avg_fulfillment = breakdown_df['Fulfillment %'].mean()
                
                with col1:
                    st.metric("📊 Total Demand", f"{total_sessions:,.0f}", 
                             help="Total sessions requested")
                with col2:
                    st.metric("✅ Fulfilled", f"{total_fulfilled:,.0f}", 
                             delta=f"{(total_fulfilled/total_sessions*100 if total_sessions > 0 else 0):.1f}%",
                             help="Sessions that got rides")
                with col3:
                    st.metric("❌ Missed", f"{total_missed:,.0f}", 
                             delta=f"-{(total_missed/total_sessions*100 if total_sessions > 0 else 0):.1f}%",
                             delta_color="inverse",
                             help="Sessions without rides")
                with col4:
                    fulfillment_color = "normal" if avg_fulfillment >= FULFILLMENT_TARGET else "inverse"
                    st.metric("📈 Avg Fulfillment", f"{avg_fulfillment:.1f}%",
                             delta=f"{avg_fulfillment - FULFILLMENT_TARGET:.1f}% vs target",
                             delta_color=fulfillment_color,
                             help=f"Target: {FULFILLMENT_TARGET}%")
                
                st.markdown("---")
    else:
        st.info("No data available for performance breakdown")
    
    st.markdown("")  # Spacing
    
    # Missed Opportunity Chart
    col_title, col_toggle = st.columns([5, 1])
    with col_title:
        st.markdown("### Missed Opportunity (Unfulfilled Sessions)")
    with col_toggle:
        missed_total_view = st.checkbox("Total", value=False, key="missed_total_toggle")
    
    sessions_for_missed = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    rides_for_missed = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    sessions_aligned_missed, rides_aligned_missed = sessions_for_missed.align(rides_for_missed, fill_value=0)
    missed_opportunity_chart = sessions_aligned_missed - rides_aligned_missed
    missed_opportunity_chart = missed_opportunity_chart.clip(lower=0)  # Ensure no negative values
    
    if len(missed_opportunity_chart) > 0:
        if missed_total_view:
            # Show total (sum of all areas)
            missed_total = missed_opportunity_chart.sum(axis=1).to_frame(name='Total')
            line_chart(missed_total, "", "Missed Sessions", show_values=st.session_state.show_chart_values)
            show_data_table(missed_total, "Missed Opportunity Data (Total)")
        else:
            # Show individual areas
            line_chart(missed_opportunity_chart, "", "Missed Sessions", show_values=st.session_state.show_chart_values)
            show_data_table(missed_opportunity_chart, "Missed Opportunity Data")
    else:
        st.info("No data for missed opportunity calculation")
    
    st.markdown("")  # Spacing
    
    # Vehicle Charts in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Effective Vehicles")
        
        eff_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Effective Active Vehicles'].sum().reset_index()
        eff_per_snapshot['bucket'] = bucket_data(eff_per_snapshot, 'timestamp', st.session_state.range_choice)
        eff_vehicles_chart = eff_per_snapshot.groupby(['bucket', 'Area'])['Effective Active Vehicles'].mean().unstack().fillna(0)
        eff_vehicles_chart = eff_vehicles_chart * 1.075
        
        if len(eff_vehicles_chart) > 0:
            line_chart(eff_vehicles_chart, "", "Vehicles", show_values=st.session_state.show_chart_values)
        else:
            st.info("No vehicle data")
    
    with col2:
        st.markdown("### Urgent Vehicles")
        
        urgent_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Urgent Vehicles'].sum().reset_index()
        urgent_per_snapshot['bucket'] = bucket_data(urgent_per_snapshot, 'timestamp', st.session_state.range_choice)
        urgent_vehicles_chart = urgent_per_snapshot.groupby(['bucket', 'Area'])['Urgent Vehicles'].mean().unstack().fillna(0)
        urgent_vehicles_chart = urgent_vehicles_chart * 1.075
        
        if len(urgent_vehicles_chart) > 0:
            line_chart(urgent_vehicles_chart, "", "Vehicles", show_values=st.session_state.show_chart_values)
        else:
            st.info("No urgent vehicle data")
    
    st.markdown("")  # Spacing
    
    # Active Vehicles
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Active Vehicles")
        
        active_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Active Vehicles'].sum().reset_index()
        active_per_snapshot['bucket'] = bucket_data(active_per_snapshot, 'timestamp', st.session_state.range_choice)
        active_vehicles_chart = active_per_snapshot.groupby(['bucket', 'Area'])['Active Vehicles'].mean().unstack().fillna(0)
        active_vehicles_chart = active_vehicles_chart * 1.075
        
        if len(active_vehicles_chart) > 0:
            line_chart(active_vehicles_chart, "", "Vehicles", show_values=st.session_state.show_chart_values)
        else:
            st.info("No active vehicle data")
    
    st.markdown("")  # Spacing
    
    # Top Neighborhoods
    st.markdown("### 🏆 Top Neighborhoods")
    
    col_toggle, col_empty = st.columns([1, 3])
    with col_toggle:
        top_metric = st.selectbox(
            "Show top 5 by:",
            ["Most Rides", "Most Sessions", "Highest Fulfillment %"],
            label_visibility="collapsed"
        )
    
    if top_metric == "Most Rides":
        neigh_rides = heat_filtered.groupby('Neighborhood Name')['Rides'].sum().sort_values(ascending=False)
        top_5_neighs = neigh_rides.head(5).index.tolist()
        
        heat_top5 = heat_filtered[heat_filtered['Neighborhood Name'].isin(top_5_neighs)].copy()
        top5_chart = heat_top5.groupby(['bucket', 'Neighborhood Name'])['Rides'].sum().unstack().fillna(0)
        
        if len(top5_chart) > 0:
            df_chart = top5_chart.reset_index()
            date_col = df_chart.columns[0]
            df_chart = df_chart.melt(date_col, var_name="Neighborhood", value_name="Rides")
            
            chart = (
                alt.Chart(df_chart)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X(f"{date_col}:T", title="", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Rides:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("Neighborhood:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[f"{date_col}:T", "Neighborhood:N", "Rides:Q"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No neighborhood data")
    else:
        st.info(f"Top Neighborhoods by {top_metric.split(' by ')[-1]} - Feature coming soon")
    
    st.markdown("")  # Spacing
    
    # ========================================================================
    # NEW ANALYTICS CHARTS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📈 Performance Analytics")
    
    # 1. VEHICLE UTILIZATION RATE
    st.markdown("### 🚗 Vehicle Utilization Rate")
    st.caption("Rides per vehicle per day - Shows efficiency and redeployment opportunities")
    
    # Calculate utilization per area
    utilization_data = []
    
    for area in selected_areas:
        area_heat = heat_filtered[heat_filtered['Area'] == area]
        
        if len(area_heat) > 0:
            # Total rides
            total_rides = area_heat['Rides'].sum()
            
            # Average vehicles (effective)
            avg_vehicles = area_heat.groupby('timestamp')['Effective Active Vehicles'].mean().mean() * 1.075
            
            # Days in period
            days_in_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            
            # Utilization = rides per vehicle per day
            utilization = (total_rides / (avg_vehicles * days_in_period)) if avg_vehicles > 0 else 0
            
            utilization_data.append({
                'Area': area,
                'Total Rides': total_rides,
                'Avg Vehicles': avg_vehicles,
                'Rides per Vehicle per Day': utilization,
                'Days': days_in_period
            })
    
    if utilization_data:
        util_df = pd.DataFrame(utilization_data)
        
        # Create bar chart
        fig_util = go.Figure()
        
        fig_util.add_trace(go.Bar(
            x=util_df['Area'],
            y=util_df['Rides per Vehicle per Day'],
            marker_color=[AREA_COLORS.get(area, "#2E86DE") for area in util_df['Area']],
            text=util_df['Rides per Vehicle per Day'].apply(lambda x: f'{x:.1f}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                          'Utilization: %{y:.1f} rides/vehicle/day<br>' +
                          'Total Rides: ' + util_df['Total Rides'].apply(lambda x: f'{x:,.0f}') + '<br>' +
                          'Avg Vehicles: ' + util_df['Avg Vehicles'].apply(lambda x: f'{x:.0f}') +
                          '<extra></extra>'
        ))
        
        # Add target line (e.g., 8 rides per vehicle per day)
        target_utilization = 8
        fig_util.add_hline(
            y=target_utilization, 
            line_dash="dash", 
            line_color="orange",
            annotation_text=f"Target: {target_utilization} rides/vehicle/day",
            annotation_position="right"
        )
        
        fig_util.update_layout(
            yaxis_title="Rides per Vehicle per Day",
            xaxis_title="",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_util, use_container_width=True)
        
        # Add metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_util = util_df['Rides per Vehicle per Day'].mean()
        total_vehicles = util_df['Avg Vehicles'].sum()
        underutilized = len(util_df[util_df['Rides per Vehicle per Day'] < target_utilization])
        best_area = util_df.loc[util_df['Rides per Vehicle per Day'].idxmax(), 'Area']
        
        with col1:
            st.metric("📊 Avg Utilization", f"{avg_util:.1f} rides/veh/day")
        with col2:
            st.metric("🚗 Total Vehicles", f"{total_vehicles:.0f}")
        with col3:
            st.metric("⚠️ Below Target", f"{underutilized} areas",
                     delta_color="inverse" if underutilized > 0 else "normal")
        with col4:
            st.metric("🏆 Best Performance", best_area)
        
        st.info(f"💡 **Insight:** Areas below {target_utilization} rides/vehicle/day may have excess capacity. Consider redeploying vehicles to high-demand areas.")
    else:
        st.info("No vehicle data available")
    
    st.markdown("")  # Spacing
    
    # 2. PEAK VS OFF-PEAK COMPARISON
    st.markdown("### ⚡ Peak vs Off-Peak Performance")
    st.caption("Compare rides and vehicle availability during peak hours vs off-peak")
    
    # Define peak hours (7-10 AM and 5-9 PM)
    def is_peak_hour(hour):
        return (7 <= hour <= 10) or (17 <= hour <= 21)
    
    # Calculate peak vs off-peak metrics from heat data (has timestamps with hours)
    peak_data = []
    
    for area in selected_areas:
        area_heat = heat_filtered[heat_filtered['Area'] == area].copy()
        
        if len(area_heat) > 0:
            # Add hour column from timestamp
            area_heat['hour'] = area_heat['timestamp'].dt.hour
            
            # Split into peak and off-peak
            peak_heat = area_heat[area_heat['hour'].apply(is_peak_hour)]
            offpeak_heat = area_heat[~area_heat['hour'].apply(is_peak_hour)]
            
            # Calculate metrics
            peak_rides = peak_heat['Rides'].sum()
            offpeak_rides = offpeak_heat['Rides'].sum()
            
            # Average vehicles during each period
            peak_vehicles = peak_heat.groupby('timestamp')['Effective Active Vehicles'].mean().mean() * 1.075 if len(peak_heat) > 0 else 0
            offpeak_vehicles = offpeak_heat.groupby('timestamp')['Effective Active Vehicles'].mean().mean() * 1.075 if len(offpeak_heat) > 0 else 0
            
            # Rides per vehicle (efficiency)
            peak_efficiency = (peak_rides / peak_vehicles) if peak_vehicles > 0 else 0
            offpeak_efficiency = (offpeak_rides / offpeak_vehicles) if offpeak_vehicles > 0 else 0
            
            # Calculate what fulfillment would be (rides / rides is 100%, but we show utilization)
            # Using rides per vehicle as a proxy for demand pressure
            peak_data.append({
                'Area': area,
                'Peak Rides': peak_rides,
                'Peak Vehicles': peak_vehicles,
                'Peak Efficiency': peak_efficiency,
                'Off-Peak Rides': offpeak_rides,
                'Off-Peak Vehicles': offpeak_vehicles,
                'Off-Peak Efficiency': offpeak_efficiency,
                'Efficiency Gap': peak_efficiency - offpeak_efficiency
            })
    
    if peak_data:
        peak_df = pd.DataFrame(peak_data)
        
        # Create grouped bar chart for rides per vehicle (efficiency)
        fig_peak = go.Figure()
        
        # Calculate total rides for percentage
        peak_df['Peak %'] = (peak_df['Peak Rides'] / (peak_df['Peak Rides'] + peak_df['Off-Peak Rides']) * 100)
        peak_df['Off-Peak %'] = (peak_df['Off-Peak Rides'] / (peak_df['Peak Rides'] + peak_df['Off-Peak Rides']) * 100)
        
        fig_peak.add_trace(go.Bar(
            name='Peak Hours (7-10 AM, 5-9 PM)',
            x=peak_df['Area'],
            y=peak_df['Peak Efficiency'],
            marker_color='#FF6B6B',
            text=peak_df['Peak Efficiency'].apply(lambda x: f'{x:.1f}'),
            textposition='outside',
            customdata=np.column_stack((
                peak_df['Peak Rides'],
                peak_df['Peak %']
            )),
            hovertemplate='<b>%{x} - Peak Hours</b><br>' +
                          'Rides/Vehicle: %{y:.1f}<br>' +
                          'Total Rides: %{customdata[0]:,.0f}<br>' +
                          '% of Daily Rides: %{customdata[1]:.1f}%' +
                          '<extra></extra>'
        ))
        
        fig_peak.add_trace(go.Bar(
            name='Off-Peak Hours',
            x=peak_df['Area'],
            y=peak_df['Off-Peak Efficiency'],
            marker_color='#20BF55',
            text=peak_df['Off-Peak Efficiency'].apply(lambda x: f'{x:.1f}'),
            textposition='outside',
            customdata=np.column_stack((
                peak_df['Off-Peak Rides'],
                peak_df['Off-Peak %']
            )),
            hovertemplate='<b>%{x} - Off-Peak Hours</b><br>' +
                          'Rides/Vehicle: %{y:.1f}<br>' +
                          'Total Rides: %{customdata[0]:,.0f}<br>' +
                          '% of Daily Rides: %{customdata[1]:.1f}%' +
                          '<extra></extra>'
        ))
        
        # Add target line (e.g., 3 rides per vehicle during that period)
        target_efficiency = 3
        fig_peak.add_hline(
            y=target_efficiency,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Target: {target_efficiency} rides/vehicle",
            annotation_position="right"
        )
        
        fig_peak.update_layout(
            barmode='group',
            yaxis_title="Rides per Vehicle",
            xaxis_title="",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_peak, use_container_width=True)
        
        # Add insights
        col1, col2, col3 = st.columns(3)
        
        avg_peak_eff = peak_df['Peak Efficiency'].mean()
        avg_offpeak_eff = peak_df['Off-Peak Efficiency'].mean()
        avg_gap = peak_df['Efficiency Gap'].mean()
        
        total_peak_rides = peak_df['Peak Rides'].sum()
        total_offpeak_rides = peak_df['Off-Peak Rides'].sum()
        avg_peak_vehicles = peak_df['Peak Vehicles'].mean()
        avg_offpeak_vehicles = peak_df['Off-Peak Vehicles'].mean()
        
        with col1:
            st.metric("⚡ Peak Rides/Vehicle", f"{avg_peak_eff:.1f}",
                     help=f"Average: {avg_peak_vehicles:.0f} vehicles serving {total_peak_rides:,.0f} rides during 7-10 AM, 5-9 PM")
        with col2:
            st.metric("💤 Off-Peak Rides/Vehicle", f"{avg_offpeak_eff:.1f}",
                     help=f"Average: {avg_offpeak_vehicles:.0f} vehicles serving {total_offpeak_rides:,.0f} rides during other hours")
        with col3:
            delta_text = f"{'Higher' if avg_gap > 0 else 'Lower'} during peak"
            st.metric("📊 Efficiency Difference", f"{abs(avg_gap):.1f}",
                     delta=delta_text,
                     delta_color="inverse" if avg_gap < 0 else "normal",
                     help="Difference in rides per vehicle between peak and off-peak")
        
        # Insights
        if avg_gap > 1.5:
            st.success(f"✅ **Insight:** Vehicles are {avg_gap:.1f} rides/vehicle MORE efficient during peak hours ({avg_peak_eff:.1f} vs {avg_offpeak_eff:.1f}). This is good! Peak demand is being handled well, though you may still have capacity for more rides.")
        elif avg_gap < -1.5:
            st.warning(f"⚠️ **Insight:** Vehicles are {abs(avg_gap):.1f} rides/vehicle LESS efficient during peak hours ({avg_peak_eff:.1f} vs {avg_offpeak_eff:.1f}). This suggests vehicles may be idle during peaks or demand is in different locations than vehicles. Review vehicle positioning during 7-10 AM and 5-9 PM.")
        else:
            st.info(f"💡 **Insight:** Similar efficiency during peak ({avg_peak_eff:.1f}) and off-peak ({avg_offpeak_eff:.1f}) hours. Demand pressure is consistent throughout the day.")
    else:
        st.info("No data available for peak comparison")

with tab2:
    st.markdown("### 🏘️ Neighborhood View")
    st.caption("View metrics and trends for individual neighborhoods")
    
    # Check if we have neighborhood data
    if 'Neighborhood Name' in heat_filtered.columns and len(heat_filtered) > 0:
        
        # Group neighborhoods by area
        for area in selected_areas:
            # Filter heat data for this area only
            area_heat = heat_filtered[heat_filtered['Area'] == area].copy()
            
            if len(area_heat) > 0:
                # Get unique neighborhoods in this area
                neighborhoods = area_heat['Neighborhood Name'].unique()
                num_neighborhoods = len(neighborhoods)
                total_rides = int(area_heat['Rides'].sum())
                
                # Get sessions for this area if available
                has_neighborhood_sessions = 'Neighborhood' in sessions_filtered.columns
                
                if has_neighborhood_sessions:
                    area_sessions = sessions_filtered[sessions_filtered['Area'] == area].copy()
                    total_sessions = int(area_sessions['sessions_count'].sum()) if len(area_sessions) > 0 else 0
                    area_fulfillment = (total_rides / total_sessions * 100) if total_sessions > 0 else 0
                    
                    checkbox_label = f"📍 **{area}** ({num_neighborhoods} neighborhoods • {total_sessions:,} sessions • {total_rides:,} rides • {area_fulfillment:.1f}% fulfillment)"
                else:
                    checkbox_label = f"📍 **{area}** ({num_neighborhoods} neighborhoods • {total_rides:,} rides)"
                
                # Area with checkbox to expand/collapse
                area_key = f"show_area_{area.replace(' ', '_')}"
                show_area = st.checkbox(
                    checkbox_label,
                    value=False,
                    key=area_key
                )
                
                if show_area:
                    # Show each neighborhood as expandable dropdown
                    for neighborhood in sorted(neighborhoods):
                        # Filter data for this specific neighborhood
                        neigh_heat = area_heat[area_heat['Neighborhood Name'] == neighborhood].copy()
                        
                        # Calculate neighborhood totals
                        neigh_rides = int(neigh_heat['Rides'].sum())
                        neigh_avg_vehicles = neigh_heat.groupby('timestamp')['Active Vehicles'].mean().mean() * 1.075
                        
                        # Get sessions for this neighborhood if available
                        has_neighborhood_sessions = 'Neighborhood' in sessions_filtered.columns
                        
                        if has_neighborhood_sessions:
                            neigh_sessions_data = sessions_filtered[
                                (sessions_filtered['Area'] == area) & 
                                (sessions_filtered['Neighborhood'] == neighborhood)
                            ].copy()
                            neigh_total_sessions = int(neigh_sessions_data['sessions_count'].sum()) if len(neigh_sessions_data) > 0 else 0
                            neigh_fulfillment_pct = (neigh_rides / neigh_total_sessions * 100) if neigh_total_sessions > 0 else 0
                            
                            expander_title = f"**{neighborhood}** ({neigh_total_sessions:,} sessions, {neigh_rides:,} rides, {neigh_fulfillment_pct:.1f}% fulfillment)"
                        else:
                            neigh_total_sessions = 0
                            neigh_fulfillment_pct = 0
                            expander_title = f"**{neighborhood}** ({neigh_rides:,} rides, {neigh_avg_vehicles:.0f} avg vehicles)"
                        
                        with st.expander(expander_title, expanded=False):
                            
                            # Neighborhood KPIs
                            if has_neighborhood_sessions and neigh_total_sessions > 0:
                                # Full metrics with sessions and fulfillment
                                n1, n2, n3, n4, n5 = st.columns(5)
                                
                                with n1:
                                    st.metric("Total Sessions", f"{neigh_total_sessions:,}")
                                
                                with n2:
                                    st.metric("Total Rides", f"{neigh_rides:,}")
                                
                                with n3:
                                    st.metric("Fulfillment", f"{neigh_fulfillment_pct:.1f}%")
                                
                                with n4:
                                    avg_eff = neigh_heat.groupby('timestamp')['Effective Active Vehicles'].mean().mean() * 1.075
                                    st.metric("Avg Effective Vehicles", f"{avg_eff:.0f}")
                                
                                with n5:
                                    avg_urgent = neigh_heat.groupby('timestamp')['Urgent Vehicles'].mean().mean() * 1.075
                                    st.metric("Avg Urgent Vehicles", f"{avg_urgent:.0f}")
                            else:
                                # Limited metrics without sessions
                                n1, n2, n3 = st.columns(3)
                                
                                with n1:
                                    st.metric("Total Rides", f"{neigh_rides:,}")
                                
                                with n2:
                                    avg_eff = neigh_heat.groupby('timestamp')['Effective Active Vehicles'].mean().mean() * 1.075
                                    st.metric("Avg Effective Vehicles", f"{avg_eff:.0f}")
                                
                                with n3:
                                    avg_urgent = neigh_heat.groupby('timestamp')['Urgent Vehicles'].mean().mean() * 1.075
                                    st.metric("Avg Urgent Vehicles", f"{avg_urgent:.0f}")
                                
                                if not has_neighborhood_sessions:
                                    st.info("💡 Sessions data is not available at neighborhood level. Update aggregation to include neighborhood assignments.")
                            
                            st.markdown("")
                            
                            # Prepare data for charts
                            neigh_heat['bucket'] = bucket_data(neigh_heat, 'timestamp', st.session_state.range_choice)
                            
                            # COMPREHENSIVE PERFORMANCE CHART (if sessions available)
                            if has_neighborhood_sessions and neigh_total_sessions > 0:
                                st.markdown("**📊 Performance Overview: Demand, Fulfillment & Missed Opportunity**")
                                
                                # Prepare comprehensive data
                                sessions_by_date = neigh_sessions_data.groupby('date')['sessions_count'].sum()
                                rides_by_date = neigh_heat.groupby('bucket')['Rides'].sum()
                                
                                perf_df = pd.DataFrame({
                                    'Date': sessions_by_date.index,
                                    'Sessions': sessions_by_date.values
                                })
                                rides_df_temp = pd.DataFrame({
                                    'Date': rides_by_date.index,
                                    'Rides': rides_by_date.values
                                })
                                
                                perf_df = perf_df.merge(rides_df_temp, on='Date', how='outer').fillna(0)
                                perf_df['Missed'] = (perf_df['Sessions'] - perf_df['Rides']).clip(lower=0)
                                perf_df['Fulfillment %'] = (perf_df['Rides'] / perf_df['Sessions'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
                                
                                if len(perf_df) > 0:
                                    fig_perf = make_subplots(specs=[[{"secondary_y": True}]])
                                    
                                    # Stacked bars
                                    fig_perf.add_trace(
                                        go.Bar(
                                            x=perf_df['Date'],
                                            y=perf_df['Rides'],
                                            name='Fulfilled',
                                            marker_color='#20BF55',
                                            hovertemplate='<b>Fulfilled</b><br>%{y:,.0f}<br>%{x|%b %d}<extra></extra>'
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    fig_perf.add_trace(
                                        go.Bar(
                                            x=perf_df['Date'],
                                            y=perf_df['Missed'],
                                            name='Missed',
                                            marker_color='#FF6B6B',
                                            hovertemplate='<b>Missed</b><br>%{y:,.0f}<br>%{x|%b %d}<extra></extra>'
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    # Fulfillment line
                                    fig_perf.add_trace(
                                        go.Scatter(
                                            x=perf_df['Date'],
                                            y=perf_df['Fulfillment %'],
                                            name='Fulfillment %',
                                            mode='lines+markers',
                                            line=dict(color='#FFC107', width=2),
                                            marker=dict(size=6),
                                            yaxis='y2',
                                            hovertemplate='<b>Fulfillment</b><br>%{y:.1f}%<br>%{x|%b %d}<extra></extra>'
                                        ),
                                        secondary_y=True
                                    )
                                    
                                    fig_perf.update_layout(
                                        barmode='stack',
                                        hovermode='x unified',
                                        height=300,
                                        showlegend=True,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        margin=dict(l=0, r=0, t=30, b=0)
                                    )
                                    
                                    fig_perf.update_xaxes(title_text="")
                                    fig_perf.update_yaxes(title_text="Count", secondary_y=False)
                                    fig_perf.update_yaxes(title_text="Fulfillment %", range=[0, 100], secondary_y=True)
                                    
                                    st.plotly_chart(fig_perf, use_container_width=True)
                                
                                st.markdown("")  # Spacing
                            
                            # Show different chart layouts based on data availability
                            if has_neighborhood_sessions and neigh_total_sessions > 0:
                                # Full layout with sessions and fulfillment
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Sessions Over Time**")
                                    sessions_by_time = neigh_sessions_data.groupby('date')['sessions_count'].sum()
                                    
                                    if len(sessions_by_time) > 0:
                                        sessions_df = sessions_by_time.reset_index()
                                        sessions_df.columns = ['Date', 'Sessions']
                                        
                                        chart = alt.Chart(sessions_df).mark_line(
                                            point=alt.OverlayMarkDef(size=80, filled=True),
                                            strokeWidth=3,
                                            color=AREA_COLORS.get(area, "#2E86DE")
                                        ).encode(
                                            x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                            y=alt.Y('Sessions:Q', title='Sessions'),
                                            tooltip=[
                                                alt.Tooltip('Date:T', title='Date', format='%b %d, %Y'),
                                                alt.Tooltip('Sessions:Q', title='Sessions', format=',.0f')
                                            ]
                                        ).properties(height=250)
                                        
                                        st.altair_chart(chart, use_container_width=True)
                                
                                with col2:
                                    st.markdown("**Rides Over Time**")
                                    rides_by_time = neigh_heat.groupby('bucket')['Rides'].sum()
                                    
                                    if len(rides_by_time) > 0:
                                        rides_df = rides_by_time.reset_index()
                                        rides_df.columns = ['Date', 'Rides']
                                        
                                        chart = alt.Chart(rides_df).mark_line(
                                            point=alt.OverlayMarkDef(size=80, filled=True),
                                            strokeWidth=3,
                                            color=AREA_COLORS.get(area, "#2E86DE")
                                        ).encode(
                                            x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                            y=alt.Y('Rides:Q', title='Rides'),
                                            tooltip=[
                                                alt.Tooltip('Date:T', title='Date', format='%b %d, %Y'),
                                                alt.Tooltip('Rides:Q', title='Rides', format=',.0f')
                                            ]
                                        ).properties(height=250)
                                        
                                        st.altair_chart(chart, use_container_width=True)
                                
                                # Fulfillment chart
                                st.markdown("**Fulfillment Over Time**")
                                sessions_by_date = neigh_sessions_data.groupby('date')['sessions_count'].sum()
                                rides_by_date = neigh_heat.groupby('bucket')['Rides'].sum()
                                
                                fulfillment_df = pd.DataFrame({
                                    'Date': sessions_by_date.index,
                                    'Sessions': sessions_by_date.values
                                })
                                rides_df_temp = pd.DataFrame({
                                    'Date': rides_by_date.index,
                                    'Rides': rides_by_date.values
                                })
                                
                                fulfillment_df = fulfillment_df.merge(rides_df_temp, on='Date', how='outer').fillna(0)
                                fulfillment_df['Fulfillment'] = (fulfillment_df['Rides'] / fulfillment_df['Sessions'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
                                
                                if len(fulfillment_df) > 0:
                                    chart = alt.Chart(fulfillment_df).mark_line(
                                        point=alt.OverlayMarkDef(size=80, filled=True),
                                        strokeWidth=3,
                                        color="#20BF55" if neigh_fulfillment_pct >= FULFILLMENT_TARGET else "#EE5A6F"
                                    ).encode(
                                        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                        y=alt.Y('Fulfillment:Q', title='Fulfillment %', scale=alt.Scale(domain=[0, 100])),
                                        tooltip=[
                                            alt.Tooltip('Date:T', title='Date', format='%b %d, %Y'),
                                            alt.Tooltip('Fulfillment:Q', title='Fulfillment %', format='.1f')
                                        ]
                                    ).properties(height=250)
                                    
                                    st.altair_chart(chart, use_container_width=True)
                                
                                # Missed Opportunity chart
                                st.markdown("**Missed Opportunity Over Time**")
                                
                                # Calculate missed opportunity (sessions - rides)
                                missed_df = fulfillment_df.copy()
                                missed_df['Missed'] = (missed_df['Sessions'] - missed_df['Rides']).clip(lower=0)
                                
                                if len(missed_df) > 0 and missed_df['Missed'].sum() > 0:
                                    chart = alt.Chart(missed_df).mark_line(
                                        point=alt.OverlayMarkDef(size=80, filled=True),
                                        strokeWidth=3,
                                        color="#FF6B6B"
                                    ).encode(
                                        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                        y=alt.Y('Missed:Q', title='Missed Sessions'),
                                        tooltip=[
                                            alt.Tooltip('Date:T', title='Date', format='%b %d, %Y'),
                                            alt.Tooltip('Missed:Q', title='Missed Sessions', format=',.0f')
                                        ]
                                    ).properties(height=250)
                                    
                                    st.altair_chart(chart, use_container_width=True)
                            else:
                                # Limited layout - just rides
                                st.markdown("**Rides Over Time**")
                                rides_by_time = neigh_heat.groupby('bucket')['Rides'].sum()
                                
                                if len(rides_by_time) > 0:
                                    rides_df = rides_by_time.reset_index()
                                    rides_df.columns = ['Date', 'Rides']
                                    
                                    chart = alt.Chart(rides_df).mark_line(
                                        point=alt.OverlayMarkDef(size=80, filled=True),
                                        strokeWidth=3,
                                        color=AREA_COLORS.get(area, "#2E86DE")
                                    ).encode(
                                        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                        y=alt.Y('Rides:Q', title='Rides'),
                                        tooltip=[
                                            alt.Tooltip('Date:T', title='Date', format='%b %d, %Y'),
                                            alt.Tooltip('Rides:Q', title='Rides', format=',.0f')
                                        ]
                                    ).properties(height=300)
                                    
                                    st.altair_chart(chart, use_container_width=True)
                            
                            # Vehicles Over Time (always show)
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Active Vehicles**")
                                active_by_time = neigh_heat.groupby('bucket')['Active Vehicles'].mean() * 1.075
                                
                                if len(active_by_time) > 0:
                                    active_df = active_by_time.reset_index()
                                    active_df.columns = ['Date', 'Vehicles']
                                    
                                    chart = alt.Chart(active_df).mark_line(
                                        point=True,
                                        strokeWidth=2,
                                        color=AREA_COLORS.get(area, "#2E86DE")
                                    ).encode(
                                        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                        y=alt.Y('Vehicles:Q', title='Vehicles'),
                                        tooltip=[
                                            alt.Tooltip('Date:T', format='%b %d, %Y'),
                                            alt.Tooltip('Vehicles:Q', format=',.0f')
                                        ]
                                    ).properties(height=250)
                                    
                                    st.altair_chart(chart, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Effective Vehicles**")
                                eff_by_time = neigh_heat.groupby('bucket')['Effective Active Vehicles'].mean() * 1.075
                                
                                if len(eff_by_time) > 0:
                                    eff_df = eff_by_time.reset_index()
                                    eff_df.columns = ['Date', 'Vehicles']
                                    
                                    chart = alt.Chart(eff_df).mark_line(
                                        point=True,
                                        strokeWidth=2,
                                        color=AREA_COLORS.get(area, "#2E86DE")
                                    ).encode(
                                        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
                                        y=alt.Y('Vehicles:Q', title='Vehicles'),
                                        tooltip=[
                                            alt.Tooltip('Date:T', format='%b %d, %Y'),
                                            alt.Tooltip('Vehicles:Q', format=',.0f')
                                        ]
                                    ).properties(height=250)
                                    
                                    st.altair_chart(chart, use_container_width=True)
                
                st.markdown("")
        
    else:
        st.info("Neighborhood data not available. Heat data must include 'Neighborhood Name' column.")

st.markdown("---")
st.caption(f"📅 Last updated: {st.session_state.data_updated.strftime('%Y-%m-%d %H:%M:%S')}")

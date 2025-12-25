import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from scipy.spatial import cKDTree
import pickle
from datetime import datetime
import gc

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

# Area colors for charts
AREA_COLORS = {
    "Alexandria": "#4F9CF9",
    "Downtown 1": "#7C4DFF",
    "Maadi": "#2DC26B",
    "Masr El Gedida": "#FF6B6B",
    "Zahraa El Maadi": "#FFA726",
}

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
        'sessions_agg': sessions_agg,  # Pre-aggregated sessions
        'heat': heat,
        'rides': rides,
        'timestamp': datetime.now()
    }
    with open(ANALYZED_DATA_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_analyzed_data():
    """Load analyzed data from disk"""
    if ANALYZED_DATA_FILE.exists():
        with open(ANALYZED_DATA_FILE, 'rb') as f:
            return pickle.load(f)
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
    
    # Remove sessions with invalid coordinates (NaN, inf, or zero)
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
    
    # Mark invalid sessions immediately
    sessions.loc[~valid_mask, 'Area'] = 'Out of Fence'
    sessions.loc[~valid_mask, 'Neighborhood'] = 'Out of Fence'
    
    # Only process valid sessions
    valid_sessions = sessions[valid_mask].copy()
    
    if len(valid_sessions) == 0:
        st.error("No valid sessions found with proper coordinates!")
        return sessions
    
    # Extract coordinates using detected column names
    session_coords = valid_sessions[[lat_col, lon_col]].values
    point_coords = all_points[["lat", "lon"]].values
    
    # Convert to radians
    session_rad = np.radians(session_coords)
    point_rad = np.radians(point_coords)
    
    # Convert to Cartesian coordinates for fast distance calculation
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
    
    # Calculate actual distances in meters
    nearest_point_coords = point_rad[nearest_indices]
    dlat = nearest_point_coords[:, 0] - session_rad[:, 0]
    dlon = nearest_point_coords[:, 1] - session_rad[:, 1]
    a = np.sin(dlat/2)**2 + np.cos(session_rad[:, 0]) * np.cos(nearest_point_coords[:, 0]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distances_meters = c * 6371000
    
    # Assign areas (within 200m threshold)
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
    
    # Update the original sessions with assignments
    sessions.loc[valid_mask, 'Area'] = valid_sessions['Area'].values
    sessions.loc[valid_mask, 'Neighborhood'] = valid_sessions['Neighborhood'].values
    
    return sessions

# ============================================================================
# HELPER FUNCTIONS - AGGREGATION
# ============================================================================

def aggregate_sessions(sessions):
    """
    Aggregate sessions to reduce from millions of rows to thousands.
    Groups by Area and Date (daily aggregation).
    """
    # Ensure timestamp exists and is valid
    if 'timestamp' not in sessions.columns:
        raise ValueError("Sessions must have 'timestamp' column before aggregation")
    
    # Remove any NaT timestamps
    sessions = sessions[sessions['timestamp'].notna()].copy()
    
    if len(sessions) == 0:
        # Return empty aggregation
        return pd.DataFrame(columns=['Area', 'date', 'sessions_count'])
    
    # Create date column (daily aggregation) - keep as datetime
    sessions['date'] = sessions['timestamp'].dt.floor('D')
    
    # Group and count
    agg = sessions.groupby(['Area', 'date']).size().reset_index(name='sessions_count')
    
    return agg

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

def line_chart(df, title="", y_label=""):
    """Create line chart with legend"""
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.melt(date_col, var_name="Series", value_name="Value")
    
    # Apply smoothing if enabled
    if st.session_state.get("smooth_enabled", False):
        win = st.session_state.get("smooth_window", 3)
        df["Value"] = df.groupby("Series")["Value"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
    
    # Get colors for series that exist in data
    series_in_data = df["Series"].unique()
    color_domain = [s for s in AREA_COLORS.keys() if s in series_in_data]
    color_range = [AREA_COLORS[s] for s in color_domain]
    
    selection = alt.selection_multi(fields=["Series"], bind="legend")
    
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=4)
        .encode(
            x=alt.X(
                f"{date_col}:T", 
                title="",
                axis=alt.Axis(
                    format="%b %d",  # Format: "Dec 03" (no time)
                    labelAngle=-45,
                    labelFontSize=14,
                    labelOverlap="greedy"  # Prevent duplicate labels
                )
            ),
            y=alt.Y("Value:Q", title=y_label, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(orient="bottom", labelFontSize=12, titleFontSize=13),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Series:N", title="Area"),
                alt.Tooltip("Value:Q", title=y_label, format=",.1f")
            ],
        )
        .add_selection(selection)
        .properties(title=title, height=400)  # Increased from 310 to 400
    )
    
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

if "smooth_enabled" not in st.session_state:
    st.session_state.smooth_enabled = False

if "smooth_window" not in st.session_state:
    st.session_state.smooth_window = 3

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

with st.sidebar.expander("➕ Add Points from File", expanded=False):
    area_for_upload = st.selectbox("Select Area", AREA_NAMES, key="area_upload_select")
    
    points_file = st.file_uploader(
        f"Upload points for {area_for_upload}",
        type=["csv", "xlsx"],
        key=f"points_upload_{area_for_upload}",
        help="File should have columns: lat, lng, point name (or neighborhood)"
    )
    
    if points_file and st.button(f"Load Points for {area_for_upload}"):
        try:
            # Load file
            if points_file.name.endswith('.csv'):
                df = pd.read_csv(points_file)
            else:
                df = pd.read_excel(points_file)
            
            # Expected columns: area, lat, lng, point name (or neighborhood)
            # Map column names (case-insensitive)
            cols_lower = {c.lower(): c for c in df.columns}
            
            lat_col = cols_lower.get('lat', cols_lower.get('latitude'))
            lng_col = cols_lower.get('lng', cols_lower.get('longitude', cols_lower.get('lon')))
            name_col = cols_lower.get('point name', cols_lower.get('neighborhood', cols_lower.get('name')))
            
            if not (lat_col and lng_col and name_col):
                st.error(f"File must have columns: 'lat', 'lng', and 'point name' (or 'neighborhood')")
                st.info(f"Found columns: {df.columns.tolist()}")
            else:
                # Add all points from file
                new_points = []
                for _, row in df.iterrows():
                    new_points.append({
                        "area": area_for_upload,
                        "neighborhood": str(row[name_col]),
                        "lat": float(row[lat_col]),
                        "lon": float(row[lng_col])
                    })
                
                # Add to existing points
                st.session_state.distribution_points.extend(new_points)
                save_distribution_points(st.session_state.distribution_points)
                
                st.success(f"✅ Added {len(new_points)} points for {area_for_upload}!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if st.session_state.distribution_points:
    st.sidebar.success(f"✅ {len(st.session_state.distribution_points)} points saved")
    
    # Show count per area
    points_df = pd.DataFrame(st.session_state.distribution_points)
    
    # Check if 'area' column exists, otherwise skip the count
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

if not st.session_state.data_loaded:
    st.sidebar.markdown("**Upload Files:**")
    st.sidebar.caption("💡 You can upload multiple files - they will be combined automatically")
    
    sfiles = st.sidebar.file_uploader("Sessions", type=["csv", "xlsx"], accept_multiple_files=True, key="sessions_upload")
    hfiles = st.sidebar.file_uploader("Heat Data", type=["csv", "xlsx"], accept_multiple_files=True, key="heat_upload")
    
    if st.sidebar.button("🚀 Analyze", type="primary", use_container_width=True):
        if not (sfiles and hfiles):
            st.sidebar.error("Upload at least one file for each type")
        elif not st.session_state.distribution_points:
            st.sidebar.error("Add distribution points first")
        else:
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # ====================================================================
                # PROCESS SESSIONS FILES (HEAVY - 14M+ rows)
                # ====================================================================
                status_text.text("📊 Processing sessions files...")
                sessions_agg_list = []
                
                for i, sfile in enumerate(sfiles):
                    progress = (i / (len(sfiles) + len(hfiles)))
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Processing sessions file {i+1}/{len(sfiles)}: {sfile.name}")
                    
                    # Load ONE file at a time
                    if sfile.name.endswith('.csv'):
                        sessions = pd.read_csv(sfile)
                    else:
                        sessions = pd.read_excel(sfile)
                    
                    status_text.text(f"📊 Loaded {len(sessions):,} sessions from {sfile.name}")
                    
                    # Parse timestamp - use Session_Date (has actual date)
                    # Created At (Local) only has time, not full datetime
                    if 'Session_Date' in sessions.columns:
                        sessions['timestamp'] = pd.to_datetime(sessions['Session_Date'], errors='coerce')
                    elif 'Created At (Local)' in sessions.columns:
                        sessions['timestamp'] = pd.to_datetime(sessions['Created At (Local)'], errors='coerce')
                    else:
                        # Try to find any date column
                        for col in sessions.columns:
                            if 'date' in col.lower():
                                sessions['timestamp'] = pd.to_datetime(sessions[col], errors='coerce')
                                break
                    
                    # Drop sessions with invalid timestamps
                    if 'timestamp' not in sessions.columns:
                        st.error(f"❌ No date column found in {sfile.name}! Need 'Session_Date' or 'Created At (Local)'")
                        continue
                    
                    invalid_timestamps = sessions['timestamp'].isna().sum()
                    if invalid_timestamps > 0:
                        st.warning(f"⚠️ Removing {invalid_timestamps:,} sessions with invalid timestamps from {sfile.name}")
                        sessions = sessions[sessions['timestamp'].notna()].copy()
                    
                    if len(sessions) == 0:
                        st.error(f"❌ No valid sessions with timestamps in {sfile.name}! Skipping file.")
                        continue
                    
                    # Assign to areas (HEAVY computation)
                    status_text.text(f"🗺️ Assigning {len(sessions):,} sessions to areas...")
                    sessions = assign_sessions_to_areas(sessions, st.session_state.distribution_points)
                    
                    # IMMEDIATELY aggregate to reduce size
                    status_text.text(f"📉 Aggregating sessions from {sfile.name}...")
                    agg = aggregate_sessions(sessions)
                    sessions_agg_list.append(agg)
                    
                    # Free memory
                    del sessions
                    gc.collect()
                    
                    status_text.text(f"✅ Processed {sfile.name} → {len(agg):,} aggregated rows")
                
                # Combine all aggregated sessions
                status_text.text("🔗 Combining aggregated sessions...")
                
                if len(sessions_agg_list) == 0:
                    st.error("❌ No valid sessions were processed from any files!")
                    st.stop()
                
                sessions_agg = pd.concat(sessions_agg_list, ignore_index=True)
                sessions_agg = sessions_agg.groupby(['Area', 'date']).sum().reset_index()
                
                status_text.text(f"✅ Total sessions aggregated: {len(sessions_agg):,} rows (from {sessions_agg['sessions_count'].sum():,} total sessions)")
                
                # Free memory
                del sessions_agg_list
                gc.collect()
                
                # ====================================================================
                # PROCESS HEATDATA FILES (2M+ rows)
                # ====================================================================
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
                    status_text.text(f"✅ Loaded {len(heat):,} rows from {hfile.name}")
                
                # Combine all heatdata
                status_text.text("🔗 Combining heatdata files...")
                heat = pd.concat(heat_list, ignore_index=True)
                
                # Parse timestamp
                heat['timestamp'] = pd.to_datetime(heat['Start Date'], errors='coerce')
                
                # Optimize dtypes
                heat['Area'] = heat['Area'].astype('category')
                
                status_text.text(f"✅ Total heatdata: {len(heat):,} rows")
                
                # Free memory
                del heat_list
                gc.collect()
                
                # ====================================================================
                # SAVE EVERYTHING
                # ====================================================================
                progress_bar.progress(0.95)
                status_text.text("💾 Saving analyzed data...")
                
                # Create empty rides dataframe (not used)
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
                
                # Wait a bit then rerun
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

# Get data from session state
sessions_agg = st.session_state.sessions_agg
heat = st.session_state.heat
rides = st.session_state.rides

# ============================================================================
# FILTERS
# ============================================================================

st.title("📊 Supply & Demand Dashboard")
st.markdown("---")
st.markdown("### 🔍 Filters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    selected_areas = st.multiselect(
        "Areas",
        AREA_NAMES,
        default=AREA_NAMES
    )

with c2:
    st.session_state.range_choice = st.selectbox("Range", ["Daily", "Weekly", "Monthly"])

with c3:
    st.session_state.smooth_enabled = st.checkbox("Smooth (Moving Average)")

with c4:
    if st.session_state.smooth_enabled:
        st.session_state.smooth_window = st.slider("Window", 2, 7, 3)

# Date range filter
date_min = sessions_agg['date'].min()
date_max = sessions_agg['date'].max()

# Handle NaT values
if pd.isna(date_min) or pd.isna(date_max):
    st.error("⚠️ No valid dates found in sessions data. Check 'Created At (Local)' column.")
    st.stop()

start_date, end_date = st.date_input(
    "Date Range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

# ============================================================================
# APPLY FILTERS
# ============================================================================

# Filter sessions_agg
sessions_filtered = sessions_agg[
    (sessions_agg['Area'].isin(selected_areas)) &
    (sessions_agg['date'] >= pd.to_datetime(start_date)) &
    (sessions_agg['date'] <= pd.to_datetime(end_date))
].copy()

# Filter heat
heat_filtered = heat[
    (heat['Area'].isin(selected_areas)) &
    (heat['timestamp'].dt.date >= start_date) &
    (heat['timestamp'].dt.date <= end_date)
].copy()

# Remove unused categories if Area is categorical
if heat_filtered['Area'].dtype.name == 'category':
    heat_filtered['Area'] = heat_filtered['Area'].cat.remove_unused_categories()

# Filter rides
if len(rides) > 0 and 'timestamp' in rides.columns:
    rides_filtered = rides[
        (rides['timestamp'].dt.date >= start_date) &
        (rides['timestamp'].dt.date <= end_date)
    ].copy()
else:
    rides_filtered = rides.copy()

# ============================================================================
# SCORECARDS
# ============================================================================

st.markdown("---")
st.markdown("### 📊 Key Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

# Total Sessions
total_sessions = int(sessions_filtered['sessions_count'].sum())
m1.metric("📍 Total Sessions", f"{total_sessions:,}")

# Within 200m (sessions not marked as Out of Fence)
sessions_in_fence = sessions_filtered[sessions_filtered['Area'] != 'Out of Fence']['sessions_count'].sum()
pct_in_fence = (sessions_in_fence / total_sessions * 100) if total_sessions > 0 else 0
m2.metric("🟢 Within 200m", f"{int(sessions_in_fence):,}", delta=f"{pct_in_fence:.1f}%")

# Total Rides (from heatdata)
total_rides = int(heat_filtered['Rides'].sum())
m3.metric("🚗 Total Rides", f"{total_rides:,}")

# Fulfillment %
fulfillment = (total_rides / total_sessions * 100) if total_sessions > 0 else 0
m4.metric("📈 Fulfillment", f"{fulfillment:.1f}%")

# Average Active Vehicles across all selected days
# Calculate per-snapshot totals (sum neighborhoods), then average across all snapshots
active_per_snapshot = heat_filtered.groupby('timestamp')['Active Vehicles'].sum()
avg_active_vehicles = active_per_snapshot.mean() * 1.075  # Apply same 7.5% adjustment
m5.metric("🚕 Avg Active Vehicles", f"{avg_active_vehicles:.1f}")

# ============================================================================
# CHARTS
# ============================================================================

st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["📊 Overview", "🏘️ Neighborhood View"])

with tab1:
    # ========================================================================
    # CHART 1: SESSIONS (FULL WIDTH)
    # ========================================================================
    st.subheader("Sessions")
    
    # Apply bucketing
    sessions_filtered['bucket'] = bucket_data(sessions_filtered, 'date', st.session_state.range_choice)
    
    # Group by bucket and area
    sessions_chart = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    
    if len(sessions_chart) > 0:
        line_chart(sessions_chart, "", "Sessions")
    else:
        st.info("No session data for selected filters")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 2: RIDES (FULL WIDTH)
    # ========================================================================
    st.subheader("Rides")
    
    # Apply bucketing
    heat_filtered['bucket'] = bucket_data(heat_filtered, 'timestamp', st.session_state.range_choice)
    
    # Group by bucket and area, SUM rides
    rides_chart = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    if len(rides_chart) > 0:
        line_chart(rides_chart, "", "Rides")
    else:
        st.info("No rides data for selected filters")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 3: FULFILLMENT % (FULL WIDTH)
    # ========================================================================
    st.subheader("Fulfillment %")
    
    # Align sessions and rides by bucket and area
    sessions_for_fulfill = sessions_filtered.groupby(['bucket', 'Area'])['sessions_count'].sum().unstack().fillna(0)
    rides_for_fulfill = heat_filtered.groupby(['bucket', 'Area'])['Rides'].sum().unstack().fillna(0)
    
    # Align indices
    sessions_aligned, rides_aligned = sessions_for_fulfill.align(rides_for_fulfill, fill_value=0)
    
    # Calculate fulfillment
    fulfillment_chart = (rides_aligned / sessions_aligned.replace(0, np.nan)) * 100
    fulfillment_chart = fulfillment_chart.fillna(0)
    
    if len(fulfillment_chart) > 0:
        line_chart(fulfillment_chart, "", "Percent")
    else:
        st.info("No data for fulfillment calculation")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 4 & 5: EFFECTIVE VEHICLES (LEFT) + URGENT VEHICLES (RIGHT)
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Effective Vehicles")
        
        # For Area: 
        # 1. Sum neighborhoods per snapshot (timestamp)
        # 2. Then average across snapshots per bucket
        # 3. Apply adjustment factor to match company dashboard
        eff_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Effective Active Vehicles'].sum().reset_index()
        eff_per_snapshot['bucket'] = bucket_data(eff_per_snapshot, 'timestamp', st.session_state.range_choice)
        eff_vehicles_chart = eff_per_snapshot.groupby(['bucket', 'Area'])['Effective Active Vehicles'].mean().unstack().fillna(0)
        
        # Adjustment factor: 7.5% increase
        eff_vehicles_chart = eff_vehicles_chart * 1.075
        
        if len(eff_vehicles_chart) > 0:
            line_chart(eff_vehicles_chart, "", "Vehicles")
        else:
            st.info("No vehicle data for selected filters")
    
    with col2:
        st.subheader("Urgent Vehicles")
        
        # Same logic with adjustment factor
        urgent_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Urgent Vehicles'].sum().reset_index()
        urgent_per_snapshot['bucket'] = bucket_data(urgent_per_snapshot, 'timestamp', st.session_state.range_choice)
        urgent_vehicles_chart = urgent_per_snapshot.groupby(['bucket', 'Area'])['Urgent Vehicles'].mean().unstack().fillna(0)
        
        # Adjustment factor: 7.5% increase
        urgent_vehicles_chart = urgent_vehicles_chart * 1.075
        
        if len(urgent_vehicles_chart) > 0:
            line_chart(urgent_vehicles_chart, "", "Vehicles")
        else:
            st.info("No urgent vehicle data")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 6: ACTIVE VEHICLES (HALF WIDTH LEFT, empty right)
    # ========================================================================
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Active Vehicles")
        
        # Same logic with adjustment factor
        active_per_snapshot = heat_filtered.groupby(['timestamp', 'Area'])['Active Vehicles'].sum().reset_index()
        active_per_snapshot['bucket'] = bucket_data(active_per_snapshot, 'timestamp', st.session_state.range_choice)
        active_vehicles_chart = active_per_snapshot.groupby(['bucket', 'Area'])['Active Vehicles'].mean().unstack().fillna(0)
        
        # Adjustment factor: 7.5% increase
        active_vehicles_chart = active_vehicles_chart * 1.075
        
        if len(active_vehicles_chart) > 0:
            line_chart(active_vehicles_chart, "", "Vehicles")
        else:
            st.info("No active vehicle data")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 7: TOP NEIGHBORHOODS
    # ========================================================================
    st.subheader("Top Neighborhoods")
    
    # Toggle for metric
    col_toggle, col_empty = st.columns([1, 3])
    with col_toggle:
        top_metric = st.selectbox(
            "Show top 5 by:",
            ["Most Sessions", "Most Rides", "Highest Fulfillment %"],
            label_visibility="collapsed"
        )
    
    if top_metric == "Most Sessions":
        # Calculate total sessions per neighborhood over date range
        # We need to get neighborhood from sessions assignments
        # For now, show message
        st.info("Top Neighborhoods by Sessions - Feature coming soon (requires neighborhood assignment tracking)")
    
    elif top_metric == "Most Rides":
        # Get neighborhoods from heatdata
        neigh_rides = heat_filtered.groupby('Neighborhood Name')['Rides'].sum().sort_values(ascending=False)
        top_5_neighs = neigh_rides.head(5).index.tolist()
        
        # Filter heat to top 5 neighborhoods
        heat_top5 = heat_filtered[heat_filtered['Neighborhood Name'].isin(top_5_neighs)].copy()
        
        # Group by bucket and neighborhood
        top5_chart = heat_top5.groupby(['bucket', 'Neighborhood Name'])['Rides'].sum().unstack().fillna(0)
        
        if len(top5_chart) > 0:
            # Create chart (without AREA_COLORS since these are neighborhoods)
            df_chart = top5_chart.reset_index()
            date_col = df_chart.columns[0]
            df_chart = df_chart.melt(date_col, var_name="Neighborhood", value_name="Rides")
            
            chart = (
                alt.Chart(df_chart)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X(f"{date_col}:T", title=""),
                    y=alt.Y("Rides:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("Neighborhood:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[f"{date_col}:T", "Neighborhood:N", "Rides:Q"],
                )
                .properties(height=310)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No neighborhood data")
    
    else:  # Highest Fulfillment %
        st.info("Top Neighborhoods by Fulfillment - Feature coming soon (requires neighborhood assignment tracking)")

# ========================================================================
# NEIGHBORHOOD VIEW TAB
# ========================================================================
with tab2:
    st.markdown("### 🏘️ Neighborhood View")
    st.caption("Detailed analytics for each neighborhood (coming soon)")
    st.info("This tab will show detailed neighborhood-level breakdowns once session-to-neighborhood assignments are tracked during aggregation.")

st.markdown("---")
st.caption(f"Last updated: {st.session_state.data_updated.strftime('%Y-%m-%d %H:%M:%S')}")

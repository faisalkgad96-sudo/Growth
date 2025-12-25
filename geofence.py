import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


# ----------------------------------------------------
# PAGE
# ----------------------------------------------------
st.set_page_config(page_title="Supply & Demand Analytics", page_icon="📊", layout="wide")
st.title("📊 Supply & Demand Analytics Dashboard")


# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------
DISPLAY_NAMES = {
    "alexandria": "Alexandria",
    "downtown": "Downtown",
    "downtown 1": "Downtown",
    "maadi": "Maadi",
    "masr el gdeida": "Masr El Gdeida",
    "masr el gedida": "Masr El Gdeida",  # Handle both spellings
    "mg": "Masr El Gdeida",
    "zahra el maadi": "Zahraa El Maadi",
    "zahraa el maadi": "Zahraa El Maadi",
}

AREA_COLORS = {
    "Alexandria": "#4F9CF9",
    "Downtown": "#7C4DFF",
    "Maadi": "#2DC26B",
    "Masr El Gdeida": "#FF6B6B",
    "Zahraa El Maadi": "#FFB020",
}


def pretty(a):
    return DISPLAY_NAMES.get(str(a).lower(), str(a).title())


# ----------------------------------------------------
# CHART UTIL
# ----------------------------------------------------
def bucket(df):
    r = st.session_state.range_choice
    if r == "Daily":
        return df["timestamp"].dt.date
    if r == "Weekly":
        return df["timestamp"].dt.to_period("W").apply(lambda x: x.start_time.date())
    return df["timestamp"].dt.to_period("M").apply(lambda x: x.start_time.date())


def line_chart(df, title="", y_label=""):
    # Reset index to get the date column (whatever it's named)
    df = df.reset_index()
    
    # Get the name of the first column (the date/bucket column)
    date_col = df.columns[0]
    
    # Melt using the first column as id
    df = df.melt(date_col, var_name="Series", value_name="Value")

    # smoothing (moving average)
    if st.session_state.get("smooth_enabled", False):
        win = st.session_state.get("smooth_window", 3)
        df["Value"] = df.groupby("Series")["Value"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )

    selection = alt.selection_multi(fields=["Series"], bind="legend")
    
    # Get colors only for series that exist in the data
    series_in_data = df["Series"].unique()
    color_domain = [s for s in AREA_COLORS.keys() if s in series_in_data]
    color_range = [AREA_COLORS[s] for s in color_domain]

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(f"{date_col}:T", title=""),
            y=alt.Y("Value:Q", title=y_label, scale=alt.Scale(zero=False)),  # Dynamic scaling
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=color_domain,  # Only colors for visible series
                    range=color_range,
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[f"{date_col}:T", "Series:N", "Value:Q"],
        )
        .add_selection(selection)
        .properties(title=title, height=310)
    )

    st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------
# DATA LOAD
# ----------------------------------------------------
import pickle
import os
from pathlib import Path

# Create data directory for persistence
DATA_DIR = Path("persistent_data")
DATA_DIR.mkdir(exist_ok=True)

DIST_POINTS_FILE = DATA_DIR / "distribution_points.pkl"
ANALYZED_DATA_FILE = DATA_DIR / "analyzed_data.pkl"

# Load saved data on startup
def load_persistent_data():
    """Load distribution points and analyzed data from disk"""
    # Load distribution points
    if DIST_POINTS_FILE.exists():
        try:
            with open(DIST_POINTS_FILE, 'rb') as f:
                saved_points = pickle.load(f)
                st.session_state.distribution_points = saved_points['points']
                st.session_state.points_updated = saved_points.get('updated')
        except Exception as e:
            st.warning(f"Could not load saved distribution points: {e}")
    
    # Load analyzed data
    if ANALYZED_DATA_FILE.exists():
        try:
            with open(ANALYZED_DATA_FILE, 'rb') as f:
                saved_data = pickle.load(f)
                st.session_state.sessions = saved_data['sessions']
                st.session_state.heat = saved_data['heat']
                st.session_state.rides = saved_data['rides']
                st.session_state.data_updated = saved_data.get('updated')
                st.session_state.data_loaded = True
        except Exception as e:
            st.warning(f"Could not load saved data: {e}")

def save_distribution_points():
    """Save distribution points to disk"""
    try:
        with open(DIST_POINTS_FILE, 'wb') as f:
            pickle.dump({
                'points': st.session_state.distribution_points,
                'updated': st.session_state.points_updated
            }, f)
    except Exception as e:
        st.error(f"Could not save distribution points: {e}")

def save_analyzed_data():
    """Save analyzed data to disk"""
    try:
        with open(ANALYZED_DATA_FILE, 'wb') as f:
            pickle.dump({
                'sessions': st.session_state.sessions,
                'heat': st.session_state.heat,
                'rides': st.session_state.rides,
                'updated': st.session_state.data_updated
            }, f)
    except Exception as e:
        st.error(f"Could not save analyzed data: {e}")

# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "distribution_points" not in st.session_state:
    st.session_state.distribution_points = []
if "points_updated" not in st.session_state:
    st.session_state.points_updated = None
if "data_updated" not in st.session_state:
    st.session_state.data_updated = None
if "rides" not in st.session_state:
    st.session_state.rides = pd.DataFrame()
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    load_persistent_data()  # Load saved data on first run

with st.sidebar:
    st.header("📂 Upload Data")
    
    # Distribution Points
    st.subheader("1. Distribution Points")
    
    # Show status first if points exist
    if st.session_state.distribution_points:
        st.success(f"✅ {len(st.session_state.distribution_points)} areas saved")
        st.caption(f"📍 {', '.join([p['area'] for p in st.session_state.distribution_points])}")
        if st.session_state.points_updated:
            st.caption(f"🕒 Updated: {st.session_state.points_updated.strftime('%m/%d %I:%M%p')}")
        if st.button("Clear Points"):
            st.session_state.distribution_points = []
            st.session_state.points_updated = None
            if DIST_POINTS_FILE.exists():
                DIST_POINTS_FILE.unlink()  # Delete saved file
            st.rerun()
    
    with st.expander("➕ Add Points", expanded=len(st.session_state.distribution_points) == 0):
        area_name = st.text_input("Area Name", key="new_area")
        points_file = st.file_uploader("Points File (CSV/Excel)", type=["csv", "xlsx"], key="new_points")
        
        if st.button("Add Points"):
            if area_name and points_file:
                try:
                    points_df = pd.read_csv(points_file) if points_file.name.endswith(".csv") else pd.read_excel(points_file)
                    points_df['area'] = area_name
                    st.session_state.distribution_points.append({'area': area_name, 'data': points_df})
                    st.session_state.points_updated = pd.Timestamp.now()
                    save_distribution_points()  # Save to disk
                    st.success(f"✅ Added {len(points_df)} points for {area_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Enter area name and upload file")
    
    st.markdown("---")
    st.subheader("2. Data Files")
    
    # Show analyzed data status
    if st.session_state.data_loaded:
        st.success("✅ Data analyzed & ready")
        if st.session_state.data_updated:
            st.caption(f"🕒 Analyzed: {st.session_state.data_updated.strftime('%m/%d %I:%M%p')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-analyze", use_container_width=True):
                st.session_state.data_loaded = False
                st.rerun()
        with col2:
            if st.button("Clear Data", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.data_updated = None
                if ANALYZED_DATA_FILE.exists():
                    ANALYZED_DATA_FILE.unlink()  # Delete saved file
                st.rerun()
    
    # Upload section - only show if no data loaded
    if not st.session_state.data_loaded:
        st.markdown("**Upload Files:**")
        st.caption("💡 You can upload multiple files for each type - they will be combined automatically")
        
        sfiles = st.file_uploader("Sessions", type=["csv", "xlsx"], key="sessions_upload", accept_multiple_files=True)
        hfiles = st.file_uploader("Heat Data", type=["csv", "xlsx"], key="heat_upload", accept_multiple_files=True)
        rfiles = st.file_uploader("Rides Data", type=["csv", "xlsx"], key="rides_upload", accept_multiple_files=True)

        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            if not (sfiles and hfiles and rfiles):
                st.error("Upload at least one file for each type")
            elif not st.session_state.distribution_points:
                st.error("Add distribution points first")
            else:
                with st.spinner("Processing..."):
                    # Load and combine sessions files
                    sessions_list = []
                    for i, sfile in enumerate(sfiles, 1):
                        df = pd.read_csv(sfile) if sfile.name.endswith(".csv") else pd.read_excel(sfile)
                        sessions_list.append(df)
                        st.toast(f"📄 Sessions file {i}/{len(sfiles)}: {sfile.name} - {len(df):,} rows")
                    sessions = pd.concat(sessions_list, ignore_index=True)
                    st.toast(f"✅ Combined {len(sfiles)} sessions file(s) - {len(sessions):,} total rows")
                    
                    # Remove any existing assignment columns to force re-assignment
                    for col in ['Assigned_Area', 'Assigned_Neighborhood', 'Within_Fence']:
                        if col in sessions.columns:
                            sessions = sessions.drop(columns=[col])
                    
                    # Load and combine heat files
                    heat_list = []
                    for i, hfile in enumerate(hfiles, 1):
                        df = pd.read_csv(hfile) if hfile.name.endswith(".csv") else pd.read_excel(hfile)
                        heat_list.append(df)
                        st.toast(f"📄 Heat file {i}/{len(hfiles)}: {hfile.name} - {len(df):,} rows, columns: {len(df.columns)}")
                    
                    # Check if all heat files have same columns
                    if len(heat_list) > 1:
                        first_cols = set(heat_list[0].columns)
                        for i, df in enumerate(heat_list[1:], 2):
                            if set(df.columns) != first_cols:
                                missing = first_cols - set(df.columns)
                                extra = set(df.columns) - first_cols
                                st.warning(f"⚠️ Heat file {i} has different columns! Missing: {missing}, Extra: {extra}")
                    
                    heat = pd.concat(heat_list, ignore_index=True)
                    st.toast(f"✅ Combined {len(hfiles)} heat file(s) - {len(heat):,} total rows")
                    
                    # Load and combine rides files
                    rides_list = []
                    for i, rfile in enumerate(rfiles, 1):
                        df = pd.read_csv(rfile) if rfile.name.endswith(".csv") else pd.read_excel(rfile)
                        rides_list.append(df)
                        st.toast(f"📄 Rides file {i}/{len(rfiles)}: {rfile.name} - {len(df):,} rows")
                    rides = pd.concat(rides_list, ignore_index=True)
                    st.toast(f"✅ Combined {len(rfiles)} rides file(s) - {len(rides):,} total rows")
                    
                    # Assign sessions to areas using KDTree
                    from scipy.spatial import cKDTree
                    
                    # Combine all distribution points
                    all_points = pd.concat([p['data'] for p in st.session_state.distribution_points], ignore_index=True)
                    
                    # Get coordinates
                    session_coords = sessions[['Latitude', 'Longitude']].values.astype(float)
                    point_coords = all_points[['lat', 'lng']].values.astype(float)
                    
                    # Validate
                    session_valid = np.isfinite(session_coords).all(axis=1)
                    sessions = sessions[session_valid].reset_index(drop=True)
                    session_coords = session_coords[session_valid]
                    
                    # Convert to radians
                    session_rad = np.radians(session_coords)
                    point_rad = np.radians(point_coords)
                    
                    # Build KDTree
                    def latlon_to_cartesian(lat_rad, lon_rad):
                        x = np.cos(lat_rad) * np.cos(lon_rad)
                        y = np.cos(lat_rad) * np.sin(lon_rad)
                        z = np.sin(lat_rad)
                        return np.column_stack([x, y, z])
                    
                    point_cartesian = latlon_to_cartesian(point_rad[:, 0], point_rad[:, 1])
                    tree = cKDTree(point_cartesian)
                    session_cartesian = latlon_to_cartesian(session_rad[:, 0], session_rad[:, 1])
                    
                    # Find nearest
                    _, nearest_indices = tree.query(session_cartesian, k=1)
                    
                    # Calculate distances
                    nearest_point_coords = point_rad[nearest_indices]
                    dlat = nearest_point_coords[:, 0] - session_rad[:, 0]
                    dlon = nearest_point_coords[:, 1] - session_rad[:, 1]
                    a = np.sin(dlat/2)**2 + np.cos(session_rad[:, 0]) * np.cos(nearest_point_coords[:, 0]) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                    distances_meters = c * 6371000
                    
                    # Assign
                    within_threshold = distances_meters <= 200
                    sessions['Assigned_Area'] = np.where(within_threshold, all_points['area'].iloc[nearest_indices].values, 'Out of Fence')
                    sessions['Assigned_Neighborhood'] = np.where(within_threshold, all_points['neighborhood'].iloc[nearest_indices].values, 'Out of Fence')
                    sessions['Within_Fence'] = np.where(within_threshold, 'Yes', 'No')
                    
                    # Show assignment summary
                    area_counts = sessions['Assigned_Area'].value_counts()
                    st.toast(f"📍 Areas found: {', '.join([f'{area} ({count:,})' for area, count in area_counts.items() if area != 'Out of Fence'])}")
                    
                    # Extract timestamps
                    for df in (sessions, heat, rides):
                        for c in df.columns:
                            if "date" in c.lower() or "time" in c.lower():
                                df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                                break

                    st.session_state.sessions = sessions
                    st.session_state.heat = heat
                    st.session_state.rides = rides
                    st.session_state.data_loaded = True
                    st.session_state.data_updated = pd.Timestamp.now()
                    save_analyzed_data()  # Save to disk for persistence
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()

if not st.session_state.data_loaded:
    st.stop()

sessions = st.session_state.sessions.copy()
heat = st.session_state.heat.copy()
rides = st.session_state.rides.copy()


# ----------------------------------------------------
# FILTERS
# ----------------------------------------------------
st.markdown("### 🔍 Filters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    areas = sorted(sessions.get("Assigned_Area", pd.Series()).dropna().unique())
    areas = [a for a in areas if a not in ['Out of Fence', 'Error']]
    selected = st.multiselect("Areas", areas, default=areas)

with c2:
    st.session_state.range_choice = st.selectbox("Range", ["Daily", "Weekly", "Monthly"])

with c3:
    st.session_state.smooth_enabled = st.checkbox("Smooth (Moving Average)")

with c4:
    if st.session_state.smooth_enabled:
        st.session_state.smooth_window = st.slider("Window", 2, 7, 3)

date_min = sessions["timestamp"].min().date()
date_max = sessions["timestamp"].max().date()

start, end = st.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

sessions = sessions[
    (sessions["timestamp"].dt.date >= start) & (sessions["timestamp"].dt.date <= end)
]
heat = heat[(heat["timestamp"].dt.date >= start) & (heat["timestamp"].dt.date <= end)]

# Only filter rides if it has data
if len(rides) > 0 and "timestamp" in rides.columns:
    rides = rides[(rides["timestamp"].dt.date >= start) & (rides["timestamp"].dt.date <= end)]

sessions = sessions[sessions["Assigned_Area"].isin(selected)]

# Filter heat data by area
heat_cols = {c.lower(): c for c in heat.columns}
heat_area_col = next((heat_cols[k] for k in ["assigned_area", "area"] if k in heat_cols), None)

if heat_area_col and len(selected) > 0:
    # Normalize heat area names
    heat['area_normalized'] = heat[heat_area_col].apply(lambda x: pretty(x) if pd.notna(x) else x)
    # Normalize selected values too to ensure matching
    selected_normalized = [pretty(s) for s in selected]
    # Filter using normalized names
    heat = heat[heat['area_normalized'].isin(selected_normalized)]

# Normalize rides area names to match selected areas
rides_cols = {c.lower(): c for c in rides.columns}
rides_area_col = next((rides_cols[k] for k in ["area", "assigned_area"] if k in rides_cols), None)

if rides_area_col and len(rides) > 0:
    # Normalize area names in rides to match sessions
    rides[rides_area_col] = rides[rides_area_col].apply(lambda x: pretty(x) if pd.notna(x) else x)
    # Filter by selected areas
    if len(selected) > 0:
        rides = rides[rides[rides_area_col].isin(selected)]

sessions["bucket"] = bucket(sessions)
heat["bucket"] = bucket(heat)
if len(rides) > 0 and "timestamp" in rides.columns:
    rides["bucket"] = bucket(rides)


# column lookups (heat_area_col already defined above)
eff_col = next(
    (heat_cols[k] for k in ["effective active vehicles", "effective_vehicles", "effective vehicles"] if k in heat_cols),
    None,
)
rides_col = next((heat_cols[k] for k in ["rides", "sessions rides", "supply"] if k in heat_cols), None)


# ----------------------------------------------------
# SCORECARDS
# ----------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Key Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

total_sessions = len(sessions)
m1.metric("📍 Total Sessions", f"{total_sessions:,}")

if "Within_Fence" in sessions.columns:
    wf = (sessions["Within_Fence"] == "Yes").sum()
    m2.metric("🟢 Within 200m", f"{wf:,}", delta=f"{(wf / total_sessions * 100 if total_sessions else 0):.1f}%")
else:
    m2.metric("🟢 Within 200m", "—")

if rides_col:
    total_rides = int(heat[rides_col].sum())
else:
    total_rides = len(heat)

m3.metric("🚗 Total Rides", f"{total_rides:,}")

fulfill = (total_rides / total_sessions * 100) if total_sessions else 0
m4.metric("📈 Fulfillment", f"{fulfill:.1f}%")

if eff_col:
    m5.metric("🚕 Avg Vehicles", f"{heat[eff_col].mean():.1f}")
else:
    m5.metric("🚕 Avg Vehicles", "—")

st.markdown("---")


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1, tab2 = st.tabs(["📊 Overview", "🏘️ Neighborhood View"])

with tab1:

    # -------------------- Sessions + Neighborhoods --------------------
    a1, a2 = st.columns(2)

    with a1:
        st.subheader("Sessions")
        t = sessions.groupby(["bucket", "Assigned_Area"]).size().unstack().fillna(0)
        t.columns = [pretty(c) for c in t.columns]
        line_chart(t, "Sessions", "Sessions")

    with a2:
        st.subheader("Top Neighborhoods")

        neigh = (
            sessions.groupby(["bucket", "Assigned_Neighborhood"])
            .size()
            .unstack()
            .fillna(0)
        )
        top = neigh.sum().sort_values(ascending=False).head(5).index

        df = neigh[top].reset_index().melt(
            "bucket", var_name="Neighborhood", value_name="Sessions"
        )

        selection = alt.selection_multi(fields=["Neighborhood"], bind="legend")

        chart = (
            alt.Chart(df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("bucket:T", title=""),
                y=alt.Y("Sessions:Q", title="Sessions", scale=alt.Scale(zero=False)),  # Dynamic scaling
                color="Neighborhood:N",
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
                tooltip=["bucket:T", "Neighborhood:N", "Sessions:Q"],
            )
            .add_selection(selection)
            .properties(title="Top Neighborhoods", height=310)
        )

        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # -------------------- Rides + Effective Vehicles --------------------
    b1, b2 = st.columns(2)

    with b1:
        st.subheader("Rides")

        if heat_area_col:
            rides = heat.groupby(["bucket", "area_normalized"]).size().unstack().fillna(0)
            rides.columns = [pretty(c) for c in rides.columns]
            line_chart(rides, "Rides", "Rides")
        else:
            st.info("No area column in heat data")

    with b2:
        st.subheader("Effective Vehicles")

        if heat_area_col and eff_col:
            eff = (
                heat.groupby(["bucket", "area_normalized"])[eff_col]
                .mean()
                .unstack()
                .fillna(0)
            )
            eff.columns = [pretty(c) for c in eff.columns]
            line_chart(eff, "Effective Vehicles", "Vehicles")
        else:
            st.info("Vehicle metrics not found")

    # -------------------- Fulfillment --------------------
    st.markdown("---")
    st.subheader("Fulfillment %")

    if heat_area_col:
        rides = heat.groupby(["bucket", "area_normalized"]).size().unstack().fillna(0)
        rides.columns = [pretty(c) for c in rides.columns]

        sess = sessions.groupby(["bucket", "Assigned_Area"]).size().unstack().fillna(0)
        sess.columns = [pretty(c) for c in sess.columns]

        rides, sess = rides.align(sess, fill_value=0)

        fulfill_df = (rides / sess.replace(0, np.nan)) * 100
        fulfill_df = fulfill_df.fillna(0)

        line_chart(fulfill_df, "Fulfillment %", "Percent")
    else:
        st.info("Cannot compute fulfillment (area missing in heat data).")
    
    # -------------------- Additional Metrics --------------------
    st.markdown("---")
    
    # DEBUG: Show what areas exist in rides data
    with st.expander("🔍 Debug: Check Area Names", expanded=False):
        st.write("**Sessions Areas:**")
        st.write(sorted(sessions['Assigned_Area'].unique()))
        
        rides_cols = {c.lower(): c for c in rides.columns}
        rides_area_col_debug = next((rides_cols[k] for k in ["area", "assigned_area"] if k in rides_cols), None)
        if rides_area_col_debug and len(rides) > 0:
            st.write("**Rides Areas (raw):**")
            st.write(sorted(rides[rides_area_col_debug].unique()))
    
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.subheader("Total Users per Area")
        
        # Look for User ID column (with space and capital letters)
        user_col = next((c for c in sessions.columns if c.lower().replace(" ", "") == "userid"), None)
        
        if user_col:
            users = sessions.groupby(["bucket", "Assigned_Area"])[user_col].nunique().unstack().fillna(0)
            users.columns = [pretty(c) for c in users.columns]
            line_chart(users, "Total Users", "Unique Users")
        else:
            st.info(f"No User ID column found. Available: {', '.join(sessions.columns[:5])}...")
    
    with row1_c2:
        st.subheader("Total Riders per Area")
        
        # Get riders from rides file
        if len(rides) > 0:
            rides_cols = {c.lower(): c for c in rides.columns}
            rides_area_col = next((rides_cols[k] for k in ["area", "assigned_area"] if k in rides_cols), None)
            rides_user_col = next((c for c in rides.columns if c.lower().replace(" ", "") == "userid"), None)
            
            if rides_area_col and rides_user_col:
                # Normalize area names
                rides_copy = rides.copy()
                rides_copy["Area_Pretty"] = rides_copy[rides_area_col].apply(pretty)
                
                riders = rides_copy.groupby(["bucket", "Area_Pretty"])[rides_user_col].nunique().unstack().fillna(0)
                line_chart(riders, "Total Riders", "Unique Riders")
            else:
                st.info("Need Area and User Id columns in rides data")
        else:
            st.info("No rides data uploaded yet")
    
    st.markdown("---")
    
    # Users with 0 rides
    st.subheader("Users with 0 Rides per Area")
    
    if user_col and len(rides) > 0:
        rides_cols = {c.lower(): c for c in rides.columns}
        rides_area_col = next((rides_cols[k] for k in ["area", "assigned_area"] if k in rides_cols), None)
        rides_user_col = next((c for c in rides.columns if c.lower().replace(" ", "") == "userid"), None)
        
        if rides_user_col and rides_area_col:
            # Get all unique users per bucket/area from sessions
            all_users = sessions.groupby(["bucket", "Assigned_Area"])[user_col].nunique().unstack().fillna(0)
            all_users.columns = [pretty(c) for c in all_users.columns]
            
            # Get users who had rides from rides data
            rides_copy = rides.copy()
            rides_copy["Area_Pretty"] = rides_copy[rides_area_col].apply(pretty)
            users_with_rides = rides_copy.groupby(["bucket", "Area_Pretty"])[rides_user_col].nunique().unstack().fillna(0)
            
            # Align and calculate
            all_users_aligned, users_with_rides_aligned = all_users.align(users_with_rides, fill_value=0)
            
            # Users with 0 rides = all users - users who had rides
            users_no_rides = all_users_aligned - users_with_rides_aligned
            users_no_rides = users_no_rides.clip(lower=0)
            
            line_chart(users_no_rides, "Users with 0 Rides", "Users")
        else:
            st.info("Need User Id and Area columns in rides data")
    else:
        st.info("Need User ID in sessions and rides data uploaded")
    
    # -------------------- Additional Metrics --------------------
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Total Riders")
        
        # Look for riders/users column in heat data
        riders_col = next((heat_cols[k] for k in ["riders", "total riders", "users", "rider id"] if k in heat_cols), None)
        
        if heat_area_col and riders_col:
            # Sum of riders by area over time
            total_riders = heat.groupby(["bucket", "area_normalized"])[riders_col].sum().unstack().fillna(0)
            total_riders.columns = [pretty(c) for c in total_riders.columns]
            line_chart(total_riders, "Total Riders", "Riders")
        else:
            st.info("Riders column not found in heat data")
    
    with c2:
        st.subheader("Riders with 0 Rides")
        
        if heat_area_col and riders_col and rides_col:
            # Riders where rides = 0
            zero_rides = heat[heat[rides_col] == 0].groupby(["bucket", heat_area_col])[riders_col].sum().unstack().fillna(0)
            zero_rides.columns = [pretty(c) for c in zero_rides.columns]
            line_chart(zero_rides, "Riders with 0 Rides", "Riders")
        else:
            st.info("Required columns not found")
    
    with c3:
        st.subheader("Total Users")
        
        # Look for users column
        users_col = next((heat_cols[k] for k in ["users", "total users", "unique users", "user id"] if k in heat_cols), None)
        
        if heat_area_col and users_col:
            total_users = heat.groupby(["bucket", "area_normalized"])[users_col].sum().unstack().fillna(0)
            total_users.columns = [pretty(c) for c in total_users.columns]
            line_chart(total_users, "Total Users", "Users")
        elif heat_area_col and riders_col:
            # Fallback to riders if users not found
            total_users = heat.groupby(["bucket", "area_normalized"])[riders_col].sum().unstack().fillna(0)
            total_users.columns = [pretty(c) for c in total_users.columns]
            line_chart(total_users, "Total Users (from Riders)", "Users")
        else:
            st.info("Users column not found")


# ----------------------------------------------------
# NEIGHBORHOOD VIEW TAB
# ----------------------------------------------------
with tab2:
    st.markdown("### 🏘️ Neighborhood View")
    st.caption("Detailed analytics for each neighborhood in every area")
    
    # Get available areas
    available_areas = sorted(sessions['Assigned_Area'].unique())
    available_areas = [a for a in available_areas if a not in ['Out of Fence', 'Error']]
    
    # Create expandable sections for each area
    for area in available_areas:
        with st.expander(f"📍 {area}", expanded=False):
            # Filter sessions for this area
            area_sessions = sessions[sessions['Assigned_Area'] == area]
            
            # Filter heatdata for this area
            if 'area_normalized' in heat.columns:
                area_heat = heat[heat['area_normalized'] == area]
            elif heat_area_col:
                area_heat = heat[heat[heat_area_col].apply(lambda x: pretty(x) if pd.notna(x) else x) == area]
            else:
                area_heat = pd.DataFrame()
            
            # Get neighborhoods in this area
            neighborhoods = sorted(area_sessions['Assigned_Neighborhood'].unique())
            neighborhoods = [n for n in neighborhoods if n not in ['Out of Fence', 'Error', 'No Neighborhood']]
            
            if len(neighborhoods) == 0:
                st.info(f"No neighborhoods found in {area}")
                continue
            
            st.markdown(f"**Select a neighborhood to view analytics** ({len(neighborhoods)} total)")
            
            # Selectbox to choose neighborhood
            selected_neighborhood = st.selectbox(
                "Neighborhood",
                neighborhoods,
                key=f"neigh_select_{area}",
                label_visibility="collapsed"
            )
            
            if selected_neighborhood:
                st.markdown(f"### 🏘️ {selected_neighborhood}")
                
                # Filter data for this specific neighborhood
                neigh_sessions = area_sessions[area_sessions['Assigned_Neighborhood'] == selected_neighborhood].copy()
                
                # Create bucket for this neighborhood's data
                if len(neigh_sessions) > 0:
                    neigh_sessions["neigh_bucket"] = bucket(neigh_sessions)
                
                neigh_heat = area_heat  # Using area-level since Arabic names don't match
                
                # Debug info
                st.caption(f"Total sessions in this neighborhood: {len(neigh_sessions)}")
                
                # Create 2x2 grid for 4 charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Chart 1: Sessions for this neighborhood
                    st.markdown(f"**Sessions**")
                    
                    if len(neigh_sessions) > 0:
                        sessions_over_time = neigh_sessions.groupby("neigh_bucket").size()
                        
                        if len(sessions_over_time) > 0:
                            # Convert to DataFrame with proper index
                            chart_df = sessions_over_time.to_frame(name=selected_neighborhood)
                            line_chart(chart_df, "", "Sessions")
                        else:
                            st.info("No sessions over time")
                    else:
                        st.warning(f"No sessions found for '{selected_neighborhood}'")
                
                with col2:
                    # Chart 2: Rides (area-level since we can't match neighborhoods)
                    st.markdown(f"**Rides** _(area-level)_")
                    
                    if len(neigh_heat) > 0 and rides_col:
                        rides_over_time = neigh_heat.groupby("bucket")[rides_col].sum()
                        
                        chart_df = pd.DataFrame({area: rides_over_time})
                        line_chart(chart_df, "", "Rides")
                    else:
                        st.info("No rides data")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Chart 3: Fulfillment %
                    st.markdown(f"**Fulfillment %**")
                    
                    if len(neigh_sessions) > 0 and len(neigh_heat) > 0 and rides_col:
                        # Demand for this neighborhood
                        neigh_demand = neigh_sessions.groupby("neigh_bucket").size()
                        # Supply for whole area (since we can't match neighborhoods)
                        area_supply = neigh_heat.groupby("bucket")[rides_col].sum()
                        
                        if len(neigh_demand) > 0 and len(area_supply) > 0:
                            # Align indices
                            demand_aligned, supply_aligned = neigh_demand.align(area_supply, fill_value=0)
                            
                            # Calculate fulfillment
                            fulfillment = (supply_aligned / demand_aligned.replace(0, np.nan)) * 100
                            fulfillment = fulfillment.fillna(0)
                            
                            # Convert to DataFrame
                            chart_df = fulfillment.to_frame(name=selected_neighborhood)
                            line_chart(chart_df, "", "Percent")
                        else:
                            st.info("Insufficient data for fulfillment calculation")
                    else:
                        st.warning("Need both sessions and rides data")
                
                with col4:
                    # Chart 4: Effective Vehicles
                    st.markdown(f"**Effective Vehicles** _(area-level)_")
                    
                    if len(neigh_heat) > 0 and eff_col:
                        vehicles_over_time = neigh_heat.groupby("bucket")[eff_col].mean()
                        
                        chart_df = pd.DataFrame({area: vehicles_over_time})
                        line_chart(chart_df, "", "Vehicles")
                    else:
                        st.info("No vehicle data")


# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# Page config
st.set_page_config(
    page_title="Ammonium Dashboard | AI Enhanced",
    layout="wide",
)

# -----------------------
# Custom CSS (only texts/tables colors)
st.markdown("""
<style>
h1, h2, h3 {
    color: #ff7e5f;
}
.stMetric {
    background-color: rgba(255, 126, 95, 0.2);
    padding: 10px;
    border-radius: 10px;
}
.stButton>button {
    background-color: #ff7e5f;
    color: white;
    border-radius: 10px;
}
.stDataFrame div {
    color: #1e3c72 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
st.title("🌊 Ammonium Water Levels - AI Enhanced Dashboard")
st.markdown("Interactive dashboard showing ammonium (NH4) levels across stations with AI-based predictions.")

# -----------------------
# Load main dataset
@st.cache_data
def load_data(file_path):
    for sep in [';', ',']:
        try:
            df = pd.read_csv(file_path, sep=sep, engine='python')
            if df.shape[1] > 1:
                return df
        except:
            continue
    raise Exception("Could not read CSV correctly. Check the separator.")

data = load_data("PB_1996_2019_NH4.csv")
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Sidebar filters
stations = data['ID_Station'].unique()
selected_station = st.sidebar.selectbox("Select Station:", stations)

min_date = data['Date'].min()
max_date = data['Date'].max()
date_range = st.sidebar.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

start_date, end_date = date_range
filtered_data = data[
    (data['ID_Station'] == selected_station) &
    (data['Date'] >= pd.to_datetime(start_date)) &
    (data['Date'] <= pd.to_datetime(end_date))
]

# -----------------------
# KPIs
st.subheader("📊 Key Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Average NH4", f"{filtered_data['NH4'].mean():.2f} mg/cub.dm")
col2.metric("Maximum NH4", f"{filtered_data['NH4'].max():.2f} mg/cub.dm")
col3.metric("Minimum NH4", f"{filtered_data['NH4'].min():.2f} mg/cub.dm")

# -----------------------
# Charts
st.subheader("📈 NH4 Over Time")
fig_ts = px.line(
    filtered_data,
    x='Date',
    y='NH4',
    title=f"Ammonium Levels at Station {selected_station}",
    markers=True,
    color_discrete_sequence=['#ff7e5f']
)
fig_ts.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_ts, use_container_width=True)

st.subheader("🌐 NH4 vs Distance")
fig_dist = px.scatter(
    filtered_data,
    x='Distance',
    y='NH4',
    color='NH4',
    size='NH4',
    color_continuous_scale='Viridis',
    title=f"Ammonium vs Distance - Station {selected_station}"
)
fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("🗂 Station Comparison")
stations_to_compare = st.multiselect("Select Stations to Compare:", stations, default=[stations[0], stations[1]])
compare_data = data[data['ID_Station'].isin(stations_to_compare)]
fig_compare = px.line(
    compare_data,
    x='Date',
    y='NH4',
    color='ID_Station',
    title="Ammonium Levels Comparison Between Selected Stations",
)
fig_compare.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_compare, use_container_width=True)

# Heatmap
heatmap_df = data.groupby('ID_Station').agg({'NH4': 'mean'}).reset_index()
station_coords = {station: (20 + i*0.5, 38 + i*0.5) for i, station in enumerate(heatmap_df['ID_Station'].unique())}
heatmap_df['Latitude'] = heatmap_df['ID_Station'].apply(lambda x: station_coords[x][0])
heatmap_df['Longitude'] = heatmap_df['ID_Station'].apply(lambda x: station_coords[x][1])

fig_map = px.scatter_mapbox(
    heatmap_df,
    lat="Latitude",
    lon="Longitude",
    color="NH4",
    size="NH4",
    hover_name="ID_Station",
    hover_data={"NH4":True},
    color_continuous_scale="Viridis",
    size_max=15,
    zoom=5,
    mapbox_style="carto-positron",
    title="NH4 Levels Heatmap Across All Stations"
)
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------
# Data Table under charts
st.subheader("📋 Data Table")
st.dataframe(filtered_data.reset_index(drop=True))

# -----------------------
# AI Prediction Section
st.subheader("🤖 AI Prediction of NH4 Levels")
train_df = load_data("train.csv")
test_df = load_data("test.csv")

# Convert numeric columns
for df in [train_df, test_df]:
    for col in df.columns:
        if col != 'Id':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Fill NaNs in features
for df in [train_df, test_df]:
    df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
feature_cols = [col for col in train_df.columns if col != 'Id' and col != train_df.columns[-1]]
target_col = train_df.columns[-1]

X_train = train_df[feature_cols]
y_train = pd.to_numeric(train_df[target_col], errors='coerce')

# Remove rows with NaN in target
mask = ~y_train.isna()
X_train = X_train[mask]
y_train = y_train[mask]

# Train RandomForest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict on user input
st.subheader("Predict NH4 for Custom Input")
user_input = {}
for col in feature_cols:
    user_input[col] = st.number_input(f"{col} NH4 (mg/cub.dm)", value=float(X_train[col].mean()))

predicted_nh4 = model.predict(pd.DataFrame([user_input]))[0]
st.success(f"Predicted NH4 Level at Target Station: {predicted_nh4:.2f} mg/cub.dm")

# -----------------------
st.markdown("---")
st.markdown("© 2025 - Ammonia Production & Electricity Conversion Project 🌊⚡")

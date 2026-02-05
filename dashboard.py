import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="🚲 Bike Sharing Demand Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stApp {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# DATA LOADING
# -----------------
@st.cache_data
def load_and_augment_data():
    # 1. Load Data
    train = pd.read_csv('bike-sharing-demand/train.csv')
    
    # 2. Augment via Seek (Simplified replication of notebook logic)
    conn = sqlite3.connect(':memory:')
    train.to_sql('train', conn, index=False, if_exists='replace')
    
    try:
        pd.read_csv('weather_external.csv').to_sql('weather_external', conn, index=False)
        pd.read_csv('holidays_external.csv').to_sql('holidays_external', conn, index=False)
    except Exception as e:
        pass # Handle gracefully if missing in dashboard scope

    try:
        with open('Data_Augmentation.sql', 'r') as file:
            query = file.read()
        train = pd.read_sql_query(query, conn)
    except:
        pass # Fallback to original if SQL fails
        
    return train

df = load_and_augment_data()

# -----------------
# FEATURE ENGINEERING
# -----------------
def engineer_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['time_bin'] = pd.cut(df['hour'], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)
    df['is_peak'] = (((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))) & (df['workingday'] == 1)).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['bad_weather'] = (df['weather'] >= 3).astype(int)
    df['humid_temp'] = df['temp'] * df['humidity']
    df['wind_weather'] = df['windspeed'] * df['weather']
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Handle missing cols from augmentation if not present
    if 'is_federal_holiday' not in df.columns: df['is_federal_holiday'] = 0
    if 'external_precipitation' not in df.columns: df['external_precipitation'] = 0.0
    if 'is_disaster' not in df.columns: df['is_disaster'] = 0
    if 'transport_disruption' not in df.columns: df['transport_disruption'] = 0
    
    # One-Hot Encoding
    # In dashboard, we might have single row input, so we need to ensure all cols exist
    df = pd.get_dummies(df, columns=['season', 'weather'], prefix=['season', 'weather'], dtype=int)
    
    # Ensure all expected columns exist (season_1..4, weather_1..4)
    for i in range(1, 5):
        if f'season_{i}' not in df.columns: df[f'season_{i}'] = 0
        if f'weather_{i}' not in df.columns: df[f'weather_{i}'] = 0
        
    return df

df_fe = engineer_features(df)

# Global list of features used for training
FEATURE_COLS = ['workingday', 'temp', 'humidity', 
                'windspeed', 'hour', 'month', 'year', 'dayofweek', 'is_peak', 
                'is_weekend', 'time_bin', 'humid_temp', 'wind_weather', 'hour_sin', 'hour_cos',
                'season_1', 'season_2', 'season_3', 'season_4',
                'weather_1', 'weather_2', 'weather_3', 'weather_4',
                'is_disaster', 'transport_disruption']

# -----------------
# MODEL TRAINING (Cached)
# -----------------
@st.cache_resource
def train_model(data):
    # Ensure columns exist in training data
    X = data[FEATURE_COLS]
    y = np.log1p(data['count'])
    
    # Train GB (Fast & Accurate enough for dashboard)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df_fe)

# -----------------
# DASHBOARD LAYOUT
# -----------------

# Sidebar
st.sidebar.image("https://emojigraph.org/media/apple/bicycle_1f6b2.png", width=50)
st.sidebar.title("Configuration")

view_mode = st.sidebar.radio("View Mode", ["Historical Analysis", "Predictor Tool"])

if view_mode == "Historical Analysis":
    st.title("📊 Historical Demand Analysis")
    st.markdown("Explore how weather, time, and holidays affected bike rentals in DC (2011-2012).")
    
    # IMPORTANT: Re-construct original labels for filtering UI since we OHE'd the dataframe
    # We can use 'month' to proxy season roughly or just reload raw for viz if needed
    # But simpler: just add back 'year' which we kept. 'season' and 'weather' are gone.
    # To filter by season, we can use season_1 (Spring), season_2 (Summer)...
    
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.multiselect("Select Year", [2011, 2012], default=[2011, 2012])
    
    # Reconstruct season series for filtering
    def get_season(row):
        if row['season_1'] == 1: return 1
        if row['season_2'] == 1: return 2
        if row['season_3'] == 1: return 3
        return 4
    
    df_fe['season_viz'] = df_fe.apply(get_season, axis=1)
    
    with col2:
        selected_season_viz = st.multiselect("Select Season", [1, 2, 3, 4], default=[1, 2, 3, 4], 
                                         format_func=lambda x: {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}[x])
    
    filtered_df = df_fe[df_fe['year'].isin(selected_year) & df_fe['season_viz'].isin(selected_season_viz)]
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Rentals", f"{filtered_df['count'].sum():,}")
    m2.metric("Avg Rentals / Hour", f"{int(filtered_df['count'].mean())}")
    
    # Charts
    st.subheader("Demand over Time")
    fig_time = px.line(filtered_df.groupby('datetime')['count'].mean().reset_index(), x='datetime', y='count', title='Daily Average Demand')
    st.plotly_chart(fig_time, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hourly Trend")
        hourly_data = filtered_df.groupby('hour')['count'].mean().reset_index()
        fig_hour = px.area(hourly_data, x='hour', y='count', color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with c2:
        st.subheader("Temperature Impact") # Weather is now OHE, harder to boxplot easily
        fig_temp = px.scatter(filtered_df, x='temp', y='count', color='temp', title='Temperature vs Demand')
        st.plotly_chart(fig_temp, use_container_width=True)

elif view_mode == "Predictor Tool":
    st.title("🔮 Demand Predictor")
    st.markdown("Simulate conditions to predict bike demand.")
    
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        p_hour = st.slider("Hour of Day", 0, 23, 17)
        p_temp = st.slider("Temperature (°C)", -10.0, 40.0, 25.0)
        p_hum = st.slider("Humidity (%)", 0, 100, 50)
    
    with input_col2:
        p_season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}[x])
        p_working = st.checkbox("Is Working Day?", value=True)
        p_weather = st.selectbox("Weather", [1, 2, 3, 4], format_func=lambda x: {1:'Clear', 2:'Mist', 3:'Light Rain', 4:'Heavy Rain'}[x])
        
    with input_col3:
        p_wind = st.slider("Windspeed", 0.0, 60.0, 10.0)
        p_holiday = st.checkbox("Is Holiday?", value=False)
        p_disaster = st.checkbox("Is Disaster Check?", value=False) # Rare but possible
        p_transport = st.checkbox("Transport Issue?", value=False)
        
    # Construct input vector
    input_data = pd.DataFrame({
        'season': [p_season],
        'holiday': [int(p_holiday)],
        'is_disaster': [int(p_disaster)],
        'transport_disruption': [int(p_transport)],
        'workingday': [int(p_working)],
        'weather': [p_weather],
        'temp': [p_temp],
        'humidity': [p_hum],
        'windspeed': [p_wind],
        'datetime': [pd.Timestamp(f"2012-01-01 {p_hour}:00:00")] # Dummy date for feature eng
    })
    
    # Eng Features
    # Note: engineer_features will OHE 'season' and 'weather'
    engineered_input = engineer_features(input_data)

    # Predict
    log_pred = model.predict(engineered_input[FEATURE_COLS])
    pred = int(np.expm1(log_pred)[0])
    
    st.success(f"### Predicted Demand: {pred} Bikes")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Hourly Demand"},
        gauge = {'axis': {'range': [None, 1000]}, 'bar': {'color': "#FF4B4B"}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

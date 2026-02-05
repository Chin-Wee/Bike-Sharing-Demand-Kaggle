import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# PAGE CONFIG
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# --- FEATURES PIPELINE (Single Source of Truth) ---
def clean_and_engineer(df):
    """Refactored Feature Engineering Logic - Must Match Notebook"""
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # Fallback if datetime is index or missing
        return df
        
    # 1. Temporal Components
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # 2. Time Bins
    df['time_bin'] = pd.cut(df['hour'], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)
    
    # 3. Peak Hours
    df['is_peak'] = 0
    peak_mask = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
    if 'workingday' in df.columns:
        df.loc[peak_mask & (df['workingday'] == 1), 'is_peak'] = 1
    
    # 4. Weekend
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 5. Domain Interactions
    if 'weather' in df.columns:
        df['bad_weather'] = (df['weather'] >= 3).astype(int)
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['humid_temp'] = df['temp'] * df['humidity']
    if 'windspeed' in df.columns and 'weather' in df.columns:
        df['wind_weather'] = df['windspeed'] * df['weather']
    
    # 6. Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 7. Flags (Handling NULLs if logic strictly depends on join presence)
    # Streamlit version might load csvs differently, but we use what we have.
    # If columns missing, fill 0
    cols = ['disruption_event_type', 'specific_holiday_name']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df['is_disaster'] = (df['disruption_event_type'] == 'disaster').astype(int)
    df['transport_disruption'] = (df['disruption_event_type'] == 'transport_disruption').astype(int)
    df['is_federal_holiday'] = df['specific_holiday_name'].notna().astype(int)
    
    return df

# --- DATA LOADING ---
@st.cache_data
def load_data():
    db_path = 'bike_sharing.db'
    conn = sqlite3.connect(db_path)
    
    # Load raw SQL query
    with open('Data_Augmentation.sql', 'r') as f:
        query = f.read()
        
    try:
        # Try to read from existing DB
        df = pd.read_sql_query(query, conn)
        
        # Check if 'count' exists (Target variable)
        # If not, we might be looking at Test data in the 'train' table (from notebook artifact)
        if 'count' not in df.columns:
            st.warning("⚠️ 'count' column missing (likely Test data in DB). Reloading Train data...")
            raise ValueError("Target column missing")
            
    except (pd.io.sql.DatabaseError, ValueError, sqlite3.OperationalError):
        # Fallback: Reload Train CSV into DB
        # Ensure we look in the correct subdirectory
        train_path = 'bike-sharing-demand/train.csv'
        if not os.path.exists(train_path):
             train_path = 'train.csv' # Fallback to root (unlikely but safe)
             
        train_df = pd.read_csv(train_path)
        
        # Clear incompatible tables if necessary or just replace
        train_df.to_sql('train', conn, index=False, if_exists='replace')
        
        # We also need external tables to be present for the SQL JOINs to work!
        # If train was missing, likely externals are too or we want to be safe
        try:
            pd.read_csv('weather_external.csv').to_sql('weather_external', conn, index=False, if_exists='replace')
            pd.read_csv('holidays_external.csv').to_sql('holidays_external', conn, index=False, if_exists='replace')
            pd.read_csv('disruptions.csv').to_sql('disruptions', conn, index=False, if_exists='replace')
        except Exception as e:
            st.warning(f"⚠️ Could not reload external data: {e}")

        # Retry Query
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    # Process
    df = clean_and_engineer(df)
    return df

@st.cache_resource
def train_model(df):
    # Simplified Model for Dashboard (RandomForest)
    features = ['workingday', 'temp', 'humidity', 'windspeed', 'hour', 'month', 'year', 
                'dayofweek', 'is_peak', 'is_weekend', 'time_bin', 'humid_temp', 
                'wind_weather', 'hour_sin', 'hour_cos', 'is_disaster', 'transport_disruption']
                
    # Align features
    X = df[features]
    y = np.log1p(df['count']) # Target log transform
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, features

# --- MAIN APP ---
def main():
    st.title("🚲 Bike Sharing Demand Dashboard")
    st.markdown("### Interactive Analysis & Prediction Tool")
    
    # Load Data
    try:
        df = load_data()
        st.success(f"Data Loaded Successfully: {df.shape[0]} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Analysis", "🔮 Prediction Playground", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Demand Patterns")
        
        col1, col2 = st.columns(2)
        with col1:
            # Hourly Trend
            hourly = df.groupby('hour')['count'].mean().reset_index()
            fig = px.line(hourly, x='hour', y='count', title="Average Demand by Hour", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Monthly Trend
            monthly = df.groupby('month')['count'].mean().reset_index()
            fig = px.bar(monthly, x='month', y='count', title="Average Demand by Month", color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
            
        # Interactive Scatter (Temp vs Count)
        st.subheader("Correlations")
        fig_scatter = px.scatter(df, x='temp', y='count', color='season', opacity=0.5, 
                               title="Temperature vs Demand (Color by Season)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.subheader("What-If Analysis")
        st.markdown("Adjust parameters to predict bike demand.")
        
        model, features = train_model(df)
        
        # Input Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            input_hour = st.slider("Hour of Day", 0, 23, 8)
            input_temp = st.slider("Temperature (C)", 0.0, 45.0, 25.0)
        with col2:
            input_humidity = st.slider("Humidity (%)", 0, 100, 50)
            input_wind = st.slider("Windspeed", 0.0, 60.0, 10.0)
        with col3:
            input_workingday = st.selectbox("Working Day?", [0, 1], index=1)
            input_weather = st.selectbox("Weather Condition", [1, 2, 3, 4], index=0)
            
        # Create Input DataFrame
        # We need to construct a single row DF with all required features
        # Note: We need to engineer the features for this single row too!
        
        # Base dict
        input_data = {
            'datetime': [datetime(2012, 7, 1, input_hour, 0, 0)], # Dummy date, year 2012
            'workingday': [input_workingday],
            'weather': [input_weather],
            'temp': [input_temp],
            'humidity': [input_humidity],
            'windspeed': [input_wind],
            'disruption_event_type': [None], # Default no disruption
            'specific_holiday_name': [None]
        }
        
        single_df = pd.DataFrame(input_data)
        # Engineer
        processed_input = clean_and_engineer(single_df)
        
        # Predict
        try:
            log_prediction = model.predict(processed_input[features])[0]
            prediction = int(np.expm1(log_prediction))
            
            st.metric(label="Predicted Bike Rentals", value=f"{prediction}", delta=None)
            
            # Confidence/Context
            st.info(f"Context: Hour {input_hour}, Temp {input_temp}C. " 
                    f"{'Peak Hour!' if processed_input['is_peak'][0] else 'Off-Peak'}")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    with tab3:
        st.dataframe(df.head(100))

if __name__ == "__main__":
    main()

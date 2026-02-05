import pandas as pd
import numpy as np
import sqlite3
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

print("⏳ Evaluating optimized models...")

# Setup data
conn = sqlite3.connect(':memory:')
conn.create_function("sin", 1, math.sin)
conn.create_function("cos", 1, math.cos)

pd.read_csv('bike-sharing-demand/train.csv').to_sql('train', conn, index=False, if_exists='replace')
pd.read_csv('weather_external.csv').to_sql('weather_external', conn, index=False)
pd.read_csv('holidays_external.csv').to_sql('holidays_external', conn, index=False)

with open('Data_Augmentation.sql', 'r') as f:
    query = f.read()
df = pd.read_sql_query(query, conn)

feature_cols = [
    'workingday', 'temp', 'humidity', 'windspeed', 'hour', 'month', 'year', 'dayofweek', 
    'is_peak', 'is_weekend', 'time_bin', 'humid_temp', 'wind_weather', 'hour_sin', 'hour_cos',
    'season_1', 'season_2', 'season_3', 'season_4',
    'weather_1', 'weather_2', 'weather_3', 'weather_4',
    'is_disaster', 'transport_disruption'
]
for c in feature_cols:
    if c not in df.columns: df[c] = 0

X = df[feature_cols]
y = np.log1p(df['count'])

# Time-based split
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
y_train_orig = df['count'].iloc[:split_idx]
y_val_orig = df['count'].iloc[split_idx:]

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# Optimized parameters
rf_params = {'n_estimators': 210, 'max_depth': 21, 'min_samples_split': 4, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1}
gb_params = {'n_estimators': 276, 'max_depth': 5, 'learning_rate': 0.14958372840580794, 'subsample': 0.9343969277612623, 'min_samples_split': 16, 'random_state': 42}
xgb_params = {'n_estimators': 922, 'max_depth': 4, 'learning_rate': 0.26043769813481843, 'subsample': 0.9135880223596955, 'colsample_bytree': 0.6165420399609431, 'reg_alpha': 4.7318289376165215, 'reg_lambda': 1.7143414756355364, 'random_state': 42, 'n_jobs': -1}

print("\n" + "="*60)
print("📊 OPTIMIZED MODEL PERFORMANCE (RMSLE - Lower is Better)")
print("="*60)

for name, params in [('Random Forest', rf_params), ('Gradient Boosting', gb_params), ('XGBoost', xgb_params)]:
    if name == 'Random Forest':
        model = RandomForestRegressor(**params)
    elif name == 'Gradient Boosting':
        model = GradientBoostingRegressor(**params)
    else:
        model = XGBRegressor(**params)
    
    model.fit(X_train, y_train)
    
    # Training RMSLE
    train_pred = np.expm1(model.predict(X_train))
    train_pred = np.maximum(0, train_pred)
    train_rmsle = rmsle(y_train_orig, train_pred)
    
    # Validation RMSLE
    val_pred = np.expm1(model.predict(X_val))
    val_pred = np.maximum(0, val_pred)
    val_rmsle = rmsle(y_val_orig, val_pred)
    
    print(f"\n{name}:")
    print(f"   Training RMSLE:   {train_rmsle:.5f}")
    print(f"   Validation RMSLE: {val_rmsle:.5f}")
    print(f"   Gap:              {val_rmsle - train_rmsle:.5f}")

print("\n" + "="*60)

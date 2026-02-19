import requests
import pandas as pd
import numpy as np
from time import sleep

# --- 1. METEOROLOGY via Open-Meteo API ---
def fetch_weather_history(lat, lon, sample_date):
    """
    Fetches historical weather for the specific location and date.
    Returns: Rain accumulation (7d, 30d) and Temp avg.
    """
    try:
        # 1. API expects ISO date
        date_obj = pd.to_datetime(sample_date, dayfirst=True)
        end_date = date_obj.strftime('%Y-%m-%d')
        start_date = (date_obj - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        # 2. Call Open-Meteo Archive
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["precipitation_sum", "temperature_2m_mean"],
            "timezone": "auto"
        }
        
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if 'daily' not in data:
            return {}
            
        df_weather = pd.DataFrame(data['daily'])
        
        # 3. Calculate Aggregates
        rain_7d = df_weather['precipitation_sum'].tail(7).sum()
        rain_30d = df_weather['precipitation_sum'].sum()
        temp_mean = df_weather['temperature_2m_mean'].mean()
        
        return {
            "rain_7d_sum": rain_7d,
            "rain_30d_sum": rain_30d,
            "temp_30d_mean": temp_mean
        }
        
    except Exception as e:
        # print(f"Weather Error: {e}")
        return {
            "rain_7d_sum": np.nan,
            "rain_30d_sum": np.nan, 
            "temp_30d_mean": np.nan
        }

# --- 2. SOIL PROPERTIES via SoilGrids REST API ---
def fetch_soil_properties(lat, lon):
    """
    Fetches soil chemistry/physics from ISRIC SoilGrids.
    """
    try:
        # Query for: Clay, pH, Cation Exchange Capacity (CEC) at surface (0-5cm)
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lat": lat,
            "lon": lon,
            "properties": ["clay", "phh2o", "cec"],
            "depths": ["0-5cm"],
            "values": "mean"
        }
        
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        
        props = data['properties']['layers']
        result = {}
        
        for layer in props:
            name = layer['name'] 
            val = layer['depths'][0]['values']['mean']
            if name == 'phh2o': val = val / 10.0 
            result[f"soil_{name}"] = val
            
        return result
        
    except Exception:
        return {
            "soil_clay": np.nan,
            "soil_phh2o": np.nan,
            "soil_cec": np.nan
        }

# --- 3. BATCH PROCESSOR ---
def fetch_environ_features(row):
    lat, lon = row['Latitude'], row['Longitude']
    date = row['Sample Date']
    
    # Sleep slightly to respect free API rate limits
    sleep(0.1) 
    
    weather = fetch_weather_history(lat, lon, date)
    soil = fetch_soil_properties(lat, lon)
    
    return {**weather, **soil}

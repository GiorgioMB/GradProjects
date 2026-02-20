import requests
import pandas as pd
import numpy as np
from time import sleep


# 1. METEOROLOGY  (Open-Meteo Archive)
def fetch_weather_history(lat, lon, sample_date):
    """
    Historical weather for a specific location + date window.
    Returns rain, temp, ET, wind, humidity, radiation, water balance.
    """
    try:
        date_obj = pd.to_datetime(sample_date, dayfirst=True)
        end_date   = date_obj.strftime('%Y-%m-%d')
        start_date = (date_obj - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "precipitation_sum",
                "temperature_2m_mean",
                "et0_fao_evapotranspiration",
                "windspeed_10m_max",
                "relative_humidity_2m_mean",
                "shortwave_radiation_sum",
            ],
            "timezone": "auto",
        }

        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        if 'daily' not in data:
            return _empty_weather()

        df_w = pd.DataFrame(data['daily'])

        rain_7d  = df_w['precipitation_sum'].tail(7).sum()
        rain_30d = df_w['precipitation_sum'].sum()
        temp_mean = df_w['temperature_2m_mean'].mean()

        # ET
        et_col = 'et0_fao_evapotranspiration'
        et_30d = df_w[et_col].sum() if et_col in df_w.columns else np.nan
        et_7d  = df_w[et_col].tail(7).sum() if et_col in df_w.columns else np.nan

        # Wind / humidity / radiation
        wind_mean = (df_w['windspeed_10m_max'].mean()
                     if 'windspeed_10m_max' in df_w.columns else np.nan)
        humidity_mean = (df_w['relative_humidity_2m_mean'].mean()
                         if 'relative_humidity_2m_mean' in df_w.columns else np.nan)
        radiation_mean = (df_w['shortwave_radiation_sum'].mean()
                          if 'shortwave_radiation_sum' in df_w.columns else np.nan)

        water_balance_30d = (rain_30d - et_30d
                             if not np.isnan(et_30d) else np.nan)

        return {
            "rain_7d_sum":         rain_7d,
            "rain_30d_sum":        rain_30d,
            "temp_30d_mean":       temp_mean,
            "et_30d_sum":          et_30d,
            "et_7d_sum":           et_7d,
            "water_balance_30d":   water_balance_30d,
            "wind_30d_mean":       wind_mean,
            "humidity_30d_mean":   humidity_mean,
            "radiation_30d_mean":  radiation_mean,
        }

    except Exception:
        return _empty_weather()


def _empty_weather():
    return {
        "rain_7d_sum":        np.nan,
        "rain_30d_sum":       np.nan,
        "temp_30d_mean":      np.nan,
        "et_30d_sum":         np.nan,
        "et_7d_sum":          np.nan,
        "water_balance_30d":  np.nan,
        "wind_30d_mean":      np.nan,
        "humidity_30d_mean":  np.nan,
        "radiation_30d_mean": np.nan,
    }


# 2. SOIL PROPERTIES  (ISRIC SoilGrids)
def fetch_soil_properties(lat, lon):
    """SoilGrids REST – often broken / returns no data for SA."""
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lat": lat, "lon": lon,
            "properties": ["clay", "phh2o", "cec"],
            "depths": ["0-5cm"],
            "values": "mean",
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        props = data['properties']['layers']
        result = {}
        for layer in props:
            name = layer['name']
            val  = layer['depths'][0]['values']['mean']
            if name == 'phh2o':
                val = val / 10.0
            result[f"soil_{name}"] = val
        return result

    except Exception:
        return {"soil_clay": np.nan, "soil_phh2o": np.nan, "soil_cec": np.nan}


# 3. Wrapper for enrich_dataset
def fetch_environ_features(row):
    lat, lon = row['Latitude'], row['Longitude']
    date = row['Sample Date']

    sleep(0.1)   # Respect API rate limits

    weather = fetch_weather_history(lat, lon, date)
    soil    = fetch_soil_properties(lat, lon)

    return {**weather, **soil}

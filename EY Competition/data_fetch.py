import time
import os
import pandas as pd
import numpy as np
import pystac_client
import planetary_computer
from odc.stac import load as stac_load
import osmnx as ox
from tqdm import tqdm
import config
import fetch_climate_soil

def get_client():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

# --- 1. TERRAIN FETCH ---
def fetch_terrain_features(lat, lon, buffer_deg=0.01):
    try:
        catalog = get_client()
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        search = catalog.search(collections=["nasadem"], bbox=bbox)
        items = list(search.item_collection())
        
        if not items: return {}

        ds = stac_load(items, bands=["elevation"], bbox=bbox).isel(time=0)
        dem = ds.elevation.values.astype(float)
        
        grad_y, grad_x = np.gradient(dem)
        res_deg = 0.00027777777 
        meters_per_deg_lat = 111132.0
        meters_per_deg_lon = 111132.0 * np.cos(np.deg2rad(lat))
        
        dy_meters = res_deg * meters_per_deg_lat
        dx_meters = res_deg * meters_per_deg_lon
        
        slope_x = grad_x / dx_meters
        slope_y = grad_y / dy_meters
        slope = np.sqrt(slope_x**2 + slope_y**2)
        aspect = np.arctan2(slope_y, -slope_x) 
        
        return {
            "elevation_mean": np.nanmean(dem),
            "slope_mean": np.nanmean(slope),     
            "slope_max": np.nanmax(slope),      
            "aspect_mean": np.nanmean(aspect), 
            "valley_depth": np.nanmax(dem) - np.nanmin(dem)
        }
    except Exception:
        return {}

# --- 2. OSM FETCH ---
def fetch_osm_context(lat, lon, dist_m=5000):
    tags = {
        'landuse': ['industrial', 'farmland', 'military', 'residential', 'mining'],
        'man_made': ['wastewater_plant', 'mineshaft', 'works'],
        'natural': ['water']
    }
    features = {
        "count_industry": 0, "count_farms": 0, "count_military": 0,
        "count_mines": 0, "count_wastewater": 0
    }
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=dist_m)
        if gdf.empty: return features
        
        if 'landuse' in gdf.columns:
            features["count_industry"] = len(gdf[gdf['landuse'] == 'industrial'])
            features["count_farms"] = len(gdf[gdf['landuse'] == 'farmland'])
            features["count_military"] = len(gdf[gdf['landuse'] == 'military'])
        
        if 'man_made' in gdf.columns:
            features["count_mines"] += len(gdf[gdf['man_made'].isin(['mineshaft', 'mining'])])
            features["count_wastewater"] += len(gdf[gdf['man_made'] == 'wastewater_plant'])
        return features
    except Exception:
        return features

# --- 3. SATELLITE FETCH ---
def fetch_temporal_satellite(row, relaxed_mode=False):
    catalog = get_client()
    
    window = config.RETRY_WINDOW_DAYS if relaxed_mode else config.SAT_WINDOW_DAYS
    cloud_lim = config.RETRY_CLOUD_MAX if relaxed_mode else config.SAT_CLOUD_MAX
    attempts = 1 if relaxed_mode else 3

    for attempt in range(attempts):
        try:
            lat, lon = row['Latitude'], row['Longitude']
            sample_date = pd.to_datetime(row['Sample Date'], dayfirst=True)
            start_date = sample_date - pd.Timedelta(days=window)
            time_range = f"{start_date.strftime('%Y-%m-%d')}/{sample_date.strftime('%Y-%m-%d')}"
            bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
            
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=time_range,
                query={"eo:cloud_cover": {"lt": cloud_lim}}
            )
            items = list(search.item_collection())
            
            if not items:
                # Return dict with CONSISTENT column order
                return {
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                    'Sample Date': str(row['Sample Date']),
                    'green': np.nan,
                    'red': np.nan,
                    'nir08': np.nan,
                    'swir16': np.nan,
                    'swir22': np.nan,
                    'NDMI': np.nan,
                    'MNDWI': np.nan,
                    'NDVI': np.nan
                }

            data = stac_load(
                items,
                bands=["green", "red", "nir08", "swir16", "swir22"],
                bbox=bbox,
                resolution=30,
                chunks={}, 
                groupby="solar_day", 
                stac_cfg={'landsat-c2-l2': {'assets': {'nir08': {'alias': 'nir'}}}}
            )
            
            median_composite = data.median(dim="time").compute()
            point_stats = median_composite.median(dim=["x", "y"])
            values = {var: float(point_stats[var].values) for var in point_stats.data_vars}
            
            eps = 1e-8
            values['NDMI'] = (values['nir08'] - values['swir16']) / (values['nir08'] + values['swir16'] + eps)
            values['MNDWI'] = (values['green'] - values['swir16']) / (values['green'] + values['swir16'] + eps)
            values['NDVI'] = (values['nir08'] - values['red']) / (values['nir08'] + values['red'] + eps)
            
            return {
                'Latitude': float(lat),
                'Longitude': float(lon),
                'Sample Date': str(row['Sample Date']),
                'green': values.get('green', np.nan),
                'red': values.get('red', np.nan),
                'nir08': values.get('nir08', np.nan),
                'swir16': values.get('swir16', np.nan),
                'swir22': values.get('swir22', np.nan),
                'NDMI': values.get('NDMI', np.nan),
                'MNDWI': values.get('MNDWI', np.nan),
                'NDVI': values.get('NDVI', np.nan)
            }

        except Exception as e:
            if attempt == attempts - 1:
                return {
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                    'Sample Date': str(row['Sample Date']),
                    'green': np.nan,
                    'red': np.nan,
                    'nir08': np.nan,
                    'swir16': np.nan,
                    'swir22': np.nan,
                    'NDMI': np.nan,
                    'MNDWI': np.nan,
                    'NDVI': np.nan
                }
            time.sleep(1 + attempt)

# --- 4. ORCHESTRATOR ---
def enrich_dataset(df, cache_path):
    """
    Iterates through locations to add Engineering, OSM, AND Environment features.
    """
    df['key'] = (
        df['Latitude'].astype(str) + "_" + 
        df['Longitude'].astype(str) + "_" + 
        df['Sample Date'].astype(str)
    )
    
    unique_rows = df.drop_duplicates(subset=['key'])
    
    if os.path.exists(cache_path):
        print(f"Loading context cache '{cache_path}'...")
        cached_df = pd.read_csv(cache_path)
        cached_df['key'] = (
            cached_df['Latitude'].astype(str) + "_" + 
            cached_df['Longitude'].astype(str) + "_" + 
            cached_df['Sample Date'].astype(str)
        )
        existing_keys = set(cached_df['key'])
    else:
        cached_df = pd.DataFrame()
        existing_keys = set()
        
    to_compute = unique_rows[~unique_rows['key'].isin(existing_keys)]
    
    if to_compute.empty:
        print("All context data found in cache.")
        if 'key' in cached_df.columns: cached_df = cached_df.drop(columns=['key'])
        if 'key' in df.columns: df = df.drop(columns=['key'])
        
        return pd.merge(df, cached_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')

    print(f"Computing Context (OSM, Terrain, Weather, Soil) for {len(to_compute)} rows...")
    
    header_needed = not os.path.exists(cache_path)
    chunk_size = 10
    new_results_accumulator = []
    
    # Iterate in chunks
    for i in tqdm(range(0, len(to_compute), chunk_size)):
        chunk = to_compute.iloc[i:i+chunk_size]
        batch_data = []
        
        for idx, row in chunk.iterrows():
            lat, lon = row['Latitude'], row['Longitude']
            
            # A. Fetch Static Data (OSM, Terrain)
            osm = fetch_osm_context(lat, lon)
            terrain = fetch_terrain_features(lat, lon)
            
            # B. Fetch Dynamic/Env Data
            env = fetch_climate_soil.fetch_environ_features(row)
            
            # C. Combine everything
            combined = {
                'Latitude': lat, 
                'Longitude': lon, 
                'Sample Date': row['Sample Date'],
                **osm, 
                **terrain,
                **env
            }
            batch_data.append(combined)
            
        batch_df = pd.DataFrame(batch_data)
        
        # Write immediately to disk
        batch_df.to_csv(cache_path, mode='a', header=header_needed, index=False)
        header_needed = False 
        
        new_results_accumulator.append(batch_df)

    if new_results_accumulator:
        new_results_df = pd.concat(new_results_accumulator, ignore_index=True)
        # Drop key from cache before concat if it exists
        if 'key' in cached_df.columns: cached_df = cached_df.drop(columns=['key'])
        final_cache = pd.concat([cached_df, new_results_df], ignore_index=True)
    else:
        final_cache = cached_df

    # Cleanup merge keys
    final_cache = final_cache.drop_duplicates(subset=['Latitude', 'Longitude', 'Sample Date'])
    if 'key' in df.columns: df = df.drop(columns=['key'])
    
    return pd.merge(df, final_cache, on=['Latitude', 'Longitude', 'Sample Date'], how='left')

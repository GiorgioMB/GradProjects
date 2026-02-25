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
import fetch_geo_features

# HELPER: Safe STAC client
def get_client():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# 1.  TERRAIN  (NASADEM via Planetary Computer)
def fetch_terrain_features(lat, lon, buffer_deg=0.01):
    try:
        catalog = get_client()
        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]
        search = catalog.search(collections=["nasadem"], bbox=bbox)
        items = list(search.item_collection())
        if not items:
            return {}

        ds  = stac_load(items, bands=["elevation"], bbox=bbox).isel(time=0)
        dem = ds.elevation.values.astype(float)

        grad_y, grad_x = np.gradient(dem)
        res_deg = 0.00027777777
        meters_per_deg_lat = 111132.0
        meters_per_deg_lon = 111132.0 * np.cos(np.deg2rad(lat))
        dy = res_deg * meters_per_deg_lat
        dx = res_deg * meters_per_deg_lon

        slope = np.sqrt((grad_x / dx)**2 + (grad_y / dy)**2)
        aspect = np.arctan2(grad_y, -grad_x)

        return {
            "elevation_mean": np.nanmean(dem),
            "elevation_std":  np.nanstd(dem),
            "slope_mean":     np.nanmean(slope),
            "slope_max":      np.nanmax(slope),
            "slope_std":      np.nanstd(slope),
            "aspect_mean":    np.nanmean(aspect),
            "valley_depth":   np.nanmax(dem) - np.nanmin(dem),
        }
    except Exception:
        return {}


# 2.  OSM context
def fetch_osm_context(lat, lon, dist_m=5000):
    tags = {
        'landuse': ['industrial', 'farmland', 'military',
                     'residential', 'mining'],
        'man_made': ['wastewater_plant', 'mineshaft', 'works'],
        'natural':  ['water'],
    }
    features = {
        "count_industry": 0, "count_farms": 0,
        "count_military": 0, "count_mines": 0,
        "count_wastewater": 0,
    }
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=dist_m)
        if gdf.empty:
            return features
        if 'landuse' in gdf.columns:
            features["count_industry"] = len(gdf[gdf['landuse'] == 'industrial'])
            features["count_farms"]    = len(gdf[gdf['landuse'] == 'farmland'])
            features["count_military"] = len(gdf[gdf['landuse'] == 'military'])
        if 'man_made' in gdf.columns:
            features["count_mines"]     += len(gdf[gdf['man_made'].isin(
                                                  ['mineshaft', 'mining'])])
            features["count_wastewater"] += len(gdf[gdf['man_made'] == 'wastewater_plant'])
        return features
    except Exception:
        return features


# 3.  SATELLITE  (Landsat C2-L2)
def fetch_temporal_satellite(row, relaxed_mode=False):
    catalog   = get_client()
    window    = config.RETRY_WINDOW_DAYS if relaxed_mode else config.SAT_WINDOW_DAYS
    cloud_lim = config.RETRY_CLOUD_MAX   if relaxed_mode else config.SAT_CLOUD_MAX
    attempts  = 1 if relaxed_mode else 3

    _empty = {
        'Latitude':    float(row['Latitude']),
        'Longitude':   float(row['Longitude']),
        'Sample Date': str(row['Sample Date']),
        'green': np.nan, 'red': np.nan, 'nir08': np.nan,
        'swir16': np.nan, 'swir22': np.nan,
        'NDMI': np.nan, 'MNDWI': np.nan, 'NDVI': np.nan,
    }

    for attempt in range(attempts):
        try:
            lat, lon = row['Latitude'], row['Longitude']
            sample_date = pd.to_datetime(row['Sample Date'], dayfirst=True)
            start_date  = sample_date - pd.Timedelta(days=window)
            time_range  = (f"{start_date.strftime('%Y-%m-%d')}/"
                           f"{sample_date.strftime('%Y-%m-%d')}")
            bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]

            search = catalog.search(
                collections=["landsat-c2-l2"], bbox=bbox,
                datetime=time_range,
                query={"eo:cloud_cover": {"lt": cloud_lim}},
            )
            items = list(search.item_collection())
            if not items:
                return _empty

            data = stac_load(
                items,
                bands=["green", "red", "nir08", "swir16", "swir22"],
                bbox=bbox, resolution=30, chunks={},
                groupby="solar_day",
                stac_cfg={'landsat-c2-l2': {
                    'assets': {'nir08': {'alias': 'nir'}}}},
            )

            median_composite = data.median(dim="time").compute()
            ps = median_composite.median(dim=["x", "y"])
            vals = {v: float(ps[v].values) for v in ps.data_vars}

            eps = 1e-8
            vals['NDMI']  = ((vals['nir08'] - vals['swir16']) /
                             (vals['nir08'] + vals['swir16'] + eps))
            vals['MNDWI'] = ((vals['green'] - vals['swir16']) /
                             (vals['green'] + vals['swir16'] + eps))
            vals['NDVI']  = ((vals['nir08'] - vals['red']) /
                             (vals['nir08'] + vals['red'] + eps))

            return {
                'Latitude':    float(lat),
                'Longitude':   float(lon),
                'Sample Date': str(row['Sample Date']),
                'green':  vals.get('green',  np.nan),
                'red':    vals.get('red',    np.nan),
                'nir08':  vals.get('nir08',  np.nan),
                'swir16': vals.get('swir16', np.nan),
                'swir22': vals.get('swir22', np.nan),
                'NDMI':   vals.get('NDMI',   np.nan),
                'MNDWI':  vals.get('MNDWI',  np.nan),
                'NDVI':   vals.get('NDVI',   np.nan),
            }
        except Exception:
            if attempt == attempts - 1:
                return _empty
            time.sleep(1 + attempt)


# 4.  ORCHESTRATOR
def enrich_dataset(df, cache_path):
    df['key'] = (
        df['Latitude'].astype(str) + "_" +
        df['Longitude'].astype(str) + "_" +
        df['Sample Date'].astype(str)
    )
    unique_rows = df.drop_duplicates(subset=['key'])

    # PART A:  OSM + Terrain + Weather + Soil
    if os.path.exists(cache_path):
        print(f"   Loading cache '{cache_path}'...")
        # Robust read: truncate extra trailing fields to expected column count
        import csv as _csv
        with open(cache_path) as _f:
            _reader = _csv.reader(_f)
            _hdr = next(_reader)
            _ncols = len(_hdr)
            _rows = [r[:_ncols] for r in _reader]
        cached_df = pd.DataFrame(_rows, columns=_hdr)
        # Convert numeric columns back from strings
        for _c in cached_df.columns:
            if _c not in ('Sample Date', 'key'):
                cached_df[_c] = pd.to_numeric(cached_df[_c], errors='coerce')
        cached_df['key'] = (
            cached_df['Latitude'].astype(str) + "_" +
            cached_df['Longitude'].astype(str) + "_" +
            cached_df['Sample Date'].astype(str)
        )
        existing_keys = set(cached_df['key'])

        # Check if cache is missing extended weather columns → rebuild those
        expected_weather = ['et_30d_sum', 'water_balance_30d', 'wind_30d_mean',
                            'humidity_30d_mean', 'radiation_30d_mean']
        missing_cols = [c for c in expected_weather if c not in cached_df.columns]
        if missing_cols:
            print(f"   Cache is missing columns {missing_cols} – will re-fetch.")
            # Force re-fetch all cached rows that lack these columns
            existing_keys = set()
            cached_df = pd.DataFrame()
    else:
        cached_df = pd.DataFrame()
        existing_keys = set()

    to_compute = unique_rows[~unique_rows['key'].isin(existing_keys)]

    if to_compute.empty:
        print("   All OSM/Terrain/Weather data found in cache.")
    else:
        print(f"   Computing OSM/Terrain/Weather/Soil for "
              f"{len(to_compute)} rows...")

        header_needed = not os.path.exists(cache_path) or cached_df.empty
        chunk_size = 10
        accumulator = []

        for i in tqdm(range(0, len(to_compute), chunk_size)):
            chunk = to_compute.iloc[i:i + chunk_size]
            batch = []
            for _, row in chunk.iterrows():
                lat, lon = row['Latitude'], row['Longitude']
                osm     = fetch_osm_context(lat, lon)
                terrain = fetch_terrain_features(lat, lon)
                env     = fetch_climate_soil.fetch_environ_features(row)
                combined = {
                    'Latitude':    lat,
                    'Longitude':   lon,
                    'Sample Date': row['Sample Date'],
                    **osm, **terrain, **env,
                }
                batch.append(combined)

            batch_df = pd.DataFrame(batch)
            batch_df.to_csv(cache_path, mode='a',
                            header=header_needed, index=False)
            header_needed = False
            accumulator.append(batch_df)

        if accumulator:
            new_df = pd.concat(accumulator, ignore_index=True)
            if not cached_df.empty:
                if 'key' in cached_df.columns:
                    cached_df = cached_df.drop(columns=['key'])
                cached_df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                cached_df = new_df

    # Merge Part A
    if 'key' in cached_df.columns:
        cached_df = cached_df.drop(columns=['key'])
    cached_df = cached_df.drop_duplicates(
        subset=['Latitude', 'Longitude', 'Sample Date'])
    if 'key' in df.columns:
        df = df.drop(columns=['key'])

    df = pd.merge(df, cached_df,
                  on=['Latitude', 'Longitude', 'Sample Date'], how='left')

    # PART B:  Geo-enrichment (WorldCover, JRC, Geology, Population,
    #          Water infrastructure, Water body type)
    geo_cache_path = config.GEO_CACHE

    df['_geo_key'] = (
        df['Latitude'].round(4).astype(str) + "_" +
        df['Longitude'].round(4).astype(str)
    )
    unique_locs = df.drop_duplicates(subset=['_geo_key'])

    if os.path.exists(geo_cache_path):
        print(f"   Loading geo cache '{geo_cache_path}'...")
        geo_cached = pd.read_csv(geo_cache_path)
        geo_cached['_geo_key'] = (
            geo_cached['Latitude'].round(4).astype(str) + "_" +
            geo_cached['Longitude'].round(4).astype(str)
        )
        existing_geo_keys = set(geo_cached['_geo_key'])
    else:
        geo_cached = pd.DataFrame()
        existing_geo_keys = set()

    to_compute_geo = unique_locs[~unique_locs['_geo_key'].isin(existing_geo_keys)]

    if to_compute_geo.empty:
        print("   All geo-enrichment data found in cache.")
    else:
        print(f"   Fetching geo-enrichment for "
              f"{len(to_compute_geo)} locations...")

        header_needed = not os.path.exists(geo_cache_path)
        chunk_size = 10
        geo_acc = []

        for i in tqdm(range(0, len(to_compute_geo), chunk_size)):
            chunk = to_compute_geo.iloc[i:i + chunk_size]
            batch = []
            for _, row in chunk.iterrows():
                geo = fetch_geo_features.fetch_geo_features(row)
                geo['Latitude']  = row['Latitude']
                geo['Longitude'] = row['Longitude']
                batch.append(geo)

            batch_df = pd.DataFrame(batch)
            batch_df.to_csv(geo_cache_path, mode='a',
                            header=header_needed, index=False)
            header_needed = False
            geo_acc.append(batch_df)

        if geo_acc:
            new_geo = pd.concat(geo_acc, ignore_index=True)
            if not geo_cached.empty:
                if '_geo_key' in geo_cached.columns:
                    geo_cached = geo_cached.drop(columns=['_geo_key'])
                geo_cached = pd.concat([geo_cached, new_geo], ignore_index=True)
            else:
                geo_cached = new_geo

    # Merge Part B
    if not geo_cached.empty:
        for c in ['_geo_key']:
            if c in geo_cached.columns:
                geo_cached = geo_cached.drop(columns=[c])
        geo_cached = geo_cached.drop_duplicates(
            subset=['Latitude', 'Longitude'])

        df['_lat_r'] = df['Latitude'].astype(float).round(4)
        df['_lon_r'] = df['Longitude'].astype(float).round(4)
        geo_cached['_lat_r'] = geo_cached['Latitude'].astype(float).round(4)
        geo_cached['_lon_r'] = geo_cached['Longitude'].astype(float).round(4)

        geo_merge_cols = [c for c in geo_cached.columns
                         if c not in ['Latitude', 'Longitude',
                                      '_lat_r', '_lon_r']]
        df = pd.merge(df,
                      geo_cached[['_lat_r', '_lon_r'] + geo_merge_cols],
                      on=['_lat_r', '_lon_r'], how='left')
        df = df.drop(columns=['_lat_r', '_lon_r'], errors='ignore')

    df = df.drop(columns=['_geo_key'], errors='ignore')
    return df

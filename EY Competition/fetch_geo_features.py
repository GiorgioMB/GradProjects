import numpy as np
import pandas as pd
import requests
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from time import sleep

#  Shared helpers
_PC_CLIENT = None

def _get_pc_client():
    global _PC_CLIENT
    if _PC_CLIENT is None:
        _PC_CLIENT = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _PC_CLIENT


def _safe_read_raster(href, bbox, band=1):
    try:
        with rasterio.open(href) as src:
            window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                                 src.transform)
            data = src.read(band, window=window)
            if data.size == 0:
                return None
            return data
    except Exception:
        return None


# ESA WorldCover 10 m  
_WC_CLASSES = {
    10: 'tree_cover',
    20: 'shrubland',
    30: 'grassland',
    40: 'cropland',
    50: 'built_up',
    60: 'bare_sparse',
    70: 'snow_ice',
    80: 'water',
    90: 'herbaceous_wetland',
    95: 'mangroves',
    100: 'moss_lichen',
}

def fetch_worldcover(lat, lon, buffer_deg=0.025):
    prefix = "lc_"
    empty = {f"{prefix}{v}": np.nan for v in _WC_CLASSES.values()}
    empty["lc_diversity"] = np.nan
    try:
        catalog = _get_pc_client()
        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]

        search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
        items = list(search.item_collection())
        if not items:
            return empty

        # Prefer 2021 version
        item = sorted(items, key=lambda x: x.id, reverse=True)[0]
        data = _safe_read_raster(item.assets["map"].href, bbox)
        if data is None:
            return empty

        total = data.size
        result = {}
        classes_present = 0
        for class_val, class_name in _WC_CLASSES.items():
            frac = float((data == class_val).sum()) / total
            result[f"{prefix}{class_name}"] = frac
            if frac > 0:
                classes_present += 1

        result["lc_diversity"] = classes_present
        return result

    except Exception:
        return empty


# JRC Global Surface Water
def fetch_surface_water(lat, lon, buffer_deg=0.025):
    empty = {
        "water_occurrence_mean": np.nan,
        "water_occurrence_max": np.nan,
        "water_fraction": np.nan,
        "water_seasonality": np.nan,
    }
    try:
        catalog = _get_pc_client()
        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]

        search = catalog.search(collections=["jrc-gsw"], bbox=bbox)
        items = list(search.item_collection())
        if not items:
            return empty

        item = items[0]

        occ = _safe_read_raster(item.assets["occurrence"].href, bbox)
        if occ is not None:
            water_pixels = occ[occ > 0]
            empty["water_occurrence_mean"] = (
                float(np.mean(water_pixels)) if water_pixels.size > 0 else 0.0
            )
            empty["water_occurrence_max"] = (
                float(np.max(water_pixels)) if water_pixels.size > 0 else 0.0
            )
            empty["water_fraction"] = float((occ > 0).sum()) / max(occ.size, 1)

        sea = _safe_read_raster(item.assets["seasonality"].href, bbox)
        if sea is not None:
            water_seasonal = sea[sea > 0]
            empty["water_seasonality"] = (
                float(np.mean(water_seasonal)) if water_seasonal.size > 0 else 0.0
            )

        return empty

    except Exception:
        return empty


# Geology / Lithology  (Macrostrat REST API)
_LITH_CATEGORIES = {
    'sedimentary': 0,
    'sedimentary rocks': 0,
    'shale': 0,
    'sandstone': 0,
    'mudstone': 0,
    'siltstone': 0,
    'limestone': 1,
    'carbonate': 1,
    'dolomite': 1,
    'chalk': 1,
    'metamorphic': 2,
    'metamorphic rocks': 2,
    'gneiss': 2,
    'schist': 2,
    'quartzite': 2,
    'marble': 2,
    'igneous': 3,
    'igneous rocks': 3,
    'volcanic': 3,
    'volcanic rocks': 3,
    'granite': 3,
    'basalt': 3,
    'dolerite': 3,
    'gabbro': 3,
    'rhyolite': 3,
    'unconsolidated': 4,
    'alluvium': 4,
    'sand': 4,
    'gravel': 4,
    'clay': 4,
    'till': 4,
}


def fetch_geology(lat, lon):
    empty = {
        "geo_lith_category": np.nan,
        "geo_rock_age_ma": np.nan,
        "geo_is_karst": np.nan,
    }
    try:
        url = "https://macrostrat.org/api/geologic_units/map"
        params = {"lat": lat, "lng": lon, "response": "long"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        records = data.get("success", {}).get("data", [])
        if not records:
            return empty

        best = records[0]
        for rec in records:
            if rec.get("lith") and rec["lith"].strip():
                best = rec
                break

        lith_raw = (best.get("lith") or "").strip().lower()
        age = best.get("b_age", np.nan)  # base age in Ma

        # Categorise lithology
        cat = _LITH_CATEGORIES.get(lith_raw, -1)
        is_karst = 1 if cat == 1 else 0  # carbonate rocks

        return {
            "geo_lith_category": cat,
            "geo_rock_age_ma": float(age) if age is not None else np.nan,
            "geo_is_karst": is_karst,
        }

    except Exception:
        return empty


# Population Density 
def fetch_population_density(lat, lon, buffer_deg=0.05):
    empty = {
        "pop_density_proxy": np.nan,
        "pop_built_up_5km": np.nan,
    }
    try:
        catalog = _get_pc_client()
        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]

        try:
            search = catalog.search(collections=["ghs-pop"], bbox=bbox)
            items = list(search.item_collection())
            if items:
                item = sorted(items, key=lambda x: x.id, reverse=True)[0]
                data = _safe_read_raster(
                    item.assets[list(item.assets.keys())[0]].href, bbox
                )
                if data is not None:
                    empty["pop_density_proxy"] = float(np.nanmean(data[data > 0])) if (data > 0).any() else 0.0
        except Exception:
            pass

        # Fallback
        search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
        items = list(search.item_collection())
        if items:
            item = sorted(items, key=lambda x: x.id, reverse=True)[0]
            data = _safe_read_raster(item.assets["map"].href, bbox)
            if data is not None:
                empty["pop_built_up_5km"] = float((data == 50).sum()) / max(data.size, 1)

        return empty

    except Exception:
        return empty


# Water Infrastructure 
def fetch_water_infrastructure(lat, lon, radius_m=10000):
    empty = {
        "dam_count_10km": 0,
        "reservoir_count_10km": 0,
        "has_dam_nearby": 0,
        "nearest_dam_dist_m": np.nan,
    }
    try:
        import osmnx as ox
        from shapely.geometry import Point

        # Search for dams, weirs, reservoirs
        tags = {
            'waterway': ['dam', 'weir'],
            'water': ['reservoir'],
        }
        gdf = ox.features_from_point((lat, lon), tags, dist=radius_m)

        if gdf.empty:
            return empty

        # Count dams + weirs
        dam_mask = gdf.get('waterway', pd.Series(dtype=str)).isin(['dam', 'weir'])
        res_mask = gdf.get('water', pd.Series(dtype=str)).isin(['reservoir'])

        dam_count = int(dam_mask.sum())
        res_count = int(res_mask.sum())

        result = {
            "dam_count_10km": dam_count,
            "reservoir_count_10km": res_count,
            "has_dam_nearby": 1 if (dam_count + res_count) > 0 else 0,
            "nearest_dam_dist_m": np.nan,
        }

        # Find nearest dam/weir distance
        dams_and_res = gdf[dam_mask | res_mask]
        if not dams_and_res.empty:
            sample_pt = Point(lon, lat)
            # Use centroid for polygons, representative point for lines
            dists = dams_and_res.geometry.apply(
                lambda g: sample_pt.distance(g.centroid) * 111_320  # rough deg->m
            )
            result["nearest_dam_dist_m"] = float(dists.min())

        return result

    except Exception:
        return empty


# Water Body Type Classification
def fetch_water_body_type(lat, lon, radius_m=5000):
    empty = {
        "wbt_river_count": 0,
        "wbt_stream_count": 0,
        "wbt_lake_count": 0,
        "wbt_canal_count": 0,
        "wbt_dominant_type": 0,
    }
    try:
        import osmnx as ox

        tags = {
            'waterway': ['river', 'stream', 'canal'],
            'water': ['river', 'lake', 'pond', 'basin'],
            'natural': ['water'],
        }
        gdf = ox.features_from_point((lat, lon), tags, dist=radius_m)

        if gdf.empty:
            return empty

        # Count each type
        waterway_col = gdf.get('waterway', pd.Series(dtype=str))
        water_col = gdf.get('water', pd.Series(dtype=str))

        river_count = int(
            waterway_col.eq('river').sum() + water_col.eq('river').sum()
        )
        stream_count = int(waterway_col.eq('stream').sum())
        lake_count = int(
            water_col.isin(['lake', 'pond', 'basin']).sum()
        )
        canal_count = int(waterway_col.eq('canal').sum())

        # Determine dominant type
        counts = {1: river_count, 2: stream_count, 3: lake_count, 4: canal_count}
        dominant = max(counts, key=counts.get) if any(counts.values()) else 0

        return {
            "wbt_river_count": river_count,
            "wbt_stream_count": stream_count,
            "wbt_lake_count": lake_count,
            "wbt_canal_count": canal_count,
            "wbt_dominant_type": dominant,
        }

    except Exception:
        return empty


# Main
def fetch_geo_features(row):
    lat, lon = row["Latitude"], row["Longitude"]

    worldcover = fetch_worldcover(lat, lon)
    surface_water = fetch_surface_water(lat, lon)
    geology = fetch_geology(lat, lon)
    sleep(0.05)

    pop = fetch_population_density(lat, lon)
    infra = fetch_water_infrastructure(lat, lon)
    sleep(0.05)
    wbt = fetch_water_body_type(lat, lon)
    sleep(0.05)
    return {**worldcover, **surface_water, **geology, **pop, **infra, **wbt}

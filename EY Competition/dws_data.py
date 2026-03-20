import os
import re
import json
import subprocess
import zipfile
import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


# STATION REGISTRY: 24 test-location DWS stations
# Format:  station_code -> (lat, lon, zip_url, csv_name_inside_zip)
STATION_REGISTRY = {
    "Q1H001": (-31.903056, 25.482222,
               "https://www.dws.gov.za/iwqs/wms/data/Q13/Q13_102439.zip",
               "Q13_102439.csv"),
    "Q2H002": (-31.905000, 25.430000,
               "https://www.dws.gov.za/iwqs/wms/data/Q21/Q21_102448.zip",
               "Q21_102448.csv"),
    "Q3H005": (-32.086390, 25.575560,
               "https://www.dws.gov.za/iwqs/wms/data/Q30/Q30_102450.zip",
               "Q30_102450.csv"),
    "Q6H003": (-32.605278, 25.885000,
               "https://www.dws.gov.za/iwqs/wms/data/Q60/Q60_102463.zip",
               "Q60_102463.csv"),
    "Q9H002": (-32.713889, 26.296667,
               "https://www.dws.gov.za/iwqs/wms/data/Q92/Q92_102479.zip",
               "Q92_102479.csv"),
    "Q9H029": (-32.761111, 26.629444,
               "https://www.dws.gov.za/iwqs/wms/data/Q94/Q94_102496.zip",
               "Q94_102496.csv"),
    "Q9H018": (-33.237780, 26.994720,
               "https://www.dws.gov.za/iwqs/wms/data/Q93/Q93_102487.zip",
               "Q93_102487.csv"),
    "N2H007": (-33.094444, 25.012778,
               "https://www.dws.gov.za/iwqs/wms/data/N22/N22_102392.zip",
               "N22_102392.csv"),
    "N3H002": (-33.001667, 25.161389,
               "https://www.dws.gov.za/iwqs/wms/data/N30/N30_102422.zip",
               "N30_102422.csv"),
    "P1H003": (-33.329167, 26.077500,
               "https://www.dws.gov.za/iwqs/wms/data/P10/P10_102430.zip",
               "P10_102430.csv"),
    "P4H001": (-33.506389, 26.744722,
               "https://www.dws.gov.za/iwqs/wms/data/P40/P40_102438.zip",
               "P40_102438.csv"),
    "R1H015": (-33.185361, 27.390750,
               "https://www.dws.gov.za/iwqs/wms/data/R10/R10_102504.zip",
               "R10_102504.csv"),
    "R2H027": (-32.991639, 27.640028,
               "https://www.dws.gov.za/iwqs/wms/data/R20/R20_102522.zip",
               "R20_102522.csv"),
    "R3H001": (-32.802778, 27.856389,
               "https://www.dws.gov.za/iwqs/wms/data/R30/R30_102526.zip",
               "R30_102526.csv"),
    "S3H013": (-32.173889, 27.372500,
               "https://www.dws.gov.za/iwqs/wms/data/S32/S32_187594.zip",
               "S32_187594.csv"),
    "S5H002": (-32.043333, 27.822778,
               "https://www.dws.gov.za/iwqs/wms/data/S50/S50_102553.zip",
               "S50_102553.csv"),
    "S6H001": (-32.579167, 27.366667,
               "https://www.dws.gov.za/iwqs/wms/data/S60/S60_102557.zip",
               "S60_102557.csv"),
    "S7H004": (-32.515278, 28.015556,
               "https://www.dws.gov.za/iwqs/wms/data/S70/S70_102568.zip",
               "S70_102568.csv"),
    "K8H005": (-34.096389, 24.439167,
               "https://www.dws.gov.za/iwqs/wms/data/K80/K80_102316.zip",
               "K80_102316.csv"),
    "K8H006": (-34.032500, 24.196389,
               "https://www.dws.gov.za/iwqs/wms/data/K80/K80_102317.zip",
               "K80_102317.csv"),
    "L7H006": (-33.731111, 24.618333,
               "https://www.dws.gov.za/iwqs/wms/data/L70/L70_102353.zip",
               "L70_102353.csv"),
    "L6H001": (-33.202778, 24.235556,
               "https://www.dws.gov.za/iwqs/wms/data/L60/L60_102349.zip",
               "L60_102349.csv"),
    "M1H012": (-33.771111, 25.386667,
               "https://www.dws.gov.za/iwqs/wms/data/M10/M10_102370.zip",
               "M10_102370.csv"),
    "T1H015": (-32.000556, 28.581667,
               "https://www.dws.gov.za/iwqs/wms/data/T13/T13_189160.zip",
               "T13_189160.csv"),
}

# DWS column name -> Competition target name
DWS_COL_MAP = {
    "EC_Phys_Water": "Electrical Conductance",
    "TAL_Diss_Water": "Total Alkalinity",
    "PO4_P_Diss_Water": "Dissolved Reactive Phosphorus",
}
_COL_FOR_TARGET = {v: k for k, v in DWS_COL_MAP.items()}

# UNIT CONVERSIONS
DWS_UNIT_CONVERSIONS = {
    "TAL_Diss_Water": 1.0,       
    "EC_Phys_Water": 10.0,       
    "PO4_P_Diss_Water": 1000.0, 
}

# Additional DWS columns that may be useful as auxiliary features
DWS_AUX_COLS = [
    "Ca_Diss_Water", "Cl_Diss_Water", "DMS_Tot_Water",
    "F_Diss_Water", "K_Diss_Water", "KJEL_N_Tot_Water",
    "Mg_Diss_Water",
    "Na_Diss_Water", "NH4_N_Diss_Water", "NO3_NO2_N_Diss_Water",
    "P_Tot_Water", "pH_Diss_Water", "Si_Diss_Water", "SO4_Diss_Water",
]

COORD_TOL = 0.005  # Degree tolerance for coordinate matching

# Runtime-populated full registry (test + training stations).
# Built by get_full_registry() on first call; all downstream functions use this.
_FULL_REGISTRY = None


# TRAINING-STATION DISCOVERY  (scrape DWS website -> match to training coords)
_ALL_REGIONS = list('ABCDEFGHJKLMNPQRSTUVWX')
_DISCOVERY_CACHE = "train_station_matches.json"
_DISCOVERY_TOL = 0.02  # degrees (~2 km), matching tolerance for training locs


def _scrape_region_stations():
    all_stations = {}
    for region in _ALL_REGIONS:
        url = (f"https://www.dws.gov.za/iwqs/wms/data/"
               f"{region}_reg_WMS_nobor.htm")
        print(f"      Scraping region {region}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "30", url],
                capture_output=True, text=True, timeout=45,
            )
            html = result.stdout
            if not html or len(html) < 100:
                print("(empty)")
                continue
        except Exception as e:
            print(f"(FAILED: {e})")
            continue

        all_links = re.findall(r'href="([^"]+)"', html, re.IGNORECASE)
        last_zip = None
        count = 0
        for link in all_links:
            if link.lower().endswith(".zip"):
                last_zip = link
            elif "Station=" in link and last_zip:
                m = re.search(r"Station=([A-Z0-9]+)", link, re.IGNORECASE)
                if m:
                    code = m.group(1).strip()
                    zu = last_zip
                    if not zu.startswith("http"):
                        zu = ("https://www.dws.gov.za/iwqs/wms/data/"
                              + zu.lstrip("./"))
                    csv_name = zu.split("/")[-1].replace(".zip", ".csv")
                    all_stations[code] = (zu, csv_name)
                    count += 1
                    last_zip = None
        print(f"({count} stations)")
    return all_stations


def _download_and_extract_coords(station, zip_url, download_dir):
    zip_path = os.path.join(download_dir, f"{station}.zip")
    try:
        subprocess.run(
            ["curl", "-sL", "--max-time", "20", "-o", zip_path, zip_url],
            capture_output=True, timeout=30,
        )
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100:
            return None, None

        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_files = [f for f in zf.namelist()
                         if "data_description" in f.lower()
                         or "macro" in f.lower()
                         or f.endswith(".txt")]
            for tf in txt_files:
                content = zf.open(tf).read().decode("utf-8", errors="replace")
                lat_m = re.search(
                    r"latitude\s+(-?\d+\.?\d*)", content, re.IGNORECASE)
                lon_m = re.search(
                    r"longitude\s+(-?\d+\.?\d*)", content, re.IGNORECASE)
                if lat_m and lon_m:
                    lat, lon = float(lat_m.group(1)), float(lon_m.group(1))
                    if -36 < lat < -20 and 16 < lon < 34:  # SA sanity check
                        return lat, lon
    except Exception:
        pass
    return None, None


def discover_training_stations(train_csv_path, dws_dir, cache_path=None):
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(train_csv_path)
                                  or ".", _DISCOVERY_CACHE)

    if os.path.exists(cache_path):
        print(f"   Loading training-station cache '{cache_path}'...")
        with open(cache_path) as f:
            matches = json.load(f)
        registry = {}
        for m in matches:
            stn = m["station"]
            if stn not in STATION_REGISTRY:
                registry[stn] = (m["lat"], m["lon"],
                                 m["zip_url"], m["csv_name"])
        print(f"   {len(registry)} training-location stations from cache")
        return registry

    print("   No training-station cache - running full DWS discovery...")

    # Load unique training coordinates
    train_df = pd.read_csv(train_csv_path)
    train_coords = set()
    for _, row in train_df.iterrows():
        train_coords.add((round(float(row["Latitude"]), 6),
                          round(float(row["Longitude"]), 6)))
    train_coords = sorted(train_coords)
    print(f"   {len(train_coords)} unique training locations")

    # Scrape all region pages
    all_stations = _scrape_region_stations()
    new_stations = {k: v for k, v in all_stations.items()
                    if k not in STATION_REGISTRY}
    print(f"   {len(all_stations)} total DWS stations found, "
          f"{len(new_stations)} not already in test registry")

    # Download ZIPs in parallel to extract coordinates
    discovery_dir = os.path.join(dws_dir, "_discovery")
    os.makedirs(discovery_dir, exist_ok=True)
    station_info = {}  # station: (lat, lon, zip_url, csv_name)

    print(f"   Downloading {len(new_stations)} ZIPs (8 workers)...")

    def _dl_one(item):
        stn, (zu, cn) = item
        lat, lon = _download_and_extract_coords(stn, zu, discovery_dir)
        return stn, lat, lon, zu, cn

    done = 0
    with ThreadPoolExecutor(max_workers=8) as exe:
        futs = {exe.submit(_dl_one, item): item[0]
                for item in new_stations.items()}
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0 or done == len(new_stations):
                print(f"      [{done}/{len(new_stations)}]")
            try:
                stn, lat, lon, zu, cn = fut.result()
                if lat is not None:
                    station_info[stn] = (lat, lon, zu, cn)
            except Exception:
                pass

    print(f"   Extracted coordinates for {len(station_info)} stations")

    # Match to training coordinates
    matches = []
    for stn, (slat, slon, zu, cn) in station_info.items():
        for tlat, tlon in train_coords:
            if (abs(slat - tlat) < _DISCOVERY_TOL
                    and abs(slon - tlon) < _DISCOVERY_TOL):
                dist = ((slat - tlat)**2 + (slon - tlon)**2)**0.5
                matches.append({
                    "station": stn, "lat": slat, "lon": slon,
                    "train_lat": tlat, "train_lon": tlon,
                    "dist": dist, "zip_url": zu, "csv_name": cn,
                })
                break  # one match per station is enough

    # Keep best match per station (closest)
    best = {}
    for m in matches:
        stn = m["station"]
        if stn not in best or m["dist"] < best[stn]["dist"]:
            best[stn] = m
    matches = sorted(best.values(), key=lambda x: x["station"])

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(matches, f, indent=2)
    print(f"   Saved {len(matches)} matched stations to '{cache_path}'")

    # Convert to registry format
    registry = {}
    for m in matches:
        stn = m["station"]
        if stn not in STATION_REGISTRY:
            registry[stn] = (m["lat"], m["lon"], m["zip_url"], m["csv_name"])
    return registry


def get_full_registry(train_csv_path=None, dws_dir=None):
    global _FULL_REGISTRY
    if _FULL_REGISTRY is not None:
        return _FULL_REGISTRY

    full = dict(STATION_REGISTRY)  # start with the 23 test stations

    if train_csv_path is not None and dws_dir is not None:
        try:
            extra = discover_training_stations(train_csv_path, dws_dir)
            full.update(extra)
        except Exception as e:
            print(f"   Warning: Training-station discovery failed (non-fatal): {e}")

    _FULL_REGISTRY = full
    print(f"   Full registry: {len(full)} stations "
          f"({len(STATION_REGISTRY)} test + "
          f"{len(full) - len(STATION_REGISTRY)} training)")
    return full


# DOWNLOAD & EXTRACT
def fetch_dws_data(dws_dir, registry=None):
    if registry is None:
        registry = _FULL_REGISTRY or STATION_REGISTRY

    os.makedirs(dws_dir, exist_ok=True)
    downloaded, skipped = 0, 0

    for station, (lat, lon, zip_url, csv_name) in registry.items():
        station_dir = os.path.join(dws_dir, station)
        csv_path = os.path.join(station_dir, csv_name)

        if os.path.exists(csv_path):
            skipped += 1
            continue

        os.makedirs(station_dir, exist_ok=True)
        zip_path = os.path.join(station_dir, f"{station}.zip")

        print(f"   Downloading {station} from DWS ...")
        try:
            subprocess.run(
                ["curl", "-sL", "-o", zip_path, zip_url],
                timeout=60, check=True,
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(station_dir)
            downloaded += 1
        except Exception as e:
            print(f"   Warning: Failed to download {station}: {e}")

    total = len(registry)
    print(f"   DWS data: {skipped} cached, {downloaded} downloaded "
          f"({total} total stations)")


def load_all_station_data(dws_dir, registry=None):
    if registry is None:
        registry = _FULL_REGISTRY or STATION_REGISTRY
    all_dws = {}
    for station, (lat, lon, _, csv_name) in registry.items():
        csv_path = os.path.join(dws_dir, station, csv_name)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, na_values=["#n/a", "", "NA"])
        df["date"] = pd.to_datetime(df["date_time"], format="mixed", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df["station"] = station
        df["Latitude"] = lat
        df["Longitude"] = lon

        # Apply unit conversions to target columns
        for dws_col, factor in DWS_UNIT_CONVERSIONS.items():
            if dws_col in df.columns and factor != 1.0:
                df[dws_col] = pd.to_numeric(df[dws_col], errors="coerce") * factor

        all_dws[station] = df
    return all_dws


# COORDINATE -> STATION LOOKUP
def coord_to_station(lat, lon, registry=None):
    if registry is None:
        registry = _FULL_REGISTRY or STATION_REGISTRY
    for station, (slat, slon, _, _) in registry.items():
        if abs(lat - slat) < COORD_TOL and abs(lon - slon) < COORD_TOL:
            return station
    return None


# STATION-LEVEL FEATURES
def build_station_features(all_dws, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = pd.Timestamp("2015-12-31")

    print(f"   Building station features (data up to "
          f"{cutoff_date.strftime('%Y-%m-%d')}) ...")

    target_dws_cols = list(_COL_FOR_TARGET.values())
    station_feats = {}

    for station, df in all_dws.items():
        hist = df[df["date"] <= cutoff_date].copy()
        if len(hist) < 3:
            hist = df.copy()  # last resort - use everything

        feats = {}

        for dws_col in target_dws_cols:
            tgt = DWS_COL_MAP[dws_col]
            pfx = tgt[:3].upper()  # TAL / ELE / DIS

            vals = hist.loc[hist[dws_col].notna(), dws_col].values
            dates = hist.loc[hist[dws_col].notna(), "date"].values

            if len(vals) < 3:
                feats[f"dws_{pfx}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
                feats[f"dws_{pfx}_std"] = np.nan
                continue

            feats[f"dws_{pfx}_mean"] = np.mean(vals)
            feats[f"dws_{pfx}_median"] = np.median(vals)
            feats[f"dws_{pfx}_std"] = np.std(vals)
            feats[f"dws_{pfx}_iqr"] = np.percentile(vals, 75) - np.percentile(vals, 25)
            feats[f"dws_{pfx}_cv"] = np.std(vals) / (np.mean(vals) + 1e-8)
            feats[f"dws_{pfx}_p10"] = np.percentile(vals, 10)
            feats[f"dws_{pfx}_p90"] = np.percentile(vals, 90)
            feats[f"dws_{pfx}_count"] = len(vals)

            ms = pd.Series(vals, index=pd.DatetimeIndex(dates))
            monthly = ms.groupby(ms.index.month).mean()
            monthly_arr = np.array([monthly.get(m, np.mean(vals)) for m in range(1, 13)])
            mc = monthly_arr - monthly_arr.mean()
            t = np.arange(12) * 2 * np.pi / 12
            cos_amp = np.sum(mc * np.cos(t)) / 6
            sin_amp = np.sum(mc * np.sin(t)) / 6
            feats[f"dws_{pfx}_season_amp"] = np.sqrt(cos_amp**2 + sin_amp**2)
            feats[f"dws_{pfx}_season_phase"] = np.arctan2(sin_amp, cos_amp)

            if len(vals) >= 10:
                ts = (pd.DatetimeIndex(dates) - pd.Timestamp("2000-01-01")
                      ).total_seconds() / (365.25 * 86400)
                try:
                    slope, _ = np.polyfit(ts.astype(float), vals, 1)
                    feats[f"dws_{pfx}_trend"] = slope
                except Exception:
                    feats[f"dws_{pfx}_trend"] = 0.0
            else:
                feats[f"dws_{pfx}_trend"] = 0.0

            for wy, label in [(2, "2y"), (5, "5y")]:
                wstart = cutoff_date - pd.DateOffset(years=wy)
                wmask = ((hist["date"] >= wstart)
                         & (hist["date"] <= cutoff_date)
                         & hist[dws_col].notna())
                wvals = hist.loc[wmask, dws_col].values
                if len(wvals) >= 2:
                    feats[f"dws_{pfx}_recent_{label}_mean"] = np.mean(wvals)
                    feats[f"dws_{pfx}_recent_{label}_std"] = np.std(wvals)
                else:
                    feats[f"dws_{pfx}_recent_{label}_mean"] = feats[f"dws_{pfx}_mean"]
                    feats[f"dws_{pfx}_recent_{label}_std"] = feats.get(f"dws_{pfx}_std", np.nan)

        for ca, cb in [("EC_Phys_Water", "TAL_Diss_Water"),
                       ("EC_Phys_Water", "PO4_P_Diss_Water"),
                       ("TAL_Diss_Water", "PO4_P_Diss_Water")]:
            pair = hist[[ca, cb]].dropna()
            corr = pair[ca].corr(pair[cb]) if len(pair) >= 10 else 0.0
            feats[f"dws_corr_{ca[:3]}_{cb[:3]}"] = corr

        station_feats[station] = feats

    feat_df = pd.DataFrame(station_feats).T
    feat_df.index.name = "station"
    feat_df = feat_df.reset_index()
    print(f"   {feat_df.shape[1]-1} features for {len(feat_df)} stations")
    return feat_df


def add_lag_features(df, all_dws, targets):
    print("   Computing DWS lag features...")

    if "date" not in df.columns or df["date"].dtype == object:
        if "Sample Date" in df.columns:
            df["date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)

    new_cols = {}
    for target in targets:
        dws_col = _COL_FOR_TARGET.get(target)
        if dws_col is None:
            continue
        pfx = target[:3].upper()

        lag_val = np.full(len(df), np.nan)
        lag_day = np.full(len(df), np.nan)
        roll_3m = np.full(len(df), np.nan)
        roll_6m = np.full(len(df), np.nan)
        roll_12m = np.full(len(df), np.nan)

        for i, row in df.iterrows():
            stn = row.get("_dws_station", None)
            if stn is None:
                stn = coord_to_station(row["Latitude"], row["Longitude"])
            if stn is None or stn not in all_dws:
                continue

            sdf = all_dws[stn]
            cur = row["date"]
            before = sdf[(sdf["date"] < cur) & sdf[dws_col].notna()].sort_values("date")
            if len(before) == 0:
                continue

            # Most recent prior measurement
            last = before.iloc[-1]
            pos = df.index.get_loc(i)
            lag_val[pos] = last[dws_col]
            lag_day[pos] = (cur - last["date"]).days

            # Rolling windows
            for wdays, arr in [(90, roll_3m), (180, roll_6m), (365, roll_12m)]:
                ws = cur - pd.Timedelta(days=wdays)
                wdata = before.loc[before["date"] >= ws, dws_col]
                if len(wdata) > 0:
                    arr[pos] = wdata.mean()

        new_cols[f"dws_{pfx}_lag_val"] = lag_val
        new_cols[f"dws_{pfx}_lag_days"] = lag_day
        new_cols[f"dws_{pfx}_roll3m"] = roll_3m
        new_cols[f"dws_{pfx}_roll6m"] = roll_6m
        new_cols[f"dws_{pfx}_roll12m"] = roll_12m

    for col, vals in new_cols.items():
        df[col] = vals
    print(f"   Added {len(new_cols)} lag/rolling features")
    return df


# AUGMENT TRAINING DATA WITH DWS ROWS
def build_augmented_rows(all_dws, test_df, date_range=("2000-01-01", "2015-12-31"),
                         train_df=None):
    print("   Building augmented DWS training rows ...")

    # Collect test dates per station for exclusion
    test_dates_by_stn = {}
    for _, row in test_df.iterrows():
        stn = coord_to_station(row["Latitude"], row["Longitude"])
        if stn:
            test_dates_by_stn.setdefault(stn, set()).add(
                pd.Timestamp(row["date"]).date() if isinstance(row["date"], pd.Timestamp)
                else pd.Timestamp(row.get("Sample Date", row["date"]),
                                  dayfirst=True).date()
            )

    # Also collect train dates per station to avoid duplicates
    train_dates_by_stn = {}
    if train_df is not None:
        if "date" not in train_df.columns or train_df["date"].dtype == object:
            if "Sample Date" in train_df.columns:
                train_df = train_df.copy()
                train_df["date"] = pd.to_datetime(
                    train_df["Sample Date"], dayfirst=True)
        for _, row in train_df.iterrows():
            stn = coord_to_station(row["Latitude"], row["Longitude"])
            if stn:
                dt = row.get("date", None)
                if dt is not None and pd.notna(dt):
                    train_dates_by_stn.setdefault(stn, set()).add(
                        pd.Timestamp(dt).date()
                    )

    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    rows = []

    for station, sdf in all_dws.items():
        registry = _FULL_REGISTRY or STATION_REGISTRY
        lat, lon, _, _ = registry[station]
        excluded = test_dates_by_stn.get(station, set())
        # Also exclude training dates at this station to avoid duplicates
        excluded = excluded | train_dates_by_stn.get(station, set())

        # Filter to date range, exclude test dates
        mask = ((sdf["date"] >= start)
                & (sdf["date"] <= end)
                & (~sdf["date"].dt.date.isin(excluded)))
        available = sdf[mask]

        for _, drow in available.iterrows():
            rd = {
                "Latitude": lat,
                "Longitude": lon,
                "Sample Date": drow["date"].strftime("%d-%m-%Y"),
                "date": drow["date"],
                "_dws_station": station,
            }
            # Target columns (already unit-converted in load_all_station_data)
            for dws_col, target in DWS_COL_MAP.items():
                v = drow.get(dws_col, np.nan)
                rd[target] = v if pd.notna(v) else np.nan
            # Same-day auxiliary columns as features
            for aux_col in DWS_AUX_COLS:
                v = drow.get(aux_col, np.nan)
                rd[f"dws_aux_{aux_col}"] = float(v) if pd.notna(v) else np.nan
            rows.append(rd)

    aug_df = pd.DataFrame(rows)
    # Drop rows where all three targets are NaN
    from config import TARGETS
    tgt_mask = aug_df[TARGETS].notna().any(axis=1)
    aug_df = aug_df[tgt_mask].reset_index(drop=True)

    # Count aux column coverage
    aux_cols = [c for c in aug_df.columns if c.startswith("dws_aux_")]
    if aux_cols:
        cov = aug_df[aux_cols].notna().mean()
        print(f"   Auxiliary column coverage in augmented rows:")
        for col in sorted(aux_cols):
            print(f"     {col}: {cov[col]*100:.0f}%")

    print(f"   {len(aug_df)} augmented rows from {len(all_dws)} stations "
          f"(test dates excluded)")
    return aug_df

# SAME-DAY AUXILIARY FEATURES FOR TEST ROWS
def add_sameday_aux_features(df, all_dws):
    print("   Adding same-day auxiliary DWS features ...")

    if "date" not in df.columns or df["date"].dtype == object:
        if "Sample Date" in df.columns:
            df["date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)

    if "_dws_station" not in df.columns:
        df["_dws_station"] = df.apply(
            lambda r: coord_to_station(r["Latitude"], r["Longitude"]),
            axis=1,
        )

    aux_col_names = [f"dws_aux_{c}" for c in DWS_AUX_COLS]
    lookup_rows = []
    for station, sdf in all_dws.items():
        sub = sdf[["date"] + [c for c in DWS_AUX_COLS if c in sdf.columns]].copy()
        sub["_dws_station"] = station
        sub["_merge_date"] = sub["date"].dt.date
        lookup_rows.append(sub)

    if not lookup_rows:
        print("   No DWS data available for aux features.")
        for col_name in aux_col_names:
            if col_name not in df.columns:
                df[col_name] = np.nan
        return df

    lookup = pd.concat(lookup_rows, ignore_index=True)
    lookup = lookup.drop_duplicates(subset=["_dws_station", "_merge_date"],
                                    keep="first")
    rename_map = {}
    for c in DWS_AUX_COLS:
        if c in lookup.columns:
            rename_map[c] = f"dws_aux_{c}"
    lookup = lookup.rename(columns=rename_map)
    keep_cols = ["_dws_station", "_merge_date"] + \
                [f"dws_aux_{c}" for c in DWS_AUX_COLS if c in rename_map.values() or f"dws_aux_{c}" in lookup.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    lookup = lookup[[c for c in keep_cols if c in lookup.columns]]

    df["_merge_date"] = df["date"].dt.date

    # If df already has some dws_aux_* columns (e.g. from augmented rows),
    # we only fill NaN values from the lookup
    already_has_aux = any(c in df.columns for c in aux_col_names)

    if already_has_aux:
        # Merge into temporary columns, then fill gaps
        lookup_renamed = lookup.rename(
            columns={c: c + "_lkp" for c in lookup.columns
                     if c.startswith("dws_aux_")})
        df = df.merge(lookup_renamed, on=["_dws_station", "_merge_date"],
                      how="left")
        for c in DWS_AUX_COLS:
            src = f"dws_aux_{c}"
            lkp = f"dws_aux_{c}_lkp"
            if lkp in df.columns:
                if src not in df.columns:
                    df[src] = np.nan
                df[src] = df[src].fillna(df[lkp])
                df.drop(columns=[lkp], inplace=True)
    else:
        for col_name in aux_col_names:
            if col_name not in df.columns:
                df[col_name] = np.nan
        df = df.merge(lookup, on=["_dws_station", "_merge_date"],
                      how="left", suffixes=("", "_lkp"))
        # Use lookup values where original is NaN
        for c in aux_col_names:
            lkp = c + "_lkp"
            if lkp in df.columns:
                df[c] = df[c].fillna(df[lkp])
                df.drop(columns=[lkp], inplace=True)

    df.drop(columns=["_merge_date"], inplace=True, errors="ignore")

    #    Rows at DWS stations that didn't get an exact date match can
    #    still benefit from the nearest DWS measurement within a window of 14 days.
    has_station = df["_dws_station"].notna()
    has_any_aux = df[aux_col_names].notna().any(axis=1) if all(
        c in df.columns for c in aux_col_names) else pd.Series(False, index=df.index)
    needs_fill = has_station & ~has_any_aux
    n_needs = needs_fill.sum()

    if n_needs > 0:
        print(f"   Nearest-date fallback for {n_needs} rows at DWS stations...")
        filled = 0
        for idx in df[needs_fill].index:
            stn = df.at[idx, "_dws_station"]
            cur = df.at[idx, "date"]
            if pd.isna(stn) or pd.isna(cur) or stn not in all_dws:
                continue
            sdf = all_dws[stn]
            # Find nearest measurement within a delta of 7 days
            dt_diff = (sdf["date"] - cur).abs()
            within = sdf[dt_diff <= pd.Timedelta(days=7)]
            if len(within) == 0:
                # Widen to 30 days
                within = sdf[dt_diff <= pd.Timedelta(days=30)]
            if len(within) == 0:
                continue
            # Pick the closest date
            nearest_idx = (within["date"] - cur).abs().idxmin()
            nearest_row = within.loc[nearest_idx]
            any_filled = False
            for aux_col in DWS_AUX_COLS:
                col_name = f"dws_aux_{aux_col}"
                if col_name in df.columns and pd.isna(df.at[idx, col_name]):
                    v = nearest_row.get(aux_col, np.nan)
                    if pd.notna(v):
                        df.at[idx, col_name] = float(v)
                        any_filled = True
            if any_filled:
                filled += 1
        print(f"   Nearest-date filled: {filled}/{n_needs} rows")

    #    Rows NOT at DWS stations can still benefit from the nearest
    #    DWS station's chemistry data, if the station is within ~0.5°
    #    (~50 km).  Uses nearest-date within a [-30,+30] days window at that station.
    has_any_aux2 = df[aux_col_names].notna().any(axis=1) if all(
        c in df.columns for c in aux_col_names) else pd.Series(False, index=df.index)
    no_station = df["_dws_station"].isna()
    needs_station_fill = no_station & ~has_any_aux2
    n_needs_stn = needs_station_fill.sum()
    NEAREST_STATION_TOL = 0.5  # degrees (~50 km)

    if n_needs_stn > 0:
        print(f"   Nearest-station aux fallback for {n_needs_stn} non-DWS rows...")
        _reg = _FULL_REGISTRY or STATION_REGISTRY
        stn_coords = {s: (lat, lon) for s, (lat, lon, _, _) in _reg.items()}
        filled_stn = 0
        for idx in df[needs_station_fill].index:
            lat, lon = df.at[idx, "Latitude"], df.at[idx, "Longitude"]
            cur = df.at[idx, "date"]
            if pd.isna(cur):
                continue
            # Find nearest station within tolerance
            dists = {s: np.sqrt((lat - sl)**2 + (lon - sn)**2)
                     for s, (sl, sn) in stn_coords.items()}
            nearest_stn = min(dists, key=dists.get)
            if dists[nearest_stn] > NEAREST_STATION_TOL:
                continue
            if nearest_stn not in all_dws:
                continue
            sdf = all_dws[nearest_stn]
            dt_diff = (sdf["date"] - cur).abs()
            within = sdf[dt_diff <= pd.Timedelta(days=30)]
            if len(within) == 0:
                continue
            nearest_didx = (within["date"] - cur).abs().idxmin()
            nearest_row = within.loc[nearest_didx]
            any_filled = False
            for aux_col in DWS_AUX_COLS:
                col_name = f"dws_aux_{aux_col}"
                if col_name in df.columns and pd.isna(df.at[idx, col_name]):
                    v = nearest_row.get(aux_col, np.nan)
                    if pd.notna(v):
                        df.at[idx, col_name] = float(v)
                        any_filled = True
            if any_filled:
                filled_stn += 1
        print(f"   Nearest-station filled: {filled_stn}/{n_needs_stn} rows")

    has_station = df["_dws_station"].notna()
    has_any_aux = df[aux_col_names].notna().any(axis=1) if all(
        c in df.columns for c in aux_col_names) else pd.Series(False, index=df.index)
    matched = has_any_aux.sum()
    print(f"   Same-day aux features: {matched}/{len(df)} rows have data "
          f"({matched/len(df)*100:.0f}%) "
          f"[{has_station.sum()} at DWS stations]")
    return df


# MERGE STATION FEATURES INTO DATAFRAME
def merge_station_features(df, station_features):
    if "_dws_station" not in df.columns:
        df["_dws_station"] = df.apply(
            lambda r: coord_to_station(r["Latitude"], r["Longitude"]),
            axis=1,
        )

    df = df.merge(
        station_features, left_on="_dws_station", right_on="station",
        how="left", suffixes=("", "_dws_dup"),
    )
    df.drop(columns=["station"], errors="ignore", inplace=True)

    # IDW-fill for rows without a station match
    feat_cols = [c for c in station_features.columns if c != "station"]
    missing = df["_dws_station"].isna()
    n_miss = missing.sum()

    if n_miss > 0:
        _reg = _FULL_REGISTRY or STATION_REGISTRY
        stn_coords = {s: (lat, lon) for s, (lat, lon, _, _) in _reg.items()}
        for idx in df[missing].index:
            lat, lon = df.at[idx, "Latitude"], df.at[idx, "Longitude"]
            dists = {s: np.sqrt((lat - sl)**2 + (lon - sn)**2)
                     for s, (sl, sn) in stn_coords.items()}
            nearest = sorted(dists, key=dists.get)[:3]
            weights = [1.0 / (dists[s] + 0.001) for s in nearest]
            tw = sum(weights)
            weights = [w / tw for w in weights]

            for col in feat_cols:
                vals, ws = [], []
                for s, w in zip(nearest, weights):
                    rm = station_features[station_features["station"] == s]
                    if len(rm) > 0:
                        v = rm.iloc[0][col]
                        if pd.notna(v):
                            vals.append(v)
                            ws.append(w)
                if vals:
                    df.at[idx, col] = np.average(vals, weights=ws)

    print(f"   Station features merged ({n_miss} rows IDW-filled)")
    return df

def prepare_dws_augmentation(dws_dir, targets, test_df, train_df=None,
                             train_csv_path=None):
    print("\n" + "=" * 60)
    print("   DWS EXTERNAL DATA INTEGRATION")
    print("=" * 60)

    registry = get_full_registry(train_csv_path=train_csv_path,
                                 dws_dir=dws_dir)
    fetch_dws_data(dws_dir, registry=registry)
    all_dws = load_all_station_data(dws_dir, registry=registry)
    print(f"   Loaded {len(all_dws)} stations, "
          f"{sum(len(d) for d in all_dws.values())} total rows")

    station_features = build_station_features(all_dws)
    aug_rows = build_augmented_rows(all_dws, test_df, train_df=train_df)

    return all_dws, station_features, aug_rows


def _get_neighbor_stations(station, registry, max_neighbors=5, max_dist_deg=2.0):
    if station not in registry:
        return []
    primary = station[0]  # first letter = primary drainage region
    slat, slon = registry[station][0], registry[station][1]
    candidates = []
    for s, (lat, lon, _, _) in registry.items():
        if s == station or s[0] != primary:
            continue
        dist = np.sqrt((lat - slat)**2 + (lon - slon)**2)
        if dist <= max_dist_deg:
            candidates.append((s, dist))
    candidates.sort(key=lambda x: x[1])
    return candidates[:max_neighbors]


def build_neighbor_features(df, all_dws, registry=None):
    if registry is None:
        registry = _FULL_REGISTRY or STATION_REGISTRY

    print("   Building neighbor/upstream features ...")

    if "date" not in df.columns or df["date"].dtype == object:
        if "Sample Date" in df.columns:
            df["date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)

    if "_dws_station" not in df.columns:
        df["_dws_station"] = df.apply(
            lambda r: coord_to_station(r["Latitude"], r["Longitude"]),
            axis=1,
        )

    target_dws_cols = list(_COL_FOR_TARGET.values())  # DWS column names
    target_names = list(DWS_COL_MAP.values())          # competition names
    prefixes = [t[:3].upper() for t in target_names]   # TAL, ELE, DIS

    # Pre-compute neighbor lists
    neighbor_cache = {}
    for stn in registry:
        neighbor_cache[stn] = _get_neighbor_stations(stn, registry)

    # Output arrays
    feat_names = []
    for pfx in prefixes:
        for agg in ["wmean", "wmin", "wmax"]:
            feat_names.append(f"nbr_{pfx}_{agg}")
    feat_names += ["nbr_n_stations", "nbr_mean_dist"]
    out = {fn: np.full(len(df), np.nan) for fn in feat_names}

    for i, (idx, row) in enumerate(df.iterrows()):
        stn = row.get("_dws_station", None)
        if stn is None or stn not in neighbor_cache:
            continue
        neighbors = neighbor_cache[stn]
        if not neighbors:
            continue

        cur_date = row["date"]
        if pd.isna(cur_date):
            continue

        # Gather target values from neighbors
        nbr_vals = {pfx: [] for pfx in prefixes}
        nbr_weights = []
        nbr_dists = []

        for nbr_stn, dist in neighbors:
            if nbr_stn not in all_dws:
                continue
            ndf = all_dws[nbr_stn]
            # Most recent measurement before current date
            before = ndf[ndf["date"] < cur_date].sort_values("date")
            if len(before) == 0:
                continue
            last = before.iloc[-1]
            w = 1.0 / (dist + 0.01)  # IDW weight

            has_any = False
            for dws_col, pfx in zip(target_dws_cols, prefixes):
                v = last.get(dws_col, np.nan)
                if pd.notna(v):
                    nbr_vals[pfx].append((float(v), w))
                    has_any = True
            if has_any:
                nbr_weights.append(w)
                nbr_dists.append(dist)

        pos = df.index.get_loc(idx)
        if nbr_dists:
            out["nbr_n_stations"][pos] = len(nbr_dists)
            out["nbr_mean_dist"][pos] = np.mean(nbr_dists)

        for pfx in prefixes:
            vals_w = nbr_vals[pfx]
            if vals_w:
                vs = [v for v, w in vals_w]
                ws = [w for v, w in vals_w]
                out[f"nbr_{pfx}_wmean"][pos] = np.average(vs, weights=ws)
                out[f"nbr_{pfx}_wmin"][pos] = min(vs)
                out[f"nbr_{pfx}_wmax"][pos] = max(vs)

    for fn, arr in out.items():
        df[fn] = arr
    n_filled = df["nbr_n_stations"].notna().sum()
    print(f"   Neighbor features: {len(feat_names)} features, "
          f"{n_filled}/{len(df)} rows have data")
    return df

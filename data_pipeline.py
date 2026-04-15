"""
Pulls ACS Census data, merges with Zillow ZHVI + ZHVF, builds features and outputs train/test CSVs for the real estate prediction model.

Outputs:
  data/output/features.csv
  data/output/train.csv
  data/output/test.csv
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────

CENSUS_API_KEY = "1cd18804a068d42c5b3d5ccecdd13c508c70f197"

DATASET_DIR = "/Users/tanaymihani/PycharmProjects/PythonProject_CS5130_Real_Estate/dataset"

ZHVI_PATH = os.path.join(DATASET_DIR, "current",
                         "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
ZHVF_PATH = os.path.join(DATASET_DIR, "future",
                         "Zip_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

OUTPUT_DIR = "data/output"

CENSUS_YEARS = [2019, 2020, 2021, 2022, 2023]
LABEL_FORECAST_COL = "2027-02-28"   # 1yr ahead column in ZHVF
TOP_QUANTILE = 0.75
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ── Census variable mappings ──────────────────────────────

# grouped by DP table so we can hit each endpoint separately
VARIABLES = {
    "DP05_0001E": "total_population",
    "DP05_0018E": "median_age",
    "DP05_0024E": "pop_65_plus",
    "DP05_0019E": "pop_18_to_34",
    "DP03_0062E": "median_household_income",
    "DP03_0003E": "employment_rate",
    "DP03_0096E": "health_insurance_pct",
    "DP04_0001E": "total_housing_units",
    "DP04_0003E": "vacancy_rate",
    "DP04_0046E": "owner_occupied_pct",
    "DP04_0134E": "median_gross_rent",
    "DP04_0089E": "median_home_value",
    "DP02_0064E": "bachelors_degree_pct",
    "DP02_0066E": "graduate_degree_pct",
    "DP02_0001E": "total_households",
}

TABLES = {
    "DP05": {k: v for k, v in VARIABLES.items() if k.startswith("DP05")},
    "DP03": {k: v for k, v in VARIABLES.items() if k.startswith("DP03")},
    "DP04": {k: v for k, v in VARIABLES.items() if k.startswith("DP04")},
    "DP02": {k: v for k, v in VARIABLES.items() if k.startswith("DP02")},
}

CENSUS_NUMERIC = [
    "total_population", "median_age", "pop_65_plus", "pop_18_to_34",
    "median_household_income", "employment_rate", "health_insurance_pct",
    "total_housing_units", "vacancy_rate", "owner_occupied_pct",
    "median_gross_rent", "median_home_value",
    "bachelors_degree_pct", "graduate_degree_pct", "total_households",
]

BASE_URL = "https://api.census.gov/data"


def section(title: str) -> None:
    return


# Fetch ACS data

def fetch_table(table, year, vars_dict):
    var_str = ",".join(vars_dict.keys())
    url = (f"{BASE_URL}/{year}/acs/acs5/profile"
           f"?get={var_str}"
           f"&for=zip%20code%20tabulation%20area:*"
           f"&key={CENSUS_API_KEY}")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={"zip code tabulation area": "zip", **vars_dict})
    df["year"] = year
    return df


def pull_census():
    all_dfs = []
    for yr in CENSUS_YEARS:
        yr_dfs = []
        for tbl, vars_dict in TABLES.items():
            df = fetch_table(tbl, yr, vars_dict)
            if df is not None:
                keep = ["zip", "year"] + list(vars_dict.values())
                yr_dfs.append(df[keep])
            time.sleep(0.5)
        if yr_dfs:
            merged = yr_dfs[0]
            for d in yr_dfs[1:]:
                merged = merged.merge(d, on=["zip", "year"], how="outer")
            all_dfs.append(merged)
    return pd.concat(all_dfs, ignore_index=True)


section("STEP 1 — Fetch ACS Census data (takes ~2 min)")
census_raw = pull_census()


# Clean census

section("STEP 2 — Clean Census data")

def clean_census(df):
    df = df.copy()
    df["zip"] = df["zip"].astype(str).str.zfill(5)

    # census API returns strings for everything
    for c in CENSUS_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


    for c in CENSUS_NUMERIC:
        if c in df.columns:
            df[c] = df[c].where(df[c] >= 0, np.nan)

    # convert raw counts into proportions
    pop = df["total_population"].replace(0, np.nan)
    hu = df["total_housing_units"].replace(0, np.nan)
    hh = df["total_households"].replace(0, np.nan)

    if "pop_65_plus" in df.columns:
        df["pct_65_plus"] = df["pop_65_plus"] / pop
    if "pop_18_to_34" in df.columns:
        df["pct_18_to_34"] = df["pop_18_to_34"] / pop
    if "vacancy_rate" in df.columns:
        df["vacancy_rate"] = df["vacancy_rate"] / hu
    if "owner_occupied_pct" in df.columns:
        df["owner_occupied_pct"] = df["owner_occupied_pct"] / hu
    if "bachelors_degree_pct" in df.columns:
        df["bachelors_degree_pct"] = df["bachelors_degree_pct"] / hh
    if "graduate_degree_pct" in df.columns:
        df["graduate_degree_pct"] = df["graduate_degree_pct"] / hh
    if "employment_rate" in df.columns:
        df["employment_rate"] = df["employment_rate"] / pop

    return df


census_clean = clean_census(census_raw)
latest_yr = census_clean["year"].max()
census_latest = census_clean[census_clean["year"] == latest_yr].copy()
census_latest.drop(columns=["year"], inplace=True)

# drop raw counts, keep derived proportions
drop_cols = [c for c in ["pop_65_plus", "pop_18_to_34"] if c in census_latest.columns]
census_latest.drop(columns=drop_cols, inplace=True)

# Load ZHVI

section("STEP 3 — Load and clean ZHVI")

zhvi_raw = pd.read_csv(ZHVI_PATH, low_memory=False)

meta_cols = [c for c in zhvi_raw.columns if not c.startswith("20")]
date_cols = [c for c in zhvi_raw.columns if c.startswith("20")]

zhvi_raw["RegionName"] = zhvi_raw["RegionName"].astype(str).str.zfill(5)

zhvi_long = zhvi_raw.melt(id_vars=meta_cols, value_vars=date_cols,
                          var_name="date", value_name="zhvi")
zhvi_long["date"] = pd.to_datetime(zhvi_long["date"])
zhvi_long.dropna(subset=["zhvi"], inplace=True)
zhvi_long.sort_values(["RegionName", "date"], inplace=True)
zhvi_long.reset_index(drop=True, inplace=True)


# ZHVI features

section("STEP 4 — ZHVI feature engineering")

max_date = zhvi_long["date"].max()

def price_at(n_months_ago):
    """Get last known price at least n months before max_date."""
    cutoff = max_date - pd.DateOffset(months=n_months_ago)
    return (zhvi_long[zhvi_long["date"] <= cutoff]
            .sort_values("date")
            .groupby("RegionName")["zhvi"]
            .last())

current = zhvi_long.groupby("RegionName")["zhvi"].last().rename("current_zhvi")
p3 = price_at(3)
p12 = price_at(12)
p36 = price_at(36)
p60 = price_at(60)

# returns
r3  = ((current - p3) / p3.replace(0, np.nan)).rename("return_3m")
r12 = ((current - p12) / p12.replace(0, np.nan)).rename("return_12m")
r36 = ((current - p36) / p36.replace(0, np.nan)).rename("return_36m")
r60 = ((current - p60) / p60.replace(0, np.nan)).rename("return_60m")

# acceleration = is short-term running ahead of long-term?
accel = ((r3 * 4) - r12).rename("acceleration")

# 12m volatility (std of monthly pct changes)
last12 = zhvi_long[zhvi_long["date"] > (max_date - pd.DateOffset(months=12))]
vol = (last12.sort_values("date")
       .groupby("RegionName")["zhvi"]
       .apply(lambda x: x.pct_change().std())
       .rename("volatility_12m"))

# zip vs metro median — relative value signal
snap = zhvi_long[zhvi_long["date"] == max_date].copy()
metro_med = snap.groupby("Metro")["zhvi"].median().rename("metro_median_zhvi")
snap = snap.join(metro_med, on="Metro")
snap["vs_metro"] = snap["zhvi"] / snap["metro_median_zhvi"].replace(0, np.nan)
vs_metro = snap.set_index("RegionName")["vs_metro"]

# grab geo metadata
geo_meta = (zhvi_long[["RegionName", "State", "City", "Metro", "CountyName", "StateName"]]
            .drop_duplicates()
            .set_index("RegionName"))

zhvi_feats = pd.concat([current, r3, r12, r36, r60, accel, vol, vs_metro, geo_meta],
                       axis=1).reset_index()


# ZHVF label

section("STEP 5 — Load ZHVF forecast label")

zhvf = pd.read_csv(ZHVF_PATH, low_memory=False)
zhvf["RegionName"] = zhvf["RegionName"].astype(str).str.zfill(5)

if LABEL_FORECAST_COL not in zhvf.columns:
    avail = [c for c in zhvf.columns if c.startswith("20")]
    raise KeyError(f"'{LABEL_FORECAST_COL}' not in ZHVF. Available: {avail}")

zhvf_label = zhvf[["RegionName", LABEL_FORECAST_COL]].rename(
    columns={LABEL_FORECAST_COL: "growth_forecast_1yr"})
zhvf_label.dropna(subset=["growth_forecast_1yr"], inplace=True)


# Merge everything

section("STEP 6 — Build master panel")

# zhvi features + census (left join, not all ZIPs have census data)
panel = zhvi_feats.merge(census_latest, left_on="RegionName", right_on="zip",
                         how="left")
panel.drop(columns=["zip"], errors="ignore", inplace=True)

# add forecast label (inner join — only keep zips with valid forecast)
panel = panel.merge(zhvf_label, on="RegionName", how="inner")


#  Validate features

section("STEP 7 — Feature validation")

id_cols = ["RegionName"]
geo_cols = [c for c in ["State", "StateName", "City", "Metro", "CountyName"]
            if c in panel.columns]

skip = set(id_cols + geo_cols + ["growth_forecast_1yr"])
# leftover cols from earlier versions of the notebook
for col in ["forecast_growth", "year"]:
    if col in panel.columns:
        skip.add(col)

feature_cols = [c for c in panel.columns
                if c not in skip and pd.api.types.is_numeric_dtype(panel[c])]

# inf → NaN → median fill
panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan)
missing = panel[feature_cols].isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing):
    for c in missing.index:
        panel[c] = panel[c].fillna(panel[c].median())


# Save features

section("STEP 8 — Save features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

out_cols = id_cols + geo_cols + feature_cols + ["growth_forecast_1yr"]
features_df = panel[out_cols].copy()
feat_path = os.path.join(OUTPUT_DIR, "features.csv")
features_df.to_csv(feat_path, index=False)


# Binary label

section("STEP 9 — Binary label (top 25%)")

threshold = panel["growth_forecast_1yr"].quantile(TOP_QUANTILE)
panel["label"] = (panel["growth_forecast_1yr"] >= threshold).astype(int)


# Train/test split

section("STEP 10 — 80/20 stratified split")

ml_cols = id_cols + geo_cols + feature_cols + ["growth_forecast_1yr", "label"]
ml_panel = panel[ml_cols].copy()

train_df, test_df = train_test_split(ml_panel, test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE,
                                     stratify=ml_panel["label"])
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Save splits

section("STEP 11 — Save train/test CSVs")

train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

section("STEP 12 — Summary")

section("DONE")
# Real Estate Investment Prediction from Demographic Trends

CS 5130 Spring 2026

Akash Setti, Tanay Mihani, Ansh Patel, Anthony Ngumah

## What this does

Predicts which US ZIP codes are likely to see high real estate growth in the next year. We pull demographic data from the Census Bureau and housing data from Zillow, merge them together, engineer a bunch of features, and then train classifiers to flag high-growth areas.

A ZIP code gets labeled as "high growth" if Zillow's 1-year forecast puts it in the top 25% nationally. We compare a simple weighted demographic index (baseline) against Logistic Regression and Random Forest to see if ML actually helps here. It does — the baseline barely beats a coin flip while Random Forest hits ~80% accuracy with a 0.87 AUC.

## Data

We use two sources:

**Zillow** — ZHVI (Zillow Home Value Index) has monthly home values for ~26k ZIP codes going back to 2000. ZHVF (Zillow Home Value Forecast) has Zillow's predicted growth percentages. Both CSVs go in `dataset/current/` and `dataset/future/` respectively. You can download them from [Zillow Research](https://www.zillow.com/research/data/).

**US Census ACS** — Pulled live through the Census API (you need an API key). We grab tables DP02, DP03, DP04, DP05 which cover demographics, economics, housing characteristics, and social data for all ZIP codes from 2019 to 2023.

## Repo structure

```
├── dataset/
│   ├── current/                <- ZHVI csv
│   └── future/                 <- ZHVF csv
├── data/
│   └── output/                 <- generated csvs (features, train, test)
├── __init__.py                 <- marks repo as a Python package
├── data_pipeline.py            <- fetches census, cleans everything, builds features, outputs train/test
├── baseline_scoring.py         <- weighted demographic index baseline
├── supervised_models.py        <- logistic regression + random forest
├── pre_process_data.ipynb      <- earlier notebook (data_pipeline.py supersedes this)
├── test_pipeline.py            <- unit tests
└── README.md
```

## Setup

Python 3.10+

```
pip install pandas numpy scikit-learn requests
```

You also need a Census API key. Get one at https://api.census.gov/data/key_signup.html and put it in `data_pipeline.py` where it says `CENSUS_API_KEY`.

The Zillow CSVs need to be downloaded manually and placed in the `dataset/` folders since they're too big for git.

## How to run

Run the pipeline first. This pulls Census data and builds the feature set + train/test split. Takes a couple minutes because of the API calls.

```
python data_pipeline.py
```

This creates three files in `data/output/`:
- `features.csv` — all 20k ZIP codes with 23 features
- `train.csv` — 80% split
- `test.csv` — 20% split

Then run the baseline:

```
python baseline_scoring.py
```

And the supervised models:

```
python supervised_models.py
```

Both scripts read from `data/output/` by default. You can pass `--train` and `--test` flags if your files are somewhere else.

## Features

We engineered 23 features total, 8 from Zillow and 15 from Census.

Zillow features capture housing market momentum: current home value, price returns over 3m/12m/36m/60m windows, an acceleration metric (short vs long term momentum), 12-month volatility, and how a ZIP compares to its metro area median.

Census features cover demographics and economics: population, median age, income, employment rate, education levels, housing vacancy, owner-occupancy rates, rent, home values, etc. Raw counts are converted to proportions where it makes sense.

## Results

|                           | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------------|----------|-----------|--------|-------|-------|
| Baseline (weighted index) | 0.559    | 0.161     | 0.149  | 0.154 | —     |
| Logistic Regression       | 0.760    | 0.538     | 0.797  | 0.643 | 0.847 |
| Random Forest             | 0.796    | 0.597     | 0.755  | 0.667 | 0.872 |

The baseline z-score normalizes Census features and computes a weighted sum — positive weights for income, education, young population, negative for vacancy and elderly population. It then uses the 75th percentile as a cutoff. It's basically useless. Both ML models significantly outperform it, with Random Forest doing slightly better across the board.

## Notes

We moved from county-level to ZIP-level analysis partway through the project because housing trends can vary a lot within a single county. This gave us more granular predictions.

Missing values are filled with column medians. We chose this over dropping rows to keep rural/sparse ZIP codes in the dataset.

The label threshold (top 25% growth) is percentile-based so it stays stable across different dataset versions.

## References

- [Zillow Research Data](https://www.zillow.com/research/data/)
- [US Census Bureau ACS API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- scikit-learn: Pedregosa et al., JMLR 12, 2011
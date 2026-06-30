# PE Target Screening — ML Model

## Overview

This project builds a machine learning model to screen and identify potential Private Equity (PE) acquisition targets from a universe of companies. It combines structured financial/firmographic data with deal history to train a classifier that predicts the likelihood of a company becoming a PE target.

## Project Structure

```
TFG_PE_Screening/
├── data/
│   ├── raw/          # Original source files (PitchBook deals, company lists, PE acquisitions)
│   └── processed/    # Cleaned and feature-engineered datasets
├── notebooks/        # Exploratory data analysis and model development
├── models/           # Saved trained models
├── outputs/          # Charts, reports, and result exports
├── venv/             # Python virtual environment
├── requirements.txt  # Package dependencies
└── README.md
```

## Data Sources

- `pitchbook_deals_3243_completo.xlsx` — PitchBook deal-level data
- `Empresas_adquiridas_por_PE_Companies.xlsx` — Known PE acquisition targets (positive labels)
- `USCompanies.csv` — Universe of US companies (candidate pool)

## Methodology

1. **Data preparation** — merge company universe with known PE deal history to create labelled training data
2. **Feature engineering** — firmographic, financial, and industry features
3. **Model training** — classification models (XGBoost, LightGBM, Logistic Regression) with cross-validation
4. **Evaluation** — precision/recall, AUC-ROC, feature importance analysis
5. **Scoring** — rank unlabelled companies by predicted PE target probability

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Launch Jupyter to run the notebooks:

```bash
jupyter lab
```

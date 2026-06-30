"""
TFG - M&A Target Screening
Script 01: Build ML Dataset
Combines PitchBook deals with Compustat universe
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz,process
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

# PitchBook M&A deals
pb_deals = pd.read_csv('data/raw/pitchbook_ma_deals.csv')
pb_deals.columns = pb_deals.columns.str.strip()
print(f"PitchBook M&A deals: {len(pb_deals)} rows")

# Compustat universe
comp = pd.read_csv('data/raw/USCompanies.csv')
print(f"Compustat: {len(comp)} rows, {comp['gvkey'].nunique()} unique companies")

# ============================================================
# 2. CLEAN COMPANY NAMES
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning company names...")
print("=" * 60)

def clean_name(name):
    if pd.isna(name):
        return ''
    name = str(name).upper()
    suffixes = [' INC', ' CORP', ' LLC', ' LTD', ' CO ', ' GROUP',
                ' HOLDINGS', ' HOLDING', ' INTERNATIONAL', ' INTL',
                ' COMPANY', ' THE', ' PLC', ' SA', ' AG', ',', '.', "'"]
    for s in suffixes:
        name = name.replace(s, ' ')
    return ' '.join(name.split()).strip()

# PitchBook company universe (for industry sector/group)
pb_companies = pd.read_csv('data/raw/pitchbook_companies.csv')
pb_companies.columns = pb_companies.columns.str.strip()
# Handle BOM in first column if present
pb_companies.columns = [c.lstrip('\ufeff') for c in pb_companies.columns]
pb_companies['company_clean'] = pb_companies['Company'].apply(clean_name)
print(f"PitchBook companies: {len(pb_companies)} rows")

pb_deals['company_clean'] = pb_deals['Companies'].apply(clean_name)
comp['company_clean'] = comp['conm'].apply(clean_name)

# Extract deal year
pb_deals['deal_year'] = pd.to_datetime(
    pb_deals['Deal Date'],
    format='%d-%b-%Y',
    errors='coerce'
).dt.year

print(f"Deal year range: {pb_deals['deal_year'].min()} - {pb_deals['deal_year'].max()}")
print(f"Deals without date: {pb_deals['deal_year'].isna().sum()}")

# ============================================================
# 3. FUZZY MATCHING
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Fuzzy matching PitchBook -> Compustat...")
print("This may take 2-3 minutes...")
print("=" * 60)

# Unique Compustat companies
comp_unique = comp[['gvkey', 'conm', 'company_clean', 'tic']].drop_duplicates('gvkey').reset_index(drop=True)
comp_names = comp_unique['company_clean'].tolist()

matches = []
no_matches = []

for idx, row in pb_deals.iterrows():
    pb_name = row['company_clean']
    if not pb_name:
        continue

    result = process.extractOne(
        pb_name,
        comp_names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=82
    )

    if result:
        match_name, score, match_idx = result
        matches.append({
            'pb_company': row['Companies'],
            'comp_company': comp_unique.iloc[match_idx]['conm'],
            'gvkey': comp_unique.iloc[match_idx]['gvkey'],
            'tic': comp_unique.iloc[match_idx]['tic'],
            'match_score': score,
            'deal_date': row['Deal Date'],
            'deal_year': row['deal_year'],
            'deal_size': row.get('Deal Size', None),
            'deal_type2': row.get('Deal Type 2', None),
            'pb_industry_code': row.get('Primary PitchBook Industry Code', None),
            'pb_investors': row.get('Investors', None),
        })
    else:
        no_matches.append(row['Companies'])

matches_df = pd.DataFrame(matches)
print(f"Matches found: {len(matches_df)}")
print(f"No match: {len(no_matches)}")
print(f"\nSample matches:")
print(matches_df[['pb_company', 'comp_company', 'match_score', 'tic']].head(10).to_string())

# ── Fuzzy-match PitchBook deals -> PitchBook companies to get sector/group ──
print("\nFuzzy-matching deals -> PitchBook company universe for sector/group...")
pb_co_names = pb_companies['company_clean'].tolist()
sector_list, group_list = [], []

for pb_name in matches_df['pb_company'].apply(clean_name):
    res = process.extractOne(pb_name, pb_co_names, scorer=fuzz.token_sort_ratio, score_cutoff=80)
    if res:
        co_row = pb_companies.iloc[res[2]]
        sector_list.append(co_row.get('Primary PitchBook Industry Sector', None))
        group_list.append(co_row.get('Primary PitchBook Industry Group', None))
    else:
        sector_list.append(None)
        group_list.append(None)

matches_df['pb_industry_sector'] = sector_list
matches_df['pb_industry_group'] = group_list
print(f"  Sector filled: {sum(s is not None for s in sector_list)}/{len(sector_list)}")

# Save matches for review
matches_df.to_csv('data/processed/pb_compustat_matches.csv', index=False)
print("\nSaved: data/processed/pb_compustat_matches.csv")

# ============================================================
# 4. BUILD PE DATASET (label = 1)
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Building PE dataset (label=1)...")
print("=" * 60)

pe_rows = []

for _, match in matches_df.iterrows():
    gvkey = match['gvkey']
    deal_year = match['deal_year']

    if pd.isna(deal_year):
        continue

    # Get financials from year BEFORE the deal
    target_year = int(deal_year) - 1
    comp_row = comp[(comp['gvkey'] == gvkey) & (comp['fyear'] == target_year)]

    if len(comp_row) > 0:
        row_data = comp_row.iloc[0].to_dict()
        row_data['ma_target'] = 1
        row_data['pb_company'] = match['pb_company']
        row_data['deal_date'] = match['deal_date']
        row_data['deal_size'] = match['deal_size']
        row_data['deal_type2'] = match['deal_type2']
        row_data['match_score'] = match['match_score']
        row_data['pb_industry_code'] = match.get('pb_industry_code', None)
        row_data['pb_investors'] = match.get('pb_investors', None)
        row_data['pb_industry_sector'] = match.get('pb_industry_sector', None)
        row_data['pb_industry_group'] = match.get('pb_industry_group', None)
        pe_rows.append(row_data)

pe_df = pd.DataFrame(pe_rows)
print(f"PE companies with financial data: {len(pe_df)}")

# ============================================================
# 5. BUILD CONTROL DATASET (label = 0)
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Building control dataset (label=0)...")
print("=" * 60)

matched_gvkeys = set(matches_df['gvkey'].tolist())
control_df = comp[~comp['gvkey'].isin(matched_gvkeys)].copy()
control_df['ma_target'] = 0
control_df['pb_company'] = None
control_df['deal_date'] = None
control_df['deal_size'] = None
control_df['deal_type2'] = None
control_df['match_score'] = None
control_df['pb_industry_code'] = None
control_df['pb_investors'] = None
control_df['pb_industry_sector'] = None
control_df['pb_industry_group'] = None
# Use Compustat SIC code as industry fallback for control group
control_df['pb_industry_code'] = control_df['pb_industry_code'].fillna(
    control_df['sich'].astype('Int64').astype(str).replace('<NA>', None)
    if 'sich' in control_df.columns else None
)

print(f"Control companies (before sampling): {control_df['gvkey'].nunique()}")

# Sample control group to 3x the size of PE group
n_sample = len(pe_df) * 3
control_df = control_df.sample(n=n_sample, random_state=42)
print(f"Control companies (after 3x sampling): {len(control_df)}")

# ============================================================
# 6. COMBINE AND CALCULATE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Combining datasets and calculating features...")
print("=" * 60)

dataset = pd.concat([pe_df, control_df], ignore_index=True)

# Convert to numeric
numeric_cols = ['sale', 'ebitda', 'oibdp', 'ni', 'at', 'lt',
                'dltt', 'dlc', 'che', 'capx', 'dp', 'wcap',
                'rect', 'ap', 'xsga', 'csho', 'prcc_f']

for col in numeric_cols:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# Calculate derived features
dataset['total_debt'] = dataset['dltt'].fillna(0) + dataset['dlc'].fillna(0)
dataset['ebitda_margin'] = dataset['ebitda'] / dataset['sale']
dataset['net_margin'] = dataset['ni'] / dataset['sale']
dataset['leverage'] = dataset['total_debt'] / dataset['at']
dataset['capex_intensity'] = dataset['capx'] / dataset['sale']
dataset['roa'] = dataset['ni'] / dataset['at']
dataset['current_ratio'] = dataset['wcap'] / dataset['lt']
dataset['market_cap'] = dataset['prcc_f'] * dataset['csho']
dataset['ev_ebitda'] = dataset['market_cap'] / dataset['ebitda']
dataset['asset_turnover'] = dataset['sale'] / dataset['at']
dataset['cash_ratio'] = dataset['che'] / dataset['at']

# Replace infinities with NaN
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

from scipy.stats import mstats
# Winsorize variables con valores extremos al percentil 1%-99%
vars_winsorize = ['ev_ebitda', 'ebitda_margin', 'net_margin']

for col in vars_winsorize:
    dataset[col] = mstats.winsorize(
        dataset[col].fillna(dataset[col].median()),
        limits=[0.01, 0.01]
    )
    print(f"{col}: Min={dataset[col].min():.2f} | Max={dataset[col].max():.2f}")

# ============================================================
# 6.5 FILTER MINIMUM SIZE TO AVOID DISTORTED RATIOS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6.5: Filtering minimum company size...")
print("=" * 60)

before = len(dataset)
dataset = dataset[dataset['at'] >= 1]    # activos mínimo 1M$
dataset = dataset[dataset['sale'] >= 1]  # ventas mínimo 1M$
after = len(dataset)

print(f"Filas antes: {before}")
print(f"Filas después: {after}")
print(f"Filas eliminadas: {before - after}")
print(f"\nLabel distribution after filter:")
print(dataset['ma_target'].value_counts())

# ============================================================
# 7. SAVE FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Saving final dataset...")
print("=" * 60)

dataset.to_csv('data/processed/dataset_ma_ml_final.csv', index=False)

print(f"\n✓ Dataset saved: data/processed/dataset_ma_ml_final.csv")
print(f"  Total rows: {len(dataset)}")
print(f"  Total columns: {len(dataset.columns)}")
print(f"\nLabel distribution:")
print(dataset['ma_target'].value_counts())
print(f"\nBalance ratio: {dataset['ma_target'].mean():.3f}")
print(f"\nMissing values in key features:")
key_features = ['sale', 'ebitda', 'at', 'leverage',
                'ebitda_margin', 'roa', 'market_cap']
print(dataset[key_features].isna().sum())

# Column coverage for new columns
print("\n" + "=" * 60)
print("NEW COLUMN COVERAGE:")
print("=" * 60)
new_cols = ['deal_size', 'pb_industry_code', 'pb_investors',
            'pb_industry_sector', 'pb_industry_group']
total = len(dataset)
targets = dataset[dataset['ma_target'] == 1]
controls = dataset[dataset['ma_target'] == 0]
for col in new_cols:
    if col in dataset.columns:
        filled_all  = dataset[col].notna().sum()
        filled_t    = targets[col].notna().sum()  if col in targets.columns else 0
        filled_c    = controls[col].notna().sum() if col in controls.columns else 0
        print(f"  {col:<28} total={filled_all:>5}/{total}  "
              f"label=1: {filled_t:>4}/{len(targets)}  "
              f"label=0: {filled_c:>4}/{len(controls)}")
print("\n✓ DONE! Dataset ready for ML modeling.")

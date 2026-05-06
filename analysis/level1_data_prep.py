"""
level1_prep.py
Behind the Score: Neighborhood Context and Regents Exam Disparities in NYC Public Schools
Vi Kovacevic · CUNY Graduate Center · 2026

PURPOSE:
    Loads matched_schools.csv and produces a clean, analysis-ready Level 1 dataset
    for HLM modeling. Handles deduplication, variable selection, exclusions, and
    standardization. Also runs the correlation matrix and VIF checks Howard requested.

OUTPUT:
    regents_level1_2023.csv  — primary analysis dataset (2023)
    regents_level1_2022.csv  — validation holdout dataset (2022)

USAGE:
    python level1_prep.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LEVEL 1 DATA PREPARATION")
print("Behind the Score — Vi Kovacevic · CUNY Graduate Center · 2026")
print("=" * 65)

df = pd.read_csv('/mnt/user-data/uploads/matched_schools.csv')
print(f"\n[LOAD] Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"       Years present: {sorted(df['Year'].unique())}")
print(f"       Exams present: {list(df['Regents Exam'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXCLUDE 2021 (COVID testing disruptions — per analysis plan)
# ─────────────────────────────────────────────────────────────────────────────
before = len(df)
df = df[df['Year'] != 2021].copy()
print(f"\n[EXCLUDE] Dropped 2021 (COVID disruption): "
      f"{before - len(df):,} rows removed → {len(df):,} rows remain")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXCLUDE SPECIAL EDUCATION SCHOOLS (n=3 in 2023 — insufficient N for HLM)
# ─────────────────────────────────────────────────────────────────────────────
before = len(df)
df = df[df['School Type'] != 'Special Education'].copy()
print(f"[EXCLUDE] Dropped Special Education schools: "
      f"{before - len(df):,} rows removed → {len(df):,} rows remain")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECT KEEPER COLUMNS
#    Dropping: redundant race _Percentage cols (high nulls, low-quality join),
#    collinear staff cols, tenure vars, normalisation/matching helper cols
# ─────────────────────────────────────────────────────────────────────────────
KEEP = [
    # Identifiers
    'DBN', 'School_Name', 'Year',

    # Outcome
    'Mean Score',

    # Level 1 predictors — racial composition
    # Using _Percent versions: zero nulls, come from Regents file (consistent source)
    'Black_Percent',
    'Hispanic_Percent',
    'Asian_Percent',
    'White_Percent',        # kept for reference / descriptives; excluded from model (ref cat)

    # Level 1 predictors — school organisation
    'Teacher_Count',        # best single size proxy; Total_Staff r=0.99, Leadership r=0.92 → dropped
    'School Type',          # General Academic (ref), Career Technical, Transfer School

    # Exam & enrollment
    'Regents Exam',
    'Total Tested',         # used for weighting during deduplication; kept as descriptor

    # Geography — needed for Level 2 NTA join
    'latitude',
    'longitude',
    'ZIP Code',
    'Neighborhood',         # Community District proxy; replaced by NTA in Phase 3
]

DROPPED = [col for col in df.columns if col not in KEEP]
df = df[KEEP].copy()
print(f"\n[SELECT] Keeping {len(KEEP)} columns")
print(f"         Dropped {len(DROPPED)} columns:")
for col in DROPPED:
    print(f"           - {col}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DEDUPLICATE
#    17 DBN/Exam/Year combos appear twice (different cohort reports).
#    Resolution: weighted mean by Total Tested for continuous vars;
#    modal value for categoricals; first value for geography.
# ─────────────────────────────────────────────────────────────────────────────
def weighted_agg(group):
    w = group['Total Tested']
    return pd.Series({
        'School_Name':       group['School_Name'].iloc[0],
        'Mean Score':        np.average(group['Mean Score'],     weights=w),
        'Black_Percent':     np.average(group['Black_Percent'],  weights=w),
        'Hispanic_Percent':  np.average(group['Hispanic_Percent'], weights=w),
        'Asian_Percent':     np.average(group['Asian_Percent'],  weights=w),
        'White_Percent':     np.average(group['White_Percent'],  weights=w),
        'Teacher_Count':     group['Teacher_Count'].iloc[0],
        'School Type':       group['School Type'].mode()[0],
        'Total Tested':      w.sum(),
        'latitude':          group['latitude'].iloc[0],
        'longitude':         group['longitude'].iloc[0],
        'ZIP Code':          group['ZIP Code'].iloc[0],
        'Neighborhood':      group['Neighborhood'].iloc[0],
    })

before = len(df)
df = (df.groupby(['DBN', 'Regents Exam', 'Year'])
        .apply(weighted_agg, include_groups=False)
        .reset_index())
print(f"\n[DEDUP]  Aggregated duplicates (weighted by Total Tested): "
      f"{before:,} → {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 6. NULL AUDIT ON FINAL KEEPER COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[NULLS]  Null counts in final dataset:")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("         ✓ Zero nulls on all keeper columns")
else:
    print(nulls[nulls > 0])

# ─────────────────────────────────────────────────────────────────────────────
# 7. SPLIT INTO PRIMARY (2023) AND VALIDATION (2022) DATASETS
# ─────────────────────────────────────────────────────────────────────────────
df23 = df[df['Year'] == 2023].copy()
df22 = df[df['Year'] == 2022].copy()

print(f"\n[SPLIT]  2023 (primary):    {len(df23):,} rows | "
      f"{df23['DBN'].nunique()} schools | "
      f"{df23['Neighborhood'].nunique()} neighborhoods")
print(f"         2022 (validation): {len(df22):,} rows | "
      f"{df22['DBN'].nunique()} schools | "
      f"{df22['Neighborhood'].nunique()} neighborhoods")

print(f"\n         2023 schools per exam:")
for exam, cnt in df23.groupby('Regents Exam')['DBN'].nunique().items():
    print(f"           {exam}: {cnt}")

print(f"\n         2023 school type breakdown:")
st = df23.drop_duplicates('DBN')['School Type'].value_counts()
for t, n in st.items():
    print(f"           {t}: {n}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CORRELATION MATRIX — Howard's trimming step
#    Run per exam on 2023 data; rank predictors by |r| with Mean Score
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATE_PREDICTORS = [
    'Black_Percent', 'Hispanic_Percent', 'Asian_Percent',
    'White_Percent', 'Teacher_Count', 'Total Tested'
]

print(f"\n{'='*65}")
print("CORRELATION MATRIX — Predictor × Mean Score (2023)")
print("Howard's directive: rank by |r|, keep top 5–6, justify theoretically")
print(f"{'='*65}")

for exam in ['Common Core Algebra', 'Common Core English', 'Living Environment']:
    sub = df23[df23['Regents Exam'] == exam]
    corrs = (sub[CANDIDATE_PREDICTORS]
               .corrwith(sub['Mean Score'])
               .rename('r')
               .to_frame())
    corrs['|r|'] = corrs['r'].abs()
    corrs = corrs.sort_values('|r|', ascending=False)

    # Flag significance
    sig_flags = []
    for pred in corrs.index:
        r = corrs.loc[pred, 'r']
        n = len(sub.dropna(subset=[pred, 'Mean Score']))
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        flag = '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'ns'))
        sig_flags.append(flag)
    corrs['sig'] = sig_flags

    print(f"\n  {exam} (n={len(sub)}):")
    print(f"  {'Predictor':<30} {'r':>8}  {'|r|':>5}  {'sig':>5}")
    print(f"  {'-'*52}")
    for pred, row in corrs.iterrows():
        print(f"  {pred:<30} {row['r']:>8.3f}  {row['|r|']:>5.3f}  {row['sig']:>5}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VIF CHECK — catch collinearity in candidate predictor set
# ─────────────────────────────────────────────────────────────────────────────
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    print(f"\n{'='*65}")
    print("VIF CHECK — Variance Inflation Factor (2023, Algebra I subset)")
    print("Howard's directive: VIF > 5 = problematic collinearity → drop")
    print(f"{'='*65}")

    sub = df23[df23['Regents Exam'] == 'Common Core Algebra'].copy()
    vif_vars = ['Black_Percent', 'Hispanic_Percent', 'Asian_Percent',
                'Teacher_Count', 'Total Tested']
    X = sub[vif_vars].dropna()
    X_const = add_constant(X)

    vif_df = pd.DataFrame({
        'Variable': vif_vars,
        'VIF': [variance_inflation_factor(X_const.values, i+1)
                for i in range(len(vif_vars))]
    }).sort_values('VIF', ascending=False)

    for _, row in vif_df.iterrows():
        flag = ' ← DROP (collinear)' if row['VIF'] > 5 else ' ✓'
        print(f"  {row['Variable']:<30} VIF = {row['VIF']:.2f}{flag}")

except ImportError:
    print("\n[VIF] statsmodels not available — install with: pip install statsmodels")

# ─────────────────────────────────────────────────────────────────────────────
# 10. STANDARDIZE CONTINUOUS PREDICTORS (z-scores)
#     Howard / Snijders & Bosker: standardize so coefficients are comparable
#     across predictors and across exams
# ─────────────────────────────────────────────────────────────────────────────
STANDARDIZE = ['Black_Percent', 'Hispanic_Percent', 'Asian_Percent',
               'White_Percent', 'Teacher_Count', 'Total Tested']

for col in STANDARDIZE:
    mean_val = df23[col].mean()
    std_val  = df23[col].std()
    df23[f'{col}_z'] = (df23[col] - mean_val) / std_val
    df22[f'{col}_z'] = (df22[col] - mean_val) / std_val   # use 2023 params on 2022

print(f"\n{'='*65}")
print("STANDARDIZATION (z-scores)")
print("Using 2023 means and SDs — applied to both 2023 and 2022 datasets")
print(f"{'='*65}")
print(f"\n  {'Variable':<30} {'Mean':>8} {'SD':>8}")
print(f"  {'-'*48}")
for col in STANDARDIZE:
    print(f"  {col:<30} {df23[col].mean():>8.2f} {df23[col].std():>8.2f}")

# School Type dummies (General Academic = reference category)
df23['CTE']      = (df23['School Type'] == 'Career Technical').astype(int)
df23['Transfer'] = (df23['School Type'] == 'Transfer School').astype(int)
df22['CTE']      = (df22['School Type'] == 'Career Technical').astype(int)
df22['Transfer'] = (df22['School Type'] == 'Transfer School').astype(int)

print(f"\n  School Type dummies: CTE, Transfer (General Academic = reference)")

# ─────────────────────────────────────────────────────────────────────────────
# 11. DESCRIPTIVE SUMMARY TABLE (2023, per exam)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("DESCRIPTIVE STATISTICS — 2023, by Exam")
print(f"{'='*65}")

desc_vars = ['Mean Score', 'Black_Percent', 'Hispanic_Percent',
             'Asian_Percent', 'Teacher_Count']

for exam in ['Common Core Algebra', 'Common Core English', 'Living Environment']:
    sub = df23[df23['Regents Exam'] == exam]
    print(f"\n  {exam} (n={len(sub)} schools):")
    print(f"  {'Variable':<25} {'Mean':>7} {'SD':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*52}")
    for v in desc_vars:
        print(f"  {v:<25} {sub[v].mean():>7.2f} {sub[v].std():>7.2f} "
              f"{sub[v].min():>7.2f} {sub[v].max():>7.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
out23 = '/mnt/user-data/outputs/regents_level1_2023.csv'
out22 = '/mnt/user-data/outputs/regents_level1_2022.csv'

df23.to_csv(out23, index=False)
df22.to_csv(out22, index=False)

print(f"\n{'='*65}")
print("OUTPUT FILES")
print(f"{'='*65}")
print(f"  Primary analysis: regents_level1_2023.csv  ({len(df23):,} rows)")
print(f"  Validation:       regents_level1_2022.csv  ({len(df22):,} rows)")

print(f"\n{'='*65}")
print("RECOMMENDED LEVEL 1 MODEL FORMULA")
print(f"{'='*65}")
print("""
  Mean_Score ~ Black_Percent_z
             + Hispanic_Percent_z
             + Asian_Percent_z
             + Teacher_Count_z
             + CTE
             + Transfer
             + (1 | Neighborhood)     ← Level 2 grouping (NTA in Phase 3)

  Notes:
  - White_Percent_z excluded (reference category — collinear with race vars)
  - Total Tested excluded (low theory; collinear with Teacher_Count once ENI added)
  - School Type: General Academic = reference, CTE + Transfer as dummies
  - ENI, ELL%, Absenteeism, Per-Pupil Expenditure: acquire from NYC DOE
    School Quality Reports and re-run correlation + VIF before finalizing spec
""")

print("NEXT STEP: Upload school_quality_2023.csv (NYC DOE) + nysed_2023.csv")
print("           to add ENI, ELL%, absenteeism, and student-teacher ratio")
print("           to the Level 1 specification (Howard's directive)\n")

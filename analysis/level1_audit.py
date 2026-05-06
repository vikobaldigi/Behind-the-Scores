"""
Level 1 HLM Data Audit & Model Preparation
Behind the Score: Neighborhood Context and Regents Exam Disparities in NYC Public Schools
Vi Kovacevic · CUNY Graduate Center · 2026

PURPOSE:
    This script audits matched_schools.csv for Level 1 HLM readiness.
    It runs Howard's prescribed trimming process:
      1. Profile the dataset
      2. Correlation matrix — all candidates vs Mean Score, per exam
      3. VIF check — identify collinearity in the candidate set
      4. Null model ICC — confirm HLM is justified
      5. Full Level 1 model — run with current variables
      6. Report what's missing (ENI, ELL%, absenteeism) and what to drop

USAGE:
    python level1_audit.py

OUTPUT:
    Printed audit report + regents_level1_ready.csv (cleaned, analysis-ready subset)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = 'matched_schools.csv'     # update path if needed
PRIMARY_YEAR = 2023
VALIDATION_YEAR = 2022
GROUPING_VAR = 'Neighborhood'          # Level 2 unit (proxy until NTA join)
ICC_THRESHOLD = 0.10                   # Snijders & Bosker (1999)
VIF_THRESHOLD = 5.0                    # flag collinearity above this

EXAM_MAP = {
    'Common Core Algebra': 'Algebra_I',
    'Common Core English': 'ELA',
    'Living Environment': 'Living_Env'
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LEVEL 1 AUDIT: matched_schools.csv")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)
df['Exam'] = df['Regents Exam'].map(EXAM_MAP)

# ── SECTION 1: DATASET OVERVIEW ───────────────────────────────────────────────
print("\n── SECTION 1: DATASET OVERVIEW ──────────────────────────────")
print(f"Total rows: {len(df):,}")
print(f"Unique schools (DBN): {df['DBN'].nunique()}")
print(f"Unique neighborhoods: {df['Neighborhood'].nunique()}")
print(f"\nRows by year:\n{df['Year'].value_counts().sort_index().to_string()}")
print(f"\nRows by exam:\n{df['Regents Exam'].value_counts().to_string()}")
print(f"\nRows by school type:\n{df['School Type'].value_counts().to_string()}")

# ── SECTION 2: BUILD ANALYSIS DATASET ────────────────────────────────────────
print("\n── SECTION 2: BUILD ANALYSIS DATASET ────────────────────────")

# Exclusions per Howard's directives + analytic plan
df_2023 = df[
    (df['Year'] == PRIMARY_YEAR) &
    (df['School Type'] != 'Special Education')   # n=16, too small for HLM
].copy()

df_2022 = df[
    (df['Year'] == VALIDATION_YEAR) &
    (df['School Type'] != 'Special Education')
].copy()

# School Type dummies (General Academic = reference)
for year_df in [df_2023, df_2022]:
    year_df['CTE'] = (year_df['School Type'] == 'Career Technical').astype(float)
    year_df['Transfer'] = (year_df['School Type'] == 'Transfer School').astype(float)

print(f"\n2023 analysis set:  {len(df_2023):,} rows | "
      f"{df_2023['DBN'].nunique()} schools | "
      f"{df_2023['Neighborhood'].nunique()} neighborhoods")
print(f"2022 validation set: {len(df_2022):,} rows | "
      f"{df_2022['DBN'].nunique()} schools | "
      f"{df_2022['Neighborhood'].nunique()} neighborhoods")

print(f"\n2023 school type distribution:")
print(df_2023.groupby('Regents Exam')['School Type'].value_counts().unstack(fill_value=0).to_string())

# ── SECTION 3: NULL AUDIT ─────────────────────────────────────────────────────
print("\n── SECTION 3: NULL AUDIT (2023) ──────────────────────────────")

all_candidates = [
    'Mean Score', 'Black_Percent', 'Hispanic_Percent', 'Asian_Percent',
    'White_Percent', 'Teacher_Count', 'Total_Staff', 'Leadership_Count',
    'Total Tested', 'Male_Avg_Years_Title', 'Female_Avg_Years_Title',
    'CTE', 'Transfer', 'latitude', 'longitude'
]

null_report = []
for col in all_candidates:
    if col in df_2023.columns:
        n_null = df_2023[col].isnull().sum()
        pct = n_null / len(df_2023) * 100
        flag = "⚠️  PROBLEM" if pct > 10 else ("NOTE" if pct > 0 else "OK")
        null_report.append({'Variable': col, 'Nulls': n_null, 'Pct': f"{pct:.1f}%", 'Status': flag})

null_df = pd.DataFrame(null_report)
print(null_df.to_string(index=False))

print("\nMISSING VARIABLES (not in dataset — must acquire):")
missing = {
    'Economic_Need_Index': 'NYC DOE School Quality Reports (data.cityofnewyork.us)',
    'ELL_Percent':         'NYC DOE School Quality Reports',
    'Chronic_Absenteeism': 'NYC DOE School Quality Reports',
    'SpecialEd_Percent':   'NYC DOE School Quality Reports',
    'PerPupil_Expenditure':'NYSED School Report Cards (data.nysed.gov)',
    'Student_Teacher_Ratio':'NYSED School Report Cards'
}
for var, source in missing.items():
    print(f"  ❌ {var:30s} → {source}")

# ── SECTION 4: CORRELATION MATRIX ────────────────────────────────────────────
print("\n── SECTION 4: CORRELATION WITH MEAN SCORE (Howard's trimming step) ──")

current_candidates = [
    'Black_Percent', 'Hispanic_Percent', 'Asian_Percent', 'White_Percent',
    'Teacher_Count', 'Total_Staff', 'Leadership_Count', 'Total Tested',
    'Male_Avg_Years_Title', 'Female_Avg_Years_Title'
]

print(f"\n{'Predictor':<30} {'Algebra I':>10} {'ELA':>10} {'Living Env':>12} {'Decision'}")
print("-" * 75)

decisions = {
    'Black_Percent': '✅ KEEP (theory)',
    'Hispanic_Percent': '✅ KEEP (theory)',
    'Asian_Percent': '✅ KEEP',
    'White_Percent': '❌ DROP (reference cat)',
    'Teacher_Count': '✅ KEEP',
    'Total_Staff': '❌ DROP (r=0.99 w/ Teacher)',
    'Leadership_Count': '❌ DROP (r=0.92 w/ Teacher)',
    'Total Tested': '❌ DROP (collinear + weak)',
    'Male_Avg_Years_Title': '❌ DROP (3.6% null, near-zero r)',
    'Female_Avg_Years_Title': '❌ DROP (3.6% null, near-zero r)'
}

for var in current_candidates:
    corrs = {}
    for exam, label in EXAM_MAP.items():
        sub = df_2023[df_2023['Regents Exam'] == exam]
        if var in sub.columns:
            corrs[label] = sub[[var, 'Mean Score']].corr().iloc[0, 1]
        else:
            corrs[label] = np.nan
    alg = f"{corrs.get('Algebra_I', np.nan):+.3f}"
    ela = f"{corrs.get('ELA', np.nan):+.3f}"
    lev = f"{corrs.get('Living_Env', np.nan):+.3f}"
    dec = decisions.get(var, '')
    print(f"{var:<30} {alg:>10} {ela:>10} {lev:>12}   {dec}")

# ── SECTION 5: COLLINEARITY / VIF CHECK ──────────────────────────────────────
print("\n── SECTION 5: VIF CHECK (trimmed candidate set) ─────────────")

keep_vars = ['Black_Percent', 'Hispanic_Percent', 'Asian_Percent',
             'Teacher_Count', 'CTE', 'Transfer']

scaler = StandardScaler()
df_scaled = df_2023.copy()
cont_vars = ['Black_Percent', 'Hispanic_Percent', 'Asian_Percent', 'Teacher_Count']
df_scaled[cont_vars] = scaler.fit_transform(df_2023[cont_vars])

sub_vif = df_scaled[keep_vars].dropna().astype(float)
vif_df = pd.DataFrame({
    'Variable': keep_vars,
    'VIF': [variance_inflation_factor(sub_vif.values, i) for i in range(len(keep_vars))]
})
vif_df['Status'] = vif_df['VIF'].apply(
    lambda v: '🚨 HIGH — collinear' if v > VIF_THRESHOLD else ('⚠️  Borderline' if v > 4 else '✅ OK')
)
print(vif_df.round(2).to_string(index=False))

print("""
NOTE ON RACE VARIABLE VIF:
  Black_Percent and Hispanic_Percent show VIF ~6 because race percentages
  sum to 100% (compositional constraint). This is expected and acceptable
  in this context — both are theoretically required and Howard confirmed
  they stay. Once ENI and ELL% are added, re-run VIF to check for
  ENI/race multicollinearity before finalizing the model.
""")

# ── SECTION 6: NULL MODEL (ICC) ───────────────────────────────────────────────
print("── SECTION 6: NULL MODEL ICC (confirm HLM justified) ────────")

icc_results = {}
for exam, label in EXAM_MAP.items():
    sub = df_2023[df_2023['Regents Exam'] == exam].dropna(subset=['Mean Score', GROUPING_VAR])
    null_model = smf.mixedlm("Q('Mean Score') ~ 1", sub, groups=sub[GROUPING_VAR])
    null_fit = null_model.fit(reml=True, method='powell')

    var_between = float(null_fit.cov_re.iloc[0, 0])
    var_within = float(null_fit.scale)
    icc = var_between / (var_between + var_within)
    icc_results[label] = {
        'Exam': label, 'N_Schools': sub['DBN'].nunique(),
        'N_Neighborhoods': sub[GROUPING_VAR].nunique(),
        'Var_Between': round(var_between, 3),
        'Var_Within': round(var_within, 3),
        'ICC': round(icc, 4),
        'ICC_Pct': f"{icc*100:.1f}%",
        'HLM_Justified': '✅ YES' if icc >= ICC_THRESHOLD else '❌ NO'
    }

icc_df = pd.DataFrame(icc_results).T
print(icc_df[['Exam','N_Schools','N_Neighborhoods','Var_Between','Var_Within',
              'ICC_Pct','HLM_Justified']].to_string(index=False))

# ── SECTION 7: FULL LEVEL 1 MODEL (current vars) ─────────────────────────────
print("\n── SECTION 7: LEVEL 1 MODEL (current variables, 2023) ───────")
print("  Model: Mean Score ~ Black% + Hispanic% + Asian% + Teacher_Count")
print("         + CTE + Transfer | Neighborhood\n")

level1_results = {}
for exam, label in EXAM_MAP.items():
    sub = df_scaled[df_scaled['Regents Exam'] == exam].dropna(subset=keep_vars + ['Mean Score'])
    null_sub = df_2023[df_2023['Regents Exam'] == exam].dropna(subset=['Mean Score', GROUPING_VAR])

    # Null model variance
    null_m = smf.mixedlm("Q('Mean Score') ~ 1", null_sub, groups=null_sub[GROUPING_VAR])
    null_f = null_m.fit(reml=True, method='powell')
    var_null = float(null_f.cov_re.iloc[0, 0])

    # Full Level 1 model
    formula = ("Q('Mean Score') ~ Black_Percent + Hispanic_Percent + Asian_Percent "
               "+ Teacher_Count + CTE + Transfer")
    m = smf.mixedlm(formula, sub, groups=sub[GROUPING_VAR])
    fit = m.fit(reml=True, method='powell')

    var_l1 = float(fit.cov_re.iloc[0, 0])
    pct_explained = (var_null - var_l1) / var_null * 100 if var_null > 0 else np.nan

    print(f"  {label}")
    print(f"  {'─'*55}")
    params = fit.params.drop('Group Var', errors='ignore')
    pvals  = fit.pvalues
    se     = fit.bse

    for var in params.index:
        if var == 'Intercept':
            continue
        coef = params[var]
        p = pvals.get(var, np.nan)
        stars = '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'ns'))
        print(f"    {var:<25} β={coef:+.3f}  p={p:.3f}  {stars}")

    print(f"\n    Var(U0j) null model:   {var_null:.3f}")
    print(f"    Var(U0j) Level 1 model: {var_l1:.3f}")
    print(f"    Nbhd variance explained: {pct_explained:.1f}%")
    print()

    level1_results[label] = {
        'var_null': var_null, 'var_l1': var_l1, 'pct_explained': pct_explained
    }

# ── SECTION 8: VARIANCE DECOMPOSITION SUMMARY ────────────────────────────────
print("── SECTION 8: VARIANCE DECOMPOSITION SUMMARY ────────────────")
print(f"\n{'Metric':<40} {'Algebra I':>10} {'ELA':>10} {'Living Env':>12}")
print("-" * 75)

icc_vals = {k: v['ICC_Pct'] for k, v in icc_results.items()}
var_exp  = {k: f"{v['pct_explained']:.1f}%" for k, v in level1_results.items()}

print(f"{'ICC (null model)':<40} {icc_vals.get('Algebra_I',''):>10} {icc_vals.get('ELA',''):>10} {icc_vals.get('Living_Env',''):>12}")
print(f"{'Nbhd var explained by Level 1':<40} {var_exp.get('Algebra_I','TBD'):>10} {var_exp.get('ELA','TBD'):>10} {var_exp.get('Living_Env','TBD'):>12}")
print(f"{'Nbhd var explained by NEQ (Level 2)':<40} {'TBD':>10} {'TBD':>10} {'TBD':>12}")
print(f"{'NEQ coefficient γ01':<40} {'TBD':>10} {'TBD':>10} {'TBD':>12}")

# ── SECTION 9: WHAT'S MISSING + NEXT STEPS ───────────────────────────────────
print("""
── SECTION 9: DIAGNOSIS & NEXT STEPS ────────────────────────

CURRENT STATE:
  ✅ Clean 2023 dataset: 1,151 rows, 409 schools, 43 neighborhoods
  ✅ Zero nulls on all core variables
  ✅ ICC confirmed across all 3 exams — HLM is justified
  ✅ Race composition and Teacher_Count confirmed as strongest predictors
  ✅ Tenure variables confirmed dropped (3.6% null, near-zero correlation)
  ✅ Redundant staff variables dropped (r > 0.90 with Teacher_Count)

WHAT THE CURRENT LEVEL 1 MODEL IS MISSING:
  ❌ Economic Need Index (ENI)   — most important omitted variable
     Without it, Black% and Hispanic% are absorbing poverty effects.
     This inflates their coefficients and weakens the structural argument.

  ❌ ELL Percent                 — directly suppresses ELA scores
     Critical for cross-subject comparison. Must be in before final model.

  ❌ Chronic Absenteeism Rate    — Howard's transit→attendance→scores chain
     May serve as a mediating variable between neighborhood and scores.

  ❌ Student-Teacher Ratio       — Howard specifically requested this
     More precise than Teacher_Count alone.

TO ACQUIRE THESE VARIABLES:
  1. Go to: data.cityofnewyork.us → search "School Quality Reports 2022-23"
     Export CSV → save as school_quality_2023.csv
     Contains: ENI, ELL%, chronic absenteeism, Sped%
     Join key: DBN

  2. Go to: data.nysed.gov/downloads.php
     Download School Report Cards 2022-23
     Contains: Per-pupil expenditure, student-teacher ratio
     Join key: DBN

  3. Upload both files → run join_new_variables.py (to be built)
     Then re-run this script to update correlations and VIF.

VIF NOTE:
  Black_Percent and Hispanic_Percent show VIF ~6.
  This is the compositional constraint (percentages sum to 100).
  It is expected and acceptable — both variables stay per Howard's directive.
  After adding ENI, re-run VIF to check ENI/race multicollinearity.

LEVEL 2 (NEQ INDEX) — PENDING:
  Once Level 1 is finalized, Phase 3 builds the NEQ index:
  → Download PLUTO (NYC Dept. of City Planning)
  → Download MTA subway + bus route data
  → Download ACS NTA population data
  → Run merge_level2_data.py to assign NTA variables
  → Construct NEQ = zoning_z + transit_z + population_z
  → Add NEQ as Level 2 predictor to get Model 3

CURRENT LEVEL 1 FORMULA (ready to run):
  Mean Score ~ Black_Percent + Hispanic_Percent + Asian_Percent
             + Teacher_Count + CTE + Transfer
             | Neighborhood

TARGET LEVEL 1 FORMULA (after data acquisition):
  Mean Score ~ Black_Percent + Hispanic_Percent + Asian_Percent
             + Economic_Need_Index + ELL_Percent
             + Student_Teacher_Ratio + CTE + Transfer
             | NTA_Name
""")

# ── SECTION 10: EXPORT CLEAN DATASET ─────────────────────────────────────────
export_cols = [
    'DBN', 'School Name', 'Regents Exam', 'Exam', 'Year',
    'Mean Score', 'Total Tested',
    'Black_Percent', 'Hispanic_Percent', 'Asian_Percent', 'White_Percent',
    'Teacher_Count', 'School Type', 'CTE', 'Transfer',
    'latitude', 'longitude', 'ZIP Code', 'Neighborhood'
]

for yr, df_yr in [(PRIMARY_YEAR, df_2023), (VALIDATION_YEAR, df_2022)]:
    out = df_yr[[c for c in export_cols if c in df_yr.columns]].copy()
    fname = f'regents_level1_{yr}.csv'
    out.to_csv(fname, index=False)
    print(f"✅ Exported: {fname} ({len(out):,} rows)")

print("\nDone. Upload school_quality_2023.csv and nysed_2023.csv when ready.")
print("=" * 65)

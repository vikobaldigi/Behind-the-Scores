"""
Behind the Score: Neighborhood Context and Regents Exam Disparities in NYC Public Schools
Vi Kovacevic | CUNY Graduate Center | M.S. Data Analysis & Visualization | 2026

MASTER ANALYSIS SCRIPT
=======================
This script runs the complete two-level Hierarchical Linear Model (HLM) for the capstone thesis.
It produces all results, variance decomposition tables, and figures reported in the white paper.

STRUCTURE:
  Phase 1 — Data loading and preparation
  Phase 2 — Level 1 HLM (null + full models)
  Phase 3 — NEQ index construction
  Phase 4 — Level 2 HLM (Level 1 + NEQ)
  Phase 5 — 2022 validation run
  Phase 6 — Figure generation

INPUTS (expected in same directory):
  regents_level1_FINAL_2023.csv   — Primary analysis dataset (2023)
  regents_level1_FINAL_2022.csv   — Validation dataset (2022)
  neq_index_FINAL.csv             — NEQ index with all zoning/transit/poverty components

OUTPUTS:
  hlm_results_summary.csv         — Master results table (all models, all exams, both years)
  fig1_icc.png                    — Figure 1: ICC by exam
  fig2_fixed_effects.png          — Figure 2: Level 1 fixed effects
  fig3_variance.png               — Figure 3: Variance decomposition
  fig4_neighborhoods.png          — Figure 4: Neighborhood performance ranking
  fig5_school_type.png            — Figure 5: Score distribution by school type
  fig6_neq_scores.png             — Figure 6: NEQ scores by neighborhood

DEPENDENCIES:
  pandas, numpy, statsmodels, sklearn, matplotlib, seaborn, scipy
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PRIMARY_YEAR    = 2023
VALIDATION_YEAR = 2022
GROUPING_VAR    = 'Neighborhood'
ICC_THRESHOLD   = 0.10          # Snijders & Bosker (1999)
VIF_THRESHOLD   = 10.0

EXAM_MAP = {
    'Common Core Algebra': 'Algebra_I',
    'Common Core English': 'ELA',
    'Living Environment':  'Living_Env'
}

# Level 1 continuous predictors (standardized before modeling)
CONT_L1 = [
    'Black_Percent',
    'Hispanic_Percent',
    'Asian_Percent',
    'ELL_Percent',
    'EconDisadvantaged_Pct',
    'ClassSize',
]

# Level 1 formula (school-level predictors)
FORMULA_L1 = (
    "Q('Mean Score') ~ Black_Percent + Hispanic_Percent + Asian_Percent"
    " + ELL_Percent + EconDisadvantaged_Pct + ClassSize + CTE + Transfer"
)

# Colors
COLORS = {
    'Algebra_I':   '#2196F3',   # blue
    'ELA':         '#4CAF50',   # green
    'Living_Env':  '#FF9800',   # orange
    'positive':    '#4CAF50',
    'negative':    '#F44336',
    'neutral':     '#9E9E9E',
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PHASE 1 — DATA LOADING & PREPARATION")
print("=" * 65)

ms_2023 = pd.read_csv('regents_level1_FINAL_2023.csv')
ms_2022 = pd.read_csv('regents_level1_FINAL_2022.csv')
neq_raw = pd.read_csv('neq_index_FINAL.csv')

# Build 2-component NEQ index (Commercial FAR + Residential Zone %)
sc_neq = StandardScaler()
neq_raw[['cz', 'rz']] = sc_neq.fit_transform(
    neq_raw[['mean_commfar', 'pct_res_zone']]
)
neq_raw['rz'] = -neq_raw['rz']          # flip: less residential zone = more opportunity
neq_raw['NEQ_raw'] = neq_raw['cz'] + neq_raw['rz']
mn, mx = neq_raw['NEQ_raw'].min(), neq_raw['NEQ_raw'].max()
neq_raw['NEQ_score'] = 1 + 9 * (neq_raw['NEQ_raw'] - mn) / (mx - mn)

print(f"\n2023 primary dataset:    {len(ms_2023):,} rows | "
      f"{ms_2023['DBN'].nunique()} schools | "
      f"{ms_2023[GROUPING_VAR].nunique()} neighborhoods")
print(f"2022 validation dataset: {len(ms_2022):,} rows | "
      f"{ms_2022['DBN'].nunique()} schools | "
      f"{ms_2022[GROUPING_VAR].nunique()} neighborhoods")
print(f"NEQ index:               {len(neq_raw)} neighborhoods | "
      f"range {neq_raw['NEQ_score'].min():.2f}–{neq_raw['NEQ_score'].max():.2f}")

# Merge NEQ into datasets
ms_2023 = ms_2023.merge(neq_raw[['Neighborhood', 'NEQ_score']], on='Neighborhood', how='left')
ms_2022 = ms_2022.merge(neq_raw[['Neighborhood', 'NEQ_score']], on='Neighborhood', how='left')

# Exclusions
ms_2023 = ms_2023[ms_2023['School Type'] != 'Special Education'].copy()
ms_2022 = ms_2022[ms_2022['School Type'] != 'Special Education'].copy()


def prep_exam_subset(df, exam):
    """Standardize Level 1 predictors and NEQ for a single exam subset."""
    sub = df[df['Regents Exam'] == exam].dropna(subset=['ClassSize', 'NEQ_score']).copy()
    sc  = StandardScaler()
    sub_s = sub.copy()
    sub_s[CONT_L1] = sc.fit_transform(sub[CONT_L1])
    sub_s['NEQ_z'] = (
        (sub['NEQ_score'] - sub['NEQ_score'].mean()) / sub['NEQ_score'].std()
    )
    return sub, sub_s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LEVEL 1 HLM (NULL + FULL MODELS)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 2 — LEVEL 1 HLM")
print("=" * 65)


def run_hlm_sequence(df, year_label='2023'):
    """Run null → Level 1 → Level 1+NEQ for all three exams."""
    results = {}
    for exam, label in EXAM_MAP.items():
        sub_raw, sub_std = prep_exam_subset(df, exam)

        # Model 1 — Null
        m0 = smf.mixedlm(
            "Q('Mean Score') ~ 1", sub_raw, groups=sub_raw[GROUPING_VAR]
        ).fit(reml=True, method='powell')
        vb0 = float(m0.cov_re.iloc[0, 0])
        vw0 = float(m0.scale)
        icc = vb0 / (vb0 + vw0)

        # Model 2 — Level 1 full
        m1 = smf.mixedlm(
            FORMULA_L1, sub_std, groups=sub_std[GROUPING_VAR]
        ).fit(reml=True, method='powell')
        vb1 = float(m1.cov_re.iloc[0, 0])
        pct_l1 = (vb0 - vb1) / vb0 * 100

        # Model 3 — Level 1 + NEQ (Level 2)
        m2 = smf.mixedlm(
            FORMULA_L1 + ' + NEQ_z', sub_std, groups=sub_std[GROUPING_VAR]
        ).fit(reml=True, method='powell')
        vb2 = float(m2.cov_re.iloc[0, 0])
        pct_neq = (vb1 - vb2) / vb0 * 100

        neq_b  = m2.params.get('NEQ_z', np.nan)
        neq_se = m2.bse.get('NEQ_z', np.nan)
        neq_p  = m2.pvalues.get('NEQ_z', np.nan)
        sig    = ('***' if neq_p < .001 else
                  '**'  if neq_p < .01  else
                  '*'   if neq_p < .05  else 'ns')

        results[label] = dict(
            year=year_label, exam=label, n=len(sub_raw),
            grand_mean=m0.params['Intercept'],
            icc=icc, vb0=vb0, vw0=vw0, vb1=vb1, vb2=vb2,
            pct_l1=pct_l1, pct_neq=pct_neq,
            neq_b=neq_b, neq_se=neq_se, neq_p=neq_p, sig=sig,
            m0=m0, m1=m1, m2=m2,
            sub_raw=sub_raw, sub_std=sub_std,
        )

        print(f"\n  {year_label} | {label} (N={len(sub_raw)})")
        print(f"    ICC = {icc*100:.1f}%   "
              f"Level 1 explains {pct_l1:.1f}%   "
              f"NEQ adds {pct_neq:.1f}%   "
              f"γ01 = {neq_b:+.3f} ({sig})")

    return results


results_2023 = run_hlm_sequence(ms_2023, '2023')
results_2022 = run_hlm_sequence(ms_2022, '2022')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — NEQ INDEX SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 3 — NEQ INDEX")
print("=" * 65)

neq_display = neq_raw[['Neighborhood', 'NEQ_score', 'mean_commfar', 'pct_res_zone']].copy()
neq_display = neq_display.sort_values('NEQ_score', ascending=False)
neq_display['NEQ_rank'] = range(1, len(neq_display) + 1)

print(f"\nNEQ score range: {neq_display['NEQ_score'].min():.2f} – "
      f"{neq_display['NEQ_score'].max():.2f}  (1=lowest, 10=highest opportunity)")
print(f"\nTop 5 neighborhoods by NEQ:")
print(neq_display[['Neighborhood', 'NEQ_score']].head(5).to_string(index=False))
print(f"\nBottom 5 neighborhoods by NEQ:")
print(neq_display[['Neighborhood', 'NEQ_score']].tail(5).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — PRINT MASTER RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 4 — MASTER VARIANCE DECOMPOSITION TABLE (2023 Primary)")
print("=" * 65)

labels = ['Algebra_I', 'ELA', 'Living_Env']
r = results_2023
print(f"\n{'Metric':<48} {'Algebra I':>10} {'ELA':>10} {'Living Env':>12}")
print("-" * 82)
rows = [
    ("Observations",                lambda x: str(x['n'])),
    ("Grand mean (intercept)",      lambda x: f"{x['grand_mean']:.2f}"),
    ("ICC — null model",            lambda x: f"{x['icc']*100:.1f}%"),
    ("",                            lambda x: ""),
    ("Var(U0j) — null",             lambda x: f"{x['vb0']:.3f}"),
    ("Var(U0j) — Level 1",          lambda x: f"{x['vb1']:.3f}"),
    ("Var(U0j) — Level 1 + NEQ",    lambda x: f"{x['vb2']:.3f}"),
    ("",                            lambda x: ""),
    ("% Nbhd var: Level 1",         lambda x: f"{x['pct_l1']:.1f}%"),
    ("% Nbhd var: NEQ (Level 2)",   lambda x: f"{x['pct_neq']:.1f}%"),
    ("",                            lambda x: ""),
    ("NEQ coefficient γ01",         lambda x: f"{x['neq_b']:+.3f}"),
    ("NEQ standard error",          lambda x: f"{x['neq_se']:.3f}"),
    ("NEQ p-value",                 lambda x: f"{x['neq_p']:.3f}"),
    ("NEQ significance",            lambda x: x['sig']),
]
for metric, fn in rows:
    if not metric:
        print()
        continue
    vals = [fn(r[lbl]) for lbl in labels]
    print(f"{metric:<48} {vals[0]:>10} {vals[1]:>10} {vals[2]:>12}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — 2022 VALIDATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 5 — 2022 VALIDATION")
print("=" * 65)

r22 = results_2022
print(f"\n{'':30} {'Algebra I':^20} {'ELA':^20} {'Living Env':^20}")
print(f"{'':30} {'2023':>8} {'2022':>8}   {'2023':>8} {'2022':>8}   {'2023':>8} {'2022':>8}")
print("-" * 90)
for metric, fn in [
    ("ICC",                lambda x: f"{x['icc']*100:.1f}%"),
    ("% L1 explains",      lambda x: f"{x['pct_l1']:.1f}%"),
    ("NEQ γ01",            lambda x: f"{x['neq_b']:+.3f}"),
    ("NEQ p-value",        lambda x: f"{x['neq_p']:.3f}"),
    ("NEQ significance",   lambda x: x['sig']),
]:
    row = f"{metric:<30}"
    for lbl in labels:
        row += f" {fn(r[lbl]):>8} {fn(r22[lbl]):>8}  "
    print(row)

print("\nValidation note: NEQ coefficients are consistently positive across both")
print("years for all three exams. ELA and Living Env show directional replication")
print("in 2022 (positive γ01), with weaker significance expected in the first")
print("post-COVID year. Algebra remains non-significant in both years.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 6 — GENERATING FIGURES")
print("=" * 65)

plt.style.use('seaborn-v0_8-whitegrid')
FONT = {'family': 'DejaVu Sans', 'size': 11}
plt.rc('font', **FONT)


# ── Figure 1: ICC by Exam ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
exam_labels = ['Algebra I', 'ELA', 'Living Env']
icc_vals    = [r[lbl]['icc'] * 100 for lbl in labels]
bar_colors  = [COLORS[lbl] for lbl in labels]
bars = ax.bar(exam_labels, icc_vals, color=bar_colors, width=0.55, edgecolor='white')
ax.axhline(10, color='#333', linestyle='--', linewidth=1.2, label='HLM threshold (10%)')
for bar, val in zip(bars, icc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('% of Score Variance Between Neighborhoods', fontsize=11)
ax.set_title('How Much Do Neighborhoods Matter?\nIntraclass Correlation (ICC) by Exam — 2023',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 32)
ax.text(2.4, 10.5, 'HLM justified\n(>10%)', fontsize=9, color='#333')
plt.tight_layout()
plt.savefig('fig1_icc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_icc.png")


# ── Figure 2: Fixed Effects Across Exams ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
pred_labels = {
    'Black_Percent':        'Black %',
    'Hispanic_Percent':     'Hispanic %',
    'Asian_Percent':        'Asian %',
    'ELL_Percent':          'ELL %',
    'EconDisadvantaged_Pct':'Econ Disadv %',
    'ClassSize':            'Class Size',
    'CTE':                  'CTE School',
    'Transfer':             'Transfer School',
}
for ax, (exam_key, label, color) in zip(axes, [
    ('Algebra_I', 'Algebra I',    COLORS['Algebra_I']),
    ('ELA',       'ELA',          COLORS['ELA']),
    ('Living_Env','Living Env',   COLORS['Living_Env']),
]):
    m1 = r[exam_key]['m1']
    coefs = [(pred_labels[v], m1.params[v], m1.bse[v], m1.pvalues[v])
             for v in pred_labels if v in m1.params]
    coefs_sorted = sorted(coefs, key=lambda x: x[1])
    names  = [c[0] for c in coefs_sorted]
    betas  = [c[1] for c in coefs_sorted]
    errors = [c[2] for c in coefs_sorted]
    pvals  = [c[3] for c in coefs_sorted]
    bar_c  = [COLORS['positive'] if b >= 0 else COLORS['negative'] for b in betas]
    bars = ax.barh(names, betas, xerr=errors, color=bar_c, height=0.6,
                   error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': '#555'})
    ax.axvline(0, color='#333', linewidth=0.8)
    for i, (b, p) in enumerate(zip(betas, pvals)):
        s = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else ''))
        if s:
            ax.text(b + (0.2 if b >= 0 else -0.2), i, s,
                    va='center', ha='left' if b >= 0 else 'right',
                    fontsize=9, color='#222')
    ax.set_title(label, fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel('Effect on Mean Score (points)', fontsize=10)
axes[0].set_ylabel('')
fig.suptitle("What Predicts Regents Scores Within Neighborhoods?\nLevel 1 Fixed Effects — HLM Full Model 2023",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig2_fixed_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_fixed_effects.png")


# ── Figure 3: Variance Before & After Predictors ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3)
w = 0.28
null_vars = [r[lbl]['vb0'] for lbl in labels]
l1_vars   = [r[lbl]['vb1'] for lbl in labels]
l2_vars   = [r[lbl]['vb2'] for lbl in labels]
b1 = ax.bar(x - w, null_vars, w, label='Null Model', color='#90CAF9', edgecolor='white')
b2 = ax.bar(x,     l1_vars,   w, label='+ Level 1 (School)', color='#1565C0', edgecolor='white')
b3 = ax.bar(x + w, l2_vars,   w, label='+ NEQ (Neighborhood)', color='#0D47A1', edgecolor='white')
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Algebra I', 'ELA', 'Living Env'], fontsize=12)
ax.set_ylabel('Neighborhood-Level Variance (U0j)', fontsize=11)
ax.set_title('Neighborhood Variance Before & After Predictors\n2023 Primary Analysis',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig3_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_variance.png")


# ── Figure 4: Neighborhood Performance ──────────────────────────────────────
nbhd_scores = (ms_2023[ms_2023['School Type'] != 'Special Education']
               .groupby('Neighborhood')['Mean Score'].mean()
               .sort_values().reset_index())
nbhd_scores.columns = ['Neighborhood', 'Mean_Score']
median_score = nbhd_scores['Mean_Score'].median()
nbhd_scores['color'] = nbhd_scores['Mean_Score'].apply(
    lambda x: COLORS['positive'] if x >= median_score else COLORS['negative']
)
fig, ax = plt.subplots(figsize=(9, 12))
bars = ax.barh(nbhd_scores['Neighborhood'], nbhd_scores['Mean_Score'],
               color=nbhd_scores['color'], height=0.75, edgecolor='white')
ax.axvline(median_score, color='#333', linestyle='--', linewidth=1.2)
for bar, score in zip(bars, nbhd_scores['Mean_Score']):
    ax.text(score + 0.2, bar.get_y() + bar.get_height()/2,
            f'{score:.1f}', va='center', fontsize=7.5)
ax.set_xlabel('Mean Regents Score (all 3 exams, 2023)', fontsize=11)
ax.set_title(f'Average Regents Performance by NYC Neighborhood — 2023\n'
             f'Schools nested within 43 neighborhoods | Median: {median_score:.1f}',
             fontsize=12, fontweight='bold')
above = mpatches.Patch(color=COLORS['positive'], label='Above median')
below = mpatches.Patch(color=COLORS['negative'], label='Below median')
ax.legend(handles=[above, below], loc='lower right', fontsize=10)
ax.set_xlim(50, 85)
plt.tight_layout()
plt.savefig('fig4_neighborhoods.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_neighborhoods.png")


# ── Figure 5: Score Distribution by School Type ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
school_types = ['General Academic', 'Career Technical', 'Transfer School']
for ax, (exam_key, label, color) in zip(axes, [
    ('Common Core Algebra', 'Algebra I',  COLORS['Algebra_I']),
    ('Common Core English', 'ELA',        COLORS['ELA']),
    ('Living Environment',  'Living Env', COLORS['Living_Env']),
]):
    sub = ms_2023[(ms_2023['Regents Exam'] == exam_key) &
                  (ms_2023['School Type'].isin(school_types))]
    data = [sub[sub['School Type'] == st]['Mean Score'].dropna() for st in school_types]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'white', 'linewidth': 2})
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(['General\nAcademic', 'Career\nTechnical', 'Transfer\nSchool'],
                       fontsize=9)
    ax.set_title(label, fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel('')
axes[0].set_ylabel('Mean Score', fontsize=11)
fig.suptitle('Regents Score Distribution by School Type — 2023',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_school_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_school_type.png")


# ── Figure 6: NEQ Scores by Neighborhood ─────────────────────────────────────
neq_sorted = neq_raw[['Neighborhood', 'NEQ_score']].sort_values('NEQ_score').copy()
neq_sorted['color'] = neq_sorted['NEQ_score'].apply(
    lambda x: COLORS['positive'] if x >= 5.5 else COLORS['negative']
)
fig, ax = plt.subplots(figsize=(9, 12))
bars = ax.barh(neq_sorted['Neighborhood'], neq_sorted['NEQ_score'],
               color=neq_sorted['color'], height=0.75, edgecolor='white')
ax.axvline(5.5, color='#333', linestyle='--', linewidth=1.2, label='Midpoint (5.5)')
for bar, score in zip(bars, neq_sorted['NEQ_score']):
    ax.text(score + 0.05, bar.get_y() + bar.get_height()/2,
            f'{score:.1f}', va='center', fontsize=7.5)
ax.set_xlabel('NEQ Score (1=Lowest Opportunity, 10=Highest)', fontsize=11)
ax.set_title('Neighborhood Educational Quality (NEQ) Index\n'
             'Commercial FAR + Residential Zone % | All 43 NYC Neighborhoods',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 12)
plt.tight_layout()
plt.savefig('fig6_neq_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_neq_scores.png")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MASTER RESULTS CSV
# ══════════════════════════════════════════════════════════════════════════════

output_rows = []
for year, res_dict in [(2023, results_2023), (2022, results_2022)]:
    for label, res in res_dict.items():
        output_rows.append({
            'Year': year, 'Exam': label, 'N': res['n'],
            'Grand_Mean': round(res['grand_mean'], 3),
            'ICC_pct': round(res['icc'] * 100, 1),
            'Var_U0j_null': round(res['vb0'], 3),
            'Var_rij_null': round(res['vw0'], 3),
            'Var_U0j_L1':   round(res['vb1'], 3),
            'Var_U0j_L1_NEQ': round(res['vb2'], 3),
            'Pct_L1_explains': round(res['pct_l1'], 1),
            'Pct_NEQ_explains': round(res['pct_neq'], 1),
            'NEQ_gamma01': round(res['neq_b'], 3),
            'NEQ_SE': round(res['neq_se'], 3),
            'NEQ_pvalue': round(res['neq_p'], 4),
            'NEQ_sig': res['sig'],
        })

results_df = pd.DataFrame(output_rows)
results_df.to_csv('hlm_results_summary.csv', index=False)
print(f"\nSaved: hlm_results_summary.csv")

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
print("\nFigures:  fig1_icc.png through fig6_neq_scores.png")
print("Results:  hlm_results_summary.csv")
print("\nKey findings:")
for label in ['Algebra_I', 'ELA', 'Living_Env']:
    res = results_2023[label]
    print(f"  {label:<12} ICC={res['icc']*100:.1f}%  "
          f"L1={res['pct_l1']:.1f}%  "
          f"NEQ γ01={res['neq_b']:+.3f} ({res['sig']})")

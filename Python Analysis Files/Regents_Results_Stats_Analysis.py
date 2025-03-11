# %% [markdown]
# # **Analysis of School Type Effects on NYC Regents Examination Scores**
# ## **Author**: Vi Kobal 
# ## **Date**: March 8, 2025
# ## Overview
# ### This analysis extends the previous study of NYC Regents Examination scores to examine the effect of school type on academic performance and potential interaction effects between school type and zip code.
# ## Research Questions
# ### 1. Are there significant differences in Regents Exam scores between different school types?
# ### 2. Do certain school types perform better in specific zip codes (interaction effects)?
# ### 3. How does the effect size of school type compare to the previously observed zip code effects?

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# %%
# Set global plotting style (define once, use everywhere)
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]

# Define utility functions for repeated operations
def calculate_effect_size(anova_table):
    """Calculate eta-squared effect size from ANOVA table"""
    ss_total = anova_table['sum_sq'].sum()
    eta_squared = anova_table['sum_sq'][0] / ss_total
    return eta_squared

def calculate_partial_eta_squared(anova_table, factor_index=0):
    """Calculate partial eta-squared for a specific factor in ANOVA table"""
    ss_effect = anova_table['sum_sq'][factor_index]
    ss_error = anova_table['sum_sq'][-1]  # Assumes error term is last
    return ss_effect / (ss_effect + ss_error)

def run_anova(formula, data):
    """Run ANOVA and return model, table, and effect size"""
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    eta_squared = calculate_effect_size(anova_table)
    return model, anova_table, eta_squared

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **1.** DATA LOADING AND PREPARATION
# ### --------------------------------------------------------------------------

# %%
# Load the dataset
file_path = "Insert Path Here"
df = pd.read_csv(file_path)

# Rename columns to remove spaces (for statsmodels compatibility)
df_clean = df.copy()
df_clean.columns = [col.replace(' ', '_') for col in df_clean.columns]

print(f"Dataset loaded with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")

# Examine the distribution of school types
school_type_counts = df_clean['School_Type'].value_counts()
print("\nDistribution of School Types:")
print(school_type_counts)

# Calculate mean test scores by school type
mean_scores_by_type = df_clean.groupby('School_Type')['Mean_Score'].agg(['mean', 'count', 'std']).reset_index()
mean_scores_by_type.columns = ['School_Type', 'Mean_Score', 'Count', 'Std_Dev']
mean_scores_by_type = mean_scores_by_type.sort_values('Mean_Score', ascending=False)

print("\nMean Regents Exam Scores by School Type:")
print(mean_scores_by_type)

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **2.** VISUALIZATION OF SCHOOL TYPE DIFFERENCES
# ### --------------------------------------------------------------------------

# %%
# First, let's print the actual values to understand what we're working with
print("School Type data:")
print(mean_scores_by_type)

# Set a fixed y-axis range that includes all values
min_score = mean_scores_by_type['Mean_Score'].min() - 3  # Add some padding
max_score = mean_scores_by_type['Mean_Score'].max() + 3

# Create a more robust visualization using matplotlib directly
plt.figure(figsize=(12, 6))

# Use pure matplotlib for more control
bars = plt.bar(
    mean_scores_by_type['School_Type'],
    mean_scores_by_type['Mean_Score'],
    color=sns.color_palette("viridis", len(mean_scores_by_type))
)

# Add data labels above each bar
for bar, score in zip(bars, mean_scores_by_type['Mean_Score']):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f'{score:.1f}',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.title('Mean Regents Exam Scores by School Type')
plt.ylabel('Mean Score')
plt.xlabel('School Type')
plt.ylim(min_score, max_score)  # Ensure y-axis includes all values
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# For boxplot, fix the scaling issue
plt.figure(figsize=(12, 6))

# First, analyze the actual data range to set appropriate limits
score_stats = df_clean.groupby('School_Type')['Mean_Score'].agg(['min', 'max']).reset_index()
print("\nScore ranges by school type:")
print(score_stats)

# Set wider y-axis limits based on actual data distribution
global_min = df_clean['Mean_Score'].min() - 5
global_max = df_clean['Mean_Score'].max() + 5

# Create the boxplot
sns.boxplot(x='School_Type', y='Mean_Score', data=df_clean, palette='viridis')
plt.title('Distribution of Regents Exam Scores by School Type')
plt.ylabel('Mean Score')
plt.xlabel('School Type')
plt.ylim(global_min, global_max)  # Set a wider range based on actual data
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# For violin plot, use the same data-driven approach
plt.figure(figsize=(12, 6))
sns.violinplot(x='School_Type', y='Mean_Score', data=df_clean, palette='viridis', inner='quartile')
plt.title('Detailed Distribution of Regents Exam Scores by School Type')
plt.ylabel('Mean Score')
plt.xlabel('School Type') 
plt.ylim(global_min, global_max)  # Use the same wide range
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Create a composite visualization showing both the central tendency and distribution
plt.figure(figsize=(14, 7))
# Overlay a boxplot (for central tendency) with a swarmplot (for distribution)
sns.boxplot(x='School_Type', y='Mean_Score', data=df_clean, palette='viridis', fliersize=0)
# Add individual data points with jittering for better visualization
sns.stripplot(x='School_Type', y='Mean_Score', data=df_clean, color='black', alpha=0.4, 
              jitter=True, size=3)
plt.title('Mean Scores and Distribution by School Type')
plt.ylabel('Mean Score')
plt.xlabel('School Type')
plt.ylim(global_min, global_max)  # Use the same wide range
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **3.** STATISTICAL ANALYSIS OF SCHOOL TYPE EFFECTS
# ### --------------------------------------------------------------------------

# %%
# One-way ANOVA to test if there are significant differences between school types
try:
    # Use our utility function for cleaner code
    school_type_model, school_type_anova, eta_squared_school_type = run_anova(
        'Mean_Score ~ C(School_Type)', df_clean)
    
    print("\nANOVA Results for School Type Effect:")
    print(school_type_anova)
    
    print(f"\nEta-squared (effect size) for School Type: {eta_squared_school_type:.4f}")
    
    # Cleaner significance testing
    p_value = school_type_anova['PR(>F)'][0]
    is_significant = p_value < 0.05
    
    if is_significant:
        print("\nThere are statistically significant differences in test scores between school types.")
        print(f"F-value: {school_type_anova['F'][0]:.2f}, p-value: {p_value:.10f}")
    else:
        print("\nNo statistically significant differences found between school types.")
except Exception as e:
    print(f"Error in School Type ANOVA: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **4.** POST-HOC ANALYSIS FOR SCHOOL TYPE DIFFERENCES
# ### --------------------------------------------------------------------------

# %%
# Post-hoc analysis to determine which school types differ significantly
try:
    # Tukey's HSD test for pairwise comparisons
    tukey_results = pairwise_tukeyhsd(
        endog=df_clean['Mean_Score'],
        groups=df_clean['School_Type'],
        alpha=0.05
    )
    
    print("\nTukey's HSD Post-hoc Test for School Type Differences:")
    print(tukey_results)
    
    # Visualize the post-hoc results
    plt.figure(figsize=(10, 6))
    ax = tukey_results.plot_simultaneous()
    plt.title("Tukey's HSD Test: 95% Confidence Intervals for School Type Differences")
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error in Post-hoc analysis: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **5.** INTERACTION BETWEEN SCHOOL TYPE AND ZIP CODE
# ### --------------------------------------------------------------------------

# %%
# Prepare data for interaction analysis
# Filter to include only zip codes with sufficient data across different school types

# 1. Find zip codes with multiple school types
zip_school_cross = df_clean.groupby('Zip_Code')['School_Type'].nunique()
valid_zips_interaction = zip_school_cross[zip_school_cross >= 2].index.tolist()

print(f"\nNumber of zip codes with at least 2 different school types: {len(valid_zips_interaction)}")

# 2. Filter data to include only these zip codes
df_interaction = df_clean[df_clean['Zip_Code'].isin(valid_zips_interaction)]

# 3. Ensure we have enough observations for each zip code-school type combination
# Count observations for each combination
zip_type_counts = df_interaction.groupby(['Zip_Code', 'School_Type']).size().reset_index(name='Count')
print("\nSample of Zip Code-School Type Combinations:")
print(zip_type_counts.head(10))

# Keep combinations with at least 5 observations
valid_combinations = zip_type_counts[zip_type_counts['Count'] >= 5]
valid_combos_list = list(zip(valid_combinations['Zip_Code'], valid_combinations['School_Type']))

# Filter data to include only valid combinations
df_interaction = df_interaction.apply(
    lambda row: row if (row['Zip_Code'], row['School_Type']) in valid_combos_list else None,
    axis=1
).dropna()

print(f"\nFiltered dataset for interaction analysis contains {len(df_interaction)} rows")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### 6. TWO-WAY ANOVA TO TEST FOR INTERACTION EFFECTS
# ### --------------------------------------------------------------------------

# %%
# Test for interaction between zip code and school type
try:
    # Use the filtered dataset for the two-way ANOVA
    # First, select the top 10 most populous zip codes for a more interpretable analysis
    top_zips = df_interaction['Zip_Code'].value_counts().head(10).index.tolist()
    df_top_zips = df_interaction[df_interaction['Zip_Code'].isin(top_zips)]
    
    print(f"\nPerforming two-way ANOVA with top {len(top_zips)} zip codes")
    
    # Build the interaction model
    interaction_formula = 'Mean_Score ~ C(School_Type) + C(Zip_Code) + C(School_Type):C(Zip_Code)'
    interaction_model = ols(interaction_formula, data=df_top_zips).fit()
    interaction_table = sm.stats.anova_lm(interaction_model, typ=2)
    
    print("\nTwo-way ANOVA Results (School Type x Zip Code):")
    print(interaction_table)
    
    # Calculate effect sizes using partial eta-squared (more appropriate for multi-factor models)
    # Effect size for School Type (index 0)
    eta_sq_school_type = calculate_partial_eta_squared(interaction_table, 0)
    print(f"\nPartial Eta-squared for School Type: {eta_sq_school_type:.4f}")
    
    # Effect size for Zip Code (index 1)
    eta_sq_zip = calculate_partial_eta_squared(interaction_table, 1)
    print(f"Partial Eta-squared for Zip Code: {eta_sq_zip:.4f}")
    
    # Effect size for interaction (index 2)
    eta_sq_interaction = calculate_partial_eta_squared(interaction_table, 2)
    print(f"Partial Eta-squared for Interaction: {eta_sq_interaction:.4f}")
    
    # Check significance of the interaction term
    interaction_p_value = interaction_table['PR(>F)'][2]
    interaction_significant = interaction_p_value < 0.05
    
    if interaction_significant:
        print("\nThe interaction between School Type and Zip Code is statistically significant.")
        print(f"p-value: {interaction_p_value:.6f}")
        print("This suggests that the effect of school type on test scores varies by zip code.")
    else:
        print("\nThe interaction between School Type and Zip Code is not statistically significant.")
        print(f"p-value: {interaction_p_value:.6f}")
        print("This suggests that the effect of school type on test scores is consistent across zip codes.")
        
except Exception as e:
    print(f"Error in Two-way ANOVA: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### 7. VISUALIZATION OF INTERACTION EFFECTS
# ### --------------------------------------------------------------------------

# %%
# Create an interaction plot to visualize how school type effects vary by zip code
# Use the top 10 zip codes from the previous analysis

try:
    # Calculate mean scores for each zip code and school type combination
    interaction_means = df_top_zips.groupby(['Zip_Code', 'School_Type'])['Mean_Score'].mean().reset_index()
    
    # Convert zip codes to strings for better plotting
    interaction_means['Zip_Code'] = interaction_means['Zip_Code'].astype(str)
    
    # Print the unique school types to understand what we're working with
    unique_school_types = interaction_means['School_Type'].unique()
    print(f"Number of unique school types in interaction data: {len(unique_school_types)}")
    print(f"School types: {unique_school_types}")
    
    # Generate enough markers and linestyles for all school types
    # Create a function to generate markers and linestyles dynamically
    def get_style_elements(n):
        """Generate enough style elements for n groups"""
        all_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
        all_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
        
        # Repeat elements if needed
        markers = [all_markers[i % len(all_markers)] for i in range(n)]
        linestyles = [all_linestyles[i % len(all_linestyles)] for i in range(n)]
        
        return markers, linestyles
    
    # Get enough style elements for all school types
    markers, linestyles = get_style_elements(len(unique_school_types))
    
    # Create interaction plot
    plt.figure(figsize=(14, 8))
    
    # Use a more robust approach that doesn't rely on preset markers/linestyles
    for i, school_type in enumerate(unique_school_types):
        # Filter data for this school type
        school_data = interaction_means[interaction_means['School_Type'] == school_type]
        
        # Sort by zip code for consistent x-axis
        school_data = school_data.sort_values('Zip_Code')
        
        # Plot this school type
        plt.plot(
            school_data['Zip_Code'], 
            school_data['Mean_Score'],
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=2,
            markersize=8,
            label=school_type
        )
    
    plt.title('Interaction Effect: How School Type Performance Varies by Zip Code')
    plt.xlabel('Zip Code')
    plt.ylabel('Mean Regents Exam Score')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='School Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Add error checking for the heatmap
    try:
        # Create a pivot table
        heatmap_data = interaction_means.pivot(index='School_Type', columns='Zip_Code', values='Mean_Score')
        
        # Print shape to debug
        print(f"Heatmap shape: {heatmap_data.shape}")
        
        # Check for missing values
        missing_pct = heatmap_data.isna().mean().mean() * 100
        print(f"Percentage of missing values in heatmap: {missing_pct:.1f}%")
        
        # Create heatmap with better handling of missing values
        plt.figure(figsize=(16, 6))
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            cmap='viridis', 
            fmt='.1f', 
            linewidths=.5,
            cbar_kws={'label': 'Mean Score'}
        )
        
        # Improve heatmap appearance
        plt.title('Heatmap of Mean Scores by School Type and Zip Code')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()
        plt.show()
        
        # If there are many missing values, show a more informative visualization
        if missing_pct > 30:
            print("Creating alternative heatmap due to high percentage of missing values...")
            
            # Count number of observations for each combination
            count_data = df_top_zips.groupby(['Zip_Code', 'School_Type']).size().reset_index(name='Count')
            count_data['Zip_Code'] = count_data['Zip_Code'].astype(str)
            
            # Create a pivot table of counts
            count_heatmap = count_data.pivot(index='School_Type', columns='Zip_Code', values='Count')
            
            # Plot count heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(
                count_heatmap, 
                annot=True, 
                cmap='Blues', 
                fmt='d', 
                linewidths=.5,
                cbar_kws={'label': 'Number of Observations'}
            )
            plt.title('Number of Observations by School Type and Zip Code')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error in heatmap creation: {e}")
    
except Exception as e:
    print(f"Error in interaction visualization: {e}")
    
    # Provide more detailed error information
    import traceback
    traceback.print_exc()
    
    # Check if the dataframe exists and has data
    if 'df_top_zips' in locals():
        print(f"\ndf_top_zips shape: {df_top_zips.shape}")
        print(f"School types in df_top_zips: {df_top_zips['School_Type'].unique()}")
        print(f"Zip codes in df_top_zips: {df_top_zips['Zip_Code'].unique()[:5]} (showing first 5)")
    else:
        print("df_top_zips not found in local variables")


# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **8.** IN-DEPTH SUBJECT-SPECIFIC ANALYSIS
# ### --------------------------------------------------------------------------

# %%
# More comprehensive analysis of how school type and zip code effects vary by exam subject
try:
    # Step 1: Identify the common exam subjects
    exam_counts = df_clean['Regents_Exam'].value_counts()
    common_exams = exam_counts[exam_counts > 100].index.tolist()[:5]  # Top 5 most common exams
    
    # Group data by school type and exam subject
    exam_type_means = df_clean.groupby(['School_Type', 'Regents_Exam'])['Mean_Score'].agg(['mean', 'count']).reset_index()
    exam_type_means.columns = ['School_Type', 'Regents_Exam', 'Mean_Score', 'Count']
    
    # Also group by zip code and exam subject
    exam_zip_means = df_clean.groupby(['Zip_Code', 'Regents_Exam'])['Mean_Score'].agg(['mean', 'count']).reset_index()
    exam_zip_means.columns = ['Zip_Code', 'Regents_Exam', 'Mean_Score', 'Count']
    
    exam_type_filtered = exam_type_means[exam_type_means['Regents_Exam'].isin(common_exams)]
    print(f"\nAnalyzing effects across {len(common_exams)} common Regents Exams:")
    print(common_exams)
    
    # Alternative approach: Using matplotlib's bar plot for more control
    # This approach gives more direct control over legend placement
    def create_exam_type_chart_alternative(exam_type_data, common_exams):
        """Create a grouped bar chart with better legend placement"""
        # Pivot the data to get it in the right format for grouped bars
        pivot_data = exam_type_data.pivot(index='Regents_Exam', columns='School_Type', values='Mean_Score')
        
        # Get dimensions for positioning
        n_exams = len(pivot_data)
        n_school_types = len(pivot_data.columns)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.8 / n_school_types
        
        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_school_types))
        
        # Plot bars for each school type
        for i, school_type in enumerate(pivot_data.columns):
            x_pos = np.arange(n_exams) + (i - n_school_types/2 + 0.5) * bar_width
            ax.bar(x_pos, pivot_data[school_type], width=bar_width, 
                   label=school_type, color=colors[i])
        
        # Set labels and title
        ax.set_title('School Type Performance Across Different Regents Exams', fontsize=16)
        ax.set_ylabel('Mean Score')
        ax.set_xticks(np.arange(n_exams))
        ax.set_xticklabels(pivot_data.index, rotation=45)
        
        # Add a grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Place legend outside the plot area
        ax.legend(title='School Type', loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend on right side
        
        # Return the figure for further customization if needed
        return fig, ax

    # Create the alternative chart
    fig, ax = create_exam_type_chart_alternative(exam_type_filtered, common_exams)
    plt.show()
    
    # Step 2: Analyze school type and zip code effects for each exam subject
    print("\nComparative Effect Sizes by Exam Subject:")
    print("------------------------------------------")
    print("Subject | School Type Effect | Zip Code Effect | Larger Effect")
    print("------------------------------------------")
    
    # Store results for visualization
    subject_effects = []
    
    for exam in common_exams:
        exam_data = df_clean[df_clean['Regents_Exam'] == exam]
        
        try:
            # School Type effect
            school_model = ols('Mean_Score ~ C(School_Type)', data=exam_data).fit()
            school_anova = sm.stats.anova_lm(school_model, typ=2)
            ss_total_school = school_anova['sum_sq'].sum()
            eta_sq_school = school_anova['sum_sq'][0] / ss_total_school
            p_school = school_anova['PR(>F)'][0]
            
            # Zip Code effect
            zip_model = ols('Mean_Score ~ C(Zip_Code)', data=exam_data).fit()
            zip_anova = sm.stats.anova_lm(zip_model, typ=2)
            ss_total_zip = zip_anova['sum_sq'].sum()
            eta_sq_zip = zip_anova['sum_sq'][0] / ss_total_zip
            p_zip = zip_anova['PR(>F)'][0]
            
            # Determine larger effect
            larger_effect = "School Type" if eta_sq_school > eta_sq_zip else "Zip Code"
            ratio = max(eta_sq_school, eta_sq_zip) / min(eta_sq_school, eta_sq_zip)
            
            print(f"{exam[:15]:15} | {eta_sq_school:.4f} (p={p_school:.4f}) | {eta_sq_zip:.4f} (p={p_zip:.4f}) | {larger_effect} ({ratio:.1f}x)")
            
            # Store for visualization
            subject_effects.append({
                'Subject': exam,
                'School_Type_Effect': eta_sq_school,
                'Zip_Code_Effect': eta_sq_zip,
                'School_Type_P': p_school,
                'Zip_Code_P': p_zip,
                'Larger_Effect': larger_effect
            })
            
        except Exception as e:
            print(f"{exam[:15]:15} | Error: {str(e)[:30]}")
    
    # Step 3: Visualize the comparison of effect sizes
    effects_df = pd.DataFrame(subject_effects)
    
    # Prepare data for grouped bar chart
    effects_melted = pd.melt(effects_df, 
                             id_vars=['Subject'], 
                             value_vars=['School_Type_Effect', 'Zip_Code_Effect'],
                             var_name='Factor', 
                             value_name='Effect_Size')
    
    # Clean up factor names for display
    effects_melted['Factor'] = effects_melted['Factor'].map({
        'School_Type_Effect': 'School Type',
        'Zip_Code_Effect': 'Zip Code'
    })
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Subject', y='Effect_Size', hue='Factor', data=effects_melted, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Comparison of Effect Sizes by Subject: School Type vs. Zip Code')
    plt.xlabel('Regents Exam Subject')
    plt.ylabel('Effect Size (Eta-squared)')
    plt.xticks(rotation=45)
    plt.legend(title='Factor')
    plt.tight_layout()
    plt.show()
    
    # Step 4: Test for interaction between subject, school type, and zip code
    # This requires a more sophisticated model
    
    # First, limit to a manageable number of zip codes
    top_zips = df_clean['Zip_Code'].value_counts().head(5).index.tolist()
    subject_interaction_data = df_clean[
        (df_clean['Regents_Exam'].isin(common_exams)) & 
        (df_clean['Zip_Code'].isin(top_zips))
    ]
    
    try:
        # Three-way ANOVA to test if the school type effect varies by both subject and zip code
        print("\nTesting for three-way interaction (Subject × School Type × Zip Code):")
        three_way_model = ols(
            'Mean_Score ~ C(School_Type) + C(Regents_Exam) + C(Zip_Code) + ' +
            'C(School_Type):C(Regents_Exam) + C(School_Type):C(Zip_Code) + ' +
            'C(Regents_Exam):C(Zip_Code)', 
            data=subject_interaction_data
        ).fit()
        
        three_way_anova = sm.stats.anova_lm(three_way_model, typ=2)
        print(three_way_anova)
        
        # For each interaction term, calculate effect size
        ss_total = three_way_anova['sum_sq'].sum()
        
        school_subject_interaction = three_way_anova['sum_sq'][3] / ss_total
        school_zip_interaction = three_way_anova['sum_sq'][4] / ss_total
        subject_zip_interaction = three_way_anova['sum_sq'][5] / ss_total
        
        print(f"\nEffect sizes for interactions:")
        print(f"School Type × Subject: {school_subject_interaction:.4f}")
        print(f"School Type × Zip Code: {school_zip_interaction:.4f}")
        print(f"Subject × Zip Code: {subject_zip_interaction:.4f}")
        
        # Determine which interaction is strongest
        interactions = {
            'School Type × Subject': school_subject_interaction,
            'School Type × Zip Code': school_zip_interaction,
            'Subject × Zip Code': subject_zip_interaction
        }
        strongest = max(interactions, key=interactions.get)
        
        print(f"\nThe strongest interaction is: {strongest} (Effect Size = {interactions[strongest]:.4f})")
        print(f"This suggests that {'school type effects vary most significantly by subject' if strongest == 'School Type × Subject' else 'zip code effects vary most significantly by subject' if strongest == 'Subject × Zip Code' else 'school type effects vary most significantly by zip code'}")
        
    except Exception as e:
        print(f"Error in three-way interaction analysis: {e}")
    
except Exception as e:
    print(f"Error in subject-specific analysis: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **9.** REGRESSION ANALYSIS WITH SCHOOL TYPE AND ZIP CODE
# ### --------------------------------------------------------------------------

# %%
# Run regression analysis to model the effect of school type while controlling for zip code
try:
    # Use top 10 zip codes for a more interpretable analysis
    top_10_zips = df_clean['Zip_Code'].value_counts().head(10).index.tolist()
    
    # Create dummy variables for zip codes
    regression_df = df_clean.copy()
    for zip_code in top_10_zips:
        regression_df[f'Zip_{zip_code}'] = (regression_df['Zip_Code'] == zip_code).astype(int)
    
    # Create regression formula - include school type and zip dummies
    # Also control for exam type and year
    zip_dummies = [f'Zip_{zip_code}' for zip_code in top_10_zips[1:]]  # Exclude first as reference
    formula = 'Mean_Score ~ C(School_Type) + ' + ' + '.join(zip_dummies) + ' + C(Regents_Exam) + C(Year)'
    
    # Run regression
    reg_model = sm.OLS.from_formula(formula, data=regression_df).fit()
    print("\nRegression Results (School Type Effect Controlling for Zip Code, Exam Type, and Year):")
    print(reg_model.summary().tables[1])  # Show only the coefficients table for brevity
    
    # Extract school type coefficients
    school_type_params = reg_model.params.filter(regex='^C\(School_Type\)')
    school_type_pvalues = reg_model.pvalues.filter(regex='^C\(School_Type\)')
    
    # Create a dataframe of school type effects
    school_effects = pd.DataFrame({
        'School_Type': [s.split('[T.')[1].rstrip(']') for s in school_type_params.index],
        'Coefficient': school_type_params.values,
        'p-value': school_type_pvalues.values
    })
    school_effects['Significant'] = school_effects['p-value'] < 0.05
    school_effects = school_effects.sort_values('Coefficient', ascending=False)
    
    print("\nSchool Type Effects (relative to reference school type):")
    print(school_effects)
    
    # Visualize school type coefficients
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='School_Type', y='Coefficient', data=school_effects, 
                     hue='Significant', palette={True: 'darkblue', False: 'lightgray'})
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('School Type Effects on Test Scores (Controlling for Zip Code, Exam Type, and Year)')
    plt.xlabel('School Type')
    plt.ylabel('Effect on Mean Score (Points)')
    plt.xticks(rotation=45)
    plt.legend(title='Statistically Significant')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error in regression analysis: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **10.** MULTILEVEL MODELING AND HIERARCHICAL EFFECTS
# ### --------------------------------------------------------------------------

# %%
# Use multilevel modeling to better understand the hierarchical nature of educational data
try:
    # Import necessary libraries for mixed-effects modeling
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    
    print("\n--- Multilevel Modeling Analysis ---")
    print("Implementing hierarchical models to account for nested data structure")
    
    # Prepare data for multilevel modeling
    # We'll use a subset of data for computational efficiency
    # Focus on top 20 zip codes by frequency
    top_zips = df_clean['Zip_Code'].value_counts().head(20).index.tolist()
    mlm_data = df_clean[df_clean['Zip_Code'].isin(top_zips)].copy()
    
    # Ensure categorical variables are properly formatted and have valid column names
    mlm_data['Zip_Code'] = mlm_data['Zip_Code'].astype('category')
    mlm_data['School_Type'] = mlm_data['School_Type'].astype('category')
    
    # Clean exam names to avoid syntax errors (replace spaces with underscores)
    mlm_data['Regents_Exam_Clean'] = mlm_data['Regents_Exam'].str.replace(' ', '_')
    mlm_data['Regents_Exam_Clean'] = mlm_data['Regents_Exam_Clean'].astype('category')
    
    print(f"\nData prepared for multilevel modeling: {len(mlm_data)} observations")
    print(f"Number of unique schools: {mlm_data['School_Name'].nunique()}")
    print(f"Number of unique zip codes: {mlm_data['Zip_Code'].nunique()}")
    
    # Model 1: Random intercepts for zip codes
    print("\nFitting Model 1: Random intercepts for zip codes")
    try:
        # Prepare model formula and fit
        md1_formula = "Mean_Score ~ C(School_Type) + C(Regents_Exam_Clean)"
        md1 = MixedLM.from_formula(md1_formula, groups="Zip_Code", data=mlm_data)
        md1_fit = md1.fit(reml=False)
        
        print("Model 1 Summary (Key Results):")
        print(md1_fit.summary().tables[1])  # Show fixed effects table
        
        # Extract variance components
        print("\nVariance Components:")
        print(f"Between-Zip Code Variance: {md1_fit.cov_re.iloc[0,0]:.4f}")
        print(f"Residual Variance: {md1_fit.scale:.4f}")
        
        # Calculate intraclass correlation (ICC)
        icc_zip = md1_fit.cov_re.iloc[0,0] / (md1_fit.cov_re.iloc[0,0] + md1_fit.scale)
        print(f"Intraclass Correlation (ZIP): {icc_zip:.4f}")
        print(f"This means approximately {icc_zip*100:.1f}% of the variance in test scores is attributable to differences between zip codes")
    except Exception as e:
        print(f"Error in Model 1: {e}")
        icc_zip = None
    
    # Model 2: Random intercepts for schools
    print("\nFitting Model 2: Random intercepts for schools")
    try:
        # Create a school identifier
        mlm_data['School_ID'] = mlm_data['School_Name'].astype('category').cat.codes
        
        # Prepare model formula and fit
        md2_formula = "Mean_Score ~ C(School_Type) + C(Regents_Exam_Clean)"
        md2 = MixedLM.from_formula(md2_formula, groups="School_ID", data=mlm_data)
        md2_fit = md2.fit(reml=False)
        
        print("Model 2 Summary (Key Results):")
        print(md2_fit.summary().tables[1])  # Show fixed effects table
        
        # Extract variance components
        print("\nVariance Components:")
        print(f"Between-School Variance: {md2_fit.cov_re.iloc[0,0]:.4f}")
        print(f"Residual Variance: {md2_fit.scale:.4f}")
        
        # Calculate intraclass correlation (ICC)
        icc_school = md2_fit.cov_re.iloc[0,0] / (md2_fit.cov_re.iloc[0,0] + md2_fit.scale)
        print(f"Intraclass Correlation (School): {icc_school:.4f}")
        print(f"This means approximately {icc_school*100:.1f}% of the variance in test scores is attributable to differences between schools")
    except Exception as e:
        print(f"Error in Model 2: {e}")
        icc_school = None
    
    # Model 3: Alternative approach to estimate variance explained by school type
    # Since the cross-classified model is problematic, we'll use a simpler approach
    print("\nModel 3: Estimating variance explained by school type")
    try:
        # Create a simple model with zip code random effects
        null_zip_formula = "Mean_Score ~ C(Regents_Exam_Clean)"
        null_zip_model = MixedLM.from_formula(null_zip_formula, groups="Zip_Code", data=mlm_data)
        null_zip_fit = null_zip_model.fit(reml=False)
        
        # Add school type to the model
        school_zip_formula = "Mean_Score ~ C(School_Type) + C(Regents_Exam_Clean)"
        school_zip_model = MixedLM.from_formula(school_zip_formula, groups="Zip_Code", data=mlm_data)
        school_zip_fit = school_zip_model.fit(reml=False)
        
        # Calculate proportion of within-zip code variance explained by school type
        var_explained = (null_zip_fit.scale - school_zip_fit.scale) / null_zip_fit.scale
        
        print("\nModel 3 Results:")
        print(f"Residual variance without School Type: {null_zip_fit.scale:.4f}")
        print(f"Residual variance with School Type: {school_zip_fit.scale:.4f}")
        print(f"Proportion of within-zip code variance explained by School Type: {var_explained:.4f}")
        print(f"This suggests School Type explains approximately {var_explained*100:.1f}% of the within-zip code variance")
    except Exception as e:
        print(f"Error in Model 3: {e}")
        var_explained = None
    
    # Visualize variance components from multilevel models
    try:
        if all(v is not None for v in [icc_zip, icc_school, var_explained]):
            # Collect variance components from different models
            variance_df = pd.DataFrame({
                'Component': ['Between-Zip Code', 'Between-School', 'Residual (Zip Code Model)', 'Residual (School Model)'],
                'Variance': [
                    md1_fit.cov_re.iloc[0,0], 
                    md2_fit.cov_re.iloc[0,0],
                    md1_fit.scale,
                    md2_fit.scale
                ],
                'Model': ['Zip Code', 'School', 'Zip Code', 'School']
            })
            
            # Plot variance components
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Component', y='Variance', hue='Model', data=variance_df, palette='viridis')
            plt.title('Variance Components from Multilevel Models')
            plt.ylabel('Variance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Compare ICC values
            icc_df = pd.DataFrame({
                'Level': ['Zip Code', 'School'],
                'ICC': [icc_zip, icc_school]
            })
            
            plt.figure(figsize=(8, 5))
            sns.barplot(x='Level', y='ICC', data=icc_df, palette='viridis')
            plt.title('Intraclass Correlation Coefficients (ICC)')
            plt.ylabel('ICC (Proportion of Variance)')
            plt.ylim(0, max(icc_zip, icc_school) * 1.2)
            
            # Add value labels
            for i, icc in enumerate([icc_zip, icc_school]):
                plt.text(i, icc + 0.01, f'{icc:.4f}', ha='center')
                
            plt.tight_layout()
            plt.show()
            
            # Now create a plot showing all the key variance measures
            summary_df = pd.DataFrame({
                'Measure': ['Zip Code ICC', 'School ICC', 'School Type Variance Explained'],
                'Value': [icc_zip, icc_school, var_explained]
            })
            
            plt.figure(figsize=(10, 6))
            bars = sns.barplot(x='Measure', y='Value', data=summary_df, palette='viridis')
            
            # Add value labels
            for i, value in enumerate([icc_zip, icc_school, var_explained]):
                plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
                
            plt.title('Comparison of Multilevel Modeling Results')
            plt.ylabel('Proportion of Variance')
            plt.tight_layout()
            plt.show()
            
            # Store variables in globals for use in other cells
            globals()['icc_zip'] = icc_zip
            globals()['icc_school'] = icc_school
            globals()['var_explained'] = var_explained
            
            print("\nMultilevel modeling results have been stored for comparison with other analyses")
    except Exception as e:
        print(f"Error in multilevel visualization: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Error in multilevel modeling analysis: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **11.** MULTILEVEL MODELING AND HIERARCHICAL EFFECT
# ### --------------------------------------------------------------------------

# %%
# Directly compare the effect sizes of school type and zip code
try:
    # Create a combined model with both factors using a clear formula
    combined_formula = 'Mean_Score ~ C(School_Type) + C(Zip_Code)'
    combined_model = ols(combined_formula, data=df_clean).fit()
    combined_anova = sm.stats.anova_lm(combined_model, typ=2)
    
    print("\nCombined ANOVA Results (School Type and Zip Code):")
    print(combined_anova)
    
    # Calculate partial eta-squared values using our utility function
    partial_eta_sq_school = calculate_partial_eta_squared(combined_anova, 0)
    partial_eta_sq_zip = calculate_partial_eta_squared(combined_anova, 1)
    
    print("\nPartial Eta-squared (Effect Size) Comparison:")
    print(f"School Type: {partial_eta_sq_school:.4f}")
    print(f"Zip Code: {partial_eta_sq_zip:.4f}")
    
    # Create dataframe for visualization
    effect_sizes = pd.DataFrame({
        'Factor': ['School Type', 'Zip Code'],
        'Partial Eta-squared': [partial_eta_sq_school, partial_eta_sq_zip]
    })
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Factor', y='Partial Eta-squared', data=effect_sizes, palette='viridis')
    plt.title('Comparison of Effect Sizes: School Type vs. Zip Code')
    plt.ylabel('Partial Eta-squared (Effect Size)')
    plt.ylim(0, max(partial_eta_sq_school, partial_eta_sq_zip) * 1.2)
    
    # Add value labels
    for i, row in enumerate(effect_sizes.itertuples()):
        plt.text(i, row._2 + 0.01, f'{row._2:.4f}', ha='center')
        
    plt.tight_layout()
    plt.show()
    
    # Calculate relative importance
    total_effect = partial_eta_sq_school + partial_eta_sq_zip
    school_relative = partial_eta_sq_school / total_effect if total_effect > 0 else 0
    zip_relative = partial_eta_sq_zip / total_effect if total_effect > 0 else 0
    
    # Create summary dataframe
    relative_importance = pd.DataFrame({
        'Factor': ['School Type', 'Zip Code'],
        'F-value': [combined_anova['F'][0], combined_anova['F'][1]],
        'p-value': [combined_anova['PR(>F)'][0], combined_anova['PR(>F)'][1]],
        'Partial Eta-squared': [partial_eta_sq_school, partial_eta_sq_zip],
        'Relative Importance': [school_relative, zip_relative]
    })
    
    print("\nRelative Importance of Factors:")
    print(relative_importance)
    
    # Add comparison with multilevel modeling results if available
    multilevel_vars = ['icc_zip', 'icc_school', 'var_explained']
    
    # Check if all required variables exist in both globals and locals
    # This handles both cases: variables defined at notebook level and within function
    vars_exist = all((var in globals() or var in locals()) for var in multilevel_vars)
    
    # If they exist, also check if they're not None
    if vars_exist:
        # Get the variables from whichever scope they exist in
        icc_zip_val = globals().get('icc_zip', locals().get('icc_zip'))
        icc_school_val = globals().get('icc_school', locals().get('icc_school'))
        var_explained_val = globals().get('var_explained', locals().get('var_explained'))
        
        # Only proceed if none of them are None
        if all(v is not None for v in [icc_zip_val, icc_school_val, var_explained_val]):
            print("\nComparison with Multilevel Modeling Results:")
            print(f"ANOVA Partial Eta-squared for School Type: {partial_eta_sq_school:.4f}")
            print(f"ANOVA Partial Eta-squared for Zip Code: {partial_eta_sq_zip:.4f}")
            print(f"Multilevel ICC for Zip Code: {icc_zip_val:.4f}")
            print(f"Multilevel ICC for School: {icc_school_val:.4f}")
            print(f"Variance explained by School Type in MLM: {var_explained_val:.4f}")
            
            # Create a more comprehensive comparison
            mlm_comparison = pd.DataFrame({
                'Method': ['ANOVA', 'ANOVA', 'Multilevel', 'Multilevel', 'Multilevel'],
                'Factor': ['School Type', 'Zip Code', 'Zip Code (ICC)', 'School (ICC)', 'School Type (VarExp)'],
                'Effect Size': [partial_eta_sq_school, partial_eta_sq_zip, icc_zip_val, icc_school_val, var_explained_val]
            })
            
            # Visualize the comparison
            plt.figure(figsize=(12, 7))
            sns.barplot(x='Factor', y='Effect Size', hue='Method', data=mlm_comparison, palette='viridis')
            plt.title('Comparison of Effect Sizes: ANOVA vs. Multilevel Modeling')
            plt.ylabel('Effect Size Measure')
            plt.xticks(rotation=45)
            plt.legend(title='Method')
            plt.tight_layout()
            plt.show()
        else:
            print("\nMultilevel modeling results not available for comparison (None values detected)")
    else:
        print("\nMultilevel modeling results not available for comparison (variables not found)")
        
except Exception as e:
    print(f"Error in effect size comparison: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **12.** TEMPORAL ANALYSIS OF DISPARITIES
# ### --------------------------------------------------------------------------

# %%
# Analyze if disparities by school type and zip code are changing over time
try:
    print("\n--- Temporal Analysis of Disparities ---")
    print("Examining whether educational disparities are increasing or decreasing over time")
    
    # Ensure 'Year' is properly formatted
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    years = sorted(df_clean['Year'].unique())
    print(f"\nYears available in dataset: {years}")
    
    # 1. School Type trends over time
    school_year_means = df_clean.groupby(['Year', 'School_Type'])['Mean_Score'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y='Mean_Score', hue='School_Type', data=school_year_means, 
                marker='o', markersize=8, linewidth=2)
    plt.title('School Type Performance Trends Over Time')
    plt.ylabel('Mean Regents Exam Score')
    plt.xlabel('Year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(years)
    plt.legend(title='School Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # 2. Calculate disparity metrics over time
    
    # Disparity Metric 1: Range between highest and lowest performing school types
    disparity_range = school_year_means.groupby('Year')['Mean_Score'].agg(lambda x: x.max() - x.min()).reset_index()
    disparity_range.columns = ['Year', 'Score_Range']
    
    # Disparity Metric 2: Standard deviation between school types
    disparity_std = school_year_means.groupby('Year')['Mean_Score'].std().reset_index()
    disparity_std.columns = ['Year', 'Score_StdDev']
    
    # Combine metrics
    disparity_metrics = pd.merge(disparity_range, disparity_std, on='Year')
    
    # Plot disparity trends
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot range on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Score Range (Max - Min)', color=color1)
    ax1.plot(disparity_metrics['Year'], disparity_metrics['Score_Range'], color=color1, 
             marker='o', markersize=8, linewidth=2, label='Score Range')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis for standard deviation
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Standard Deviation between School Types', color=color2)
    ax2.plot(disparity_metrics['Year'], disparity_metrics['Score_StdDev'], color=color2, 
             marker='s', markersize=8, linewidth=2, linestyle='--', label='Std Dev')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and grid
    plt.title('Trends in School Type Disparities Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(years)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Test if disparity trends are statistically significant
    try:
        from scipy import stats
        
        # Linear regression for trend analysis
        x = disparity_metrics['Year']
        y_range = disparity_metrics['Score_Range']
        y_std = disparity_metrics['Score_StdDev']
        
        # Regression for range
        slope_range, intercept_range, r_value_range, p_value_range, std_err_range = stats.linregress(x, y_range)
        
        # Regression for standard deviation
        slope_std, intercept_std, r_value_std, p_value_std, std_err_std = stats.linregress(x, y_std)
        
        print("\nTrend Analysis for Disparities:")
        print(f"Score Range Trend: Slope = {slope_range:.4f}, p-value = {p_value_range:.4f}")
        print(f"Standard Deviation Trend: Slope = {slope_std:.4f}, p-value = {p_value_std:.4f}")
        
        if p_value_range < 0.05:
            trend_description = "increasing" if slope_range > 0 else "decreasing"
            print(f"The range between highest and lowest performing school types is significantly {trend_description} over time.")
        else:
            print("There is no statistically significant trend in the range between school types over time.")
            
        if p_value_std < 0.05:
            trend_description = "increasing" if slope_std > 0 else "decreasing"
            print(f"The standard deviation between school types is significantly {trend_description} over time.")
        else:
            print("There is no statistically significant trend in the standard deviation between school types over time.")
    except Exception as e:
        print(f"Error in trend analysis: {e}")
    
    # 4. Effect size by year
    print("\nSchool Type Effect Size by Year:")
    effect_sizes_by_year = []
    
    for year in years:
        year_data = df_clean[df_clean['Year'] == year]
        
        try:
            # ANOVA for school type effect
            year_model = ols('Mean_Score ~ C(School_Type)', data=year_data).fit()
            year_anova = sm.stats.anova_lm(year_model, typ=2)
            
            # Calculate effect size
            ss_total_year = year_anova['sum_sq'].sum()
            eta_sq_year = year_anova['sum_sq'][0] / ss_total_year
            
            p_value_year = year_anova['PR(>F)'][0]
            
            effect_sizes_by_year.append({
                'Year': year,
                'Eta_Squared': eta_sq_year,
                'p_value': p_value_year,
                'Significant': p_value_year < 0.05
            })
            
            print(f"Year {year}: Eta-squared = {eta_sq_year:.4f}, p-value = {p_value_year:.6f}")
        except Exception as e:
            print(f"Year {year}: Error in analysis - {e}")
    
    # Plot effect sizes over time
    if effect_sizes_by_year:
        effect_size_df = pd.DataFrame(effect_sizes_by_year)
        
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x='Year', y='Eta_Squared', data=effect_size_df, 
                         palette=[('darkblue' if sig else 'lightgray') for sig in effect_size_df['Significant']])
        
        # Add data labels
        for i, row in enumerate(effect_size_df.itertuples()):
            plt.text(i, row.Eta_Squared + 0.005, f'{row.Eta_Squared:.4f}', ha='center')
        
        plt.title('School Type Effect Size (Eta-squared) by Year')
        plt.ylabel('Effect Size (Eta-squared)')
        plt.xlabel('Year')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
except Exception as e:
    print(f"Error in temporal analysis: {e}")

# %% [markdown]
# ### --------------------------------------------------------------------------
# ### **13.** TREE-BASED ANALYSIS OF EDUCATIONAL DISPARITIES
# ### --------------------------------------------------------------------------

# %%
# Use decision trees to identify the most important predictors of Regents Exam performance
try:
    # Import necessary libraries for tree-based analysis
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    print("\n--- Tree-Based Analysis of Educational Disparities ---")
    print("Using decision trees to identify and visualize the most important predictors")
    
    # Prepare data for tree-based analysis
    # Select relevant columns
    tree_data = df_clean.copy()
    
    # Remove rows with missing values
    tree_data = tree_data.dropna(subset=['Mean_Score', 'School_Type', 'Zip_Code', 'Regents_Exam'])
    
    # Limit to most common exams for more reliable results
    if 'common_exams' in globals() and len(globals()['common_exams']) > 0:
        common_exams = globals()['common_exams']
        tree_data = tree_data[tree_data['Regents_Exam'].isin(common_exams)]
    else:
        # Identify the most common exam types
        exam_counts = tree_data['Regents_Exam'].value_counts()
        common_exams = exam_counts[exam_counts > 100].index.tolist()[:5]
        tree_data = tree_data[tree_data['Regents_Exam'].isin(common_exams)]
    
    print(f"\nPrepared data for tree analysis: {len(tree_data)} rows")
    
    # Define features and target
    X = tree_data[['School_Type', 'Zip_Code', 'Regents_Exam']]
    y = tree_data['Mean_Score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")
    
    # Define preprocessing for categorical features
    categorical_features = ['School_Type', 'Zip_Code', 'Regents_Exam']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Build decision tree pipeline
    tree_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
    ])
    
    # Train the model
    print("\nTraining decision tree model...")
    tree_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = tree_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance on test set:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    # Extract feature importances
    tree_model = tree_pipeline.named_steps['regressor']
    feature_names = tree_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Map feature importance back to original categories
    importances = tree_model.feature_importances_
    
    # Group importances by category
    school_type_importance = sum(imp for name, imp in zip(feature_names, importances) if 'School_Type' in name)
    zip_code_importance = sum(imp for name, imp in zip(feature_names, importances) if 'Zip_Code' in name)
    exam_importance = sum(imp for name, imp in zip(feature_names, importances) if 'Regents_Exam' in name)
    
    # Create importances dataframe
    importance_df = pd.DataFrame({
        'Feature': ['School Type', 'Zip Code', 'Exam Type'],
        'Importance': [school_type_importance, zip_code_importance, exam_importance]
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("\nOverall feature importance:")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    # Use a proper color list instead of 'viridis' string
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', fontsize=11)
    
    plt.title('Relative Importance of Factors in Predicting Regents Exam Scores', fontsize=14)
    plt.ylabel('Importance', fontsize=12)
    plt.ylim(0, max(importance_df['Importance'])*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create a simpler tree for visualization
    print("\nCreating a simplified decision tree for visualization...")
    
    # Select a smaller subset of data for a more interpretable tree visualization
    # Focus on the most common zip codes and most common exam
    common_zips = tree_data['Zip_Code'].value_counts().head(3).index.tolist()
    most_common_exam = tree_data['Regents_Exam'].value_counts().index[0]
    
    viz_data = tree_data[
        (tree_data['Zip_Code'].isin(common_zips)) & 
        (tree_data['Regents_Exam'] == most_common_exam)
    ].copy()
    
    # Create a simple tree with just school type and zip code
    X_viz = viz_data[['School_Type', 'Zip_Code']]
    y_viz = viz_data['Mean_Score']
    
    # Use the same preprocessing approach
    categorical_viz = ['School_Type', 'Zip_Code']
    viz_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_viz)
        ])
    
    # Create a very simple tree for visualization
    X_viz_encoded = viz_preprocessor.fit_transform(X_viz)
    viz_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
    viz_tree.fit(X_viz_encoded, y_viz)
    
    # Get feature names for the plot - convert to list since sklearn requires a list
    feature_names_viz = viz_preprocessor.get_feature_names_out().tolist()
    
    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(viz_tree, feature_names=feature_names_viz, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree for Predicting {most_common_exam} Scores', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Create a more detailed feature importance breakdown
    # Look at specific schools and zip codes with highest importance
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    # Show top 10 specific features
    print("\nTop 10 specific predictors of Regents Exam performance:")
    for i, row in feature_imp.head(10).iterrows():
        # Clean up the feature name for display
        feature = row['Feature'].replace('cat__', '')
        print(f"{feature}: {row['Importance']:.4f}")
    
    # Compare tree-based results with previous analyses
    print("\nComparison with Previous Analyses:")
    
    # Calculate relative importance percentages
    total_importance = importance_df['Importance'].sum()
    school_type_pct = school_type_importance / total_importance
    zip_code_pct = zip_code_importance / total_importance
    exam_pct = exam_importance / total_importance
    
    print(f"Tree model: School Type explains {school_type_pct:.1%} of predictive power")
    print(f"Tree model: Zip Code explains {zip_code_pct:.1%} of predictive power")
    print(f"Tree model: Exam Type explains {exam_pct:.1%} of predictive power")
    
    # Compare to ANOVA results if available
    try:
        if all(var in globals() for var in ['partial_eta_sq_school', 'partial_eta_sq_zip']):
            print(f"\nANOVA: School Type effect size (partial eta-squared): {globals()['partial_eta_sq_school']:.4f}")
            print(f"ANOVA: Zip Code effect size (partial eta-squared): {globals()['partial_eta_sq_zip']:.4f}")
            
            total_eta = globals()['partial_eta_sq_school'] + globals()['partial_eta_sq_zip']
            school_eta_pct = globals()['partial_eta_sq_school'] / total_eta
            zip_eta_pct = globals()['partial_eta_sq_zip'] / total_eta
            
            print(f"ANOVA: School Type explains {school_eta_pct:.1%} of variance")
            print(f"ANOVA: Zip Code explains {zip_eta_pct:.1%} of variance")
    except:
        print("ANOVA results not available for comparison")
    
    # Compare to multilevel modeling results if available
    try:
        if all(var in globals() for var in ['icc_zip', 'icc_school', 'var_explained']):
            print(f"\nMultilevel: Zip Code ICC: {globals()['icc_zip']:.4f}")
            print(f"Multilevel: School ICC: {globals()['icc_school']:.4f}")
            print(f"Multilevel: Variance explained by School Type: {globals()['var_explained']:.4f}")
    except:
        print("Multilevel modeling results not available for comparison")
        
    # Create a comparative visualization if both ANOVA and tree results are available
    try:
        if all(var in globals() for var in ['partial_eta_sq_school', 'partial_eta_sq_zip']):
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Factor': ['School Type', 'Zip Code'],
                'Tree Importance': [school_type_pct, zip_code_pct],
                'ANOVA Effect': [school_eta_pct, zip_eta_pct]
            })
            
            # Melt for easier plotting
            comparison_melted = pd.melt(
                comparison_df, 
                id_vars='Factor', 
                value_vars=['Tree Importance', 'ANOVA Effect'],
                var_name='Method',
                value_name='Relative Importance'
            )
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Factor', y='Relative Importance', hue='Method', data=comparison_melted)
            plt.title('Comparison of Factor Importance: Tree Model vs. ANOVA', fontsize=14)
            plt.ylabel('Relative Importance (%)')
            plt.ylim(0, 1)
            
            # Format y-axis as percentage
            from matplotlib.ticker import PercentFormatter
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            
            # Add data labels
            for i, p in enumerate(plt.gca().patches):
                height = p.get_height()
                plt.text(p.get_x() + p.get_width()/2., height + 0.02,
                        f'{height:.1%}', ha='center')
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Unable to create comparison visualization: {e}")
    
    print("\nTree Analysis Conclusions:")
    print("1. The tree-based analysis provides a complementary perspective on factor importance")
    print("2. Decision trees capture non-linear relationships and interactions automatically")
    print("3. The relative importance of factors largely confirms our previous findings")
    if 'partial_eta_sq_school' in globals() and 'partial_eta_sq_zip' in globals():
        if (school_type_pct > zip_code_pct and globals()['partial_eta_sq_school'] > globals()['partial_eta_sq_zip']) or \
           (zip_code_pct > school_type_pct and globals()['partial_eta_sq_zip'] > globals()['partial_eta_sq_school']):
            print("4. There is strong agreement between tree-based and ANOVA approaches on the relative importance of factors")
        else:
            print("4. There are some differences between tree-based and ANOVA results, suggesting complex relationships in the data")
            
except Exception as e:
    print(f"Error in tree-based analysis: {e}")
    import traceback
    traceback.print_exc()

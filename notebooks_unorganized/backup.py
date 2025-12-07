plt.tight_layout()
plt.show()

# Refit with optimal k to get the final parameters
k_baseline = best_k
combined_data['p_D'] = k_baseline * combined_data['Baseline_DEM_SHARE']
combined_data['p_R'] = k_baseline * (1 - combined_data['Baseline_DEM_SHARE'])
p_D_fixed = combined_data['p_D'].values
p_R_fixed = combined_data['p_R'].values

result = minimize(neg_log_likelihood, best_params, bounds=bounds, method='L-BFGS-B', 
                  options={'maxiter': 2000, 'ftol': 1e-9})

g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt = result.x[0:5]
h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt = result.x[5:10]

# === SENSITIVITY ANALYSIS FOR WYOMING, TEXAS, AND OHIO 2016 ===
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS: WYOMING, TEXAS, OHIO 2016")
print("Testing how different spending levels affect the distribution")
print("="*80)

# Test 16 different spending scenarios
test_scenarios = [
    (0, 0, "No spending"),
    (50, 0, "D=50M, R=0"),
    (100, 0, "D=100M, R=0"),
    (200, 0, "D=200M, R=0"),
    (0, 50, "D=0, R=50M"),
    (0, 100, "D=0, R=100M"),
    (0, 200, "D=0, R=200M"),
    (50, 50, "D=50M, R=50M"),
    (100, 100, "D=100M, R=100M"),
    (200, 200, "D=200M, R=200M"),
    (100, 50, "D=100M, R=50M"),
    (50, 100, "D=50M, R=100M"),
    (200, 100, "D=200M, R=100M"),
    (100, 200, "D=100M, R=200M"),
    (200, 50, "D=200M, R=50M"),
    (50, 200, "D=50M, R=200M"),
]

# States to analyze
test_states = ['WYOMING', 'TEXAS', 'OHIO']

for state_name in test_states:
    state_data = all_states_data[(all_states_data['state'] == state_name) & 
                                 (all_states_data['year'] == 2016)]
    
    if len(state_data) == 0:
        print(f"\n{state_name} 2016 data not found, skipping...")
        continue
    
    state_row = state_data.iloc[0]
    N_state = state_row['total_partisan']
    p_D_state = state_row['p_D']
    p_R_state = state_row['p_R']
    actual_dem_share = state_row['DEM_SHARE']
    baseline_dem_share = state_row['Baseline_DEM_SHARE']
    
    print(f"\n{state_name} 2016:")
    print(f"  Population: {N_state:,.0f}")
    print(f"  Baseline Dem Share: {baseline_dem_share:.3f}")
    print(f"  Actual Dem Share: {actual_dem_share:.3f}")
    print(f"  p_D = {p_D_state:.2f}, p_R = {p_R_state:.2f}")
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    x_range = np.linspace(0, 1, 500)
    
    for idx, (D_test, R_test, label) in enumerate(test_scenarios):
        ax = axes[idx]
        
        # Calculate a and b for this scenario
        g_val = g(D_test, g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt)
        h_val = h(R_test, h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt)
        
        a_test = (1.0 / N_state) * g_val + p_D_state
        b_test = (1.0 / N_state) * h_val + p_R_state
        
        expected_share = a_test / (a_test + b_test)
        
        # Plot Beta PDF
        pdf_vals = beta.pdf(x_range, a_test, b_test)
        ax.plot(x_range * 100, pdf_vals, 'b-', lw=2.5)
        
        # Mark expected value
        ax.axvline(expected_share * 100, color='r', linestyle='--', lw=2, 
                   label=f'E[share]={expected_share:.3f}')
        
        # Mark baseline (for reference on first plot)
        if idx == 0:
            ax.axvline(baseline_dem_share * 100, color='g', linestyle=':', lw=2,
                      label=f'Baseline={baseline_dem_share:.3f}')
        
        # Mark actual 2016 result (for reference)
        ax.axvline(actual_dem_share * 100, color='orange', linestyle='-.', lw=1.5,
                  alpha=0.7, label=f'Actual={actual_dem_share:.3f}')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Democratic Vote Share (%)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{label}\na={a_test:.2f}, b={b_test:.2f}', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'{state_name} 2016: How Ad Spending Affects Distribution\n'
        f'(Using optimal k={best_k}, Population={N_state:,.0f}, Baseline Dem Share={baseline_dem_share:.3f})',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize

# Load data
county_df = pd.read_csv('countypres_2000-2024.csv')
ad_df = pd.read_csv('ad_data.csv')

# --- Calculate Democratic Vote Share by State and Year ---
state_partisan_votes = county_df[county_df['party'].isin(['DEMOCRAT', 'REPUBLICAN'])].groupby(
    ['year', 'state', 'party']
)['candidatevotes'].sum().unstack(fill_value=0).reset_index()

state_partisan_votes['DEM_SHARE'] = state_partisan_votes['DEMOCRAT'] / (
    state_partisan_votes['DEMOCRAT'] + state_partisan_votes['REPUBLICAN'])
state_partisan_votes['total_partisan'] = state_partisan_votes['DEMOCRAT'] + state_partisan_votes['REPUBLICAN']

# Sort by state and year
state_partisan_votes = state_partisan_votes.sort_values(['state', 'year']).reset_index(drop=True)

# Calculate weighted 3-year prior Democratic share (baseline partisan lean)
def past_3_inverse_weights(series):
    s = series.shift(1)
    values = np.full(series.shape[0], np.nan)
    weights = np.array([1, 1/2, 1/3])
    for i in range(3, len(s)+1):
        window = s.iloc[i-3:i].values
        if np.isnan(window).any():
            continue
        weighted_avg = np.dot(window, weights) / np.sum(weights)
        values[i-1] = weighted_avg
    return pd.Series(values, index=series.index)

state_partisan_votes['Baseline_DEM_SHARE'] = (
    state_partisan_votes.groupby('state')['DEM_SHARE'].transform(past_3_inverse_weights)
)

# Standardize state names
state_partisan_votes['state'] = state_partisan_votes['state'].str.upper()
ad_df['State'] = ad_df['State'].str.upper()

# Merge with ad spending
combined_data = ad_df.merge(
    state_partisan_votes[['year', 'state', 'total_partisan', 'DEM_SHARE', 'Baseline_DEM_SHARE']],
    left_on=['year', 'State'],
    right_on=['year', 'state'],
    how='inner'
)

# Filter out Iowa 2016
combined_data = combined_data[~((combined_data['State'] == 'IOWA') & (combined_data['year'] == 2016))]

# Remove rows with missing data (needs baseline from 3 prior elections)
combined_data = combined_data.dropna(subset=['D-ad-spending', 'R-ad-spending', 'DEM_SHARE', 'total_partisan', 'Baseline_DEM_SHARE'])

# === PARAMETER SEARCH FOR OPTIMAL k_baseline ===

print("\n" + "="*80)
print("SEARCHING FOR OPTIMAL k_baseline")
print("="*80)

k_values = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
results_by_k = []

for k_baseline in k_values:
    print(f"\nTrying k_baseline = {k_baseline}...")
    
    # Recompute p_D and p_R with this k
    combined_data['p_D'] = k_baseline * combined_data['Baseline_DEM_SHARE']
    combined_data['p_R'] = k_baseline * (1 - combined_data['Baseline_DEM_SHARE'])
    
    # Update fixed arrays
    p_D_fixed = combined_data['p_D'].values
    p_R_fixed = combined_data['p_R'].values
    
    # Optimize with this k
    result_k = minimize(neg_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', 
                        options={'maxiter': 2000, 'ftol': 1e-9})
    
    results_by_k.append({
        'k_baseline': k_baseline,
        'neg_log_likelihood': result_k.fun,
        'success': result_k.success,
        'params': result_k.x.copy()
    })
    
    print(f"  Neg Log-Likelihood: {result_k.fun:.4f}, Success: {result_k.success}")

# Find the best k
results_df = pd.DataFrame(results_by_k)
best_idx = results_df['neg_log_likelihood'].idxmin()
best_k = results_df.loc[best_idx, 'k_baseline']
best_nll = results_df.loc[best_idx, 'neg_log_likelihood']
best_params = results_df.loc[best_idx, 'params']

print("\n" + "="*80)
print("k_baseline SEARCH RESULTS")
print("="*80)
print(results_df[['k_baseline', 'neg_log_likelihood', 'success']].to_string(index=False))

print("\n" + "="*80)
print(f"OPTIMAL k_baseline = {best_k}")
print(f"Best Neg Log-Likelihood = {best_nll:.4f}")
print("="*80)

# Plot how spending functions change with k
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

D_range_viz = np.linspace(0, np.max(D_spending) * 1.2, 200)
R_range_viz = np.linspace(0, np.max(R_spending) * 1.2, 200)

# Plot g(D) for different k values
for i, row in results_df.iterrows():
    if row['success']:
        k_val = row['k_baseline']
        params = row['params']
        g1, g2, g3, g_log, g_exp = params[0:5]
        
        g_curve = g(D_range_viz, g1, g2, g3, g_log, g_exp)
        
        alpha = 0.3 if k_val != best_k else 1.0
        lw = 1.5 if k_val != best_k else 3.0
        label = f'k={k_val}' + (' (BEST)' if k_val == best_k else '')
        
        axes[0].plot(D_range_viz, g_curve, alpha=alpha, lw=lw, label=label)

axes[0].set_xlabel('Democratic Ad Spending (millions)', fontsize=11)
axes[0].set_ylabel('g(D)', fontsize=11)
axes[0].set_title('Democratic Spending Effect Across Different k', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot h(R) for different k values
for i, row in results_df.iterrows():
    if row['success']:
        k_val = row['k_baseline']
        params = row['params']
        h1, h2, h3, h_log, h_exp = params[5:10]
        
        h_curve = h(R_range_viz, h1, h2, h3, h_log, h_exp)
        
        alpha = 0.3 if k_val != best_k else 1.0
        lw = 1.5 if k_val != best_k else 3.0
        label = f'k={k_val}' + (' (BEST)' if k_val == best_k else '')
        
        axes[1].plot(R_range_viz, h_curve, alpha=alpha, lw=lw, label=label, color=f'C{i}')

axes[1].set_xlabel('Republican Ad Spending (millions)', fontsize=11)
axes[1].set_ylabel('h(R)', fontsize=11)
axes[1].set_title('Republican Spending Effect Across Different k', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Also plot likelihood vs k
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(results_df['k_baseline'], results_df['neg_log_likelihood'], 'o-', lw=2, markersize=8)
ax.axvline(best_k, color='r', linestyle='--', lw=2, label=f'Optimal k = {best_k}')
ax.set_xlabel('k_baseline', fontsize=12)
ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
ax.set_title('Model Fit Quality vs Baseline Strength Parameter', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Refit with optimal k
k_baseline = best_k
combined_data['p_D'] = k_baseline * combined_data['Baseline_DEM_SHARE']
combined_data['p_R'] = k_baseline * (1 - combined_data['Baseline_DEM_SHARE'])

print(f"\nComputed p_D range: [{combined_data['p_D'].min():.2f}, {combined_data['p_D'].max():.2f}]")
print(f"Computed p_R range: [{combined_data['p_R'].min():.2f}, {combined_data['p_R'].max():.2f}]")

# Extract arrays
D_spending = combined_data['D-ad-spending'].values
R_spending = combined_data['R-ad-spending'].values
dem_share = combined_data['DEM_SHARE'].values
total_partisan = combined_data['total_partisan'].values
p_D_fixed = combined_data['p_D'].values
p_R_fixed = combined_data['p_R'].values
states = combined_data['state'].values
years = combined_data['year'].values

# Get unique states and create state indices
unique_states = sorted(combined_data['state'].unique())
state_to_idx = {s: i for i, s in enumerate(unique_states)}

n_obs = len(D_spending)

print(f"\nNumber of observations: {n_obs}")
print(f"Number of states: {len(unique_states)}")
print(f"Democratic share range: [{np.min(dem_share):.4f}, {np.max(dem_share):.4f}]")
print(f"D spending range: [{np.min(D_spending):.2f}, {np.max(D_spending):.2f}] million")
print(f"R spending range: [{np.min(R_spending):.2f}, {np.max(R_spending):.2f}] million")

# Clip dem_share to avoid Beta distribution issues at boundaries
eps = 1e-7
dem_share_clipped = np.clip(dem_share, eps, 1 - eps)

# === MODEL: Absolute Democratic Share with FIXED State Baselines ===
# a = (1/N) * g(D) + p_D[state] (where p_D is FIXED from historical baseline)
# b = (1/N) * h(R) + p_R[state] (where p_R is FIXED from historical baseline)
# 
# CRITICAL: g(0) = 0 and h(0) = 0 so that with no spending, we get baseline!
# Remove constant terms from g and h

def g(D, g1, g2, g3, g_log, g_exp):
    """Democratic spending effect - NO CONSTANT TERM, g(0) = 0"""
    return g1 * D + g2 * D**2 + g3 * D**3 + g_log * np.log(D + 1) + g_exp * (np.exp(-D) - 1)

def h(R, h1, h2, h3, h_log, h_exp):
    """Republican spending effect - NO CONSTANT TERM, h(0) = 0"""
    return h1 * R + h2 * R**2 + h3 * R**3 + h_log * np.log(R + 1) + h_exp * (np.exp(-R) - 1)

def neg_log_likelihood(params):
    """
    params = [g1, g2, g3, g_log, g_exp, h1, h2, h3, h_log, h_exp]
    Only 10 parameters now! No constant terms, and p_D/p_R are fixed.
    """
    # Unpack parameters
    g1, g2, g3, g_log, g_exp = params[0:5]
    h1, h2, h3, h_log, h_exp = params[5:10]
    
    # Non-negativity constraints on ALL spending coefficients
    if np.any(np.array([g1, g2, g3, g_log, h1, h2, h3, h_log]) < 0):
        return 1e12
    
    # Calculate a and b for each observation
    g_vals = g(D_spending, g1, g2, g3, g_log, g_exp)
    h_vals = h(R_spending, h1, h2, h3, h_log, h_exp)
    
    # Use FIXED p_D and p_R
    a_vals = (1.0 / total_partisan) * g_vals + p_D_fixed
    b_vals = (1.0 / total_partisan) * h_vals + p_R_fixed
    
    # Ensure positive parameters
    if np.any(a_vals <= 0) or np.any(b_vals <= 0):
        return 1e12
    
    # Log-likelihood
    ll = np.sum(beta.logpdf(dem_share_clipped, a_vals, b_vals))
    
    return -ll

# Initialize parameters - only 10 now (no constant terms)
initial_params = np.array([
    1e6, 0, 0, 1e6, 0,    # g parameters (no g0)
    1e6, 0, 0, 1e6, 0     # h parameters (no h0)
])

# RELAXED Bounds: Remove minimum constraints, allow model to find natural solution
# Still keep non-negativity for interpretability (except exp term)
bounds = [
    (0, 1e13), (0, 1e12), (0, 1e10), (0, 1e13), (-1e12, 1e12),  # g (fully relaxed)
    (0, 1e13), (0, 1e12), (0, 1e10), (0, 1e13), (-1e12, 1e12)   # h (fully relaxed)
]

print("\n" + "="*80)
print("FITTING ABSOLUTE DEMOCRATIC SHARE MODEL")
print("Response: Democratic vote share [0, 1]")
print("Predictors: Spending effects g(D) and h(R) with FIXED state baselines p_D, p_R")
print("State baselines computed from weighted 3-year historical average")
print("="*80)

# Final fit with best k and best parameters from search
result = minimize(neg_log_likelihood, best_params, bounds=bounds, method='L-BFGS-B', 
                  options={'maxiter': 2000, 'ftol': 1e-9})

# Extract fitted parameters
g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt = result.x[0:5]
h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt = result.x[5:10]

print("\n" + "="*80)
print("OPTIMIZED PARAMETERS")
print("="*80)
print(f"\nDemocratic spending effect g(D):")
print(f"  g(D) = {g1_opt:.2e}*D + {g2_opt:.2e}*D^2 + {g3_opt:.2e}*D^3")
print(f"         + {g_log_opt:.2e}*log(D+1) + {g_exp_opt:.2e}*(exp(-D)-1)")
print(f"  g(0) = 0 [by construction]")

print(f"\nRepublican spending effect h(R):")
print(f"  h(R) = {h1_opt:.2e}*R + {h2_opt:.2e}*R^2 + {h3_opt:.2e}*R^3")
print(f"         + {h_log_opt:.2e}*log(R+1) + {h_exp_opt:.2e}*(exp(-R)-1)")
print(f"  h(0) = 0 [by construction]")

print(f"\nFinal negative log-likelihood: {result.fun:.4f}")
print(f"Optimization success: {result.success}")
print(f"Optimization message: {result.message}")

# Calculate fitted values
g_vals_fitted = g(D_spending, g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt)
h_vals_fitted = h(R_spending, h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt)

a_fitted = (1.0 / total_partisan) * g_vals_fitted + p_D_fixed
b_fitted = (1.0 / total_partisan) * h_vals_fitted + p_R_fixed

predicted_share = a_fitted / (a_fitted + b_fitted)

print("\n" + "="*80)
print("MODEL FIT STATISTICS")
print("="*80)
print(f"a ranges from {a_fitted.min():.2f} to {a_fitted.max():.2f}")
print(f"b ranges from {b_fitted.min():.2f} to {b_fitted.max():.2f}")

residuals = dem_share - predicted_share
print(f"\nResidual statistics:")
print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"  Max absolute error: {np.max(np.abs(residuals)):.4f}")

# Visualize fit
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Predicted vs Observed
axes[0, 0].scatter(predicted_share, dem_share, alpha=0.5, s=60, c=combined_data['Baseline_DEM_SHARE'], cmap='RdBu_r')
axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Predicted Democratic Share', fontsize=11)
axes[0, 0].set_ylabel('Observed Democratic Share', fontsize=11)
axes[0, 0].set_title('Model Fit: Predicted vs Observed', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Baseline Dem Share')

# 2. State baselines
state_baseline_summary = combined_data.groupby('state').agg({
    'p_D': 'mean',
    'p_R': 'mean'
}).reset_index()
state_baseline_summary['p_D - p_R'] = state_baseline_summary['p_D'] - state_baseline_summary['p_R']
state_baseline_summary = state_baseline_summary.sort_values('p_D - p_R')

axes[0, 1].barh(range(len(state_baseline_summary)), state_baseline_summary['p_D - p_R'], alpha=0.7)
axes[0, 1].set_yticks(range(len(state_baseline_summary)))
axes[0, 1].set_yticklabels(state_baseline_summary['state'], fontsize=7)
axes[0, 1].set_xlabel('p_D - p_R (Dem lean →)', fontsize=11)
axes[0, 1].set_title('State Partisan Baselines (Fixed from History)', fontsize=12, fontweight='bold')
axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Spending effect curves
D_range = np.linspace(0, np.max(D_spending) * 1.2, 200)
R_range = np.linspace(0, np.max(R_spending) * 1.2, 200)

g_curve = g(D_range, g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt)
h_curve = h(R_range, h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt)

axes[1, 0].plot(D_range, g_curve, 'b-', lw=2.5, label='g(D)')
axes[1, 0].set_xlabel('Democratic Ad Spending (millions)', fontsize=11)
axes[1, 0].set_ylabel('g(D)', fontsize=11)
axes[1, 0].set_title('Democratic Spending Effect', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(R_range, h_curve, 'r-', lw=2.5, label='h(R)')
axes[1, 1].set_xlabel('Republican Ad Spending (millions)', fontsize=11)
axes[1, 1].set_ylabel('h(R)', fontsize=11)
axes[1, 1].set_title('Republican Spending Effect', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- NOW CREATE STATE-BY-STATE BETA PDF PLOTS ---

# Get ALL states and years (including those without ad spending data)
all_state_votes = state_partisan_votes.copy()

# Compute baselines for ALL states
all_state_votes['p_D'] = k_baseline * all_state_votes['Baseline_DEM_SHARE']
all_state_votes['p_R'] = k_baseline * (1 - all_state_votes['Baseline_DEM_SHARE'])

# Merge with ad data (left join to keep all states)
all_states_data = all_state_votes.merge(
    ad_df[['year', 'State', 'D-ad-spending', 'R-ad-spending']],
    left_on=['year', 'state'],
    right_on=['year', 'State'],
    how='left'
)

# Fill missing ad spending with 0
all_states_data['D-ad-spending'] = all_states_data['D-ad-spending'].fillna(0)
all_states_data['R-ad-spending'] = all_states_data['R-ad-spending'].fillna(0)

# Filter to years 2012-2024
years_to_plot = [2012, 2016, 2020, 2024]
plot_data = all_states_data[all_states_data['year'].isin(years_to_plot)].copy()

# Remove rows without baselines
plot_data = plot_data.dropna(subset=['p_D', 'p_R', 'total_partisan'])

# Calculate a and b for each state-year
D_plot = plot_data['D-ad-spending'].values
R_plot = plot_data['R-ad-spending'].values
N_plot = plot_data['total_partisan'].values
p_D_plot = plot_data['p_D'].values
p_R_plot = plot_data['p_R'].values

g_plot = g(D_plot, g1_opt, g2_opt, g3_opt, g_log_opt, g_exp_opt)
h_plot = h(R_plot, h1_opt, h2_opt, h3_opt, h_log_opt, h_exp_opt)

plot_data['a'] = (1.0 / N_plot) * g_plot + p_D_plot
plot_data['b'] = (1.0 / N_plot) * h_plot + p_R_plot

# Create state plots
unique_plot_states = sorted(plot_data['state'].unique())
n_plot_states = len(unique_plot_states)
n_cols = 5
n_rows = (n_plot_states + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

x_range = np.linspace(0, 1, 500)

# Colors by year
year_colors = {2012: '#9467bd', 2016: '#1f77b4', 2020: '#ff7f0e', 2024: '#2ca02c'}

for idx, state in enumerate(unique_plot_states):
    ax = axes[idx]
    state_entries = plot_data[plot_data['state'] == state]
    
    for _, row in state_entries.iterrows():
        a_val = row['a']
        b_val = row['b']
        dem_share_actual = row['DEM_SHARE']
        year = int(row['year'])
        d_spend = row['D-ad-spending']
        r_spend = row['R-ad-spending']
        color = year_colors.get(year, 'gray')
        
        # Plot Beta PDF
        pdf_vals = beta.pdf(x_range, a_val, b_val)
        
        ax.plot(x_range * 100, pdf_vals,
                linewidth=2, color=color,
                label=f'{year} (D=${d_spend:.0f}M, R=${r_spend:.0f}M)',
                alpha=0.8)
        
        # Overlay observed Dem share
        ax.axvline(x=dem_share_actual * 100,
                   linestyle='--', linewidth=2,
                   alpha=0.7, color=color)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Democratic Vote Share (%)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{state}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_plot_states, len(axes)):
    axes[idx].axis('off')

plt.suptitle(
    'Beta PDF Predictions vs Observed Democratic Vote Share by State\n'
    '(Fixed state baselines from weighted 3-year history, 2012-2024)',
    fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("Fitted parameters saved for simulation:")
print("  g1, g2, g3, g_log, g_exp (NO g0 - g(0) = 0)")
print("  h1, h2, h3, h_log, h_exp (NO h0 - h(0) = 0)")
print("  k_baseline (for computing p_D and p_R from historical baseline)")
print("="*80)
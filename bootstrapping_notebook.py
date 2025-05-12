import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm.notebook import tqdm  # Jupyter-specific progress bar
from statsmodels.stats.multitest import multipletests

def bootstrap_individual_models(file1, file2, n_iterations=10000, alpha=0.05, metrics=None, random_seed=42):
    """
    Perform bootstrapping on individual models to get confidence intervals for each metric
    
    Parameters:
    -----------
    file1, file2 : str
        Paths to the CSV files containing the metrics
    n_iterations : int
        Number of bootstrap iterations
    alpha : float
        Significance level for confidence intervals
    metrics : list or None
        List of metrics to compare (default: all numeric columns except the first)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict: Results containing means and confidence intervals for each model and metric
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Start timing
    start_time = time.time()
    
    # Load the data
    print(f"Loading data from {file1} and {file2}")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Get model names from filenames (without extension)
    model1_name = file1.split('/')[-1].split('.')[0]
    model2_name = file2.split('/')[-1].split('.')[0]
    
    # If metrics not specified, use all numeric columns except the first (assuming first is ID)
    if metrics is None:
        # Identify numeric columns
        numeric_cols = df1.select_dtypes(include=np.number).columns.tolist()
        # Remove the first column if it exists and is numeric
        if len(numeric_cols) > 0 and numeric_cols[0] == df1.columns[0]:
            numeric_cols = numeric_cols[1:]
        metrics = numeric_cols
    
    print(f"Performing bootstrap on metrics: {metrics}")
    
    # Prepare arrays for faster computation
    data1 = {metric: df1[metric].values for metric in metrics}
    data2 = {metric: df2[metric].values for metric in metrics}
    
    n1 = len(df1)
    n2 = len(df2)
    
    # Pre-allocate arrays for results
    bootstrap_means1 = {metric: np.zeros(n_iterations) for metric in metrics}
    bootstrap_means2 = {metric: np.zeros(n_iterations) for metric in metrics}
    
    # Function to compute sample statistics (vectorized)
    def compute_stats(data_array):
        # Remove NaN values for reliable statistics
        valid_data = data_array[~np.isnan(data_array)]
        if len(valid_data) == 0:
            return np.nan
        return np.mean(valid_data)
    
    # Original means
    orig_means1 = {metric: np.nanmean(data1[metric]) for metric in metrics}
    orig_means2 = {metric: np.nanmean(data2[metric]) for metric in metrics}
    
    # Main bootstrap loop
    print(f"Running {n_iterations} bootstrap iterations...")
    for i in tqdm(range(n_iterations)):
        # Generate bootstrap sample indices
        indices1 = np.random.randint(0, n1, size=n1)
        indices2 = np.random.randint(0, n2, size=n2)
        
        # Compute means for bootstrapped samples for each metric
        for metric in metrics:
            bootstrap_sample1 = data1[metric][indices1]
            bootstrap_sample2 = data2[metric][indices2]
            
            bootstrap_means1[metric][i] = compute_stats(bootstrap_sample1)
            bootstrap_means2[metric][i] = compute_stats(bootstrap_sample2)
    
    # Compute results
    results = {
        model1_name: {},
        model2_name: {}
    }
    
    # Calculate p-values for differences
    p_values = {}
    
    for metric in metrics:
        # Confidence intervals for model 1
        sorted_means1 = np.sort(bootstrap_means1[metric])
        lower_idx = int(n_iterations * (alpha/2))
        upper_idx = int(n_iterations * (1 - alpha/2))
        
        results[model1_name][metric] = {
            'mean': orig_means1[metric],
            'ci_lower': sorted_means1[lower_idx],
            'ci_upper': sorted_means1[upper_idx]
        }
        
        # Confidence intervals for model 2
        sorted_means2 = np.sort(bootstrap_means2[metric])
        results[model2_name][metric] = {
            'mean': orig_means2[metric],
            'ci_lower': sorted_means2[lower_idx],
            'ci_upper': sorted_means2[upper_idx]
        }
        
        # Calculate difference and p-value
        bootstrap_diffs = bootstrap_means1[metric] - bootstrap_means2[metric]
        orig_diff = orig_means1[metric] - orig_means2[metric]
        
        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )
        
        # Store p-value
        p_values[metric] = p_value
    
    # Adjust p-values for multiple comparisons
    metrics_list = list(p_values.keys())
    p_values_list = [p_values[m] for m in metrics_list]
    _, p_adjusted, _, _ = multipletests(p_values_list, method='fdr_bh')
    
    # Add adjusted p-values to results
    results['p_values'] = {}
    results['p_values_adjusted'] = {}
    for i, metric in enumerate(metrics_list):
        results['p_values'][metric] = p_values[metric]
        results['p_values_adjusted'][metric] = p_adjusted[i]
    
    elapsed_time = time.time() - start_time
    print(f"Bootstrap analysis completed in {elapsed_time:.2f} seconds")
    
    # Add metadata
    results['model1_name'] = model1_name
    results['model2_name'] = model2_name
    results['metrics'] = metrics
    
    return results

def plot_bootstrap_model_comparison(results, figsize=(14, 10), plot_title=None):
    """
    Plot confidence intervals for each metric and model
    
    Parameters:
    -----------
    results : dict
        Results dictionary from bootstrap_individual_models
    figsize : tuple
        Figure size
    plot_title : str or None
        Optional plot title
    
    Returns:
    --------
    matplotlib figure
    """
    metrics = results['metrics']
    model1_name = results['model1_name']
    model2_name = results['model2_name']
    
    # Set up the figure and axes
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Set a title for the overall figure if provided
    if plot_title:
        fig.suptitle(plot_title, fontsize=16)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e']  # First color is for model1, second for model2
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Data for plotting
        models = [model1_name, model2_name]
        means = [results[model1_name][metric]['mean'], results[model2_name][metric]['mean']]
        ci_lower = [results[model1_name][metric]['ci_lower'], results[model2_name][metric]['ci_lower']]
        ci_upper = [results[model1_name][metric]['ci_upper'], results[model2_name][metric]['ci_upper']]
        
        # Calculate error bar sizes
        yerr_lower = [m - l for m, l in zip(means, ci_lower)]
        yerr_upper = [u - m for m, u in zip(means, ci_upper)]
        yerr = [yerr_lower, yerr_upper]
        
        # Plot
        x = np.arange(len(models))
        ax.bar(x, means, width=0.6, color=colors, alpha=0.7)
        ax.errorbar(x, means, yerr=yerr, fmt='none', color='black', capsize=5)
        
        # Add value labels
        for j, value in enumerate(means):
            ax.text(j, value + yerr_upper[j] + 0.01 * max(means), 
                    f"{value:.4f}", ha='center', va='bottom', fontsize=10)
        
        # Add p-value
        p_val = results['p_values'][metric]
        p_adj = results['p_values_adjusted'][metric]
        
        # Determine significance
        sig_str = ""
        if p_adj < 0.001:
            sig_str = "***"
        elif p_adj < 0.01:
            sig_str = "**"
        elif p_adj < 0.05:
            sig_str = "*"
        
        # Add p-value annotation
        ax.text(0.5, 0.9, f"p={p_val:.4f} (adj: {p_adj:.4f}) {sig_str}", 
                transform=ax.transAxes, ha='center', fontsize=10)
        
        # Customize plot
        ax.set_title(f"{metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limit starting from 0 or slightly below if values are negative
        y_min = min(0, min(ci_lower) * 1.1)
        y_max = max(ci_upper) * 1.1
        ax.set_ylim(y_min, y_max)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if plot_title:
        plt.subplots_adjust(top=0.95)  # Make space for the title
        
    return fig

def efficient_bootstrap_comparison(file1, file2, n_iterations=1000, alpha=0.05, metrics=None, random_seed=42):
    """
    Efficiently perform bootstrapping to compare two result sets
    
    Parameters:
    -----------
    file1, file2 : str
        Paths to the CSV files containing the metrics
    n_iterations : int
        Number of bootstrap iterations
    alpha : float
        Significance level for confidence intervals
    metrics : list or None
        List of metrics to compare (default: all numeric columns except the first)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict: Results containing differences, confidence intervals, and p-values
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Start timing
    start_time = time.time()
    
    # Load the data
    print(f"Loading data from {file1} and {file2}")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # If metrics not specified, use all numeric columns except the first (assuming first is ID)
    if metrics is None:
        # Identify numeric columns
        numeric_cols = df1.select_dtypes(include=np.number).columns.tolist()
        # Remove the first column if it exists and is numeric
        if len(numeric_cols) > 0 and numeric_cols[0] == df1.columns[0]:
            numeric_cols = numeric_cols[1:]
        metrics = numeric_cols
    
    print(f"Performing bootstrap comparison on metrics: {metrics}")
    
    # Prepare arrays for faster computation
    data1 = {metric: df1[metric].values for metric in metrics}
    data2 = {metric: df2[metric].values for metric in metrics}
    
    n1 = len(df1)
    n2 = len(df2)
    
    # Pre-allocate arrays for results
    bootstrap_diffs = {metric: np.zeros(n_iterations) for metric in metrics}
    
    # Function to compute sample statistics (vectorized)
    def compute_stats(data_array):
        # Remove NaN values for reliable statistics
        valid_data = data_array[~np.isnan(data_array)]
        if len(valid_data) == 0:
            return np.nan
        return np.mean(valid_data)
    
    # Main bootstrap loop
    print(f"Running {n_iterations} bootstrap iterations...")
    for i in tqdm(range(n_iterations)):
        # Generate bootstrap sample indices
        indices1 = np.random.randint(0, n1, size=n1)
        indices2 = np.random.randint(0, n2, size=n2)
        
        # Compute differences between bootstrapped samples for each metric
        for metric in metrics:
            bootstrap_sample1 = data1[metric][indices1]
            bootstrap_sample2 = data2[metric][indices2]
            
            mean1 = compute_stats(bootstrap_sample1)
            mean2 = compute_stats(bootstrap_sample2)
            
            bootstrap_diffs[metric][i] = mean1 - mean2
    
    # Compute results
    results = {}
    for metric in metrics:
        # Original difference
        orig_diff = np.mean(data1[metric]) - np.mean(data2[metric])
        
        # Sort bootstrap differences for percentile method
        sorted_diffs = np.sort(bootstrap_diffs[metric])
        
        # Compute confidence intervals
        lower_idx = int(n_iterations * (alpha/2))
        upper_idx = int(n_iterations * (1 - alpha/2))
        ci_lower = sorted_diffs[lower_idx]
        ci_upper = sorted_diffs[upper_idx]
        
        # Approximate p-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs[metric] >= 0),
            np.mean(bootstrap_diffs[metric] <= 0)
        )
        
        results[metric] = {
            'original_diff': orig_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': (ci_lower > 0) or (ci_upper < 0)  # CI doesn't contain 0
        }
    
    elapsed_time = time.time() - start_time
    print(f"Bootstrap analysis completed in {elapsed_time:.2f} seconds")
    
    return results

def plot_difference_bootstrap_results(results, title="Bootstrap Comparison Results"):
    """
    Plot the bootstrap comparison results with confidence intervals
    
    Parameters:
    -----------
    results : dict
        Results dictionary from efficient_bootstrap_comparison
    title : str
        Plot title
    """
    metrics = list(results.keys())
    diffs = [results[m]['original_diff'] for m in metrics]
    ci_lower = [results[m]['ci_lower'] for m in metrics]
    ci_upper = [results[m]['ci_upper'] for m in metrics]
    
    # Calculate error bar sizes
    yerr_lower = [d - l for d, l in zip(diffs, ci_lower)]
    yerr_upper = [u - d for d, u in zip(diffs, ci_upper)]
    yerr = [yerr_lower, yerr_upper]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot differences
    x = np.arange(len(metrics))
    ax.errorbar(x, diffs, yerr=yerr, fmt='o', capsize=5, 
                ecolor='black', color='blue', markersize=8)
    
    # Add zero line
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Difference (Model 1 - Model 2)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    
    # Add significance indicators
    for i, metric in enumerate(metrics):
        if results[metric]['significant']:
            ax.text(i, diffs[i], '*', fontsize=20, 
                    horizontalalignment='center', color='green')
    
    # Add p-values
    for i, metric in enumerate(metrics):
        p_val = results[metric]['p_value']
        ax.text(i, diffs[i] + yerr_upper[i] + 0.01, f'p={p_val:.3f}', 
                horizontalalignment='center', fontsize=8)
    
    plt.tight_layout()
    return fig

# Example usage in a Jupyter notebook:
"""
# Import this notebook as a module
%run bootstrap_notebook.py

# Run individual model bootstrap
results = bootstrap_individual_models(
    "base_mean_dict.csv", 
    "ft_mean_dict.csv",
    n_iterations=10000
)

# Plot model comparisons with confidence intervals
fig = plot_bootstrap_model_comparison(results, 
                                     plot_title="Model Performance Comparison with 95% CIs")
plt.savefig("model_comparison.png", dpi=300)

# Optionally, also run difference-based bootstrapping
diff_results = efficient_bootstrap_comparison(
    "base_mean_dict.csv", 
    "ft_mean_dict.csv",
    n_iterations=10000
)

# Plot differences with confidence intervals
fig2 = plot_difference_bootstrap_results(diff_results, 
                                        title="Difference Between Models (with 95% CIs)")
plt.savefig("model_differences.png", dpi=300)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def load_dataset(file_path):
    """Load the dataset from a Parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None



def inspect_structure(df):
    """Inspect the basic structure of the DataFrame."""
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- First 3 Rows ---")
    print(df.head(3))
    print("\n--- Column Names ---")
    print(df.columns.tolist())
    print("\n--- Value Counts for Label Column ---")
    print(df['label'].value_counts())



def inspect_actual_values(df):
    """Inspect one sample entry to understand structure."""
    print("\n--- Inspecting actual values ---")
    first_entry = df['Combined'].iloc[0]
    print("Entry type:", type(first_entry))

    if isinstance(first_entry, np.ndarray):
        print("Entry shape:", first_entry.shape)
        if first_entry.ndim == 1:
            print("Entry example (first value):", first_entry[:1])
        else:
            print("Entry has unexpected number of dimensions.")
    elif isinstance(first_entry, list):
        print("Entry length:", len(first_entry))
        if isinstance(first_entry[0], np.ndarray):
            print("First inner array shape:", first_entry[0].shape)
            print("Entry example (first array):", first_entry[0])
        else:
            print("First entry is not a NumPy array.")
    else:
        print("Unexpected entry format.")



def plot_label_distribution(df):
    label_counts = df['label'].value_counts().sort_index()
    print("\n--- Label Counts ---")
    print(label_counts)

    # Prepare a small DataFrame for seaborn
    df_counts = label_counts.reset_index()
    df_counts.columns = ['label', 'count']

    # Bar plot with explicit hue
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=df_counts,
        x='label',
        y='count',
        hue='label',
        palette='pastel',
        dodge=False
    )
    ax.get_legend().remove()  # remove the redundant legend
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Pie chart (unchanged)
    plt.figure(figsize=(5, 5))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    plt.title("Label Distribution (Pie Chart)")
    plt.tight_layout()
    plt.show()




def plot_sequence_lengths(df):
    lengths = df['Combined'].apply(lambda seq: len(seq))
    print("\n--- Sequence Length Stats ---")
    print(lengths.describe())

    plt.figure(figsize=(8, 4))
    sns.histplot(lengths, bins=20, kde=True)
    plt.title("Distribution of Sequence Lengths (T)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def compute_feature_statistics(df):
    """
    Compute mean, std, min, max per feature (averaged over time for each sample).
    
    Returns a DataFrame of shape (126, 4).
    """
    # For each sample, compute time‑averaged feature vector
    feature_matrix = np.stack([entry.mean(axis=0) for entry in df['Combined']])
    stats = {
        'mean': feature_matrix.mean(axis=0),
        'std': feature_matrix.std(axis=0),
        'min': feature_matrix.min(axis=0),
        'max': feature_matrix.max(axis=0),
    }
    return pd.DataFrame(stats, index=[f'feat_{i}' for i in range(feature_matrix.shape[1])])

def plot_feature_distributions(df, feature_idxs=None):
    """
    Plot histograms and boxplots for selected features.
    feature_idxs: list of feature indices (0–125). Defaults to [0,1,2,3,4].
    """
    if feature_idxs is None:
        feature_idxs = list(range(5))
    # build a long-form DataFrame for seaborn
    records = []
    for idx, entry in enumerate(df['Combined']):
        avg = entry.mean(axis=0)
        for f in feature_idxs:
            records.append({'sample': idx, 'feature': f, 'value': avg[f], 'label': df.at[idx, 'label']})
    plot_df = pd.DataFrame(records)
    
    for f in feature_idxs:
        # histogram
        plt.figure(figsize=(6,4))
        sns.histplot(
            plot_df.query("feature==@f"),
            x='value',
            hue='label',
            kde=True,
            element='step',
            common_norm=False
        )
        plt.title(f'Feature {f} Distribution by Label')
        plt.xlabel('Average Value')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
        
        # boxplot with explicit hue and legend removal
        plt.figure(figsize=(4,6))
        ax = sns.boxplot(
            x='label',
            y='value',
            hue='label',
            data=plot_df.query("feature==@f"),
            palette='pastel',
            dodge=False
        )
        ax.get_legend().remove()   # remove the redundant legend
        plt.title(f'Feature {f} Boxplot by Label')
        plt.xlabel('Label')
        plt.ylabel('Average Value')
        plt.tight_layout()
        plt.show()

def plot_pca_tsne(df, use='pca', n_components=2):
    """
    Perform dimensionality reduction on time‑averaged feature vectors.
    use: 'pca' or 'tsne'
    """
    X = np.stack([entry.mean(axis=0) for entry in df['Combined']])
    y = df['label'].values

    if use=='pca':
        dr = PCA(n_components=n_components, random_state=0)
        X2 = dr.fit_transform(X)
        title = 'PCA'
    else:
        dr = TSNE(n_components=n_components, random_state=0)
        X2 = dr.fit_transform(X)
        title = 't-SNE'

    plt.figure(figsize=(6,6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y, palette='bright', legend='full')
    plt.title(f'{title} of Samples (time‑averaged features)')
    plt.xlabel(f'{title} Component 1')
    plt.ylabel(f'{title} Component 2')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.show()



# --- Temporal Dynamics ---

def plot_random_samples(df, feature_idxs=[0,1,2], n_samples=3, seed=0):
    """
    Plot raw time‑series (line plots) for a few random samples and selected features.
    """
    np.random.seed(seed)
    samples = np.random.choice(len(df), size=n_samples, replace=False)
    for idx in samples:
        seq = np.stack(df['Combined'].iloc[idx])  # (T, 126)
        T = seq.shape[0]
        plt.figure(figsize=(8, 3))
        for f in feature_idxs:
            plt.plot(range(T), seq[:, f], label=f"feat {f}")
        plt.title(f"Sample {idx} (Label {df.at[idx, 'label']})")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend(ncol=len(feature_idxs), bbox_to_anchor=(1,1), loc="upper left")
        plt.tight_layout()
        plt.show()

def compare_label_dynamics(df, feature_idx=0):
    """
    Overlay average time‑series for Positive vs Negative samples for one feature.
    """
    # stack into arrays
    pos = [np.stack(x)[:, feature_idx] for x, y in zip(df['Combined'], df['label']) if y==1]
    neg = [np.stack(x)[:, feature_idx] for x, y in zip(df['Combined'], df['label']) if y==0]
    # pad/truncate to same length if needed
    T_max = max(max(len(p) for p in pos), max(len(n) for n in neg))
    pos_mat = np.array([np.pad(p, (0, T_max-len(p)), 'edge') for p in pos])
    neg_mat = np.array([np.pad(n, (0, T_max-len(n)), 'edge') for n in neg])
    avg_pos = pos_mat.mean(axis=0)
    avg_neg = neg_mat.mean(axis=0)

    plt.figure(figsize=(8,4))
    plt.plot(avg_pos, label="Positive", linewidth=2)
    plt.plot(avg_neg, label="Negative", linewidth=2)
    plt.title(f"Average Dynamics for Feature {feature_idx}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_signals(df, feature_idxs=[0,1,2]):
    """
    Plot average time‑series for a small set of features, separated by label.
    """
    for f in feature_idxs:
        compare_label_dynamics(df, feature_idx=f)


# --- Correlation & Redundancy ---

def feature_correlation_heatmap(df):
    """
    Compute correlation matrix on time‑averaged features and plot heatmap.
    """
    # time‑average each sample → (n_samples, 126)
    X = np.stack([np.stack(x).mean(axis=0) for x in df['Combined']])
    corr = pd.DataFrame(X).corr()
    plt.figure(figsize=(8,8))
    sns.heatmap(corr, cmap='vlag', center=0, square=True, cbar_kws={"shrink":.5})
    plt.title("Feature‑to‑Feature Correlation (time‑averaged)")
    plt.tight_layout()
    plt.show()
    return corr

def identify_low_variance_features(df, threshold=1e-3):
    """
    Return list of features whose variance (over all samples and time) is below threshold.
    """
    # flatten across samples and time: shape (n_samples*T, 126)
    all_vals = np.vstack([np.stack(x) for x in df['Combined']])
    variances = all_vals.var(axis=0)
    low_var = np.where(variances < threshold)[0].tolist()
    print(f"Low‑variance features (var < {threshold}): {low_var}")
    return low_var


# --- Step 6: Missing & Outlier Handling ---

def find_nan_or_constant_sequences(df):
    """
    Identify any sample sequences containing NaNs or zero variance (constant).
    """
    nan_idxs = []
    const_idxs = []
    for i, x in enumerate(df['Combined']):
        arr = np.stack(x)
        if np.isnan(arr).any():
            nan_idxs.append(i)
        if np.allclose(arr.var(axis=0), 0):
            const_idxs.append(i)
    print(f"Samples with NaNs: {nan_idxs}")
    print(f"Samples with constant signal: {const_idxs}")
    return nan_idxs, const_idxs

def plot_outlier_feature_distribution(df, feature_idx=0):
    """
    Boxplot of all values for one feature across all samples to visualize outliers.
    """
    # collect all frames
    vals = np.hstack([np.stack(x)[:, feature_idx] for x in df['Combined']])
    plot_df = pd.DataFrame({
        'value': vals,
        'group': 'all'
    })

    plt.figure(figsize=(4,6))
    ax = sns.boxplot(
        x='group',
        y='value',
        hue='group',
        data=plot_df,
        palette='pastel',
        dodge=False
    )
    # only remove legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_xlabel('')  # no x-label
    plt.title(f"Outlier Detection for Feature {feature_idx}")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()



def main():
    # -------------------------------------------
    # Step-by-Step Dataset Analysis Overview
    # -------------------------------------------
    # 1. Load dataset from Parquet
    # 2. Inspect structure: shape, dtypes, column names
    # 3. Inspect sample values: array shapes, types, example content
    # 4. Plot label distribution (class balance)
    # 5. Plot sequence length distribution (frames per sample)
    # 6. Compute feature-wise statistics: mean, std, min, max
    # 7. Plot feature distributions (boxplots) for selected features
    # 8. Dimensionality reduction (PCA and t-SNE) for visual separation
    # 9. Plot random sample sequences over time for selected features
    # 10. Plot average signal over time (per feature, per label)
    # 11. Compute and visualize feature–feature correlation matrix
    # 12. Identify and print top correlated feature pairs
    # 13. Identify low-variance features (potentially redundant)
    # 14. Detect samples with NaNs or constant values
    # 15. Plot boxplot of selected feature to visualize outliers
    # -------------------------------------------

    path = "Augmented_Dataset.parquet"
    df = load_dataset(path)
    
    if df is None:
        return

    # Step 2–3
    inspect_structure(df)
    inspect_actual_values(df)

    # Step 4–5
    plot_label_distribution(df)
    plot_sequence_lengths(df)

    # Step 6–8
    stats_df = compute_feature_statistics(df)
    print("\n--- Feature Statistics (first 10 features) ---")
    print(stats_df.head(10))
    plot_feature_distributions(df, feature_idxs=[0,1,2])
    plot_pca_tsne(df, use='pca')
    plot_pca_tsne(df, use='tsne')

    # Step 9–10
    plot_random_samples(df, feature_idxs=[0,1,2], n_samples=3)
    plot_average_signals(df, feature_idxs=[0,1,2])

    # Step 11–13
    corr_matrix = feature_correlation_heatmap(df)
    print("\n--- Correlation Matrix (first 5 features) ---")
    print(corr_matrix.iloc[:5, :5])
    flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    top = flat.abs().unstack().sort_values(ascending=False).dropna().head(10)
    print("\n--- Top 10 feature–feature correlations ---")
    print(top)
    low_var_feats = identify_low_variance_features(df, threshold=1e-3)
    print(f"\nLow‑variance features: {low_var_feats}")

    # Step 14–15
    nan_idxs, const_idxs = find_nan_or_constant_sequences(df)
    print(f"\nSamples with NaNs: {nan_idxs}")
    print(f"Samples with constant signal: {const_idxs}")
    plot_outlier_feature_distribution(df, feature_idx=0)

if __name__ == "__main__":
    main()

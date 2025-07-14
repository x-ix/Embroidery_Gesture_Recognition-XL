import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
rng = np.random.default_rng(seed=42)  # Seed once per runtime



def retrieve_df(file_path):
    df = pd.read_parquet(file_path)
    return df



def print_head(df, n=5):
    print(df.head(n))



def first_entry(df):
    # Print the shape of the first item in 'Top-View'
    first_entry = df['Top-View'].iloc[0]
    try:
        first_item = np.stack(first_entry)
        print("Shape of first item in 'Top-View':", first_item.shape)
    except Exception as e:
        print("Failed to stack first item. Error:", e)



def largest_matrix_check(df):
    # Determine the largest shape in views
    max_shape = (0, 0)
    for view in ['Top-View', 'Bottom-View']:
        for entry in df[view]:
            try:
                arr = np.stack(entry)
                shape = arr.shape
                if shape[0] > max_shape[0] or (shape[0] == max_shape[0] and shape[1] > max_shape[1]):
                    max_shape = shape
            except Exception as e:
                print(f"Skipping entry due to error: {e}")
    print("Largest shape is:", max_shape)



def identify_mismatching_pairs(df):
    outliers = []
    for idx, row in df.iterrows():
        try:
            top_shape = np.stack(row['Top-View']).shape
        except Exception as e:
            top_shape = f"Error: {e}"
        try:
            bottom_shape = np.stack(row['Bottom-View']).shape
        except Exception as e:
            bottom_shape = f"Error: {e}"
        if top_shape != bottom_shape:
            outliers.append(idx)
    return outliers



def drop_outliers(df, outliers):
    # drops rows where outliers have been detected
    if len(outliers) != 0:
        for i in range(len(outliers)):
                df.drop(index=[outliers[i]], inplace=True, errors='ignore')
    return df



def concatenate_views(df):
    concatenated = []
    indices = []
    for idx, row in df.iterrows():
        try:
            top = np.stack(row['Top-View'])
            bottom = np.stack(row['Bottom-View'])
            if top.shape[0] == bottom.shape[0]:
                combined = np.concatenate([top, bottom], axis=1)
                concatenated.append(combined)
                indices.append(idx)
            else:
                print(f"Skipping index {idx} due to shape mismatch: {top.shape} vs {bottom.shape}")
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
    return pd.DataFrame({'Combined': concatenated}, index=indices)



def print_combined_sizes(df):
    for idx, matrices in df['Combined'].items():
        try:
            arr = np.stack(matrices)
            print(f"Index {idx}: shape = {arr.shape}")
        except Exception as e:
            print(f"Index {idx}: failed to stack – {e}")



def jitter(X, sigma=0.01):
    X_jit = X + rng.normal(loc=0., scale=sigma, size=X.shape)
    return np.clip(X_jit, 0, 1)



def scaling(X, scale_range=(0.98, 1.02)):  # tighter range
    factor = rng.uniform(*scale_range)
    X_scaled = X * factor
    return np.clip(X_scaled, 0, 1)



def time_warp(X, max_warp=0.15):  # warp in time only, resample to original size
    T, D = X.shape
    warp_factor = rng.uniform(1 - max_warp, 1 + max_warp)
    new_T = max(5, int(T * warp_factor))
    interp = interp1d(np.linspace(0, 1, T), X, axis=0, kind='linear', fill_value="extrapolate")
    X_warped = interp(np.linspace(0, 1, new_T))

    # Resample back to original length
    interp_back = interp1d(np.linspace(0, 1, new_T), X_warped, axis=0, kind='linear', fill_value="extrapolate")
    return np.clip(interp_back(np.linspace(0, 1, T)), 0, 1)



def magnitude_warp(X, sigma=0.1, knot=4):  # smaller sigma
    T, D = X.shape
    orig_steps = np.linspace(0, T - 1, knot + 2)
    random_factors = rng.normal(loc=1.0, scale=sigma, size=(knot + 2, D))
    warp = np.array([
        CubicSpline(orig_steps, random_factors[:, d])(np.arange(T)) for d in range(D)
    ]).T
    X_warped = X * warp
    return np.clip(X_warped, 0, 1)



def noise_injection(X, sigma=0.03):  # gentler noise
    drift = rng.normal(loc=0.0, scale=sigma, size=(1, X.shape[1]))
    trend = np.linspace(0, 1, X.shape[0]).reshape(-1, 1)
    injected = X + trend * drift
    return np.clip(injected, 0, 1)



def augment_within_bounds(X, methods, soft_clip_margin=0.1):
    """
    Augments a sequence X with given methods.
    Keeps values mostly within [min−margin, max+margin] of the original sample.
    """
    original_min = X.min()
    original_max = X.max()
    lower_bound = original_min - soft_clip_margin
    upper_bound = original_max + soft_clip_margin

    X_aug = X.copy()
    for method in methods:
        X_aug = method(X_aug)

    # Soft bounding without hard clipping
    return np.clip(X_aug, lower_bound, upper_bound)



def augment_instance(X):
    methods = [lambda x: jitter(x, sigma=0.015),
               lambda x: scaling(x, scale_range=(0.97, 1.03)),
               lambda x: magnitude_warp(x, sigma=0.08),
               lambda x: noise_injection(x, sigma=0.02),
               lambda x: time_warp(x, max_warp=0.1)]
    
    return augment_within_bounds(X, methods)



def fill(concat_df, label):
    augmented_rows = []
    for _ in range(75 - len(concat_df)):
        orig = concat_df['Combined'].sample(n=1, random_state=rng.integers(0, 1_000_000)).values[0]
        aug = augment_instance(np.array(orig))
        augmented_rows.append([aug])

    # Create new DataFrame with augmented data
    df_aug = pd.DataFrame(augmented_rows, columns=['Combined'])
    
    df = pd.concat([concat_df, df_aug], ignore_index=True)
    df['label'] = label
    return df



def main():
    df = retrieve_df(r"results_(positive)\landmarks_(positive).parquet")
    df2 = retrieve_df(r"results_(negative)\landmarks_(negative).parquet")
    #first_entry(df)
    #first_entry(df2)
    largest_matrix_check(df)
    largest_matrix_check(df2)
    df_outliers = identify_mismatching_pairs(df)
    df2_outliers = identify_mismatching_pairs(df2)
    cleaned_df = drop_outliers(df, df_outliers)
    cleaned_df2 = drop_outliers(df2, df2_outliers)
    df_concatenated = concatenate_views(cleaned_df)
    df2_concatenated = concatenate_views(cleaned_df2)

    augmented_df = fill(df_concatenated, 1)
    augmented_df2 = fill(df2_concatenated, 0)

    final_df = pd.concat([augmented_df, augmented_df2], ignore_index=True)
    final_df['Combined'] = final_df['Combined'].apply(lambda x: np.array(x, dtype=np.float32))
    final_df['Combined'] = final_df['Combined'].apply(lambda x: x.tolist())
    print_head(final_df, 80)
    print(f"Dataset length: {len(final_df)}")

    final_df.to_parquet("Augmented_Dataset.parquet", index=False)


if __name__ == '__main__':
    main()





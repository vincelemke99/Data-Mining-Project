import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

def print_size_status(df, step_name):
    print(f"\n=== Size Status After {step_name} ===")
    print(f"Total rows: {len(df)}")
    print("Size value counts:")
    print(df['size'].value_counts(dropna=False))
    nan_count = df['size'].isna().sum()
    print(f"NaN count: {nan_count}")
    if nan_count > 0:
        print("\nSample of rows with NaN size:")
        print(df[df['size'].isna()].head()[['manufacturer', 'model', 'year', 'fuel', 'type', 'cylinders']])

# Step 1: Read the cleaned cylinders data
df = pd.read_csv('Project/data/vehicles_cylinders_cleaned.csv')
print_size_status(df, "Initial Load")

# Step 2: Show missing values for each column
print("\nMissing values per column:")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_percent': missing_percent})
print(missing_df[missing_df.missing_count > 0].sort_values(by='missing_count', ascending=False))

# Step 3: Analyze patterns in existing size values
print("\n=== Analysis of Existing Size Values ===")

# 3.1: Size distribution by vehicle type
print("\nSize distribution by vehicle type:")
print(pd.crosstab(df['type'], df['size'], margins=True))

# 3.2: Size distribution by number of cylinders
print("\nSize distribution by number of cylinders:")
print(pd.crosstab(df['cylinders'], df['size'], margins=True))

# 3.3: Size distribution by top 10 manufacturers
top_manufacturers = df['manufacturer'].value_counts().head(10).index
print("\nSize distribution by top 10 manufacturers:")
print(pd.crosstab(df[df['manufacturer'].isin(top_manufacturers)]['manufacturer'], 
                 df[df['manufacturer'].isin(top_manufacturers)]['size'], 
                 margins=True))

# 3.4: Analyze missing size values by type
print("\nMissing size values by vehicle type:")
missing_by_type = df[df['size'].isna()]['type'].value_counts()
print(missing_by_type)
print("\nPercentage of missing size by type:")
print((missing_by_type / df['type'].value_counts() * 100).round(2))

# 3.5: Analyze missing size values by cylinders
print("\nMissing size values by number of cylinders:")
missing_by_cylinders = df[df['size'].isna()]['cylinders'].value_counts()
print(missing_by_cylinders)
print("\nPercentage of missing size by cylinders:")
print((missing_by_cylinders / df['cylinders'].value_counts() * 100).round(2))

# 3.6: Sample of rows with known size values
print("\nSample of rows with known size values:")
print(df[df['size'].notna()].head(10)[['manufacturer', 'model', 'year', 'fuel', 'type', 'cylinders', 'size']])

# Step 4: Impute missing 'size' values using strong mode patterns
mode_threshold = 0.8
min_group_size = 10
impute_count = 0

groups = df.groupby(['type', 'cylinders'])
for (type_val, cyl_val), group in groups:
    known = group['size'].dropna()
    if len(known) >= min_group_size:
        mode_val = known.mode()
        if len(mode_val) == 1 and (known.value_counts().iloc[0] / len(known)) >= mode_threshold:
            mask = (df['type'] == type_val) & (df['cylinders'] == cyl_val) & (df['size'].isna())
            n_to_impute = mask.sum()
            if n_to_impute > 0:
                df.loc[mask, 'size'] = mode_val.iloc[0]
                impute_count += n_to_impute
                print(f"Imputed {n_to_impute} missing 'size' values for type '{type_val}' and cylinders '{cyl_val}' with '{mode_val.iloc[0]}'")

print(f"\nTotal 'size' values imputed using strong mode patterns: {impute_count}")
print_size_status(df, "Strong Mode Imputation")

# Step 5: Profile remaining missing 'size' values
print("\n=== Profiling Remaining Missing 'size' Values ===")
missing_size = df[df['size'].isna()]
print(f"Rows still missing 'size': {len(missing_size)}")
print("\nTop 10 manufacturers with missing 'size':")
print(missing_size['manufacturer'].value_counts().head(10))
print("\nTop 10 models with missing 'size':")
print(missing_size['model'].value_counts().head(10))
print("\nTop 10 (manufacturer, model) pairs with missing 'size':")
print(missing_size.groupby(['manufacturer', 'model']).size().sort_values(ascending=False).head(10))

# Step 6: Impute missing 'size' using strong mode patterns for manufacturer, model
mode_threshold = 0.8
min_group_size = 10
impute_count = 0

groups = df.groupby(['manufacturer', 'model'])
for (manu, model), group in groups:
    known = group['size'].dropna()
    if len(known) >= min_group_size:
        mode_val = known.mode()
        if len(mode_val) == 1 and (known.value_counts().iloc[0] / len(known)) >= mode_threshold:
            mask = (df['manufacturer'] == manu) & (df['model'] == model) & (df['size'].isna())
            n_to_impute = mask.sum()
            if n_to_impute > 0:
                df.loc[mask, 'size'] = mode_val.iloc[0]
                impute_count += n_to_impute
                print(f"Imputed {n_to_impute} missing 'size' values for manufacturer '{manu}', model '{model}' with '{mode_val.iloc[0]}'")

print(f"\nTotal 'size' values imputed using (manufacturer, model) strong mode patterns: {impute_count}")
print_size_status(df, "(Manufacturer, Model) Mode Imputation")

# Step 8: Study Patterns in Remaining Missing 'size' Values
print("\n=== Studying Patterns in Remaining Missing 'size' Values ===")
missing_size = df[df['size'].isna()]
print(f"Rows still missing 'size': {len(missing_size)}")
print("\nTop 10 manufacturers with missing 'size':")
print(missing_size['manufacturer'].value_counts().head(10))
print("\nTop 10 models with missing 'size':")
print(missing_size['model'].value_counts().head(10))
print("\nMissing 'size' values by vehicle type:")
print(missing_size['type'].value_counts())
print("\nMissing 'size' values by number of cylinders:")
print(missing_size['cylinders'].value_counts())
print("\nTop 10 (manufacturer, model) pairs with missing 'size':")
print(missing_size.groupby(['manufacturer', 'model']).size().sort_values(ascending=False).head(10))
print("\nStep 8: Pattern Study for Remaining Missing 'size' Values completed.")

# Step 9: Efficient Robust Multi-Feature Imputation
print("\n=== Step 9: Efficient Robust Multi-Feature Imputation ===")
print("Precomputing modes for each feature...")
type_mode = df.groupby('type')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
cyl_mode = df.groupby('cylinders')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
manu_mode = df.groupby('manufacturer')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
model_mode = df.groupby('model')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()

feature_weights = {
    'type': 0.4,
    'cylinders': 0.3,
    'manufacturer': 0.2,
    'model': 0.1
}

def fast_predict_size(row):
    votes = {}
    t = row['type']
    if pd.notna(t) and t in type_mode and pd.notna(type_mode[t]):
        votes[type_mode[t]] = votes.get(type_mode[t], 0) + feature_weights['type']
    c = row['cylinders']
    if pd.notna(c) and c in cyl_mode and pd.notna(cyl_mode[c]):
        votes[cyl_mode[c]] = votes.get(cyl_mode[c], 0) + feature_weights['cylinders']
    m = row['manufacturer']
    if pd.notna(m) and m in manu_mode and pd.notna(manu_mode[m]):
        votes[manu_mode[m]] = votes.get(manu_mode[m], 0) + feature_weights['manufacturer']
    mo = row['model']
    if pd.notna(mo) and mo in model_mode and pd.notna(model_mode[mo]):
        votes[model_mode[mo]] = votes.get(model_mode[mo], 0) + feature_weights['model']
    if not votes:
        return np.nan
    max_vote = max(votes.values())
    if max_vote >= 0.4:
        return max(votes.items(), key=lambda x: x[1])[0]
    return np.nan

mask = df['size'].isna()
print(f"Imputing {mask.sum()} missing 'size' values using multi-feature voting...")
imputed = df.loc[mask].apply(fast_predict_size, axis=1)
df.loc[mask, 'size'] = imputed

print("\nImputation complete. Final value counts:")
print(df['size'].value_counts(dropna=False))
print(f"NaN count: {df['size'].isna().sum()}")

# Save the final imputed dataset
final_save_path = 'vehicles_size_cleaned_label_missing.csv'
df.to_csv(final_save_path, index=False)
print(f"\nFinal cleaned dataset with imputed 'size' saved to '{final_save_path}'")

# Step 10: Analyze Patterns in Remaining Missing 'size' Values
print("\n=== Step 10: Pattern Analysis of Remaining Missing 'size' Values ===")
remaining_missing = df[df['size'].isna()]
print(f"Rows still missing 'size': {len(remaining_missing)}")
print("\nTop 10 manufacturers with missing 'size':")
print(remaining_missing['manufacturer'].value_counts().head(10))
print("\nTop 10 models with missing 'size':")
print(remaining_missing['model'].value_counts().head(10))
print("\nMissing 'size' values by vehicle type:")
print(remaining_missing['type'].value_counts())
print("\nMissing 'size' values by number of cylinders:")
print(remaining_missing['cylinders'].value_counts())
print("\nTop 10 (manufacturer, model) pairs with missing 'size':")
print(remaining_missing.groupby(['manufacturer', 'model']).size().sort_values(ascending=False).head(10))
print("\nStep 10: Pattern Analysis for Remaining Missing 'size' Values completed.")

# Step 11: Impute using additional strong pairwise patterns
print("\n=== Step 11: Impute using additional strong pairwise patterns ===")
mode_threshold = 0.9
min_group_size = 10

def impute_by_pair(pair_cols, pair_name):
    groups = df.groupby(pair_cols)
    impute_count = 0
    for pair_vals, group in groups:
        known = group['size'].dropna()
        if len(known) >= min_group_size:
            mode_val = known.mode()
            if len(mode_val) == 1 and (known.value_counts().iloc[0] / len(known)) >= mode_threshold:
                mask = (df[pair_cols[0]] == pair_vals[0]) & (df[pair_cols[1]] == pair_vals[1]) & (df['size'].isna())
                n_to_impute = mask.sum()
                if n_to_impute > 0:
                    df.loc[mask, 'size'] = mode_val.iloc[0]
                    impute_count += n_to_impute
                    print(f"Imputed {n_to_impute} missing 'size' values for {pair_name} {pair_vals} with '{mode_val.iloc[0]}'")
    print(f"Total 'size' values imputed using {pair_name} strong mode patterns: {impute_count}")

to_try = [(['manufacturer', 'type'], '(manufacturer, type)'),
          (['manufacturer', 'cylinders'], '(manufacturer, cylinders)'),
          (['model', 'cylinders'], '(model, cylinders)')]
for cols, name in to_try:
    impute_by_pair(cols, name)

print_size_status(df, "After Additional Pairwise Imputation")

# Step 12: Data-driven imputation for models with strong internal evidence
model_size_counts = df[df['size'].notna()].groupby('model')['size'].value_counts()
model_total_counts = df[df['size'].notna()].groupby('model')['size'].count()
model_mode = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
model_mode_count = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
model_mode_ratio = (model_mode_count / model_total_counts).fillna(0)
strong_models = model_mode_ratio[model_mode_ratio >= 0.9].index
strong_model_map = model_mode.loc[strong_models]
mask_strong_model = df['size'].isna() & df['model'].isin(strong_models)
df.loc[mask_strong_model, 'size'] = df.loc[mask_strong_model, 'model'].map(strong_model_map)
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[mask_strong_model, 'size_imputation_status'] = 'imputed_strong_model'
print(f"Imputed {mask_strong_model.sum()} missing 'size' values using data-driven strong model mapping.")
df.to_csv('Project/vehicles_cleaned_with_data_driven_model_map.csv', index=False)
total_rows = len(df)
print(f"Total rows: {total_rows}")
print("Size value counts:")
print(df['size'].value_counts(dropna=False))

# Step 13: Robust profiling for new imputation opportunities among remaining NaNs
def profile_strong_patterns_for_remaining_nans():
    print("\n=== Step 13: Profiling for Strong Patterns in Remaining NaNs ===")
    features = ['manufacturer', 'model', 'cylinders', 'fuel', 'year']
    df = pd.read_csv('Project/vehicles_cleaned_with_data_driven_model_map.csv')
    missing = df[df['size'].isna()]
    found_any = False
    for r in [2, 3]:
        print(f"\n--- Checking all {r}-feature combinations ---")
        for combo in combinations(features, r):
            group = missing.groupby(list(combo))
            for name, sub in group:
                if len(sub) < 3:
                    continue
                mask = (df[list(combo)] == pd.Series(name, index=combo)).all(axis=1) & df['size'].notna()
                known = df.loc[mask, 'size']
                if len(known) < 3:
                    continue
                mode = known.mode()
                if len(mode) == 1:
                    mode_count = (known == mode.iloc[0]).sum()
                    ratio = mode_count / len(known)
                    if ratio >= 0.95:
                        found_any = True
                        print(f"Combo {combo}, value {name}: mode={mode.iloc[0]}, ratio={ratio:.2f}, missing={len(sub)}, known={len(known)}")
    if not found_any:
        print("No strong patterns found for further imputation.")
    total_rows = len(df)
    print(f"\nTotal rows: {total_rows}")
    print("Size value counts:")
    print(df['size'].value_counts(dropna=False))

if __name__ == "__main__":
    profile_strong_patterns_for_remaining_nans()

# Step 14: Data-driven imputation for (manufacturer, cylinders) pairs with extremely strong mode
pair = ['manufacturer', 'cylinders']
known = df[df['size'].notna()]
grouped = known.groupby(pair)['size']
pair_mode = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
pair_mode_count = grouped.agg(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
pair_total_count = grouped.count()
pair_mode_ratio = (pair_mode_count / pair_total_count).fillna(0)
strong_pairs = pair_mode_ratio[(pair_mode_ratio >= 0.95) & (pair_total_count >= 3)].index
strong_pair_map = pair_mode.loc[strong_pairs]
mask_strong_pair = df['size'].isna() & df.set_index(pair).index.isin(strong_pairs)
def impute_pair(row):
    key = (row['manufacturer'], row['cylinders'])
    if key in strong_pair_map:
        return strong_pair_map[key]
    return row['size']
df.loc[mask_strong_pair, 'size'] = df.loc[mask_strong_pair].apply(impute_pair, axis=1)
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[mask_strong_pair, 'size_imputation_status'] = 'imputed_strong_manufacturer_cylinders'
print(f"Imputed {mask_strong_pair.sum()} missing 'size' values using strong (manufacturer, cylinders) mode patterns.")
total_rows = len(df)
print(f"Total rows: {total_rows}")
print("Size value counts:")
print(df['size'].value_counts(dropna=False))

# Step 15: Label remaining missing 'size' values as 'missing'
remaining_missing_count = df['size'].isna().sum()
df['size'] = df['size'].fillna('missing')
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[df['size'] == 'missing', 'size_imputation_status'] = 'missing_label'
print(f"\nLabeled {remaining_missing_count} remaining missing 'size' values as 'missing'.")

# Final summary
print("\n=== Final Summary After Labeling Missing 'size' ===")
total_rows = len(df)
print(f"Total rows: {total_rows}")
print("Size value counts:")
print(df['size'].value_counts(dropna=False))
missing_count = (df['size'] == 'missing').sum()
print(f"'missing' label count: {missing_count}")

# Save the final imputed dataset
final_save_path = 'vehicles_size_cleaned_labeled_missing.csv'
df.to_csv(final_save_path, index=False)
print(f"\nFinal cleaned dataset with labeled missing 'size' saved to '{final_save_path}'") 
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import chi2_contingency
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
# For each type, cylinders group, if there are at least 10 known values 
# and the mode covers at least 80%, impute missing 'size' with the mode.
mode_threshold = 0.8
min_group_size = 10
impute_count = 0

# Implement imputation based on type-cylinder combinations
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
# Only impute if there are at least 10 known values and the mode covers at least 80%.
mode_threshold = 0.8
min_group_size = 10
impute_count = 0

# Implement imputation based on manufacturer-model combinations
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
# --------------------------------------------------

print("\n=== Studying Patterns in Remaining Missing 'size' Values ===")
missing_size = df[df['size'].isna()]
print(f"Rows still missing 'size': {len(missing_size)}")

# Analyze patterns by manufacturer
print("\nTop 10 manufacturers with missing 'size':")
print(missing_size['manufacturer'].value_counts().head(10))

# Analyze patterns by model
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
# --------------------------------------------------
# This step implements a robust multi-feature imputation:
# - Precompute the mode of 'size' for each value of type, cylinders, manufacturer, and model
# - For each missing row, aggregate weighted votes from these features
# - Impute only if the highest vote meets the confidence threshold (>= 0.4)
# - All logic is preserved, but performance is greatly improved


print("\n=== Step 9: Efficient Robust Multi-Feature Imputation ===")

# Precompute modes for each feature
print("Precomputing modes for each feature...")
type_mode = df.groupby('type')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
cyl_mode = df.groupby('cylinders')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
manu_mode = df.groupby('manufacturer')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
model_mode = df.groupby('model')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()

feature_weights = {
    'type': 0.4,        # Vehicle type shows strongest correlation with size (40% importance)
    'cylinders': 0.3,   # Engine configuration is second most important (30% importance)
    'manufacturer': 0.2, # Brand-specific patterns contribute 20% to size prediction
    'model': 0.1        # Model-specific variations have 10% influence on size
}

def fast_predict_size(row):
    votes = {}
    # Type-based prediction
    t = row['type']
    if pd.notna(t) and t in type_mode and pd.notna(type_mode[t]):
        votes[type_mode[t]] = votes.get(type_mode[t], 0) + feature_weights['type']
    # Cylinder-based prediction
    c = row['cylinders']
    if pd.notna(c) and c in cyl_mode and pd.notna(cyl_mode[c]):
        votes[cyl_mode[c]] = votes.get(cyl_mode[c], 0) + feature_weights['cylinders']
    # Manufacturer-based prediction
    m = row['manufacturer']
    if pd.notna(m) and m in manu_mode and pd.notna(manu_mode[m]):
        votes[manu_mode[m]] = votes.get(manu_mode[m], 0) + feature_weights['manufacturer']
    # Model-based prediction
    mo = row['model']
    if pd.notna(mo) and mo in model_mode and pd.notna(model_mode[mo]):
        votes[model_mode[mo]] = votes.get(model_mode[mo], 0) + feature_weights['model']
    
    if not votes:
        return np.nan
    
    max_vote = max(votes.values())
    if max_vote >= 0.4:  # Confidence threshold
        return max(votes.items(), key=lambda x: x[1])[0]
    return np.nan

# Apply multi-feature imputation
mask = df['size'].isna()
print(f"Imputing {mask.sum()} missing 'size' values using multi-feature voting...")
imputed = df.loc[mask].apply(fast_predict_size, axis=1)
df.loc[mask, 'size'] = imputed

print("\nImputation complete. Final value counts:")
print(df['size'].value_counts(dropna=False))
print(f"NaN count: {df['size'].isna().sum()}")

# Save the final imputed dataset
final_save_path = 'vehicles_size_cleaned.csv'
df.to_csv(final_save_path, index=False)
print(f"\nFinal cleaned dataset with imputed 'size' saved to '{final_save_path}'")

# Step 10: Analyze Patterns in Remaining Missing 'size' Values
# --------------------------------------------------
# After all imputation steps, profile the remaining NaN 'size' values to see if any patterns remain.
# This helps decide whether further imputation, flagging, or dropping is appropriate.

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
# --------------------------------------------------
# For each of the following pairs, if there are at least 10 known values and the mode covers at least 90%,
# impute missing 'size' with the mode for that group:
#   - (manufacturer, type)
#   - (manufacturer, cylinders)
#   - (model, cylinders)
# This is a final, conservative imputation step for rows where type is missing but other features are present.

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

# Apply for each pair
to_try = [(['manufacturer', 'type'], '(manufacturer, type)'),
          (['manufacturer', 'cylinders'], '(manufacturer, cylinders)'),
          (['model', 'cylinders'], '(model, cylinders)')]
for cols, name in to_try:
    impute_by_pair(cols, name)

print_size_status(df, "After Additional Pairwise Imputation")

# Step 12: Data-driven imputation for models with strong internal evidence

# 1. For each model, check if the mode covers ≥90% of known size values
model_size_counts = df[df['size'].notna()].groupby('model')['size'].value_counts()
model_total_counts = df[df['size'].notna()].groupby('model')['size'].count()
model_mode = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
model_mode_count = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
model_mode_ratio = (model_mode_count / model_total_counts).fillna(0)

# 2. Only use models where the mode is dominant (e.g., ≥90%)
strong_models = model_mode_ratio[model_mode_ratio >= 0.9].index
strong_model_map = model_mode.loc[strong_models]

# 3. Impute missing size for these strong models
mask_strong_model = df['size'].isna() & df['model'].isin(strong_models)
df.loc[mask_strong_model, 'size'] = df.loc[mask_strong_model, 'model'].map(strong_model_map)

# Optionally, track imputation method
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[mask_strong_model, 'size_imputation_status'] = 'imputed_strong_model'

print(f"Imputed {mask_strong_model.sum()} missing 'size' values using data-driven strong model mapping.")

# Save the updated file
df.to_csv('Project/vehicles_cleaned_with_data_driven_model_map.csv', index=False)

# Print summary of the new data after step 12
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
                # Find all known size values for this group in the full data
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
    # Print summary of the current data
    total_rows = len(df)
    print(f"\nTotal rows: {total_rows}")
    print("Size value counts:")
    print(df['size'].value_counts(dropna=False))

if __name__ == "__main__":
    profile_strong_patterns_for_remaining_nans()

# Step 14: Data-driven imputation for (manufacturer, cylinders) pairs with extremely strong mode
# -------------------------------------------------------------------
# Rationale:
# - For some (manufacturer, cylinders) pairs, the vast majority of known rows in your data have the same 'size'.
# - If the mode covers >=95% of known values for a pair, it is reasonable to impute missing 'size' for that pair.
# - This step increases coverage while minimizing risk of misclassification, as only very strong patterns are used.
# - All imputations are based strictly on your dataset, not external knowledge.
#
# Process:
# 1. For each (manufacturer, cylinders) pair, calculate the mode and its dominance ratio among known 'size' values.
# 2. For pairs where the mode covers >=95% of known values and there are at least 3 known, impute missing 'size' with the mode.
# 3. Track the number of imputations and print a summary after this step.

# 1. Calculate mode and dominance for each (manufacturer, cylinders) pair
pair = ['manufacturer', 'cylinders']
known = df[df['size'].notna()]
grouped = known.groupby(pair)['size']
pair_mode = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
pair_mode_count = grouped.agg(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
pair_total_count = grouped.count()
pair_mode_ratio = (pair_mode_count / pair_total_count).fillna(0)

# 2. Select strong pairs (>=95% dominance, at least 3 known)
strong_pairs = pair_mode_ratio[(pair_mode_ratio >= 0.95) & (pair_total_count >= 3)].index
strong_pair_map = pair_mode.loc[strong_pairs]

# 3. Impute missing 'size' for these strong pairs
mask_strong_pair = df['size'].isna() & df.set_index(pair).index.isin(strong_pairs)
def impute_pair(row):
    key = (row['manufacturer'], row['cylinders'])
    if key in strong_pair_map:
        return strong_pair_map[key]
    return row['size']
df.loc[mask_strong_pair, 'size'] = df.loc[mask_strong_pair].apply(impute_pair, axis=1)

# Optionally, track imputation method
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[mask_strong_pair, 'size_imputation_status'] = 'imputed_strong_manufacturer_cylinders'

print(f"Imputed {mask_strong_pair.sum()} missing 'size' values using strong (manufacturer, cylinders) mode patterns.")

# Print summary after Step 14
total_rows = len(df)
print(f"Total rows: {total_rows}")
print("Size value counts:")
print(df['size'].value_counts(dropna=False))

# Step 15: Random Forest Imputation for Remaining Missing 'size' Values
# --------------------------------------------------------------------
# Rationale:
# - After exhausting all robust, pattern-based, and data-driven imputation methods, some missing 'size' values remain.
# - Random Forest classifier can predict 'size' using other features, capturing complex patterns.
# - All imputations are flagged for transparency.

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare data
features = ['manufacturer', 'model', 'cylinders', 'fuel', 'year']
df_rf = df.copy()
encoders = {}
for col in features + ['size']:
    le = LabelEncoder()
    df_rf[col] = df_rf[col].astype(str)
    df_rf[col] = le.fit_transform(df_rf[col])
    encoders[col] = le

# Split into train (known size) and predict (missing size)
train_mask = df['size'].notna()
X_train = df_rf.loc[train_mask, features]
y_train = df_rf.loc[train_mask, 'size']
X_pred = df_rf.loc[~train_mask, features]

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict missing 'size' values
size_pred = clf.predict(X_pred)

# Decode predictions back to original categories
size_le = encoders['size']
df.loc[~train_mask, 'size'] = size_le.inverse_transform(size_pred)

# Optionally, flag these imputations
if 'size_imputation_status' not in df.columns:
    df['size_imputation_status'] = 'original'
df.loc[~train_mask, 'size_imputation_status'] = 'imputed_rf'

print("After Random Forest imputation:")
print(df['size'].value_counts(dropna=False))

# Step 16: Final Summary After Random Forest Imputation
# ----------------------------------------------------
# Print a summary of the final imputation results, including total rows, size value counts, and NaN count.

print("\n=== Final Summary After Random Forest Imputation ===")
total_rows = len(df)
print(f"Total rows: {total_rows}")
print("Size value counts:")
print(df['size'].value_counts(dropna=False))
nan_count = df['size'].isna().sum()
print(f"NaN count: {nan_count}")

# Step 17: Bias Analysis of Random Forest Imputation
# -------------------------------------------------
# This step analyzes whether the Random Forest imputation introduced any bias by comparing:
# 1. Overall distributions of original vs imputed values
# 2. Distributions by manufacturer (top 5 manufacturers with most imputations)
# 3. Distributions by number of cylinders
# 4. Distributions by year (grouped by decades)
# 5. Statistical test (Chi-square) to check if distributions are significantly different
#
# This analysis helps us understand if our imputation method is maintaining the natural
# distribution of vehicle sizes in the dataset or if it's introducing bias.

print("\n=== Bias Analysis of Random Forest Imputation ===")

# 1. Compare overall distributions
print("\n1. Distribution Comparison:")
print("Original values distribution:")
original_dist = df[df['size_imputation_status'] == 'original']['size'].value_counts(normalize=True)
print(original_dist)
print("\nImputed values distribution:")
imputed_dist = df[df['size_imputation_status'] == 'imputed_rf']['size'].value_counts(normalize=True)
print(imputed_dist)

# 2. Compare distributions by manufacturer
print("\n2. Distribution by Manufacturer (Top 5 manufacturers with most imputations):")
top_imputed_manufacturers = df[df['size_imputation_status'] == 'imputed_rf']['manufacturer'].value_counts().head()
for manu in top_imputed_manufacturers.index:
    print(f"\nManufacturer: {manu}")
    print("Original distribution:")
    print(df[(df['manufacturer'] == manu) & (df['size_imputation_status'] == 'original')]['size'].value_counts(normalize=True))
    print("Imputed distribution:")
    print(df[(df['manufacturer'] == manu) & (df['size_imputation_status'] == 'imputed_rf')]['size'].value_counts(normalize=True))

# 3. Compare distributions by cylinders
print("\n3. Distribution by Cylinders:")
print("Original values by cylinders:")
print(pd.crosstab(df[df['size_imputation_status'] == 'original']['cylinders'], 
                 df[df['size_imputation_status'] == 'original']['size'], 
                 normalize='index'))
print("\nImputed values by cylinders:")
print(pd.crosstab(df[df['size_imputation_status'] == 'imputed_rf']['cylinders'], 
                 df[df['size_imputation_status'] == 'imputed_rf']['size'], 
                 normalize='index'))

# 4. Compare distributions by year
print("\n4. Distribution by Year (grouped by decades):")
df['decade'] = (df['year'] // 10) * 10
print("Original values by decade:")
print(pd.crosstab(df[df['size_imputation_status'] == 'original']['decade'], 
                 df[df['size_imputation_status'] == 'original']['size'], 
                 normalize='index'))
print("\nImputed values by decade:")
print(pd.crosstab(df[df['size_imputation_status'] == 'imputed_rf']['decade'], 
                 df[df['size_imputation_status'] == 'imputed_rf']['size'], 
                 normalize='index'))

# 5. Statistical test for distribution similarity
print("\n5. Chi-square test for distribution similarity:")
contingency = pd.crosstab(df['size_imputation_status'], df['size'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"Chi-square statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Interpretation: If p-value < 0.05, distributions are significantly different (potential bias)")

# Remove temporary column
df = df.drop('decade', axis=1)

# Step 18: Balanced Random Forest Imputation
# -----------------------------------------
# This step implements a more sophisticated approach to address any bias found in Step 17.
# Key improvements over the basic Random Forest (Step 15):
# 1. Uses class weights based on original distribution to prevent bias
# 2. Adds more granular feature engineering:
#    - Cylinders categories: small, medium, large, v8, v10+
#    - Year categories: vintage, classic, modern, recent
# 3. Uses cross-validation with stratification
# 4. Analyzes feature importance
# 5. Uses a more complex Random Forest model with:
#    - 200 estimators (vs 100 in Step 15)
#    - Maximum depth of 10
#    - Minimum samples per leaf of 5
#    - Class weights to balance the predictions
#
# This approach ensures that the imputed values maintain the natural distribution
# of vehicle sizes in the dataset, addressing any bias found in the previous imputation.

print("\n=== Balanced Random Forest Imputation ===")

# Calculate class weights based on original distribution
original_dist = df[df['size_imputation_status'] == 'original']['size'].value_counts(normalize=True)
class_weights = {size: 1/prop for size, prop in original_dist.items()}
print("\nClass weights based on original distribution:")
for size, weight in class_weights.items():
    print(f"{size}: {weight:.2f}")

# Map class_weights to encoded integer labels
size_le = encoders['size']
class_weight_encoded = {size_le.transform([k])[0]: v for k, v in class_weights.items()}

# Prepare features with more granular encoding
features = ['manufacturer', 'model', 'cylinders', 'fuel', 'year']
df_balanced = df.copy()

# Enhanced feature engineering
df_balanced['cylinders_category'] = pd.cut(df_balanced['cylinders'], 
                                         bins=[-float('inf'), 3, 4, 6, 8, float('inf')],
                                         labels=['small', 'medium', 'large', 'v8', 'v10+'])
df_balanced['year_category'] = pd.cut(df_balanced['year'],
                                    bins=[-float('inf'), 1970, 1990, 2010, float('inf')],
                                    labels=['vintage', 'classic', 'modern', 'recent'])

# Update features list
features.extend(['cylinders_category', 'year_category'])

# Encode categorical features
encoders = {}
for col in features + ['size']:
    le = LabelEncoder()
    df_balanced[col] = df_balanced[col].astype(str)
    df_balanced[col] = le.fit_transform(df_balanced[col])
    encoders[col] = le

# Split into train and predict sets
train_mask = df['size'].notna()
X_train = df_balanced.loc[train_mask, features]
y_train = df_balanced.loc[train_mask, 'size']
X_pred = df_balanced.loc[~train_mask, features]

# Train balanced Random Forest
clf_balanced = RandomForestClassifier(
    n_estimators=200,
    class_weight=class_weight_encoded,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Cross-validation with stratification
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_balanced, X_train, y_train, cv=cv, scoring='f1_weighted')
print(f"\nCross-validation scores (F1-weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
clf_balanced.fit(X_train, y_train)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': clf_balanced.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature importance:")
print(feature_importance)

# Predict missing values only if there are any left
if X_pred.shape[0] > 0:
    size_pred = clf_balanced.predict(X_pred)
    # Decode predictions
    size_le = encoders['size']
    df.loc[~train_mask, 'size'] = size_le.inverse_transform(size_pred)
    # Update imputation status
    df.loc[~train_mask, 'size_imputation_status'] = 'imputed_balanced_rf'
    print("\nNew distribution after balanced imputation:")
    print(df['size'].value_counts(normalize=True))
    print("\nComparison with original distribution:")
    print("Original distribution:")
    print(original_dist)
    print("\nNew distribution:")
    print(df[df['size_imputation_status'] == 'imputed_balanced_rf']['size'].value_counts(normalize=True))
else:
    print("No missing values left to impute with Balanced Random Forest.")

# Remove temporary columns
df = df.drop(['cylinders_category', 'year_category'], axis=1)

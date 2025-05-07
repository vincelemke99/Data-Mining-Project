import pandas as pd
import numpy as np

def print_cylinder_status(df, step_name):
    print(f"\n=== Cylinder Status After {step_name} ===")
    print(f"Total rows: {len(df)}")
    print("Cylinder value counts:")
    print(df['cylinders'].value_counts(dropna=False))
    nan_count = df['cylinders'].isna().sum()
    print(f"NaN count: {nan_count}")
    if nan_count > 0:
        print("\nSample of rows with NaN cylinders:")
        print(df[df['cylinders'].isna()].head()[['manufacturer', 'model', 'year', 'fuel', 'type']])

# Step 1: Read the original data
df_original = pd.read_csv('Project/data/vehicles.csv')
df = df_original.copy()
print_cylinder_status(df, "Initial Load")

# Step 2: Show missing values for each column
# so i can see the missing values for each column and the percentage of missing values 
# to make a decision on how to handle the missing values for cylinders
print("Data shape:", df.shape)
print(df.head())
print("\nMissing values per column:")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_percent': missing_percent})
print(missing_df[missing_df.missing_count > 0].sort_values(by='missing_count', ascending=False))

# Step 3: Standardize cylinder values to numeric
print("\nStep 3: Standardizing cylinder values")
print("Before standardization - unique values in cylinders:")
print(df['cylinders'].value_counts(dropna=False))

def convert_cylinders(val):
    try:
        if isinstance(val, str) and val.split()[0].isdigit():
            return int(val.split()[0])
        else:
            return val  # leave as is ("other", NaN, etc.)
    except Exception:
        return val

df['cylinders'] = df['cylinders'].apply(convert_cylinders)
print("\nAfter standardization - unique values in cylinders:")
print(df['cylinders'].value_counts(dropna=False))

# Step 4: Impute missing cylinder values
# In this step, we address missing values in the 'cylinders' column using a multi-stage approach:
#   1. Impute electric vehicles (set cylinders to 0 for EVs).
#   2. Impute using the most common value (mode) for each manufacturer-model combination.
#   3. For remaining NaNs, impute using manufacturer-year mode.
#   4. For any still-missing, impute using type-year mode.
#   5. Finally, impute any remaining missing values using strong patterns (see function below).
#   6. Drop any rows where 'cylinders' is still missing.
# Each stage prints a summary of how many values were imputed and the logic used.

# Make a copy for imputation
df_imputed = df.copy()
df_imputed['imputation_method'] = 'original'
# Track initial missing values for summary
initial_missing = df_imputed['cylinders'].isna().sum()

# 1. Impute electric vehicles (set cylinders to 0 for EVs)
electric_mask = (df_imputed['cylinders'].isna()) & (df_imputed['fuel'] == 'electric')
df_imputed.loc[electric_mask, 'cylinders'] = 0
df_imputed.loc[electric_mask, 'imputation_method'] = 'electric'
print(f"\nImputed {electric_mask.sum():,} electric vehicles to 0 cylinders")

# 2. Impute using the most common value (mode) for each manufacturer-model combination
known_cylinders = df_imputed[
    (df_imputed['cylinders'].notna()) & (df_imputed['cylinders'] != 'other')
].groupby(['manufacturer', 'model'])['cylinders'].agg(
    lambda x: int(x.mode()[0]) if len(x.mode()) > 0 else np.nan
).to_dict()

original_nan = df_imputed['cylinders'].isna()
def impute_by_manufacturer_model(row):
    if pd.isna(row['cylinders']):
        key = (row['manufacturer'], row['model'])
        if key in known_cylinders:
            row['imputation_method'] = 'manufacturer_model'
            return known_cylinders[key]
    return row['cylinders']

df_imputed['cylinders'] = df_imputed.apply(impute_by_manufacturer_model, axis=1)
manufacturer_model_mask = original_nan & df_imputed['cylinders'].notna() & (df_imputed['imputation_method'] == 'original')
df_imputed.loc[manufacturer_model_mask, 'imputation_method'] = 'manufacturer_model'
print(f"Imputed {manufacturer_model_mask.sum():,} values using manufacturer-model mapping")

# 3. For remaining NaN values, impute using manufacturer-year mode
original_nan = df_imputed['cylinders'].isna()
known_cylinders_by_year = df_imputed[
    (df_imputed['cylinders'].notna()) & (df_imputed['cylinders'] != 'other')
].groupby(['manufacturer', 'year'])['cylinders'].agg(
    lambda x: int(x.mode()[0]) if len(x.mode()) > 0 else np.nan
).to_dict()

def impute_by_manufacturer_year(row):
    if pd.isna(row['cylinders']):
        key = (row['manufacturer'], row['year'])
        if key in known_cylinders_by_year:
            row['imputation_method'] = 'manufacturer_year'
            return known_cylinders_by_year[key]
    return row['cylinders']

df_imputed['cylinders'] = df_imputed.apply(impute_by_manufacturer_year, axis=1)
manufacturer_year_mask = original_nan & df_imputed['cylinders'].notna() & (df_imputed['imputation_method'] == 'original')
df_imputed.loc[manufacturer_year_mask, 'imputation_method'] = 'manufacturer_year'
print(f"Imputed {manufacturer_year_mask.sum():,} values using manufacturer-year mapping")

# 4. For any still-missing, impute using type-year mode
original_nan = df_imputed['cylinders'].isna()
known_cylinders_by_type = df_imputed[
    (df_imputed['cylinders'].notna()) & (df_imputed['cylinders'] != 'other')
].groupby(['type', 'year'])['cylinders'].agg(
    lambda x: int(x.mode()[0]) if len(x.mode()) > 0 else np.nan
).to_dict()

def impute_by_type_year(row):
    if pd.isna(row['cylinders']):
        key = (row['type'], row['year'])
        if key in known_cylinders_by_type:
            row['imputation_method'] = 'type_year'
            return known_cylinders_by_type[key]
    return row['cylinders']

df_imputed['cylinders'] = df_imputed.apply(impute_by_type_year, axis=1)
type_year_mask = original_nan & df_imputed['cylinders'].notna() & (df_imputed['imputation_method'] == 'original')
df_imputed.loc[type_year_mask, 'imputation_method'] = 'type_year'
print(f"Imputed {type_year_mask.sum():,} values using type-year mapping")

# 5. Finally, impute any remaining missing values using strong patterns
def impute_strong_patterns(df_imputed):
    """
    Impute remaining missing cylinder values using strong data patterns:
      - For commercial vehicles (e.g., Freightliner, International, Peterbilt, Blue Bird), impute if 80%+ of known values agree and at least 10 known.
      - For any other model-fuel combinations, impute if 80%+ of known values agree and at least 10 known.
    This is the final stage of imputation after other strategies have been applied.
    """
    print("\n=== Strong Pattern Imputation ===")
    initial_missing = df_imputed['cylinders'].isna().sum()
    print(f"Initial missing values: {initial_missing:,}")
    
    # 1. Analyze and impute commercial vehicles and buses
    commercial_keywords = ['freightliner', 'international', 'peterbilt', 'blue bird', 'cascadia', '4300', '579']
    commercial_mask = (df_imputed['cylinders'].isna()) & (
        df_imputed['model'].str.contains('|'.join(commercial_keywords), case=False, na=False)
    )
    
    # Get known values for commercial vehicles
    known_commercial = df_imputed[
        (~df_imputed['cylinders'].isna()) & (df_imputed['cylinders'] != 'other') &
        (df_imputed['model'].str.contains('|'.join(commercial_keywords), case=False, na=False))
    ]
    
    print("\nAnalyzing known cylinder values for commercial vehicles:")
    print("\nBy fuel type:")
    print(known_commercial.groupby('fuel')['cylinders'].value_counts())
    
    # Only impute if we have strong patterns in the data
    mode_threshold = 0.8
    min_group_size = 10
    
    # Group by model and fuel to find strong patterns
    commercial_patterns = known_commercial.groupby(['model', 'fuel'])['cylinders'].agg([
        ('count', 'count'),
        ('mode', lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan),
        ('mode_pct', lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0)
    ])
    
    # Filter for strong patterns
    strong_patterns = commercial_patterns[
        (commercial_patterns['count'] >= min_group_size) & 
        (commercial_patterns['mode_pct'] >= mode_threshold)
    ]
    
    print("\nStrong patterns found in commercial vehicles:")
    print(strong_patterns)
    
    # Impute based on strong patterns
    impute_count = 0
    for (model, fuel), row in strong_patterns.iterrows():
        mask = (
            (df_imputed['model'].str.contains(model, case=False, na=False)) & 
            (df_imputed['fuel'] == fuel) & 
            df_imputed['cylinders'].isna()
        )
        n_to_impute = mask.sum()
        if n_to_impute > 0:
            df_imputed.loc[mask, 'cylinders'] = row['mode']
            df_imputed.loc[mask, 'imputation_method'] = 'commercial_pattern'
            impute_count += n_to_impute
            print(f"Imputed {n_to_impute} values for {model} ({fuel}) with {row['mode']} cylinders")
    
    print(f"\nImputed {impute_count:,} commercial vehicle values based on data patterns")
    
    # 2. Impute using model-fuel combinations with high confidence
    mode_threshold = 0.8
    min_group_size = 10
    
    # Get unique model-fuel combinations with missing values
    missing_combos = df_imputed[df_imputed['cylinders'].isna()][['model', 'fuel']].drop_duplicates()
    impute_count = 0
    
    for _, row in missing_combos.iterrows():
        # Get known values for this combo
        combo_data = df_imputed[
            (df_imputed['model'] == row['model']) & 
            (df_imputed['fuel'] == row['fuel']) &
            (~df_imputed['cylinders'].isna()) & (df_imputed['cylinders'] != 'other')
        ]['cylinders']
        
        if len(combo_data) >= min_group_size:
            # Calculate mode and its percentage
            value_counts = combo_data.value_counts()
            if len(value_counts) > 0:
                mode_val = value_counts.index[0]
                mode_pct = value_counts.iloc[0] / len(combo_data)
                
                # Only impute if mode percentage meets threshold
                if mode_pct >= mode_threshold:
                    # Impute
                    mask = (
                        (df_imputed['model'] == row['model']) & 
                        (df_imputed['fuel'] == row['fuel']) & 
                        df_imputed['cylinders'].isna()
                    )
                    n_to_impute = mask.sum()
                    if n_to_impute > 0:
                        df_imputed.loc[mask, 'cylinders'] = mode_val
                        df_imputed.loc[mask, 'imputation_method'] = 'model_fuel_mode'
                        impute_count += n_to_impute
    
    print(f"Imputed {impute_count:,} values using model-fuel combinations")
    
    # 3. Final analysis
    remaining_missing = df_imputed['cylinders'].isna().sum()
    print(f"\nFinal missing values: {remaining_missing:,}")
    print("\nImputation methods used in final step:")
    print(df_imputed[df_imputed['imputation_method'].isin(['commercial_pattern', 'model_fuel_mode'])]['imputation_method'].value_counts())
    
    return df_imputed

df_imputed = impute_strong_patterns(df_imputed)

df_imputed['cylinders'] = df_imputed['cylinders'].fillna('missing')
print("\nCylinder value counts after labeling remaining NaNs as 'missing':")
print(df_imputed['cylinders'].value_counts(dropna=False))
"""""
Most likely, we will use KNN imputation instead of dropping the entire Row

# 
# 6. Drop any rows where 'cylinders' is still missing
rows_before_drop = len(df_imputed)
df_imputed = df_imputed.dropna(subset=['cylinders'])
rows_after_drop = len(df_imputed)
rows_dropped = rows_before_drop - rows_after_drop



print(f"\nDropped {rows_dropped:,} rows with remaining missing cylinder values")
print(f"Final dataset size: {rows_after_drop:,} rows")
"""
# Print cylinder value counts after imputation
print("\nCylinder value counts after imputation :")
print(df_imputed['cylinders'].value_counts(dropna=False))

# Final summary
total_rows = len(df_imputed)
remaining_missing = (df_imputed['cylinders'] == 'missing').sum()

print("\n=== Final Imputation Summary ===")
print(f"Initial missing values: {initial_missing:,} ({(initial_missing/len(df)*100):.2f}%)")
print(f"Final 'missing' labels: {remaining_missing:,} ({(remaining_missing/total_rows*100):.2f}%)")
print(f"\nImputation methods used:")
print(df_imputed['imputation_method'].value_counts())

print("\n=== Detailed Analysis of Remaining 'missing' Labels ===")
print(f"\nTotal 'missing' labels: {remaining_missing:,} ({(remaining_missing/total_rows*100):.2f}%)")

# Select rows labeled as 'missing' in cylinders
remaining_missing = df_imputed[df_imputed['cylinders'] == 'missing']

# Analyze by manufacturer
print("\nMissing values by manufacturer:")
print(remaining_missing['manufacturer'].value_counts().head(10))

# Analyze by vehicle type
print("\nMissing values by vehicle type:")
print(remaining_missing['type'].value_counts().head(10))

# Analyze by fuel type
print("\nMissing values by fuel type:")
print(remaining_missing['fuel'].value_counts().head(10))

# Analyze by year
print("\nMissing values by year (top 10 years):")
print(remaining_missing['year'].value_counts().sort_index().tail(10))

# Check for patterns in model names
print("\nSample of model names with missing cylinders (after filtering):")
print(remaining_missing['model'].value_counts().head(10))

# Print cylinder value counts after imputation
print("\nCylinder value counts after imputation (including 'other' and 'missing'):")
print(df_imputed['cylinders'].value_counts(dropna=False))

# Step 5: Investigate and handle the 'other' category in cylinders

# 1. Profile the 'other' category
other_rows = df_imputed[df_imputed['cylinders'] == 'other']
print(f"\nNumber of rows with 'other' in cylinders: {len(other_rows)}")

print("\nTop 20 models for 'other' cylinders:")
print(other_rows['model'].value_counts().head(20))

print("\nTop 10 manufacturers for 'other' cylinders:")
print(other_rows['manufacturer'].value_counts().head(10))

print("\nValue counts for 'type' in 'other' cylinders:")
print(other_rows['type'].value_counts())

print("\nValue counts for 'fuel' in 'other' cylinders:")
print(other_rows['fuel'].value_counts())

print("\nValue counts for 'year' in 'other' cylinders:")
print(other_rows['year'].value_counts().head(20))

print("\nSample rows with 'other' in cylinders:")
print(other_rows[['manufacturer', 'model', 'year', 'fuel', 'type']].head(20))

print("\nCrosstab: Model vs. Fuel for 'other' cylinders:")
print(pd.crosstab(other_rows['model'], other_rows['fuel']).head(20))

print("\nCrosstab: Model vs. Type for 'other' cylinders:")
print(pd.crosstab(other_rows['model'], other_rows['type']).head(20))

print("\nCrosstab: Manufacturer vs. Type for 'other' cylinders:")
print(pd.crosstab(other_rows['manufacturer'], other_rows['type']).head(20))

# 2. Drop non-vehicle entries in the 'other' group
non_vehicle_keywords = ['finance', 'program', 'special']
mask_non_vehicle = (
    (df_imputed['cylinders'] == 'other') &
    (df_imputed['model'].str.lower().str.contains('|'.join(non_vehicle_keywords), na=False))
)
df_imputed = df_imputed[~mask_non_vehicle]

# 3. Assign 0 cylinders to all electric vehicles in the 'other' group
mask_other_electric = (df_imputed['cylinders'] == 'other') & (df_imputed['fuel'].str.lower() == 'electric')
df_imputed.loc[mask_other_electric, 'cylinders'] = 0
df_imputed.loc[mask_other_electric, 'imputation_method'] = 'other_electric'

# 4. Re-profile the 'other' category after cleaning
other_rows_cleaned = df_imputed[df_imputed['cylinders'] == 'other']
print(f"\nNumber of rows with 'other' in cylinders after cleaning: {len(other_rows_cleaned)}")

print("\nTop 20 models for 'other' cylinders after cleaning:")
print(other_rows_cleaned['model'].value_counts().head(20))

print("\nTop 10 manufacturers for 'other' cylinders after cleaning:")
print(other_rows_cleaned['manufacturer'].value_counts().head(10))

print("\nValue counts for 'type' in 'other' cylinders after cleaning:")
print(other_rows_cleaned['type'].value_counts())

print("\nValue counts for 'fuel' in 'other' cylinders after cleaning:")
print(other_rows_cleaned['fuel'].value_counts())

print("\nValue counts for 'year' in 'other' cylinders after cleaning:")
print(other_rows_cleaned['year'].value_counts().head(20))

print("\nSample rows with 'other' in cylinders after cleaning:")
print(other_rows_cleaned[['manufacturer', 'model', 'year', 'fuel', 'type']].head(20))

# Step 6: Further cleaning and targeted imputation of 'other' in cylinders
# In this step, we address remaining ambiguous 'other' values in the 'cylinders' column using a conservative, data-driven approach:
#   - For each (model, fuel) combination in 'other', if there are at least 10 known numeric cylinder values and at least 80% of them agree (i.e., the mode covers 80%+), we impute the mode value for 'cylinders'.
#   - This ensures that imputation is only performed when there is a strong, reliable pattern in the data, minimizing the risk of introducing bias or error.
#   - Any remaining 'other' values after this step are considered too ambiguous or rare to impute and are left as is (or flagged/dropped in subsequent steps).
#   - This approach is analogous to the strong-pattern imputation logic used for commercial vehicles in previous steps, but is applied more broadly to all (model, fuel) combinations.
#   - The thresholds (min 10 known, 80%+ agreement) are chosen to balance completeness with accuracy and transparency.

# 1. Remove any remaining non-vehicle entries in the 'other' group
non_vehicle_keywords = [
    'finance', 'program', 'special', 'bad credit', 'trailer', 'carry on trailer', 'rack', 'shasta', 'jayco', 'avenger', 'credit', '0k', 'plymouth valiant'
]
mask_non_vehicle = (
    (df_imputed['cylinders'] == 'other') &
    (df_imputed['model'].str.lower().str.contains('|'.join(non_vehicle_keywords), na=False))
)
df_imputed = df_imputed[~mask_non_vehicle]

# 2. Targeted imputation for 'other' where strong patterns exist
# For each (model, fuel) combo in 'other', if there are at least 10 known numeric values and 80%+ are the same, impute that value
other_rows = df_imputed[df_imputed['cylinders'] == 'other']
for model in other_rows['model'].dropna().unique():
    for fuel in other_rows['fuel'].dropna().unique():
        known = df_imputed[
            (df_imputed['model'] == model) &
            (df_imputed['fuel'] == fuel) &
            (df_imputed['cylinders'] != 'other')
        ]['cylinders']
        if len(known) >= 10:
            mode_val = known.mode()
            if len(mode_val) == 1 and (known.value_counts().iloc[0] / len(known)) >= 0.8:
                mask = (
                    (df_imputed['model'] == model) &
                    (df_imputed['fuel'] == fuel) &
                    (df_imputed['cylinders'] == 'other')
                )
                df_imputed.loc[mask, 'cylinders'] = mode_val.iloc[0]
                df_imputed.loc[mask, 'imputation_method'] = 'other_strong_pattern'
                print(f"Imputed {mask.sum()} 'other' values for model '{model}' and fuel '{fuel}' with {mode_val.iloc[0]}")

# 3. Profile the 'other' category after further cleaning/imputation
other_rows_final = df_imputed[df_imputed['cylinders'] == 'other']
print(f"\nNumber of rows with 'other' in cylinders after further cleaning/imputation: {len(other_rows_final)}")
print("\nTop 20 models for 'other' cylinders after further cleaning/imputation:")
print(other_rows_final['model'].value_counts().head(20))
print("\nTop 10 manufacturers for 'other' cylinders after further cleaning/imputation:")
print(other_rows_final['manufacturer'].value_counts().head(10))
print("\nValue counts for 'type' in 'other' cylinders after further cleaning/imputation:")
print(other_rows_final['type'].value_counts())
print("\nValue counts for 'fuel' in 'other' cylinders after further cleaning/imputation:")
print(other_rows_final['fuel'].value_counts())
print("\nValue counts for 'year' in 'other' cylinders after further cleaning/imputation:")
print(other_rows_final['year'].value_counts().head(20))
print("\nSample rows with 'other' in cylinders after further cleaning/imputation:")
print(other_rows_final[['manufacturer', 'model', 'year', 'fuel', 'type']].head(20))

# Step 7: Drop all rows where cylinders is 'other'
# After all data-driven imputation steps (requiring at least 10 known values and 80%+ agreement for strong patterns),
# any remaining 'other' values in the 'cylinders' column are considered too ambiguous or rare to impute reliably.
# We drop these rows to ensure the 'cylinders' column is fully numeric and high-quality for downstream analysis.
# This avoids introducing bias from speculative guesses and ensures the workflow remains transparent and reproducible.
# The number of rows dropped and the new dataset size are reported for transparency.
rows_before = len(df_imputed)
df_imputed = df_imputed[df_imputed['cylinders'] != 'other']
rows_after = len(df_imputed)
print(f"\nDropped {rows_before - rows_after} rows where cylinders was 'other'.")
print(f"Final dataset size: {rows_after}")

print("\nCylinder value counts after dropping 'other':")
print(df_imputed['cylinders'].value_counts(dropna=False))

# Save the fully cleaned and imputed dataset to CSV
output_path = "Project/data/vehicles_cylinders_cleaned.csv"
df_imputed.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to {output_path}")


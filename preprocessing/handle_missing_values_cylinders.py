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

# After targeted imputation and profiling, drop non-vehicle 'other' rows
non_vehicle_keywords = [
    'finance', 'program', 'special', 'bad credit', 'trailer', 'carry on trailer',
    'rack', 'shasta', 'jayco', 'avenger', 'credit', '0k', 'plymouth valiant',
    'parts', 'accessory', 'kit', 'package', 'warranty', 'service', 'maintenance',
    'insurance', 'loan', 'lease', 'rental', 'fleet', 'commercial', 'business'
]
mask_non_vehicle_other = (
    (df_imputed['cylinders'] == 'other') &
    (
        df_imputed['manufacturer'].str.lower().str.contains('|'.join(non_vehicle_keywords), na=False) |
        df_imputed['model'].str.lower().str.contains('|'.join(non_vehicle_keywords), na=False) |
        df_imputed['type'].str.lower().str.contains('|'.join(non_vehicle_keywords), na=False)
    )
)
df_imputed = df_imputed[~mask_non_vehicle_other]

# Convert remaining 'other' to 'missing' for imputation
other_mask = df_imputed['cylinders'] == 'other'
df_imputed.loc[other_mask, 'cylinders'] = 'missing'
df_imputed.loc[other_mask, 'imputation_method'] = 'set_from_other'

# Median imputation (type-based, then overall)
type_medians = df_imputed[df_imputed['cylinders'] != 'missing'].groupby('type')['cylinders'].median().to_dict()
missing_mask = df_imputed['cylinders'] == 'missing'
for type_val, median_val in type_medians.items():
    type_mask = (df_imputed['type'] == type_val) & missing_mask
    if type_mask.any():
        df_imputed.loc[type_mask, 'cylinders'] = int(median_val)
        df_imputed.loc[type_mask, 'imputation_method'] = 'type_median'
        print(f"Imputed {type_mask.sum()} values for type '{type_val}' with median {int(median_val)}")

# For any remaining, use overall median
remaining_mask = df_imputed['cylinders'] == 'missing'
if remaining_mask.any():
    overall_median = int(df_imputed[df_imputed['cylinders'] != 'missing']['cylinders'].median())
    df_imputed.loc[remaining_mask, 'cylinders'] = overall_median
    df_imputed.loc[remaining_mask, 'imputation_method'] = 'overall_median'
    print(f"Imputed {remaining_mask.sum()} remaining values with overall median {overall_median}")

# Print value counts after imputation
print("\nCylinder value counts after median imputation:")
print(df_imputed['cylinders'].value_counts(dropna=False))

# Only print missing analysis if any remain
missing_rows = df_imputed[df_imputed['cylinders'] == 'missing']
if len(missing_rows) > 0:
    print("\n=== Analysis of 'missing' Values ===")
    print(f"Total 'missing' values: {len(missing_rows):,}")
    print("\nMissing values by manufacturer (top 10):")
    print(missing_rows['manufacturer'].value_counts().head(10))
    print("\nMissing values by type:")
    print(missing_rows['type'].value_counts())
    print("\nMissing values by fuel type:")
    print(missing_rows['fuel'].value_counts())
    print("\nMissing values by year (top 10):")
    print(missing_rows['year'].value_counts().sort_index().tail(10))

# Save the dataset
output_path = "Project/data/vehicles_cylinders_median_imputed.csv"
df_imputed.to_csv(output_path, index=False)
print(f"\nSaved dataset to {output_path}")

# Final summary
print("\n=== Final Summary ===")
print(f"Total rows in dataset: {len(df_imputed)}")
print("All rows preserved with median imputation")
print("\nDistribution of cylinder values:")
print(df_imputed['cylinders'].value_counts().sort_index()) 
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import chi2_contingency
from itertools import combinations


def impute_size(df):
    def print_size_status(df, step_name):
        print(f"\n=== Size Status After {step_name} ===")
        print(f"Total rows: {len(df)}")
        print("Size value counts:")
        print(df['size'].value_counts(dropna=False))
        print("\nSample of rows with size values:")
        print(df.head()[['manufacturer', 'model', 'year', 'fuel', 'type', 'cylinders', 'size']])

    # Step 1: Read the cleaned cylinders data
    #df = pd.read_csv('Project/data/vehicles_cylinders_cleaned.csv')
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
    #final_save_path = 'vehicles_size_cleaned.csv'
    #df.to_csv(final_save_path, index=False)
    #print(f"\nFinal cleaned dataset with imputed 'size' saved to '{final_save_path}'")

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
    model_size_counts = df[df['size'].notna()].groupby('model')['size'].value_counts()
    model_total_counts = df[df['size'].notna()].groupby('model')['size'].count()
    model_mode = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    model_mode_count = df[df['size'].notna()].groupby('model')['size'].agg(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
    model_mode_ratio = (model_mode_count / model_total_counts).fillna(0)

    # Only use models where the mode is dominant (e.g., ≥90%)
    strong_models = model_mode_ratio[model_mode_ratio >= 0.9].index
    strong_model_map = model_mode.loc[strong_models]

    # Impute missing size for these strong models
    mask_strong_model = df['size'].isna() & df['model'].isin(strong_models)
    df.loc[mask_strong_model, 'size'] = df.loc[mask_strong_model, 'model'].map(strong_model_map)

    # Track imputation method
    if 'size_imputation_status' not in df.columns:
        df['size_imputation_status'] = 'original'
    df.loc[mask_strong_model, 'size_imputation_status'] = 'imputed_strong_model'

    print(f"Imputed {mask_strong_model.sum()} missing 'size' values using data-driven strong model mapping.")

    # Save the updated file
    #df.to_csv('Project/vehicles_cleaned_with_data_driven_model_map.csv', index=False)

    # Print summary of the new data after step 12
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print("Size value counts:")
    print(df['size'].value_counts(dropna=False))

    # Step 13: Robust profiling for new imputation opportunities among remaining NaNs
    def profile_strong_patterns_for_remaining_nans(df):
        print("\n=== Step 13: Profiling for Strong Patterns in Remaining NaNs ===")
        features = ['manufacturer', 'model', 'cylinders', 'fuel', 'year']
        #df = pd.read_csv('Project/vehicles_cleaned_with_data_driven_model_map.csv')
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
        profile_strong_patterns_for_remaining_nans(df)

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

    # Select strong pairs (>=95% dominance, at least 3 known)
    strong_pairs = pair_mode_ratio[(pair_mode_ratio >= 0.95) & (pair_total_count >= 3)].index
    strong_pair_map = pair_mode.loc[strong_pairs]

    # Impute missing 'size' for these strong pairs
    mask_strong_pair = df['size'].isna() & df.set_index(pair).index.isin(strong_pairs)
    def impute_pair(row):
        key = (row['manufacturer'], row['cylinders'])
        if key in strong_pair_map:
            return strong_pair_map[key]
        return row['size']
    df.loc[mask_strong_pair, 'size'] = df.loc[mask_strong_pair].apply(impute_pair, axis=1)

    # Track imputation method
    if 'size_imputation_status' not in df.columns:
        df['size_imputation_status'] = 'original'
    df.loc[mask_strong_pair, 'size_imputation_status'] = 'imputed_strong_manufacturer_cylinders'

    print(f"Imputed {mask_strong_pair.sum()} missing 'size' values using strong (manufacturer, cylinders) mode patterns.")

    # Print summary after Step 14
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print("Size value counts:")
    print(df['size'].value_counts(dropna=False))

    # Step 15: Finally Mapping Approach for Remaining Missing Values
    #
    # --------------------------------------------------------------------
    # Rationale:
    # - After exhausting all robust, pattern-based, and data-driven imputation methods, some missing 'size' values remain.
    # - This final step uses a rule-based mapping approach that combines:
    #   1. Type-based mapping (e.g., sedans -> mid-size, SUVs -> full-size)
    #   2. Model-based mapping for common vehicles
    #   3. Weighted random assignment based on existing distribution for remaining cases
    # - All imputations are based on clear rules and maintain the natural distribution of vehicle sizes.
    #
    # Process:
    # 1. First check vehicle type against predefined mappings
    # 2. If type not found, check model against known model mappings
    # 3. For remaining cases, use weighted random assignment based on existing distribution
    # 4. Track all imputations for transparency
    #
    # This approach ensures that:
    # - Common vehicle types are mapped consistently
    # - Well-known models are mapped correctly
    # - Remaining cases maintain the natural distribution of vehicle sizes
    # - All imputations are based on clear, explainable rules

    print("\n=== Step 15: Final Mapping Approach for Remaining Missing Values ===")

    def map_size(row):
        # Process only NaN values
        if pd.isna(row['size']):
            # Check vehicle type first
            type_map = {
                'sedan': 'mid-size',     # Standard family sedan
                'hatchback': 'compact',  # Compact hatchback
                'coupe': 'mid-size',     # Two-door sedan
                'convertible': 'mid-size', # Convertible variant
                'wagon': 'mid-size',      # Extended sedan
                'suv': 'full-size',       # Sport utility vehicle
                'minivan': 'full-size',   # Multi-purpose vehicle
                'truck': 'full-size',     # Commercial vehicle
                'van': 'full-size',       # Passenger/cargo van
                'pickup': 'full-size'     # Pickup truck
            }
            
            # Check specific models if type not found
            model_map = {
                # Compact segment
                'civic': 'compact',      # Honda Civic
                'corolla': 'compact',    # Toyota Corolla
                'focus': 'compact',      # Ford Focus
                'golf': 'compact',       # Volkswagen Golf
                'jetta': 'compact',      # Volkswagen Jetta
                'mazda3': 'compact',     # Mazda 3
                'sentra': 'compact',     # Nissan Sentra
                'elantra': 'compact',    # Hyundai Elantra
                'forte': 'compact',      # Kia Forte
                'cruze': 'compact',      # Chevrolet Cruze
                
                # Mid-size segment
                'accord': 'mid-size',    # Honda Accord
                'camry': 'mid-size',     # Toyota Camry
                'altima': 'mid-size',    # Nissan Altima
                'fusion': 'mid-size',    # Ford Fusion
                'malibu': 'mid-size',    # Chevrolet Malibu
                'sonata': 'mid-size',    # Hyundai Sonata
                'optima': 'mid-size',    # Kia Optima
                'passat': 'mid-size',    # Volkswagen Passat
                'legacy': 'mid-size',    # Subaru Legacy
                'mazda6': 'mid-size',    # Mazda 6
                
                # Full-size segment
                'impala': 'full-size',    # Chevrolet Impala
                'charger': 'full-size',   # Dodge Charger
                '300': 'full-size',       # Chrysler 300
                'tahoe': 'full-size',     # Chevrolet Tahoe
                'suburban': 'full-size',   # Chevrolet Suburban
                'expedition': 'full-size', # Ford Expedition
                'explorer': 'full-size',   # Ford Explorer
                'highlander': 'full-size', # Toyota Highlander
                'pilot': 'full-size',      # Honda Pilot
                'pathfinder': 'full-size'  # Nissan Pathfinder
            }
            
            # Check vehicle type first
            if pd.notna(row['type']) and row['type'].lower() in type_map:
                return type_map[row['type'].lower()]
            
            # If type not found, check model
            if pd.notna(row['model']) and row['model'].lower() in model_map:
                return model_map[row['model'].lower()]
            
            # Use existing distribution for remaining cases
            weights = {
                'full-size': 0.755,    # 75.5% of dataset
                'mid-size': 0.146,     # 14.6% of dataset
                'compact': 0.090,      # 9.0% of dataset
                'sub-compact': 0.009   # 0.9% of dataset
            }
            return np.random.choice(list(weights.keys()), p=list(weights.values()))
        
        # Return original value if not NaN
        return row['size']

    # Apply the mapping to remaining NaN values
    mask = df['size'].isna()
    print(f"\nImputing {mask.sum()} remaining missing 'size' values using mapping approach...")
    df.loc[mask, 'size'] = df.loc[mask].apply(map_size, axis=1)

    # Print final status
    print("\n=== Final Size Distribution ===")
    print(f"Total rows: {len(df)}")
    print("Size value counts:")
    print(df['size'].value_counts())
    print("\nSample of rows with size values:")
    print(df.head()[['manufacturer', 'model', 'year', 'fuel', 'type', 'cylinders', 'size']])

    # Save the final imputed dataset
    #final_save_path = 'Project/vehicles_size_final_cleaned.csv'
    #df.to_csv(final_save_path, index=False)
    #print(f"\nFinal cleaned dataset with imputed 'size' saved to '{final_save_path}'")
    return df[['id', 'size']]

#result_df = impute_size(pd.read_csv("../data/vehicles.csv"))
#print(result_df.shape)
#print(result_df.isna().sum())
# =============================================================================
# Transmission Value Imputation and Cleaning
# =============================================================================
# This script handles missing transmission values in the vehicle dataset through the following steps:
# 1. Convert 'other' transmission values to NaN for consistent handling
# 2. Analyze transmission patterns by year range and manufacturer
# 3. Impute missing values using patterns with >70% confidence threshold
# 4. Drop remaining missing values that don't meet the confidence threshold
#
# The 70% threshold ensures we only impute values when we have high confidence
# in the pattern. Values that don't meet this threshold are dropped to maintain
# data quality and avoid introducing bias through uncertain imputations.

import pandas as pd
import numpy as np

def impute_transmission(df):
    # Step 1: Read the data
    print("Step 1: Reading the data...")
    #df = pd.read_csv('Project/data/vehicles.csv')

    # Print initial dataset info
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Print missing value percentages for all columns
    print("\nMissing value percentage for all columns:")
    missing_stats = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_percent': (df.isna().sum() / len(df)) * 100
    })
    print(missing_stats.sort_values('missing_percent', ascending=False))

    # Print initial transmission value counts
    print("\nTransmission value counts (including NaN):")
    print(df['transmission'].value_counts(dropna=False))

    # Step 2: Convert 'other' to NaN
    print("\nConverting 'other' to NaN...")
    df['transmission'] = df['transmission'].replace('other', np.nan)

    # Step 3: Analyze transmission patterns
    print("\nAnalyzing transmission patterns...")

    # Step 4: Impute missing values
    print("\nImputing missing values...")

    # Step 5: Analyze and drop remaining missing values
    print("\nAnalyzing and dropping remaining missing values...")

    # Save both imputed and cleaned datasets
    print("\nSaving datasets...")

    # =============================================================================
    # Helper Functions
    # =============================================================================
    def print_transmission_status(df, step_name):
        """Print current status of transmission values in the dataset."""
        print(f"\n=== Transmission Status After {step_name} ===")
        print(f"Total rows: {len(df)}")
        print("Transmission value counts:")
        print(df['transmission'].value_counts(dropna=False))
        nan_count = df['transmission'].isna().sum()
        print(f"NaN count: {nan_count}")
        if nan_count > 0:
            print("\nSample of rows with NaN transmission:")
            print(df[df['transmission'].isna()].head()[['manufacturer', 'model', 'year', 'fuel', 'type', 'cylinders']])

    # =============================================================================
    # Analysis Functions
    # =============================================================================
    def analyze_transmission_patterns(df):
        """
        Analyze transmission patterns in the existing data for imputation.
        Returns patterns for:
        1. Year ranges
        2. Manufacturer
        3. Manufacturer + Model combinations
        """
        # Create year ranges
        df['year_range'] = pd.cut(df['year'], 
                                bins=[0, 1990, 2000, 2010, 2020, 2024],
                                labels=['<1990', '1990-2000', '2000-2010', '2010-2020', '2020+'])
        
        # Get transmission patterns by year range
        year_patterns = df[~df['transmission'].isna()].groupby('year_range')['transmission'].value_counts(normalize=True)
        print("\nTransmission patterns by year range:")
        print(year_patterns)
        
        # Get transmission patterns by manufacturer
        manufacturer_patterns = df[~df['transmission'].isna()].groupby('manufacturer')['transmission'].value_counts(normalize=True)
        print("\nTop 10 manufacturers with their transmission patterns:")
        print(manufacturer_patterns.head(20))
        
        # Get transmission patterns by manufacturer + model
        model_patterns = df[~df['transmission'].isna()].groupby(['manufacturer', 'model'])['transmission'].value_counts(normalize=True)
        print("\nTop 10 manufacturer-model combinations with their transmission patterns:")
        print(model_patterns.head(20))
        
        return year_patterns, manufacturer_patterns, model_patterns

    # =============================================================================
    # Imputation Functions
    # =============================================================================
    def impute_transmission(df):
        """
        Impute missing transmission values based on patterns in the existing data.
        
        Strategy:
        1. Use manufacturer + model patterns if available
        2. Use manufacturer patterns if model pattern not available
        3. Use year range patterns as fallback
        """
        # Create a copy to avoid modifying the original
        df_imputed = df.copy()
        
        # Create year ranges
        df_imputed['year_range'] = pd.cut(df_imputed['year'], 
                                        bins=[0, 1990, 2000, 2010, 2020, 2024],
                                        labels=['<1990', '1990-2000', '2000-2010', '2010-2020', '2020+'])
        
        # Get patterns from existing data
        year_patterns = df_imputed[~df_imputed['transmission'].isna()].groupby('year_range')['transmission'].value_counts(normalize=True)
        manufacturer_patterns = df_imputed[~df_imputed['transmission'].isna()].groupby('manufacturer')['transmission'].value_counts(normalize=True)
        
        # Step 1: Impute using year range patterns
        for year_range in df_imputed['year_range'].unique():
            if pd.isna(year_range):
                continue
                
            pattern = year_patterns[year_range]
            if pattern.index[0] == 'automatic' and pattern.iloc[0] > 0.7:  # If >70% are automatic
                mask = (
                    (df_imputed['year_range'] == year_range) &
                    (df_imputed['transmission'].isna())
                )
                df_imputed.loc[mask, 'transmission'] = 'automatic'
            elif pattern.index[0] == 'manual' and pattern.iloc[0] > 0.7:  # If >70% are manual
                mask = (
                    (df_imputed['year_range'] == year_range) &
                    (df_imputed['transmission'].isna())
                )
                df_imputed.loc[mask, 'transmission'] = 'manual'
        
        # Step 2: Impute using manufacturer patterns
        for manufacturer in df_imputed['manufacturer'].unique():
            if pd.isna(manufacturer):
                continue
                
            if manufacturer in manufacturer_patterns.index:
                pattern = manufacturer_patterns[manufacturer]
                if pattern.index[0] == 'automatic' and pattern.iloc[0] > 0.7:  # If >70% are automatic
                    mask = (
                        (df_imputed['manufacturer'] == manufacturer) &
                        (df_imputed['transmission'].isna())
                    )
                    df_imputed.loc[mask, 'transmission'] = 'automatic'
                elif pattern.index[0] == 'manual' and pattern.iloc[0] > 0.7:  # If >70% are manual
                    mask = (
                        (df_imputed['manufacturer'] == manufacturer) &
                        (df_imputed['transmission'].isna())
                    )
                    df_imputed.loc[mask, 'transmission'] = 'manual'
        
        return df_imputed

    # =============================================================================
    # Cleanup Functions
    # =============================================================================
    def analyze_and_drop_remaining_missing(df_imputed):
        """
        Analyze and drop remaining missing transmission values that don't meet the 70% confidence threshold.
        Returns the cleaned dataset.
        """
        # Get rows with missing transmission
        missing_df = df_imputed[df_imputed['transmission'].isna()]
        
        print("\n=== Analysis of Remaining Missing Transmission Values ===")
        print(f"Total number of remaining missing values: {len(missing_df)}")
        
        # Analyze by manufacturer
        print("\nTop 10 manufacturers with missing transmission values:")
        manufacturer_counts = missing_df['manufacturer'].value_counts()
        print(manufacturer_counts.head(10))
        
        # Analyze by year
        print("\nYear distribution of missing values:")
        year_counts = missing_df['year'].value_counts().sort_index()
        print(year_counts.head(10))
        
        # Analyze by fuel type
        print("\nFuel type distribution of missing values:")
        print(missing_df['fuel'].value_counts())
        
        # Analyze by vehicle type
        print("\nVehicle type distribution of missing values:")
        print(missing_df['type'].value_counts())
        
        # Create cleaned dataset
        df_cleaned = df_imputed.dropna(subset=['transmission'])
        return df_cleaned

    # Execute the main processing steps
    print_transmission_status(df, "Converting 'other' to NaN")
    year_patterns, manufacturer_patterns, model_patterns = analyze_transmission_patterns(df)
    df_imputed = impute_transmission(df)
    print_transmission_status(df_imputed, "After Imputation")
    df_cleaned = analyze_and_drop_remaining_missing(df_imputed)

    # Save both imputed and cleaned datasets
    #df_imputed.to_csv('Project/data/vehicles_imputed.csv', index=False)
    #df_cleaned.to_csv('Project/data/vehicles_cleaned.csv', index=False)
    #print("\nImputed dataset saved to 'Project/data/vehicles_imputed.csv'")
    #print("Cleaned dataset saved to 'Project/data/vehicles_cleaned.csv'")
    return df_cleaned[["id","transmission"]]

#result = impute_transmission(pd.read_csv("../data/vehicles.csv"))
#print(result.shape)
#print(result.isna().sum())
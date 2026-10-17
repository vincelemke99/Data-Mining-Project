import pandas as pd
import test


def remove_iqr_outliers(df, column, factor=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
def calculate_skew(df, column):
    return df[column].skew()


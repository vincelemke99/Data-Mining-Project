import pandas as pd

# Assumption is that the data is already split into train and test sets
train_df = pd.read_csv("../data/train_data_preprocessed.csv")
test_df = pd.read_csv("../data/test_data_preprocessed.csv")

def impute_condition(df):
    """
    replaces missing values with the condition "missing"
    """
    transformed_column = df["condition"].fillna("missing")
    return transformed_column


train_df["condition"] = impute_condition(train_df)
test_df["condition"] = impute_condition(test_df)


print("Missing values in train set after imputation:")
print(train_df.isna().sum())
print("Missing values in test set after imputation:")
print(test_df.isna().sum())

train_df.to_csv("../data/train_data_preprocessed.csv", index=False)
test_df.to_csv("../data/test_data_preprocessed.csv", index=False)
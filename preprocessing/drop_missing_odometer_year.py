import pandas as pd
import test

train_set = pd.read_csv("../data/train_data_preprocessed.csv")
test_set = pd.read_csv("../data/test_data_preprocessed.csv")

print(train_set["condition"].value_counts())
def drop_missing_odometer_year(train_set, test_set):
    """
    Drop rows with missing values in the 'odometer' and 'year' columns from both train and test sets.
    """

    missing_train = train_set[train_set[['year', 'odometer']].isnull().any(axis=1)]
    missing_test  = test_set[test_set[['year', 'odometer']].isnull().any(axis=1)]


    train_clean = train_set.drop(missing_train.index)
    test_clean  = test_set.drop(missing_test.index)


    missing_train.to_csv("../data/dropped_train_odometer_year_rows.csv", index=False)
    missing_test.to_csv("../data/dropped_test_odometer_year_rows.csv", index=False)


    train_clean.to_csv("../data/train_data_preprocessed.csv", index=False)   
    test_clean.to_csv("../data/test_data_preprocessed.csv", index=False)



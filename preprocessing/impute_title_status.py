
import pandas as pd

# this function will be used for preprocessing missing values
def impute_title_status(raw_data):
    """
    Returns the column "title_status" with filled "missing" if the observation is missing.
    """
    transformed_column = raw_data["title_status"].fillna("missing")
    raw_data["title_status_imputed"] = transformed_column
    return raw_data[["id", "title_status_imputed"]]


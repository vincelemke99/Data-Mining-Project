import pandas as pd

#raw_data = pd.read_csv("../data/vehicles.csv")

#print(raw_data["type"].value_counts())

def impute_type(dataframe):
    """This function maps the attribute "type" in a three-step process: it first looks uses the attribute "model" by finding the most common non-null "type" for each model.  
    It then fills in missing type values for rows that have the same model. If there are still missing values, manufacturer is then used in a similar process. 
    Finally, it fills in any remaining missing values with "missing". The function returns the transformed column."""

    dataframe_copy = dataframe.copy()

    # Most common type per model (from known rows)
    model_type_map = dataframe_copy[dataframe['type'].notnull()].groupby('model')['type'].agg(lambda x: x.mode().iloc[0])

    # Fill in missing 'type' using this mapping
    dataframe_copy.loc[dataframe_copy['type'].isnull(), 'type'] = dataframe_copy.loc[dataframe_copy['type'].isnull(), 'model'].map(model_type_map)

    # print(dataframe_copy["type"].isna().sum())

    # Most common type per manufacturer
    manu_type_map = dataframe_copy[dataframe_copy['type'].notnull()].groupby('manufacturer')['type'].agg(lambda x: x.mode().iloc[0])

    # Fill in remaining missing values
    dataframe_copy.loc[dataframe_copy['type'].isnull(), 'type'] = dataframe_copy.loc[dataframe_copy['type'].isnull(), 'manufacturer'].map(manu_type_map)

    # print(dataframe_copy["type"].isna().sum())

    dataframe_copy['type'].fillna('missing', inplace=True)

    return dataframe_copy[["id", "type"]]

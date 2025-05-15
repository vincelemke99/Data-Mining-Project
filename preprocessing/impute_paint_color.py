import pandas as pd

# Assumption is that the data is already split into train and test sets
def impute_mode_paint_color(raw_data):
    """
    returns transformed column in which missing values are replaced with most common label => white.
    """
    transformed_column = raw_data["paint_color"].fillna(raw_data["paint_color"].mode()[0])
    raw_data["paint_color_imputed"] = transformed_column
    return raw_data[["id", "paint_color_imputed"]]

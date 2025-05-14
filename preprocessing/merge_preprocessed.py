import dask.dataframe as dd

# Load main file (model and manufacturer imputed)
df_main = dd.read_csv('Used-Car-Data-Mining/preprocessing/vehicles_imputed_extended.csv')

# Load other preprocessed files
print('Loading other files...')
df_cylinders = dd.read_csv('Project/data/vehicles_cylinders_median_imputed.csv', usecols=['id', 'cylinders'])
df_size = dd.read_csv('Project/data/vehicles_size_final_cleaned.csv', usecols=['id', 'size'])
df_transmission = dd.read_csv('Project/data/vehicles_transmission_cleaned.csv', usecols=['id', 'transmission'])
df_type = dd.read_csv('Used-Car-Data-Mining/preprocessing/vehicles_type_imputed.csv', usecols=['id', 'type'])
df_title_status = dd.read_csv('Used-Car-Data-Mining/preprocessing/vehicles_title_status_imputed.csv', usecols=['id', 'title_status'])
df_paint_color = dd.read_csv('Used-Car-Data-Mining/preprocessing/vehicles_paint_color_imputed.csv', usecols=['id', 'paint_color'])
# drive missing. when doing drive, type needs to be imputed first

print('Merging files...')
df_merged = df_main.merge(df_cylinders, on='id', how='inner')
df_merged = df_merged.merge(df_size, on='id', how='inner')
df_merged = df_merged.merge(df_transmission, on='id', how='inner')
df_merged = df_merged.merge(df_type, on='id', how='inner')
df_merged = df_merged.merge(df_title_status, on='id', how='inner')
df_merged = df_merged.merge(df_paint_color, on='id', how='inner')

print('Columns after merging:')
print(df_merged.columns)

# Rename columns to remove _x and _y suffixes if present
def clean_column_names(columns):
    return [col.replace('_x', '').replace('_y', '') for col in columns]
df_merged.columns = clean_column_names(df_merged.columns)

# Save the cleaned, fully-complete file (compute to pandas, then save)
df_merged.compute().to_csv('Used-Car-Data-Mining/data/vehicles_merged_no_missing.csv', index=False)
print("Saved merged file with all columns and values as 'Used-Car-Data-Mining/data/vehicles_merged_no_missing.csv'.") 
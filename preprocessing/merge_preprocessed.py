import pandas as pd 
import model_imputation
import drive_imputation
import manufacturer_imputation
import impute_title_status
import handle_missing_values_transmission
import handle_missing_values_size
import impute_paint_color
import type_imputation
import handle_missing_values_cylinders

# read split datasets
train_df = pd.read_csv("../data/train_data.csv")
test_df = pd.read_csv("../data/test_data.csv")

train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

print("size of test data: ", test_df.shape)
# transform train and test data SEPARATELY to avoid data leakage
# ORDER: model, manufacturer, type, drive, rest as below, FINAL one = size 

# model only needs input df, no other prerequisites. remember it drops all rows Nulls in model column ==> merge via id
# manufacturer: make sure to drop remaining nulls => done in code
# drive has no missing values. when doing drive, type needs to be imputed first
# size depends on cylinders and other attribtues, so input argument needs to be preprocessed
# transmission already dropped nulls in preprocessing
print("============ Preprocessing train split ===========")

df_model = model_imputation.impute_model(train_df_copy)

print("1. join model with training set, drop model column and rename model_imputed to model")
df_train_merged = train_df.merge(df_model, on="id", how="inner")
df_train_merged = df_train_merged.drop(columns=["model"], axis=1)
df_train_merged = df_train_merged.rename(columns={"model_imputed": "model"})

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

train_df_copy = df_train_merged.copy()
df_manufacturer = manufacturer_imputation.impute_manufacturer(train_df_copy)

print("2. join manufacturer with training set, drop manufacturer column and rename manufacturer_imputed to manufacturer")
df_train_merged = df_train_merged.merge(df_manufacturer, on="id", how="inner")
df_train_merged = df_train_merged.drop(columns=["manufacturer"], axis=1)
df_train_merged = df_train_merged.rename(columns={"manufacturer_imputed": "manufacturer"})

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

df_type = type_imputation.impute_type(df_train_merged)

print("3. join type with training set, drop type column")
df_train_merged = df_train_merged.drop(columns=["type"], axis=1)
df_train_merged = df_train_merged.merge(df_type, on="id", how="inner")

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

train_df_copy = df_train_merged.copy()
df_drive = drive_imputation.impute_drive(train_df_copy)

print("4. join drive with training set, drop drive column and rename drive_imputed to drive")
df_train_merged = df_train_merged.merge(df_drive, on="id", how="inner")
df_train_merged = df_train_merged.drop(columns=["drive"], axis=1)
df_train_merged = df_train_merged.rename(columns={"drive_imputed": "drive"})

print(df_train_merged.isna().sum())
print(df_train_merged.shape) 
# drop rows with missing values in drive column
df_train_merged = df_train_merged.dropna(subset=["drive"])

df_cylinders = handle_missing_values_cylinders.impute_cylinders(df_train_merged)

print("5. join cylinders with training set, drop cylinders column")
df_train_merged = df_train_merged.drop(columns=["cylinders"], axis=1)
df_train_merged = df_train_merged.merge(df_cylinders, on="id", how="inner")

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

train_df_copy = df_train_merged.copy()
df_title_status = impute_title_status.impute_title_status(train_df_copy)

print("6. join title_status with training set, drop title_status column")
df_train_merged = df_train_merged.merge(df_title_status, on="id", how="inner")
df_train_merged = df_train_merged.drop(columns=["title_status"], axis=1)
df_train_merged = df_train_merged.rename(columns={"title_status_imputed": "title_status"})

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

train_df_copy = df_train_merged.copy()
df_paint_color = impute_paint_color.impute_mode_paint_color(train_df_copy)

print("7. join paint_color with training set, drop paint_color column")
df_train_merged = df_train_merged.merge(df_paint_color, on="id", how="inner")
df_train_merged = df_train_merged.drop(columns=["paint_color"], axis=1)
df_train_merged = df_train_merged.rename(columns={"paint_color_imputed": "paint_color"})

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

df_transmission = handle_missing_values_transmission.impute_transmission(df_train_merged)

print("8. join transmission with training set, drop transmission column")
df_train_merged = df_train_merged.drop(columns=["transmission"], axis=1)
df_train_merged = df_train_merged.merge(df_transmission, on="id", how="inner")

print(df_train_merged.isna().sum())
print(df_train_merged.shape)

#train_df_copy = df_train_merged.copy()
#df_size = handle_missing_values_size.impute_size(train_df_copy)

print("9. join size with training set, drop size column")
#df_train_merged = df_train_merged.drop(columns=["size"], axis=1)
#df_train_merged = df_train_merged.merge(df_size, on="id", how="inner")


print("FINISHED WITH TRAINING SET")
print("============ Preprocessing Test Split ===========")

df_model = model_imputation.impute_model(test_df_copy)

print("1. join model with test set, drop model column and rename model_imputed to model")
df_test_merged = test_df.merge(df_model, on="id", how="inner")
df_test_merged = df_test_merged.drop(columns=["model"], axis=1)
df_test_merged = df_test_merged.rename(columns={"model_imputed": "model"})

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

test_df_copy = df_test_merged.copy()
df_manufacturer = manufacturer_imputation.impute_manufacturer(test_df_copy)

print("2. join manufacturer with test set, drop manufacturer column and rename manufacturer_imputed to manufacturer")
df_test_merged = df_test_merged.merge(df_manufacturer, on="id", how="inner")
df_test_merged = df_test_merged.drop(columns=["manufacturer"], axis=1)
df_test_merged = df_test_merged.rename(columns={"manufacturer_imputed": "manufacturer"})

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

df_type = type_imputation.impute_type(df_test_merged)

print("3. join type with test set, drop type column")
df_test_merged = df_test_merged.drop(columns=["type"], axis=1)
df_test_merged = df_test_merged.merge(df_type, on="id", how="inner")

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

test_df_copy = df_test_merged.copy()
df_drive = drive_imputation.impute_drive(test_df_copy)

print("4. join drive with test set, drop drive column and rename drive_imputed to drive")
df_test_merged = df_test_merged.merge(df_drive, on="id", how="inner")
df_test_merged = df_test_merged.drop(columns=["drive"], axis=1)
df_test_merged = df_test_merged.rename(columns={"drive_imputed": "drive"})

print(df_test_merged.isna().sum())
print(df_test_merged.shape) 
# drop rows with missing values in drive column
df_test_merged = df_test_merged.dropna(subset=["drive"])

df_cylinders = handle_missing_values_cylinders.impute_cylinders(df_test_merged)

print("5. join cylinders with training set, drop cylinders column")
df_test_merged = df_test_merged.drop(columns=["cylinders"], axis=1)
df_test_merged = df_test_merged.merge(df_cylinders, on="id", how="inner")

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

test_df_copy = df_test_merged.copy()
df_title_status = impute_title_status.impute_title_status(test_df_copy)

print("6. join title_status with training set, drop title_status column")
df_test_merged = df_test_merged.merge(df_title_status, on="id", how="inner")
df_test_merged = df_test_merged.drop(columns=["title_status"], axis=1)
df_test_merged = df_test_merged.rename(columns={"title_status_imputed": "title_status"})

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

test_df_copy = df_test_merged.copy()
df_paint_color = impute_paint_color.impute_mode_paint_color(test_df_copy)

print("7. join paint_color with training set, drop paint_color column")
df_test_merged = df_test_merged.merge(df_paint_color, on="id", how="inner")
df_test_merged = df_test_merged.drop(columns=["paint_color"], axis=1)
df_test_merged = df_test_merged.rename(columns={"paint_color_imputed": "paint_color"})

print(df_test_merged.isna().sum())
print(df_test_merged.shape)

df_transmission = handle_missing_values_transmission.impute_transmission(df_test_merged)

print("8. join transmission with training set, drop transmission column")
df_test_merged = df_test_merged.drop(columns=["transmission"], axis=1)
df_test_merged = df_test_merged.merge(df_transmission, on="id", how="inner")

print(df_test_merged.isna().sum())
print(df_test_merged.shape)


#test_df_copy = df_test_merged.copy()
#df_size = handle_missing_values_size.impute_size(test_df_copy)

print("9. join size with test set, drop size column")
#df_test_merged = df_test_merged.drop(columns=["size"], axis=1)
#df_test_merged = df_test_merged.merge(df_size, on="id", how="inner")

print("FINISHED WITH TEST SET")

print("SAVING FILES")
df_train_merged.to_csv("../data/train_data_preprocessed.csv", index=False)
df_test_merged.to_csv("../data/test_data_preprocessed.csv", index=False)
import pandas as pd


train_df = pd.read_csv("../data/train_data_preprocessed.csv")

def remove_iqr_outliers(df):
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    return df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
    
def calculate_skew(df):
    return df["price"].skew()

df_new = remove_iqr_outliers(train_df)

print(df_new.shape)
print(df_new.describe())


df_new.to_csv("../data/train_data_preprocessed_no_outliers.csv", index=False)



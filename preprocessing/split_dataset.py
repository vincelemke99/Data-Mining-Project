import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gdrive_pull_push

# Split dataset into a training and test set with a ratio of 80:20
raw_data = pd.read_csv("../data/vehicles.csv")
X_train, X_test, y_train, y_test = train_test_split(raw_data.drop(columns=["price"]), raw_data["price"], test_size=0.2, random_state=42)


# combine X_train and y_train into a single DataFrame
train_data = pd.concat([X_train, y_train], axis=1)

# combine X_test and y_test into a single DataFrame
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv("../data/train_data.csv", index=False)
test_data.to_csv("../data/test_data.csv", index=False)

# save the training and test data in the google cloud using the gdrive_pull_push module s
service = gdrive_pull_push.get_service()
gdrive_pull_push.upload_file(service, "../data/train_data.csv", "train_data.csv")
gdrive_pull_push.upload_file(service, "../data/test_data.csv", "test_data.csv")
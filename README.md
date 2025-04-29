# Used-Car-Data-Mining

Welcome to the Data Mining project revolving around the Craig's List-based used-car dataset. The goal of this project is to predict the price of listed vehicles.

## Dataset

The dataset consists of roughly 426,000 raw listings. The original dataset can be found on Kaggle:  
https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

To simplify access and avoid handling large files through GitHub, we host the dataset on a shared Google Drive associated with our team account.

### Accessing the Dataset

There are two ways to access the dataset:

#### **Option 1: Manual Download**

1. Log in to the **team Google account (team.dataset@gmail.com)**.
2. Navigate to the shared folder named `used-car-data-mining-project`.
3. Download the dataset file (e.g., `vehicles.csv`) to your computer.
4. Move the file into the `data/` directory of this project.

#### **Option 2: Use the Python Script**

We provide a script to automatically pull the dataset from Google Drive using the Google Drive API.

##### Prerequisites:

- Install the following dependencies (or look at requirements.txt):

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client httplib2
```

- `credentials.json` file must be present in the project directory.
- The file is shared via Google Drive and should **not be committed to Git**.
  ===> Make sure to keep credentials.json in .gitignore

##### Setup:

1. Download `credentials.json` from the shared Google Drive folder.
2. Place it in the root directory of this project
3. Run the script, which is in the data, to download the dataset:

```bash
python data/fetch_dataset.py
```

##### Reminder

The Google Drive only has 15 GB of FREE storage space. The upload function in the script can be used to NEW upload files to the drive.

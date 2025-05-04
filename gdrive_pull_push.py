from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import os

# If modifying these SCOPES, delete the file token.json
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '../credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def download_file(service, file_id, destination_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download progress: {int(status.progress() * 100)}%")
    print("Download complete.")

def upload_file(service, file_path, file_name, parent_folder_id=None):
    file_metadata = {'name': file_name}
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded file with ID: {file.get('id')}")
    return file.get('id')

if __name__ == '__main__':
    FILE_ID = '1HNc9VMiR7GwQSC6tsBPaBD53o8kmyjVd'
    LOCAL_PATH = 'dataset.csv'
    UPLOAD_NAME = 'dataset_uploaded.csv'

    creds = authenticate()
    #service = build('drive', 'v3', credentials=creds)

    #download_file(service, FILE_ID, LOCAL_PATH)
    # UPLOAD CREATES A NEW FILE
    #upload_file(service, LOCAL_PATH, UPLOAD_NAME)

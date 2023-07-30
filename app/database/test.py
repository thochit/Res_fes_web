from __future__ import print_function

import os.path

import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'app/database/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')

def search_file(folder_id, csv_file):
    """Recursively searches for image files in a folder and its subfolders in Google Drive
    and saves the file ID and name in a CSV file.

    Args:
        folder_id (str): The ID of the folder in Google Drive.
        csv_file (str): The name of the CSV file to save the results.

    Returns:
        list: A list of file objects found in the specified folder and its subfolders.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'app/database/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        files = []
        page_token = None
        while True:
            response = service.files().list(q=f"'{folder_id}' in parents and mimeType='image/jpeg'",
                                            spaces='drive',
                                            fields='nextPageToken, files(id, name)',
                                            pageToken=page_token).execute()
            for file in response.get('files', []):
                file_id = file.get('id')
                file_name = file.get('name')
                print(f'Found file: {file_name}, {file_id}')
                files.append({'ID': file_id, 'Name': file_name})
                # Save the file ID and name to the CSV file
                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['ID', 'Name'])
                    writer.writerow({'ID': file_id, 'Name': file_name})
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        subfolders = service.files().list(q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'",
                                          spaces='drive',
                                          fields='files(id)').execute().get('files', [])
        for subfolder in subfolders:
            subfolder_id = subfolder.get('id')
            subfolder_files = search_file(subfolder_id, csv_file)
            files.extend(subfolder_files)

    except HttpError as error:
        print(f'An error occurred: {error}')
        files = None

    return files

if __name__ == '__main__':
    # main()
    folder_id = '14VAMXo3OJC3M_cVx7SZsSinmWFKup0sP'
    csv_file = 'file_details.csv'
    search_file(folder_id, csv_file)



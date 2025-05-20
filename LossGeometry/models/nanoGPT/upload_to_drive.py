#!/usr/bin/env python3
"""
Upload the out-spectral.zip file to Google Drive.

This script uses PyDrive to authenticate with Google Drive and upload the specified file.
"""

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

def upload_to_drive(file_path, folder_id=None):
    """
    Upload a file to Google Drive
    
    Args:
        file_path: Path to the file to upload
        folder_id: ID of the folder to upload to (optional)
    
    Returns:
        File ID of the uploaded file
    """
    print(f"Uploading {file_path} to Google Drive...")
    
    # Authenticate with Google Drive
    gauth = GoogleAuth()
    
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    
    if gauth.credentials is None:
        # Authenticate if they're not available
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    
    drive = GoogleDrive(gauth)
    
    # Create file object and set title to the filename
    file_name = os.path.basename(file_path)
    drive_file = drive.CreateFile({'title': file_name})
    
    # If folder ID is provided, set it as parent
    if folder_id:
        drive_file['parents'] = [{'id': folder_id}]
    
    # Set the file content and upload
    drive_file.SetContentFile(file_path)
    drive_file.Upload()
    
    print(f"Successfully uploaded {file_name} to Google Drive with ID: {drive_file['id']}")
    print(f"View it at: https://drive.google.com/file/d/{drive_file['id']}")
    
    return drive_file['id']

if __name__ == "__main__":
    # Path to the zip file
    zip_path = "out-spectral.zip"
    
    # Check if the file exists
    if not os.path.exists(zip_path):
        print(f"Error: File {zip_path} not found.")
        print("Please ensure you're running this script from the LossGeometry/models/nanoGPT directory.")
        exit(1)
    
    # Upload the file to Google Drive
    file_id = upload_to_drive(zip_path) 
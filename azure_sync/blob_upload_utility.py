
from azure.storage.blob import BlobServiceClient
# Paste your connection string (from Key1)
CONNECTION_STRING = "xxxxxxxxxxxxxxx"
# Container (Bucket 1)
CONTAINER_NAME = "bucket1"
# File to upload (from your VM)
FILE_PATH = "entity_keys/server_key.zip"
# Name in cloud
BLOB_NAME = "server_key.zip"
def upload_to_bucket():
    try:
        # Create Blob service client
        blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        # Get blob client
        blob_client = blob_service.get_blob_client(
            container=CONTAINER_NAME,
            blob=BLOB_NAME
        )
        # Upload file
        with open(FILE_PATH, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print("Upload successful: server_key.zip -> bucket1")
    except Exception as e:
        print("Upload failed:", e)
if __name__ == "__main__":
    upload_to_bucket()

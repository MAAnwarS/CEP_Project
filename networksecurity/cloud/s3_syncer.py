import os

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        if not folder or not aws_bucket_url:
            print("Error: Folder path or AWS bucket URL is missing.")
            return
        command = f'aws s3 sync "{folder}" "{aws_bucket_url}"'
        print(f"Running: {command}")
        os.system(command)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        if not folder or not aws_bucket_url:
            print("Error: Folder path or AWS bucket URL is missing.")
            return
        command = f'aws s3 sync "{aws_bucket_url}" "{folder}"'
        print(f"Running: {command}")
        os.system(command)

def load_data_cloud_storage():
    from io import BytesIO
    import pandas as pd
    from google.cloud import storage
    storage_client = Client.from_service_account_json(“Location of JSON file”, project=’project name’)
    bucket = storage_client.get_bucket(“bucket name”)
    path = “Image path”
    filename = “%s%s” % (‘’,path)
    blob = self.bucket.blob(“<GCP Folder Name>/{}.jpg”.format(<Image Name on GCP>))
    blob.content_type = “image/jpeg”
    with open(path, ‘rb’) as f:
    blob.upload_from_file(f)
    print(“Image Uploaded : “)

import os
from minio import Minio
from minio.error import S3Error

def main():
    # MinIO setup
    client = Minio(
        "localhost:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    
    bucket_name = "mlpipeline"
    file_path = "data/processed/enriched_products.json"
    object_name = "enriched_products.json"
    
    # Check if bucket exists, create if not
    try:
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            print(f"Created bucket '{bucket_name}'")
        else:
            print(f"Bucket '{bucket_name}' already exists")
            
        # Upload the file
        if os.path.exists(file_path):
            client.fput_object(bucket_name, object_name, file_path)
            print(f"Successfully uploaded {file_path} to minio://{bucket_name}/{object_name}")
        else:
            print(f"Error: Local file {file_path} does not exist.")
            
    except S3Error as err:
        print("MinIO Error:", err)

if __name__ == "__main__":
    main()

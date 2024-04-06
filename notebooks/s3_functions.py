import os
import io
import boto3
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def list_s3_contents(file_path, access_key_id=None, secret_access_key=None, bucket_name=None):
    """
    List the contents of an S3 bucket path, prioritizing directories first,
    then files in alphabetical order.

    Args:
        file_path (str): The S3 bucket path to list.
        access_key_id (str): AWS access key ID.
        secret_access_key (str): AWS secret access key.
        bucket_name (str): Name of the S3 bucket.
    """
    # Use global variables if specific credentials are not provided
    if access_key_id is None:
        access_key_id = globals().get('aws_access_key_id', '')
    if secret_access_key is None:
        secret_access_key = globals().get('aws_secret_access_key', '')
    if bucket_name is None:
        bucket_name = globals().get('s3_bucket_name', '')

    # Initialize a boto3 client
    s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    # Add a trailing slash if not present to properly emulate directory behavior
    if not file_path.endswith('/'):
        file_path += '/'

    # List objects in the specified path
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=file_path, Delimiter='/')

    # Collect directories (CommonPrefixes) and files
    directories = [cp['Prefix'] for cp in response.get('CommonPrefixes', [])]
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'] != file_path]

    # Sort directories and files separately
    directories.sort()
    files.sort()

    # Combine directories and files for listing
    all_contents = directories + files

    # Print or return the sorted list
    for item in all_contents:
        print(item)

    return all_contents


def load_from_s3(file_path, access_key_id=None, secret_access_key=None, bucket_name=None):
    '''
    Download a file from the S3 bucket location

    Args:
        file_path (str): The path of the file within the S3 bucket.
        access_key_id (str, optional): The AWS access key ID. Defaults to global aws_access_key_id variable.
        secret_access_key (str, optional): The AWS secret access key. Defaults to global aws_secret_access_key variable.
        bucket_name (str, optional): The name of the S3 bucket. Defaults to global s3_bucket_name variable.

    Returns:
        io.BytesIO: A BytesIO object containing the file content.
    '''
    # Use global variables if specific credentials are not provided
    if access_key_id is None:
        access_key_id = globals().get('aws_access_key_id', '')
    if secret_access_key is None:
        secret_access_key = globals().get('aws_secret_access_key', '')
    if bucket_name is None:
        bucket_name = globals().get('s3_bucket_name', '')

    # Initialize a boto3 s3 client with credentials from the .env file
    s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    # Use the client to grab the data
    f_obj = s3_client.get_object(Bucket=bucket_name, Key=file_path)

    # Set f to the body of the file object
    f = io.BytesIO(f_obj['Body'].read())

    s3_client.close()

    return f


def write_to_s3(file_path, data, access_key_id=None, secret_access_key=None, bucket_name=None, pickle_file=False):
    '''
    Upload a file to the S3 bucket location

     Args:
        file_path (str): The path to store the data within the S3 bucket.
        data ([pd.DataFrame, Any]): The data to upload. Can be a Pandas DataFrame or any picklable object.
        access_key_id (str, optional): The AWS access key ID. Defaults to global aws_access_key_id variable.
        secret_access_key (str, optional): The AWS secret access key. Defaults to global aws_secret_access_key variable.
        bucket_name (str, optional): The name of the S3 bucket. Defaults to global s3_bucket_name variable.
        pickle_file (bool, optional): Whether to pickle the data before uploading. Defaults to False.
    '''
    # Use global variables if specific credentials are not provided
    if access_key_id is None:
        access_key_id = globals().get('aws_access_key_id', '')
    if secret_access_key is None:
        secret_access_key = globals().get('aws_secret_access_key', '')
    if bucket_name is None:
        bucket_name = globals().get('s3_bucket_name', '')

    # Open the S3 client
    s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    if pickle_file:
        pickled_data = pickle.dumps(data)
        s3_client.put_object(Bucket=bucket_name, Key=file_path, Body=pickled_data)

    elif isinstance(data, pd.DataFrame):
        with io.BytesIO() as buffer:
            data.to_parquet(buffer)
            buffer.seek(0)
            s3_client.put_object(Bucket=bucket_name, Key=file_path, Body=buffer)

    else:
        s3_client.close()
        raise ValueError("Unsuppored data type for upload")

    s3_client.close()
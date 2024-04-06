import os
import io
import boto3
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv, dotenv_values
from pathlib import Path

def load_env_vars():
    global aws_access_key_id, aws_secret_access_key, s3_bucket_name, neptune_project, neptune_api_key, mapbox_api_key

    # Determine ecosystem and load appropriate variables
    if os.getenv('DEEPNOTE_RUNTIME_UUID'):
        deepnote = True

        # Since we're in Deepnote, it has native S3 integration, so this is not required to be set,
        #   but we're setting values here for the s3_functions.py file, which assumes some defaults
        aws_access_key_id = 'Not_Valid'
        aws_secret_access_key = 'Not_Valid'
        s3_bucket_name = 'Not_Valid'

        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        os.environ['S3_BUCKET_NAME'] = aws_secret_access_key
        neptune_project = os.getenv('NEPTUNE_PROJECT')
        neptune_api_key = os.getenv('NEPTUNE_API_TOKEN')
        mapbox_api_key = os.getenv('MAPBOX_TOKEN')

        print('Running on Deepnote with Env and S3 integrations, skipping dotenv')
    else:
        deepnote = False
        print('Loading dotenv file')

        # Private file contains non-public variable configurations for local development.  Not loaded to github.
        # variables.env can be populated with user specific API and Access keys and is empty by default.  Loaded to github.
        private_vars_path = Path("../private_variables.env")
        var_path = Path("../variables.env")

        # Use the private vars if exists, otherwise use the public vars file
        env_path = private_vars_path if private_vars_path.exists() else var_path

        # Load the environment variables from the env path
        load_dotenv(env_path)

        # Get the environment variables as a dict that can be returned
        #env_dict = dotenv_values(env_path)

        # Establish the variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        neptune_project = os.getenv('NEPTUNE_PROJECT')
        neptune_api_key = os.getenv('NEPTUNE_API_TOKEN')
        mapbox_api_key = os.getenv('MAPBOX_TOKEN')

    env_dict = {
        'aws_access_key_id': aws_access_key_id,
        'aws_secret_access_key': aws_secret_access_key,
        's3_bucket_name': s3_bucket_name,
        'neptune_project': neptune_project,
        'neptune_api_key': neptune_api_key,
        'mapbox_api_key': mapbox_api_key,
    }

    return deepnote, env_dict
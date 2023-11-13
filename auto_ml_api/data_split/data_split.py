from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
import pandas as pd
import numpy as np

# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

project_path = os.path.join(path,'tools')

sys.path.insert(0, project_path)
from db_tools import get_dataset, save_split_strategy
from supported_data_formats import read_supported
from oauth2_tools import verify_token
@require_http_methods(["POST"])
@csrf_exempt
def split_data_api(request):
    # Parse the request body as JSON
    request_body = json.loads(request.body.decode())
    verify_token(request)
    if verify_token=="failed":
        return JsonResponse({"message":"session expired"}, status=401)    
    # Get the file path from the request body
    id_dataset = request_body.get('idDataset')
    # Get the train and test percentages from the request body
    train_percentage = request_body.get('train')
    test_percentage = request_body.get('test')
    seed = request_body.get('seed')
    shuffle = request_body.get('shuffle')
    if not all([id_dataset, train_percentage, test_percentage, seed]) and shuffle is not None:
        return JsonResponse({"message":'field is required',"success":False}, status=400)

    file_path, filename = get_dataset(id_dataset)
    
    file_path=dataset_path_prefix+file_path
    # df = pd.read_csv(file_path) #  CHANGED: Mar 6, 2023
    df = read_supported(file_path) #  CHANGED: Mar 6, 2023
    
    # Get the names of the columns in the DataFrame
    columns = df.columns   
    # Calculate the number of rows for the training and testing sets
    num_rows = df.shape[0]
    num_train_rows = np.floor(num_rows * train_percentage / 100)
    num_test_rows = np.floor(num_rows * test_percentage / 100)
    
    save_split_strategy(id_dataset,train_percentage,test_percentage,seed,shuffle)
    response = {
    'columns': columns.tolist(),
    'test_rows': num_test_rows,
    'train_rows': num_train_rows,
    "success":True,
    "message":""
    }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
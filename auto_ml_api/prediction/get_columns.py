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
from supported_data_formats import read_supported
from db_tools import get_dataset
from get_variables_types_names import get_variables
from oauth2_tools import verify_token
from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter
@require_http_methods(["GET"])
@csrf_exempt
def get_columns(request):

    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get columns",context=context) as span:
        try:
            method="GET"  
            span.set_attribute("http.method", method)
            # Get the file path from the params
            id_dataset=request.GET.get('idDataset')
            if id_dataset is None:
                response={"data":[],"message":'id is required',"success":False}
                return JsonResponse(response, status=400)
            verify_token(request)
            if verify_token=="failed":
                return JsonResponse({"message":"session expired"}, status=401) 
            # Get path of the descriptive statistics JSON and the data file
            file_path, filename = get_dataset(id_dataset)
            file_path=dataset_path_prefix+file_path
            
            #Load the data CSV file 
            df = read_supported(file_path)
            clean_data_path= os.path.join(dataset_path_prefix,"dataset/"+filename+".json")

            #check if the JSON file exists
            if os.path.exists(clean_data_path):
                
                #Load the JSON file
                with open(clean_data_path) as f:
                    clean_data = json.load(f)
                    
                columns_data= get_variables(df, clean_data)
            else:
                response={"data":[],"message":'JSON descriptive statistics file does not exist',"success":False}
                return JsonResponse(response, status=400) 
            
            response={
                    "data":columns_data,
                    "message":"",
                    "success":True
                    }
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
        except Exception as e :
            status=400
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            response={
                "data":[],
                "success":False,
                "message":str(e)
            }
            span.record_exception(e)
    span.end()   
    return JsonResponse(response, status=status, json_dumps_params={'indent': 2})
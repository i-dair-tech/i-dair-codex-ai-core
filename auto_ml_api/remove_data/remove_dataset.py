from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os 

# Get the current working directory
path=os.getcwd()

# Construct the path to the tools and training folders
tools_path = os.path.join(path,'tools')

sys.path.insert(0, tools_path)
from db_tools import delete_trained_models_by_dataset,delete_dataset
from oauth2_tools import verify_token

@require_http_methods(["DELETE"])
@csrf_exempt
def remove_dataset(request):
    
    dataset_id=request.GET.get('dataset_id')
    if not dataset_id:
        return JsonResponse({"message": "dataset id is empty"}, status=400)
    verify_token(request)
    
    if verify_token=="failed":
        return JsonResponse({"message":"session expired"}, status=401) 
    
    try:
        delete_trained_models_by_dataset(dataset_id)
        delete_dataset(dataset_id)
    except Exception:
        return JsonResponse({"message":"Somthing wrong in the db queries"}, status=400) 
    
    response = {
        "data": [],
        "message":"",
        "success": True
    }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
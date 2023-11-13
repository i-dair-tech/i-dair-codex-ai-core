from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
import urllib.parse
# Get the current working directory
path=os.getcwd()

# Construct the path to the tools and training folders
tools_path = os.path.join(path,'tools')

sys.path.insert(0, tools_path)

from oauth2_tools import verify_token
from tools.validate_train import ValidateTrain
import logging

logger = logging.getLogger("main")
@require_http_methods(["GET"])
@csrf_exempt
def validate_hyperparameter(request):
    
    model_id=request.GET.get('model_id')
    task=request.GET.get('task')
    hyperparameters_str=request.GET.get("hyperparameters")
    decoded_hyperparameters_str = urllib.parse.unquote(hyperparameters_str)
    
    try:
        hyperparameters = json.loads(decoded_hyperparameters_str)
        print(f"hyperparameters \n {hyperparameters}")
    except json.JSONDecodeError:
        return JsonResponse({"message": "Invalid hyperparameters JSON format"}, status=400)
    
    verify_token(request)
    
    if verify_token=="failed":
        return JsonResponse({"message":"session expired"}, status=401) 
    
    try:
        model_id = int(model_id)
    except ValueError:
        return JsonResponse({"message":"Invalid item ID"}, status=400)
    
    train_validation , message =ValidateTrain(hyperparameters=hyperparameters,model_id=model_id,task=task).validate()
    if train_validation:
        response={
                "data":[],
                "message":"The hyperparameter combination is valid",
                "success":True
                }
        return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
    else:
        return JsonResponse({"message":f"Invalid Hyperparameter Combinations Detected:{message}"}, status=400)
    

    

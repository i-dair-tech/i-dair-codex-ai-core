from django.http import JsonResponse
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

from oauth2_tools import get_user_id
from db_tools import get_original_hyperparameters,save_params

@require_http_methods(["POST"])
@csrf_exempt
def save_hyperparameters(request):
    # Parse the request body as JSON
    request_body = json.loads(request.body.decode())
    user_id=get_user_id(request)
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    model_id= request_body.get('model_id')
    params=request_body.get('params')
    original_hyperparameters=get_original_hyperparameters(model_id)
    n_iter=params["n_iter"]
    folds_number=params["folds_number"]
    if not n_iter:
        n_iter=original_hyperparameters["n_iter"]
    
    if not folds_number:
        folds_number=original_hyperparameters["folds_number"]
    
    del params["n_iter"]   
    del params["folds_number"]  
      
    for key in params:
        if not params[key] or len(params[key])==0:
            params[key]=original_hyperparameters[key]
    params["n_iter"]=n_iter
    params["folds_number"]=folds_number
    params_json=json.dumps(params)
    save_params(user_id,model_id,params_json)
    response={
        "data":[],
        "message":"The hyperparameters have been successfully saved.",
        "success":True
    }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
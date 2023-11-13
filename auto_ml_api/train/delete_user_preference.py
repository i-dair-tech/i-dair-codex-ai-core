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

from oauth2_tools import get_user_id
from db_tools import delete_preference

@require_http_methods(["DELETE"])
@csrf_exempt
def delete_user_preference(request,model_id):
    # Parse the request body as JSON
    #request_body = json.loads(request.body.decode())
    user_id=get_user_id(request)
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    
    try:
        model_id = int(model_id)
    except ValueError:
        return JsonResponse({"message":"Invalid item ID"}, status=400)
    print(model_id)
    delete_preference(user_id,model_id)
    
    response={
        "data":[],
        "success":True
    }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
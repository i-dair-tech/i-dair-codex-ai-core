from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
# Get the current working directory
path=os.getcwd()
tools_path = os.path.join(path, 'tools')

# Add the paths to the tools and training folders to the module search path
sys.path.insert(0, tools_path)
from oauth2_tools import get_user_id
from db_tools import assigned_dataset_validation,assign_dataset_to_group

@require_http_methods(["POST"])
@csrf_exempt
def assign_dataset(request):
    user_id=get_user_id(request)
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    request_body = json.loads(request.body.decode())
    dataset_id=request_body.get('id_dataset')
    group_id=request_body.get('group_id')
    if not all([dataset_id,group_id]):
        return JsonResponse({"message":'field is required',"success":False}, status=400)
    data_is_assigned=assigned_dataset_validation(dataset_id,group_id)
    if data_is_assigned:
        return JsonResponse({"data":[],"message":"The dataset has already been assigned to a group","success":True}, status=200, 
                            json_dumps_params={'indent': 2}
                            )
    assign_dataset_to_group(dataset_id,group_id)
    response={"data":[],
              "message":"The dataset has been assigned to the group",
              "success":True
            }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
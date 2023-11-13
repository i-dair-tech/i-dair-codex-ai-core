from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
import json
# Get the current working directory
path=os.getcwd()
tools_path = os.path.join(path, 'tools')
group_management_path=os.path.join(path,"auto_ml_api/group_management")
# Construct the path to the tools folders
sys.path.insert(0, tools_path)
sys.path.insert(0, group_management_path)
# Add the paths to the tools and training folders to the module search path
sys.path.insert(0, tools_path)
from oauth2_tools import verify_token
from db_tools import group_existence_check,insert_user_in_group,delete_old_members
from create_group import members_validation
@require_http_methods(["POST"])
@csrf_exempt
def invite_member(request):
    request_body = json.loads(request.body.decode())
    group_id=request_body.get("group_id")
    verify_token(request)
    if verify_token=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    emails=request_body.get("emails")
    privilege="member"
    member_ids,not_found_users=members_validation(emails)
    if not_found_users:
        return JsonResponse({"message":f"{not_found_users} not found","success":False}, status=400)
    
    group_existence=group_existence_check(group_id=group_id)
    if group_existence==False:
        return JsonResponse({"message":"The group you are referring to does not exist","success":False}, status=400)
    delete_old_members(group_id)
    if emails:
        for member_id in member_ids:
            insert_user_in_group(member_id,privilege,group_id=group_id)
    response={
            "message":"The members has been edited successfully",
            "success":True
            }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
   
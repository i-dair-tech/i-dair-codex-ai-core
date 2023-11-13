from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
import json
# Get the current working directory
path=os.getcwd()
tools_path = os.path.join(path, 'tools')
# Construct the path to the tools folders
sys.path.insert(0, tools_path)
# Add the paths to the tools and training folders to the module search path
sys.path.insert(0, tools_path)
from oauth2_tools import get_user_id
from db_tools import group_existence_check,insert_user_in_group,email_existence_check

def members_validation(members):
    not_found_users=[]  
    member_ids=[]
    for member in members:
        email_existence,member_id=email_existence_check(member)
        
        if email_existence==True:
            member_ids.append(member_id)
        else:
            not_found_users.append(member)
    return member_ids,not_found_users
 
@require_http_methods(["POST"])
@csrf_exempt
def create_group(request):
    
    request_body = json.loads(request.body.decode())
    group_name=request_body.get("group_name")
    user_id=get_user_id(request)
    members=request_body.get("members")
    privilege="owner"
    
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401) 
    
    member_ids,not_found_users=members_validation(members)
    if not_found_users:
        return JsonResponse({"message":f"{not_found_users} not found","success":False}, status=400)
                
    group_existence=group_existence_check(user_id,privilege,group_name=group_name)
    if group_existence:
        return JsonResponse({"message":"The group name already exists","success":False}, status=400)
    
    group_id=insert_user_in_group(user_id,privilege,group_name=group_name,return_group_id=True)
    privilege="member"
    if members:
        for member_id in member_ids:
            insert_user_in_group(member_id,privilege,group_id=group_id)

    response={
            "data":{"group_id":group_id},
            "message":"The group has been created",
            "success":True
            }
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})
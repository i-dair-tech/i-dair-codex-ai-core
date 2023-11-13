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
from db_tools import get_groups_user_info,get_user_email,get_user_groups_ids
def format_groups_user_info_dict(data):
    result = []
    group_dict = {}
    for item in data:
        group_id = item['group_id']
        item['user_id']=get_user_email(item['user_id'])
        if group_id not in group_dict:
            group_dict[group_id] = {
                    'groupId': group_id,
                    'name': item['name'],
                    'members': []
                }
        if item['privilege'] == 'owner':
            group_dict[group_id]['owner'] = item['user_id']
        else:
            group_dict[group_id]['members'].append(item['user_id'])
    result = list(group_dict.values())
    return result

@require_http_methods(["GET"])
@csrf_exempt
def get_groups(request):
    
    user_id=get_user_id(request)
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    user_groups=get_user_groups_ids(user_id)
    group_ids=','.join(str(item['group_id']) for item in user_groups)
    if not group_ids:
        return JsonResponse({"data":{"userGroups":[]},"message":"","success":True}, status=200, json_dumps_params={'indent': 2})
    groups_user_info=get_groups_user_info(group_ids)
    formatted_groups_user_info=format_groups_user_info_dict(groups_user_info)
    groups_info={"userGroups":formatted_groups_user_info}
    
    response={"data":groups_info,
              "message":"",
              "success":True
            }
    
    return JsonResponse(response, status=200, json_dumps_params={'indent': 2})

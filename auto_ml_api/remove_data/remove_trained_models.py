from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import os
import sys

# Get the current working directory
path = os.getcwd()

# Construct the path to the tools and training folders
tools_path = os.path.join(path, 'tools')

sys.path.insert(0, tools_path)

from oauth2_tools import get_user_id
from db_tools import delete_trained_models,delete_training_session

@require_http_methods(["DELETE"])
@csrf_exempt
def remove_trained_models(request):
    try:
        # Parse the request body as JSON
        request_data = json.loads(request.body.decode())

        # Get the 'trained_models' parameter from the JSON request
        trained_models = request_data.get('trained_models')
        training_sessions = request_data.get('training_sessions')

        if not trained_models or not training_sessions:
            return JsonResponse({"message": 'field is required', "success": False}, status=400)

        user_id = get_user_id(request)

        if user_id == "failed":
            return JsonResponse({"message": "session expired"}, status=401)

        trained_models_str = ",".join(str(trained_model) for trained_model in trained_models)
        training_sessions_str = ",".join(str(training_session) for training_session in training_sessions)
        # Call the remove_trained_models function with trained_models
        delete_trained_models(trained_models_str)
        delete_training_session(training_sessions_str)

        response = {
            "data": [],
            "success": True
        }

        return JsonResponse(response, status=200, json_dumps_params={'indent': 2})

    except json.JSONDecodeError:
        return JsonResponse({"message": "Invalid JSON in request body"}, status=400)

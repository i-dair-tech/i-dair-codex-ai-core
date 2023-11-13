from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import os
from opentelemetry import trace
path = os.getcwd()
tools_path = os.path.join(path, 'tools')
sys.path.insert(0, tools_path)

from oauth2_tools import verify_token
from trace_propagation import get_context
from trace_status_setter import status_setter

@require_http_methods(["GET"])
@csrf_exempt
def get_models_params(request):
    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get models params",context=context) as span:
        try:
            method="GET"    
            span.set_attribute("http.method", method)
            path= os.getcwd()
            verify_token(request)
            if verify_token=="failed":
                return JsonResponse({"message":"session expired"}, status=401) 
            params_path=os.path.join(path,"auto_ml_api/train/params_types.json")
            
            with open(params_path,"r") as file:
                file_content = file.read()
                params=json.loads(file_content)
            response={
                "data":params,
                "success":True
            }
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
        except Exception as e :
            status=400
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
            span.record_exception(e)
            response={
                "data":[],
                "success":False,
                "message":str(e)
            } 
    span.end()
    return JsonResponse(response, status=status, json_dumps_params={'indent': 2})
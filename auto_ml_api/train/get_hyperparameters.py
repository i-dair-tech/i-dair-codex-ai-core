from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from opentelemetry import trace

import sys
import os
path = os.getcwd()

tools_path = os.path.join(path, 'tools')
sys.path.insert(0, tools_path)

from db_tools import get_default_hyperparameter
from oauth2_tools import get_user_id
from trace_propagation import get_context
from trace_status_setter import status_setter

@require_http_methods(["GET"])
@csrf_exempt
def get_hyperparameters(request):

    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get hyperparameters",context=context) as span:
        try:
            method="GET"    
            span.set_attribute("http.method", method)

            user_id=get_user_id(request)
            if user_id=="failed":
                return JsonResponse({"message":"session expired"}, status=401)
            models_params= get_default_hyperparameter(user_id)
            response={
                "data":models_params,
                "success":True
            }
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
        except Exception as e :
            status=400
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            response={
                "data":[],
                "success":False,
                "message":str(e)
            }
            span.record_exception(e)

    span.end()
    return JsonResponse(response, status=status, json_dumps_params={'indent': 2})

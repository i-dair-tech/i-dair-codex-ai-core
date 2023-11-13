from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json

from django.views.decorators.csrf import csrf_exempt
import sys
import os 

# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
project_path = os.path.join(path,'tools')

sys.path.insert(0, project_path)
from db_tools import get_dataset, delete_dataset
from statistics_shared_task import generate_stats
from opentelemetry import trace
from oauth2_tools import get_user_id
import logging
from trace_propagation import get_context
from trace_status_setter import status_setter

logger = logging.getLogger("main")

def remove_files(paths):
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


@require_http_methods(["POST"])
@csrf_exempt
def clean_data_api(request):
    user_id=get_user_id(request)
    logger.info("User requested to get statistics",extra={'user_id': user_id})
    tracer = trace.get_tracer(__name__)
    
    context=get_context(request)
    
    with tracer.start_span("Generate Statistics",context=context) as span:
        current_span_context = span.get_span_context()
        current_trace_id=format(current_span_context.trace_id,'x')
        current_span_id = format(current_span_context.span_id,'x')
        method="POST"
        
        span.set_attribute("http.method", method)
        # Parse the request body as JSON
        request_body = json.loads(request.body.decode())
        
        # Get the file path from the request body
        id_dataset = request_body.get('idDataset')
        if id_dataset is None:
            logger.error("Invalid dataset ID ",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
            response={"data":[],"message":'id is required',"success":False}
            return JsonResponse(response, status=400)
        try:
            # Load the JSON data from the file
            file_path, filename = get_dataset(id_dataset)
            clean_data_path= os.path.join(dataset_path_prefix,"dataset/"+filename+".json")
            filename=filename.split(".")[0]

                
            results_dict=generate_stats.apply_async(args=(filename,file_path,clean_data_path,user_id,current_trace_id,current_span_id)).get() 
            if results_dict['error']:
                raise Exception

            with open(results_dict['file_path_json']) as f:
                clean_data = json.load(f)
                
            response={
                "data":clean_data,
                "message":"",
                "success":True
                }
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)


            logger.info("Generating statistics json file  done successfully",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
        except Exception as e:

            paths=[results_dict['file_path_json'],results_dict['file_path']]
            remove_files(paths)
            
            delete_dataset(id_dataset)
            status=400
            span.set_attribute("http.status_code", str(status))

            status_setter(span,False)
            response={"data":[],"message":"Oops! It seems there's an issue with your dataset.","success":False}

            if results_dict['error']:
                error_message=results_dict['error_value']
            else:
                error_message=e
            span.record_exception(error_message)
            logger.error("Generating statistics json failed",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
        span.end()
    return JsonResponse(response, status=status, json_dumps_params={'indent': 2})
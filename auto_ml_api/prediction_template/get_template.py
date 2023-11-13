from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import pandas as pd
import os
import sys
import json
import base64
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
parent_folder=os.path.dirname(path)+"/"
tools_path=os.path.join(parent_folder,"tools")
sys.path.insert(0,tools_path)
from db_tools import get_dataset
from supported_data_formats import read_supported
from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter
@require_http_methods(["POST"])
@csrf_exempt
def get_template(request):

    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get template",context=context) as span:
        try:
            method="GET"  
            span.set_attribute("http.method", method)
    
            # Parse the request body as JSON
            request_body=json.loads(request.body.decode())
            
            # Get the request body content
            id_dataset= request_body.get("idDataset")
            target=request_body.get("target")
            
            if not all ([id_dataset,target]):
                return JsonResponse({"message": 'Data or target does not exist', "success": False}, status=400)
            
            file_path,_= get_dataset(id_dataset)
            df_path=parent_folder+file_path
            df=read_supported(df_path)
            df=df.head()
            df=df.drop([target],axis=1)
            df.to_csv('prediction_template.csv', index=False)
            with open('prediction_template.csv','rb') as file :
                csv_file = file.read()
            
            response = {
            "success":True,
            "message":""
                }    
            
            file_contents = {}
            file_contents["prediction_template.csv"]= base64.b64encode(csv_file).decode('utf-8')
            os.remove("prediction_template.csv")
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
            file_contents=None

    span.end()        
    return JsonResponse({'file_contents': file_contents,'data':response,"status":status })
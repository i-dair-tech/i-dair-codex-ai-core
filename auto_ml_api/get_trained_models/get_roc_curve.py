from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import joblib

import mlflow

import glob
baseurl_mlflow = os.environ.get("BASE_URL_MLFLOW_REST_API")

# Get the current working directory
path = os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

project_path = os.path.join(path, 'tools') 


sys.path.insert(0, project_path)
from db_tools import get_trained_model_by_id
from plots_tools import figures_unifier
from oauth2_tools import verify_token
import plotly.express as px
import plotly.io as pio
from set_mlflow_tracking_uri import set_tracking_uri
from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter


def remove_files(file_extensions=None):
    files=[]
    for file_extension in file_extensions:
        file_to_delete=glob.glob(f"*.{file_extension}")
        files=files+file_to_delete

    for file in files:
        os.remove(file)

@require_http_methods(["GET"])
@csrf_exempt
def get_roc_curve(request):
    print("started")
    tracer = trace.get_tracer(__name__)

    context=get_context(request)
    
    with tracer.start_span("Get roc curve",context=context) as span:
        try:
            method="GET"    
            span.set_attribute("http.method", method)
            
            set_tracking_uri()
            
            verify_token(request)
            if verify_token=="failed":
                return JsonResponse({"message":"session expired"}, status=401)

            models_ids=request.GET.get('models_ids')
            models_ids=models_ids.split("-")
            models_id_list=','.join(str(item)for item in models_ids)
            trained_models=get_trained_model_by_id(models_id_list)
            plots=[]
            model_without_plot=[]
            for trained_model in trained_models:
                mlflow_result=mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="roc_curve.joblib", dst_path=".")
                roc_curve_list=joblib.load('roc_curve.joblib')
                if roc_curve_list[1]:
                    roc_curve=roc_curve_list[0]
                    plots.append(roc_curve)
                else:
                    model_without_plot.append(trained_model["name"])
                remove_files(["joblib"])
               
            unified_fig=figures_unifier(plots)
            pio.write_json(unified_fig, 'roc_curve.json')
            with open('roc_curve.json') as json_file:
                unified_fig = json.load(json_file)
            remove_files(["json"])
            info=""
            if model_without_plot:
                model_without_plot=", ".join([model for model in model_without_plot])
                info="Unable to generate the "+model_without_plot+" ROC curve. It appears the probability hyperparameter is set to 'false.' Adjusting this setting may resolve the issue."
            
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
            message="ROC curve generated"
            response_data=unified_fig
            success=True
        except Exception as e :
            info=""
            remove_files(["json","joblib"])
            status=400
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            span.record_exception(e)
            message=str("e")
            response_data=None
            success=False

    span.end()   
    return JsonResponse({'data': response_data,"success":success,"message":message,"info":info,"status":status})
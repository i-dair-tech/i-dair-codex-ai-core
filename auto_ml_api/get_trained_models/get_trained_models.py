from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import pandas as pd
import requests 
import mlflow

import glob
import joblib
import plotly.express as px
import plotly.io as pio


baseurl_mlflow = os.environ.get("BASE_URL_MLFLOW_REST_API")
# Get the current working directory
path = os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

project_path = os.path.join(path, 'tools')
training_path = os.path.join(path,'training')

sys.path.insert(0, project_path)

from db_tools import get_all_trained_model_by_dataset,get_all_trained_model_in_progress,get_all_trained_model_in_progress_for_group,get_all_trained_model_in_progress_by_user,get_completed_trained_model_by_user,change_seen_status_of_session,get_not_completed_trained_model_by_user
from oauth2_tools import get_user_id,verify_token
from set_mlflow_tracking_uri import set_tracking_uri

from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter
from set_decimals import set_decimals

def remove_downloded_files():
    csv_files=glob.glob("*.csv")
    json_files= glob.glob("*.json")
    files=csv_files+json_files
    for file in files:
        os.remove(file)

@require_http_methods(["GET"])
@csrf_exempt
def get_trained_models(request):
    set_tracking_uri()
    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get trained models",context=context) as span:
        try:
            method="GET"    
            span.set_attribute("http.method", method)

            user_id=get_user_id(request)
            if user_id=="failed":
                return JsonResponse({"message":"session expired"}, status=401)    

            trained_models = get_completed_trained_model_by_user(user_id)
            
            
            for trained_model in trained_models:
                api_url = baseurl_mlflow+"/runs/get?run_id="+trained_model['run_id']
                result = requests.get(api_url)
                metrics=result.json()['run']['data']['metrics']    
                trained_model['metrics'] = metrics 
                if  trained_model["task"]!="clustering":
                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="importance_scores.json", dst_path=".")
                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="importance_fig.json", dst_path=".")
                    # Opening JSON file
                    with open('importance_scores.json') as json_file:
                        importance_scores = json.load(json_file)
                    with open('importance_fig.json') as json_file:
                        importance_fig = json.load(json_file)
                    trained_model['importance_scores'] = importance_scores
                    trained_model['plot'] = importance_fig

                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="bootstrap_estimates.csv", dst_path=".")
                    bootstrap_estimates=pd.read_csv("bootstrap_estimates.csv")
                    grouped_df = bootstrap_estimates.groupby(['model', 'metric']).agg({
                        'lower': 'mean',
                        'median': 'first', 
                        'upper': 'mean'
                    }).reset_index()
                    grouped_df.columns = ['model', 'metric', 'lower', 'median', 'upper']
                    
                    for metric in metrics :
                        quantiles_metric_df=grouped_df[grouped_df["metric"]==metric["key"]]
                        metric["value"]=f"{set_decimals(quantiles_metric_df['median'].to_string(index=False))}[{set_decimals(quantiles_metric_df['lower'].to_string(index=False))},{set_decimals(quantiles_metric_df['upper'].to_string(index=False))}]"

                mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="feature_engineering_tracking.joblib", dst_path=".")
                mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="param_space.joblib", dst_path=".")
                feature_engineering_tracking_dict=joblib.load("feature_engineering_tracking.joblib")
                param_space=joblib.load("param_space.joblib")
                result = requests.get(api_url)
                parameters=result.json()['run']['data']['params']
                trained_model['feature_engineering']=feature_engineering_tracking_dict
                settings=[]
                extra_values_to_delete=[]
                for parameter in parameters:
                    if parameter["key"] in param_space:
                        parameter["param_space"]= param_space[parameter["key"]]
                    else:
                        settings.append(parameter)
                        extra_values_to_delete.append(parameter)
                        
                parameters=[parameter for parameter in parameters if parameter not in extra_values_to_delete]    
                trained_model['settings']=settings        
                trained_model['parameters']=parameters
                
                os.remove("feature_engineering_tracking.joblib")
                os.remove("param_space.joblib")

                #mark the models as seen
                change_seen_status_of_session(trained_model['session_id'])
                
            remove_downloded_files()

            not_completed_models=get_not_completed_trained_model_by_user(user_id)   
            for not_completed_model in not_completed_models:
                #mark the models as seen
                change_seen_status_of_session(not_completed_model['session_id'])

            response = {
            'data': {"trained_models":trained_models},
            "success":True,
            "message":""
            }
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
        except Exception as e :
            remove_downloded_files()
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

@require_http_methods(["GET"])
@csrf_exempt
def get_trained_models_by_dataset(request):
    # set Tracking Uri
    set_tracking_uri()
    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Get trained models by dataset",context=context) as span:
        try:
            # Parse the request body as JSON
            id_dataset=request.GET.get('idDataset')
            method="GET"    
            span.set_attribute("http.method", method)

            verify_token(request)
            if verify_token=="failed":
                return JsonResponse({"message":"session expired"}, status=401)     
            if not all([id_dataset]):
                return JsonResponse({"message": 'dataset is required', "success": False}, status=400)

            trained_models = get_all_trained_model_by_dataset(id_dataset)
            
            for trained_model in trained_models:  
                api_url = baseurl_mlflow+"/runs/get?run_id="+trained_model['run_id'] 
                result = requests.get(api_url)
                metrics=result.json()['run']['data']['metrics']     
                trained_model['metrics'] = metrics  
                if trained_model["task"]!="clustering":
                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="importance_scores.json", dst_path=".")
                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="importance_fig.json", dst_path=".")
                    # Opening JSON file
                    with open('importance_scores.json') as json_file:
                        importance_scores = json.load(json_file)
                    with open('importance_fig.json') as json_file:
                        importance_fig = json.load(json_file)
                        
                    trained_model['importance_scores'] = importance_scores
                    trained_model['plot'] = importance_fig
                    mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="bootstrap_estimates.csv", dst_path=".")
                    bootstrap_estimates=pd.read_csv("bootstrap_estimates.csv")
                    grouped_df = bootstrap_estimates.groupby(['model', 'metric']).agg({
                        'lower': 'mean',
                        'median': 'first', 
                        'upper': 'mean'
                    }).reset_index()
                    grouped_df.columns = ['model', 'metric', 'lower', 'median', 'upper']
                    
                    for metric in metrics :
                        quantiles_metric_df=grouped_df[grouped_df["metric"]==metric["key"]]
                        metric["value"]=f"{set_decimals(quantiles_metric_df['median'].to_string(index=False))}[{set_decimals(quantiles_metric_df['lower'].to_string(index=False))},{set_decimals(quantiles_metric_df['upper'].to_string(index=False))}]"

                mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="feature_engineering_tracking.joblib", dst_path=".")
                mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="param_space.joblib", dst_path=".")
                feature_engineering_tracking_dict=joblib.load("feature_engineering_tracking.joblib")
                param_space=joblib.load("param_space.joblib")
                
                trained_model['feature_engineering']=feature_engineering_tracking_dict
                parameters=result.json()['run']['data']['params']
                settings=[]
                extra_values_to_delete=[]
                for parameter in parameters:
                    if parameter["key"] in param_space:
                        parameter["param_space"]= param_space[parameter["key"]]
                    else:
                        settings.append(parameter)
                        extra_values_to_delete.append(parameter)
                        
                parameters=[parameter for parameter in parameters if parameter not in extra_values_to_delete]    
                trained_model['settings']=settings        
                trained_model['parameters']=parameters
                    
                trained_model['parameters']=parameters
                os.remove("feature_engineering_tracking.joblib")
                os.remove("param_space.joblib")
                
            response = {
            'data': {"trained_models":trained_models},
            "success":True,
            "message":""
            }
            remove_downloded_files()
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)

        except Exception as e :
            remove_downloded_files()
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




@require_http_methods(["GET"])
@csrf_exempt
def get_group_training_in_progress(request):
    user_id=get_user_id(request)
    id_group=request.GET.get('idGroup')
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    if id_group=="0":
        trained_models = get_all_trained_model_in_progress(user_id)
    else:
        trained_models = get_all_trained_model_in_progress_for_group(id_group)
    response = {
    'data': {"trained_models":trained_models,"user_id":user_id},
    "success":True,
    "message":""
    }

    return JsonResponse(response, status=200)


@require_http_methods(["GET"])
@csrf_exempt
def get_training_in_progress_by_user(request):
    
    user_id=get_user_id(request)
    if user_id=="failed":
        return JsonResponse({"message":"session expired"}, status=401)
    trained_models = get_all_trained_model_in_progress_by_user(user_id)
    response = {
    'data': {"trained_models":trained_models,"user_id":user_id},
    "success":True,
    "message":""
    }

    return JsonResponse(response, status=200)

    


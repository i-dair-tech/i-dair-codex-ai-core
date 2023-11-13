from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import pandas as pd
import requests 
import mlflow
import base64
import joblib
import glob
baseurl_mlflow = os.environ.get("BASE_URL_MLFLOW_REST_API")

# Get the current working directory
path = os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

project_path = os.path.join(path, 'tools')

sys.path.insert(0, project_path)
from db_tools import get_trained_model_by_id
from oauth2_tools import verify_token
import plotly.express as px
import plotly.io as pio
from set_mlflow_tracking_uri import set_tracking_uri
from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter

def remove_files(plot_files=None):
    csv_files = glob.glob("*.csv")

    if plot_files:
        files=plot_files+csv_files
    else:
        files=csv_files

    for file in files:
        os.remove(file)

@require_http_methods(["POST"])
@csrf_exempt
def plot_compare(request):

    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Plot compare",context=context) as span:
        try:
            method="POST"    
            span.set_attribute("http.method", method)
    
            request_body = json.loads(request.body.decode())
            
            set_tracking_uri()
            
            verify_token(request)
            if verify_token=="failed":
                return JsonResponse({"message":"session expired"}, status=401) 
            file_contents={}
            models_ids= request_body.get("models_ids") 
            models_id_list=','.join(str(item)for item in models_ids)
            trained_models=get_trained_model_by_id(models_id_list)
            all_bootstrap_estimates_df = pd.DataFrame([])
            for trained_model in trained_models:
                session_name=trained_model["session_name"]
                
                mlflow.artifacts.download_artifacts(run_id=trained_model['run_id'], artifact_path="bootstrap_estimates.csv", dst_path=".")
                bootstrap_estimates=pd.read_csv("bootstrap_estimates.csv")
                bootstrap_estimates["session"]=[session_name for i in range(bootstrap_estimates.shape[0])]
                all_bootstrap_estimates_df = pd.concat([all_bootstrap_estimates_df, pd.DataFrame(bootstrap_estimates)])
                
            all_bootstrap_estimates_df=all_bootstrap_estimates_df.drop(["lower","median","upper"],axis=1)
            
            filename_to_save="plot-associated-data.csv"
            all_bootstrap_estimates_df.to_csv("plot-associated-data.csv",index=False)
            metrics=all_bootstrap_estimates_df["metric"].unique().tolist()
            for metric in metrics:
                filtered_df = all_bootstrap_estimates_df[all_bootstrap_estimates_df['metric'].str.contains(metric, case=False)]
                bootstrap_fig = px.box(filtered_df, x="model", y="score", color="model")
                bootstrap_fig.update_traces(quartilemethod="exclusive")
                bootstrap_fig.update_yaxes(matches=None)
                bootstrap_fig.update_layout(yaxis_title=f"{metric} score")
                bootstrap_fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
                pio.write_json(bootstrap_fig, f'{metric}-bootstrap_fig.json')
            plot_files = glob.glob("*.json")
            plots=[]
            for plot_file in plot_files:
                metric=plot_file.split("-")[0]
                with open(plot_file) as json_file:
                    bootstrap_fig = json.load(json_file)
                plots.append({"metric":metric,"fig":bootstrap_fig})
            with open(filename_to_save,'rb') as file :
                csv_file = file.read()
                file_contents[filename_to_save]= base64.b64encode(csv_file).decode('utf-8') 
            remove_files(plot_files)
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
            message=""
            response_data=plots
            success=True
        except Exception as e :
            remove_files()
            status=400
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            span.record_exception(e)
            message=str("e")
            file_contents=None
            response_data=None
            success=False

    span.end()   
    return JsonResponse({'file_contents': file_contents,'data': response_data,"success":success,"message":message,"status":status})
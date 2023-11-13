import mlflow.pyfunc
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from db_tools import get_trained_model_by_id
import numpy as np
from db_tools import get_dataset
import os 
import sys
import pandas as pd
import shutil
import io 
import copy
import base64
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
project_path = os.path.join(path,'data_tools')
tools_path = os.path.join(path, 'tools')
sys.path.insert(0, project_path)
sys.path.insert(0, tools_path)
from oauth2_tools import get_user_id
import joblib
import glob
from opentelemetry import trace
from trace_propagation import get_context
from trace_status_setter import status_setter
import logging
from set_mlflow_tracking_uri import set_tracking_uri
logger = logging.getLogger("main")

def generate_predictions_df(id_model,target,predictions,df,model_name,session_name):
    model_list=[model_name for i in range(0,len(predictions))]
    session_name_list=[session_name for i in range(0,len(predictions))]
    df.insert(0,"session_name",session_name_list)
    df.insert(0,f"{target}_predictions",predictions)
    df.insert(0,"model_name",model_list)
    df.to_csv(f"model_{id_model}.csv",index=False)
    model_info={f"model_{id_model}":model_name}
    return model_info

def add_all_models_prediction(prediction_files,file_contents):
    predictions_dfs=[]
    for prediction_file in prediction_files:
        filename=prediction_file.split("/")[-1]
        
        file_df=pd.read_csv(filename)
        predictions_dfs.append(file_df)
        
        with open(prediction_file,'rb') as file :
            csv_file = file.read()
            file_contents[filename]= base64.b64encode(csv_file).decode('utf-8')
    
    all_models_prediction=pd.concat(predictions_dfs)
    
    return all_models_prediction
    
             
def remove_artifacts():
    #Remove downloaded artifacts files    
    joblib_files= glob.glob("*.joblib")
    json_files=glob.glob("*.json")
    csv_files=glob.glob("*.csv")
    files=joblib_files+json_files+csv_files  
    for file in files:
        os.remove(file)
    folder_path= "model"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    #shutil.rmtree(folder_path) 
    
def all_models_prediction_to_bytes(all_models_prediction,file_contents):
    
    all_models_prediction.to_csv("all_models_prediction.csv",index=False)
    with open("all_models_prediction.csv",'rb') as file :
        all_models_prediction_bytes = file.read()
    file_contents["all_models_prediction.csv"]= base64.b64encode(all_models_prediction_bytes).decode('utf-8')        
    os.remove("all_models_prediction.csv")

def check_prediction_method(is_file,id_model,target,predictions,prediction_df_copy,model_predictions,models_info):
    if is_file=="true":
        model_info=generate_predictions_df(id_model,target,predictions,prediction_df_copy,model_predictions["model_name"],model_predictions["session_name"])
        models_info.update(model_info)
    else :
        model_predictions["predictions"]=predictions

def select_features(df,id_model): 
    prediction_nb_columns=df.shape[1]
    if prediction_nb_columns > 1:
        #Get feature selection method
        if id_model!=11:
            feature_selection_method= "recursive_feature_elimination.joblib"
        else:
            feature_selection_method= "univariate_selection.joblib"
        
    feature_selection= joblib.load(feature_selection_method)
    df= feature_selection.transform(df)
    return df
@require_http_methods(["POST"])
@csrf_exempt
def prediction(request): # Compliant
    # set Tracking Uri
    set_tracking_uri()

    tracer = trace.get_tracer(__name__)

    context=get_context(request)

    with tracer.start_span("Prediction",context=context) as span:
        current_span_context = span.get_span_context()
        current_trace_id=format(current_span_context.trace_id,'x')
        current_span_id = format(current_span_context.span_id,'x')
        method="POST"  
        span.set_attribute("http.method", method)
        
        # Parse the request body as JSON
        request_body= request.POST
        user_id=get_user_id(request)
        if user_id=="failed":
            logger.error("session expired",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
            return JsonResponse({"message":"session expired"}, status=401) 
        # Get the file path from the request body
        id_models=json.loads(request_body.get('idModels')) 
        is_file=request_body.get("isFile")
        if not all([id_models,is_file]):
            logger.error("missing field in the request body",extra={'user_id': user_id ,"trace_id":current_trace_id,"span_id":current_span_id})
            return JsonResponse({"message": 'missing field in the request body', "success": False}, status=400)
        if is_file=="true":
            file_stream=request.FILES['data']
            file_stream = io.BytesIO(file_stream.read())
            prediction_df=pd.read_csv(file_stream)
            prediction_df_cols=prediction_df.columns.tolist()
            logger.info("User requested to launch a prediction with file",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
        else:    
            x_predict= json.loads(request_body.get('data'))
            file_contents = {}
            #Convert x_predict into DataFrame
            prediction_df= pd.DataFrame(x_predict)
            prediction_df_cols=prediction_df.columns.tolist()
            logger.info("User requested to launch a prediction with form",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
            
        file_contents = {}
        data=[]
        df=None
        try:
            models_info={}
            for id_model in id_models:
                df=copy.deepcopy(prediction_df)
                prediction_df_copy=copy.deepcopy(prediction_df)
                trained_model= get_trained_model_by_id(id_model)
                run_id= trained_model[0]["run_id"]
                task= trained_model[0]["task"]
                target=trained_model[0]["target"]
                model_id=trained_model[0]["id_model"]
                session_name=trained_model[0]["session_name"]
                session_id=trained_model[0]["session_id"]
                logger.info(f"prediction with model: name= {trained_model[0]['name']}, id= {model_id} in session/task: id= {session_id}, name= {session_name}",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
                if(task!="clustering"):
                    
                    #Download artifacts 
                    artifact_path = "."
                    mlflow.artifacts.download_artifacts(run_id=run_id,dst_path=artifact_path)
                    
                    #load OneHot Encoding joblib file
                    original_cols=joblib.load("original_cols_list.joblib")

                    if prediction_df_cols != original_cols and is_file=="true":
                        return JsonResponse({"message": 'wrong file structure', "success": False}, status=400)
                    
                    # Find files that have col_ in beginning  with the .joblib extension
                    labels_encoders={}    
                    files = glob.glob("col_*.joblib")
                    for file in files:
                        file_array= file.split("_")
                        column_name= file_array[1:-1]
                        if isinstance(column_name, list)>0:
                            column_name="".join([val+"_" if i<(len(column_name)-1) else val for i , val in enumerate(column_name)])
                        labels_encoders[column_name]=file
                        
                    # Apply label encoder to the columns
                    categorical_cols= joblib.load("categorical_cols.joblib")
                    for idx, val in labels_encoders.items():
                        label_encoder= joblib.load(val)
                        df[idx]= label_encoder.transform(df[idx])
                        
                    # Encode the categorical columns with one hot encoded
                    ohe= joblib.load("ohe.joblib")
                    one_hot_encoded_cols= ohe.transform(df[categorical_cols])
                    df= np.concatenate((one_hot_encoded_cols, df.drop(categorical_cols, axis=1)), axis=1)

                    # Apply Standardization
                    standard_scaler = joblib.load("StandardScaler.joblib")
                    df= standard_scaler.transform(df)
                    
                    #df= select_features(df,model_id)
                                    
                #Load the model
                model= mlflow.pyfunc.load_model(f'runs:/{run_id}/model')
                
                #Use the loaded model to predict
                prediction= model.predict(df).tolist()
                model_predictions={}
                model_predictions["id_model"]=id_model
                model_predictions["email"]=trained_model[0]["email"]
                model_predictions["model_name"]=trained_model[0]["name"]
                model_predictions["session_name"]=session_name
                
                #Get the classes names of the predicition result
                if task=="classification":
                    classes= trained_model[0]["classes"]
                    json_classes= json.loads(classes)
                    reversed_json_classes={v: k for k, v in json_classes.items()}
                    predictions=[reversed_json_classes[idx] for idx in prediction]
                    check_prediction_method(is_file,id_model,target,predictions,prediction_df_copy,model_predictions,models_info)
                        
                else:
                    check_prediction_method(is_file,id_model,target,prediction,prediction_df_copy,model_predictions,models_info)
                        
                data.append(model_predictions)   
            prediction_files = glob.glob("model_*.csv") 
            if is_file=="true":
                all_models_prediction=add_all_models_prediction(prediction_files,file_contents)
                remove_artifacts()
                all_models_prediction_to_bytes(all_models_prediction,file_contents)
            remove_artifacts()

            json_response = {
            'data': data,
            "success":True,
            "message":""}
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
            logger.info("prediction done",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})

                  
        except Exception as e:
            status=500
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            remove_artifacts()
            json_response={"data":[],"message":'server error',"success":False}
            file_contents={}
            logger.error(f"prediction failed due: {str(e)}",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
    span.end()    
    
    return JsonResponse({'file_contents': file_contents,'data':json_response,"status":status }) 


from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
import shutil
from django.views.decorators.csrf import csrf_exempt
import sys
import os 
import copy


# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

# Construct the path to the tools and training folders
tools_path = os.path.join(path,'tools')
training_path = os.path.join(path,'training')

# Add the paths to the tools and training folders to the module search path
sys.path.insert(0, tools_path)
sys.path.insert(0, training_path)
from oauth2_tools import get_user_id

from db_tools import get_default_hyperparameter
from trace_propagation import get_context
from trace_status_setter import status_setter
from opentelemetry import trace

from db_tools import create_train_session,change_train_session_status,get_preferences,update_failed_session
from train  import train_models
from celery.result import AsyncResult
import logging

logger = logging.getLogger("main")

@require_http_methods(["POST"])
@csrf_exempt
def cancel_train(request):
    request_body = json.loads(request.body.decode())
    session_id = request_body.get('sessionId')
    user_id=get_user_id(request)
    logger.info(f"User requested to cancel a train session/task[{session_id}]",extra={'user_id': user_id})
    
    if user_id=="failed":
        logger.error("session expired",extra={'user_id': user_id})
        return JsonResponse({"message":"session expired"}, status=401)    
    
    result = AsyncResult(str(session_id))
    # Revoke the task
    result.revoke(terminate=True)
    change_train_session_status(session_id,"canceled")
    if os.path.exists(str(session_id)):
        shutil.rmtree(str(session_id))
    logger.info(f"session/task[{session_id}] is canceled",extra={'user_id': user_id})
    return JsonResponse({"status":"canceled"}, status=200) 
    
@require_http_methods(["POST"])
@csrf_exempt
def train(request):
    tracer = trace.get_tracer(__name__)
    context=get_context(request)
    with tracer.start_span("Train",context=context) as span:
        current_span_context = span.get_span_context()
        current_trace_id=format(current_span_context.trace_id,'x')
        current_span_id = format(current_span_context.span_id,'x')
        method="POST"
        span.set_attribute("http.method", method)

        # Parse the request body as JSON
        request_body = json.loads(request.body.decode())
        user_id=get_user_id(request)
        logger.info("User requested to launch a train",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
        if user_id=="failed":
            logger.error("session expired",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
            return JsonResponse({"message":"session expired"}, status=401)    
        id_dataset = request_body.get('idDataset')
        
        train_percentage = request_body.get('train_percentage')
        test_percentage = request_body.get('test_percentage')
        seed=request_body.get('seed')
        shuffle=request_body.get('shuffle')
        task=request_body.get('task')
        target=request_body.get('target',"")
        models= request_body.get('models')
        session_name= request_body.get('sessionName')
        
        if not all([id_dataset,models,task, train_percentage, test_percentage, seed]) and shuffle is not None:
            logger.error("missing fields in the request body",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
            return JsonResponse({"message":'field is required',"success":False}, status=400)

        for model in models:
            id_model=model["id_model"]
            
        models_id=[]
        param_dicts=[]
        param_dict={}
        split_strategy={}
        for model in models: 
            id_model=model["id_model"]
            default=model["default"]
            models_id.append(id_model)
            if default:
                params=get_preferences(user_id,id_model)
                params=json.loads(params)
            else:
                params=model.get("params")
                
            n_iter=params["n_iter"]
            optimizer=params.get("parameter_search",["BayesianOptimization"])

            optimizer_existance="parameter_search" in params
            if optimizer_existance:
                del params["parameter_search"]
                
            del params["n_iter"]
            for key in dict(params):
                if not params[key]:
                    del params[key]
            param_dict={
                        "feature_selection":model["featureSelection"],
                        "param_space":params,
                        "optimizer":optimizer[0],
                        "n_iter":n_iter,
                        }
            split_strategy={
                            "train_percentage":(train_percentage/100),
                            "test_percentage":(test_percentage/100),
                            "seed":seed,
                            "shuffle":shuffle  
                            }
            param_dicts.append(param_dict)
        
        try:
            logger.info(f"Models ids {models_id}requested by the user",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
            session_id=create_train_session(models_id,id_dataset,user_id,task,target,param_dicts,session_name)
        
            res=train_models.apply_async(args=(user_id,models_id,id_dataset,param_dicts,split_strategy,task,session_id,target), task_id=str(session_id))   
            print(f"res.result {res.result}")
            print(f"res.get {res.get()}")
            res_bool=bool(res.get() or res.result)
            if res_bool:
                logger.error(f"celery task/session[{session_id}] failed",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
                raise Exception
            else:
                logger.info(f"celery task/session[{session_id}] succeeded",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
            status=200
            span.set_attribute("http.status_code", str(status))
            status_setter(span,True)
            response={
            "message":"The training has been done successfully",
            "success":True
            }
        except Exception as e:
            status=500
            span.set_attribute("http.status_code", str(status))
            status_setter(span,False)
            if os.path.exists(str(session_id)):
                shutil.rmtree(str(session_id))
            hp_error="Possible issue with Hyperparameter combination or search"
            if  hp_error in str(res.result) or hp_error in str(res.get()) or res_bool:
                if res.result!=None:
                    message=res.result
                else:
                    message=res.get()
                error=copy.deepcopy(message)
                if ':' in str(message):
                    message=str(message).split(':')[0]+'.'
                update_failed_session(session_id,str(message))
                logger.error(f"{(str(message))} in task/session[{session_id}] ",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
                span.record_exception(error)
            else:
                message="The training has been canceled due to an error."
                update_failed_session(session_id,message)
                logger.error(f"{message}: {e}, in task/session[{session_id}] ",extra={'user_id': user_id,"trace_id":current_trace_id,"span_id":current_span_id})
                span.record_exception(e)
            
            result = AsyncResult(str(session_id))
            result.revoke(terminate=True, signal='SIGQUIT')
            change_train_session_status(session_id,"Failed")
            response={
            "message":message,
            "success":False
            }
    span.end()

    return JsonResponse(response, status=status, json_dumps_params={'indent': 2})

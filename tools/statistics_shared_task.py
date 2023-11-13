import json
import os
from supported_data_formats import read_supported
from pandas_profiling_cleaner import clean
import pandas_profiling
from pandas_profiling.utils.cache import cache_file
from celery import shared_task
import logging
logger = logging.getLogger("main")
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)
@shared_task(bind=True)
def generate_stats(self,filename,file_path,clean_data_path,user_id,current_trace_id,current_span_id):
    try:
        results_dict={}
        file_path_json=os.path.join(dataset_path_prefix,f"dataset/{filename}.json")
        if os.path.exists(clean_data_path):
            logger.info("Getting a generated statistics json file",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
            with open(clean_data_path) as f:
                clean_data = json.load(f)
            results_dict={"file_path":clean_data_path}
        else:
            logger.info("Started generating statistics json file ",extra={'user_id': user_id, "trace_id":current_trace_id,"span_id":current_span_id})
            file_path=dataset_path_prefix+file_path
            df = read_supported(file_path)
            profile_report = df.profile_report(html={"style": {"full_width": True}},title=filename)
            profile_report.to_file(file_path_json)
            
            with open(file_path_json) as f:
                data = json.load(f)
            os.remove(file_path_json)
            
            # Return the cleaned data as a JSON response
            clean_data=clean(data,df) 
            with open(clean_data_path, "w") as f:
                json.dump(clean_data, f,  indent=2) 
    
        results_dict["file_path_json"]=file_path_json
        results_dict["error"]=False
        results_dict["error_value"]=None
        results_dict["file_path"]=file_path
        return results_dict
    except Exception as e :  
        file_path_json=os.path.join(dataset_path_prefix,f"dataset/{filename}.json")
        results_dict={"file_path":file_path,"file_path_json":file_path_json,"error":True, "error_value":e}  
        return results_dict
from plots_tools import generate_plots
import copy
import math
import json

def replace_nan_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_nan_values(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = replace_nan_values(item)
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def clean(data,df):    
    # Create a deep copy of the data
    clean_data = copy.deepcopy(data)
    
    # Remove keys that do not start with "table" or "var"
    to_remove = [k for k in clean_data.keys() if not k.startswith('table') and not k.startswith('var')]
    for k in to_remove:
        del clean_data[k]
    
    # Remove certain keys from the variables sub-dictionary and add HTML plot in plot key
    remove = ["character", "value_counts", "alias", "script", "n_category", "histogram"]
    for sub_key in data["variables"].keys():
        for target_key in data["variables"][sub_key].keys():
            for k in remove:
                if k in target_key:
                    if clean_data["variables"][sub_key]["type"]!="Boolean" or target_key=="value_counts_without_nan":
                        del clean_data["variables"][sub_key][target_key]
                    
                    if clean_data["variables"][sub_key]["type"]=="Numeric":
                        clean_data["variables"][sub_key]["plot"]=generate_plots(df, sub_key, "histogram",num=True)
                        
                    else:
                        clean_data["variables"][sub_key]["plot"]=generate_plots(df, sub_key, "histogram")
    clean_data= replace_nan_values (clean_data)                 
    return clean_data
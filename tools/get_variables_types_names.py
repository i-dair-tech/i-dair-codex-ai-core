from db_tools import get_dataset
import pandas as pd 
import os 
import json

from supported_data_formats import read_supported

# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

def get_variables(df, clean_data=None ,data_preprocessing=False,id_dataset=None):  
    if data_preprocessing:
        # Get path of the descriptive statistics JSON and the data file
        file_path, filename = get_dataset(id_dataset)
        file_path=dataset_path_prefix+file_path
    
        #Load the data CSV file 
        # df = pd.read_csv(file_path): CHANGED: Mar 6, 2023
        df = read_supported(file_path) #  CHANGED: Mar 6, 2023
        clean_data_path= os.path.join(dataset_path_prefix,"dataset/"+filename+".json")
        #Load the JSON file
        with open(clean_data_path) as f:
            clean_data = json.load(f)
    #Get the variables names
    clean_data_keys=clean_data['variables'].keys()
    columns_data={}
    for subkey in clean_data_keys:
        variable_type=clean_data['variables'][subkey]['type']
            
        if variable_type=="Categorical" or variable_type=="Boolean":
            unique_values=df[subkey].unique().tolist()
            columns_data[subkey]={"type":variable_type,"values":unique_values}
                
        else:
            columns_data[subkey]={"type":variable_type}
    return columns_data
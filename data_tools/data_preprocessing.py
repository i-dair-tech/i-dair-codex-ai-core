from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler, Normalizer, OneHotEncoder
import numpy as np
import copy
import pandas as pd 
import joblib
import os 
import sys

# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
tools_path = os.path.join(path,'tools')
sys.path.insert(0, tools_path)
from get_variables_types_names import get_variables

data_preprocessing_method={}
encoder_steps=[]
def is_categorical(series,df,id_dataset,target):
    columns_data=get_variables(df,data_preprocessing=True,id_dataset=id_dataset)
    columns_type=columns_data[target]["type"]
    return np.issubdtype(series.dtype, np.number) and columns_type=="Categorical"

def unique_elements(lst):
    unique_set = set()
    unique_dict = {}
    k=0
    for elem in lst:
        if elem not in unique_set:
            unique_set.add(elem)
            unique_dict[str(elem)] = k
            k=k+1
    return unique_dict
def unique_elements_for_numbers(lst):
    unique_set = set()
    unique_dict = {}
    for elem in lst:
        if elem not in unique_set:
            unique_set.add(elem)
            unique_dict[str(elem)] = int(elem)
    return unique_dict

def encoder(session,task,df,id_dataset=None,target=""):
    global encoder_steps
    if task!="clustering":
        x=df.drop(target, axis=1)
        y=df[target]
        
        # Create a LabelEncoder object
        y_encoder = LabelEncoder()
        encoder_steps.append("LabelEncoder")
        unique_classes={}
        if y.dtypes==object or y.dtypes == 'category':
            if (["yes","no "] in y.unique()) and (len(y.unique())==2):
                unique_classes={"yes":1,"no":0}
            else:
                unique_classes=unique_elements(y.to_numpy())
            # Encode the categorical variable 'y'
            y= y_encoder.fit_transform(y)
            y = pd.DataFrame(y,columns=[target])
        elif is_categorical(y,df,id_dataset,target):
            unique_classes=unique_elements_for_numbers(y.to_numpy())
            
        # Identify the categorical columns in the dataset
        categorical_feature_mask = x.dtypes==object
        categorical_cols = x.columns[categorical_feature_mask].tolist()
        joblib.dump(categorical_cols,f"{session}/categorical_cols.joblib")
        joblib.dump(x.columns.tolist(),f"{session}/original_cols_list.joblib")
        
        # Encode the categorical columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            x[col] = encoder.fit_transform(x[col])
            filename=session+"/col_"+col+"_LabelEncoder.joblib"
            joblib.dump(encoder,filename)
            
        # Create a OneHotEncoder object
        ohe = OneHotEncoder(sparse=False)
        encoder_steps.append("OneHotEncoder")
        # Encode the categorical columns
        ohe_transformer = ohe.fit(x[categorical_cols])
        filename=f"{session}/ohe.joblib"
        joblib.dump(ohe,filename)
        x_encoded=ohe_transformer.transform(x[categorical_cols])
        x_encoded = np.concatenate((x_encoded, x.drop(categorical_cols, axis=1)), axis=1)
        if task=="classification":
            columns = ohe_transformer.get_feature_names_out(categorical_cols).tolist() + x.drop(categorical_cols, axis=1).columns.tolist()
            x_encoded = pd.DataFrame(x_encoded, columns=columns)
            return x_encoded , y,unique_classes,
        else:
            columns = ohe_transformer.get_feature_names_out(categorical_cols).tolist() + x.drop(categorical_cols, axis=1).columns.tolist()
            x_encoded = pd.DataFrame(x_encoded, columns=columns)
            return x_encoded , y
        
    elif task=="clustering":
        x= copy.deepcopy(df)
        # Identify the categorical columns in the dataset
        categorical_feature_mask = x.dtypes==object
        categorical_cols = x.columns[categorical_feature_mask].tolist()

        
        for col in categorical_cols:
            encoder = LabelEncoder()
            x[col] = encoder.fit_transform(x[col])
            filename=session+"/"+col+"_LabelEncoder.joblib"
            joblib.dump(encoder,filename)
            
        
        # Create a OneHotEncoder object
        ohe = OneHotEncoder(sparse=False)

        # Encode the categorical columns
        ohe_transformer = ohe.fit(x[categorical_cols])
        filename=f"{session}/ohe.joblib"
        joblib.dump(ohe,filename)
        x_encoded=ohe_transformer.transform(x[categorical_cols])
        x_encoded = np.concatenate((x_encoded, x.drop(categorical_cols, axis=1)), axis=1)
        columns = ohe_transformer.get_feature_names_out(categorical_cols).tolist() + x.drop(categorical_cols, axis=1).columns.tolist()
        x_encoded = pd.DataFrame(x_encoded, columns=columns)        
        return x_encoded

def scaler_clustering(session,x_train):
    global encoder_steps
    global data_preprocessing_method
    # Apply MinMaxScaler
    encoder_steps.append("MinMaxScaler")
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(x_train)
    filename=f"{session}/MinMaxScaler.joblib"
    joblib.dump(scaler,filename)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    # Apply Standardization
    encoder_steps.append("Standardization")
    sc_x = StandardScaler()
    sc_x = sc_x.fit(x_train)
    filename=f"{session}/StandardScaler.joblib"
    joblib.dump(sc_x,filename)
    x_train = pd.DataFrame(sc_x.transform(x_train), columns=x_train.columns)
    # Transform the data using the absolute value function
    x_train = np.abs(x_train)
    # Apply Normalization
    encoder_steps.append("Normalization")
    norm_x = Normalizer()
    norm_x = norm_x.fit(x_train)
    filename=f"{session}/Normalizer.joblib"
    joblib.dump(norm_x,filename)
    x_train = pd.DataFrame(norm_x.transform(x_train), columns=x_train.columns)
    encoder_steps = list(dict.fromkeys(encoder_steps))
    data_preprocessing_method["data_preprocessing"]=encoder_steps
    joblib.dump(data_preprocessing_method,f"{session}/feature_engineering_tracking.joblib")

    return x_train   

#you must split the data before use scaler function
def scaler(session,x_train, x_test):
    global encoder_steps
    global data_preprocessing_method

    # Apply Standardization
    encoder_steps.append("Standardization")
    sc_x = StandardScaler()
    sc_x = sc_x.fit(x_train)
    x_train = pd.DataFrame(sc_x.transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(sc_x.transform(x_test), columns=x_test.columns)
    filename=f"{session}/StandardScaler.joblib"
    joblib.dump(sc_x,filename)

    encoder_steps = list(dict.fromkeys(encoder_steps))
    data_preprocessing_method["data_preprocessing"]=encoder_steps
    
    joblib.dump(data_preprocessing_method,f"{session}/feature_engineering_tracking.joblib")
    return x_train, x_test